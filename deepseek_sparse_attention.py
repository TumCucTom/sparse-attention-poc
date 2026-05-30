#!/usr/bin/env python3
"""
DeepSeek Sparse Attention (DSA) Implementation

Based on the article description:
- Uses MLA (Multi-head Latent Attention) instead of GQA
- Has a compression stage before sparse selection
- Block-level selection similar to CSA but operates on real KV
- Differs from M3 in that it uses compression + attention on compressed representation

Key differences from M3:
1. MLA instead of GQA - uses low-rank KV compression
2. Attention selection happens on compressed latent space
3. Decoupling of compression and selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek-style Sparse Attention with:
    1. Low-rank KV compression (MLA style)
    2. Sparse selection on compressed representation
    3. Attention on selected compressed KV blocks
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        top_k_blocks: int = 4,
        compression_dim: int = 64,  # Low-rank dimension for KV compression
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.compression_dim = compression_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        # Main projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # MLA-style low-rank KV compression
        # Instead of full K/V, we compress to low-rank latent vectors
        self.k_compress = nn.Linear(num_kv_heads * head_dim, compression_dim, bias=False)
        self.v_compress = nn.Linear(num_kv_heads * head_dim, compression_dim, bias=False)

        # Query compression for selection (to match compressed KV dimension)
        self.q_compress = nn.Linear(num_heads * head_dim, compression_dim, bias=False)

        # Selection network: learn which compressed blocks to attend to
        self.selection_net = nn.Sequential(
            nn.Linear(compression_dim, compression_dim),
            nn.GELU(),
            nn.Linear(compression_dim, 1)
        )

    def _compress_kv(self, k_flat: torch.Tensor, v_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV using low-rank projection.
        k_flat: [batch, seq_len, num_kv_heads * head_dim] - already flattened
        v_flat: [batch, seq_len, num_kv_heads * head_dim] - already flattened
        Returns: compressed_k, compressed_v: [batch, seq_len, compression_dim]
        """
        # Compress to low-rank
        k_compressed = self.k_compress(k_flat)  # [B, seq_len, compression_dim]
        v_compressed = self.v_compress(v_flat)  # [B, seq_len, compression_dim]

        return k_compressed, v_compressed

    def _select_blocks(
        self,
        q_compressed: torch.Tensor,
        k_compressed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select top-k blocks based on selection network scores.
        q_compressed: [batch, seq_len, compression_dim]
        k_compressed: [batch, seq_len, compression_dim]
        Returns: selected block indices [batch, num_heads, top_k]
        """
        batch_size, seq_len, comp_dim = q_compressed.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Pad sequences to block boundary
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len > 0:
            # Pad the sequence dimension (dim 1) with zeros
            q_padded = torch.zeros(batch_size, num_blocks * self.block_size, comp_dim, device=q_compressed.device, dtype=q_compressed.dtype)
            q_padded[:, :seq_len, :] = q_compressed
            k_padded = torch.zeros(batch_size, num_blocks * self.block_size, comp_dim, device=k_compressed.device, dtype=k_compressed.dtype)
            k_padded[:, :seq_len, :] = k_compressed
        else:
            q_padded = q_compressed
            k_padded = k_compressed

        # Reshape to blocks: [B, num_blocks, block_size, comp_dim]
        q_blocks = q_padded.view(batch_size, num_blocks, self.block_size, comp_dim)
        k_blocks = k_padded.view(batch_size, num_blocks, self.block_size, comp_dim)

        # Average pool blocks: [B, num_blocks, comp_dim]
        q_block_avg = q_blocks.mean(dim=2)  # Average over sequence within block
        k_block_avg = k_blocks.mean(dim=2)

        # Expand query blocks to match all key blocks for scoring
        # q_block_avg: [B, num_blocks, comp_dim] -> [B, num_blocks, 1, comp_dim]
        # scores computed between each query block and each key block

        # Compute selection scores using learned network
        # For each query block, compute scores against all key blocks
        q_selected = q_block_avg.unsqueeze(2)  # [B, num_blocks, 1, comp_dim]
        k_expanded = k_block_avg.unsqueeze(1)  # [B, 1, num_blocks, comp_dim]

        # Combine and compute selection scores
        combined = q_selected + k_expanded  # [B, num_blocks, num_blocks, comp_dim]
        scores = self.selection_net(combined).squeeze(-1)  # [B, num_blocks, num_blocks]

        # Clamp scores to prevent fp16 overflow
        scores = scores.float().clamp(-1e4, 1e4)

        # Apply causal mask (each query block can only attend to previous key blocks)
        causal_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, device=scores.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask.view(1, num_blocks, num_blocks), -1e9)

        # Get top-k blocks for the first query block (used globally)
        _, topk_blocks = scores[:, 0, :].topk(self.top_k_blocks, dim=-1)

        # Debug
        print(f"DEBUG _select_blocks: seq_len={seq_len}, num_blocks={num_blocks}, topk_blocks={topk_blocks}, max={topk_blocks.max().item()}, min={topk_blocks.min().item()}")

        # Return [batch, top_k] - same blocks for all heads
        return topk_blocks

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse attention on selected KV blocks.
        q: [batch, num_heads, seq_len, head_dim]
        k/v: [batch, num_kv_heads, seq_len, head_dim]
        selected_blocks: [batch, top_k] - block indices
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size
        num_kv_heads = k.shape[1]
        actual_k = selected_blocks.shape[-1]

        # Repeat KV heads for GQA
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert block indices to position indices
        # selected_blocks: [batch, top_k] - same blocks for all heads
        block_offsets = torch.arange(block_size, device=q.device, dtype=torch.long).view(1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size  # [batch, 1, top_k]
        position_indices = (block_base + block_offsets).view(batch_size, 1, actual_k * block_size)  # [batch, 1, top_k*block]
        position_indices = position_indices.expand(-1, num_heads, -1)  # [batch, num_heads, top_k*block]

        # Clamp to valid range
        position_indices = position_indices.clamp(0, seq_len - 1)

        # Expand position indices for gather
        gather_idx = position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, top_k*block, head_dim]
        k_selected = torch.gather(k_rep, 2, gather_idx)  # [B, H, top_k*block, head_dim]
        v_selected = torch.gather(v_rep, 2, gather_idx)

        # Compute attention
        q_f = q.float()
        k_f = k_selected.float()
        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * self.scale

        # Causal mask
        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)
        scores = scores.masked_fill(~causal_mask, -1e9)

        # Softmax
        attn = F.softmax(scores, dim=-1)
        v_f = v_selected.float()
        out = torch.matmul(attn, v_f)

        return out.to(dtype=q.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Cast to match weight dtype
        weight_dtype = self.q_proj.weight.dtype
        x = hidden_states.to(dtype=weight_dtype)

        # Main QKV projections (before permute for _compress_kv)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compress for selection (use unpermuted format)
        q_reshaped = q.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        k_reshaped = k.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        v_reshaped = v.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        k_compressed, v_compressed = self._compress_kv(k_reshaped, v_reshaped)
        q_compressed = self.q_compress(q_reshaped)

        # Select blocks
        selected_blocks = self._select_blocks(q_compressed, k_compressed)

        # Sparse attention
        out = self._sparse_attention(q, k, v, selected_blocks)

        # Reshape and output
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.o_proj(out)

        # Return 3 values to match Qwen2Attention.forward signature
        return out.to(dtype=hidden_states.dtype), None, past_key_value


class StandardAttention(nn.Module):
    """Standard dense attention for comparison."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        seq_len_q = q.shape[2]
        causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_q, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len_q, seq_len_q), -1e9)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep.float())

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


if __name__ == "__main__":
    # Test DSA module
    print("Testing DeepSeek Sparse Attention...")

    batch_size, seq_len = 1, 128
    hidden_size = 512
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64

    attn = DeepSeekSparseAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=16,
        top_k_blocks=4,
        compression_dim=64,
    )

    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    output, _ = attn(x)
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")
    print("DSA test passed!")