#!/usr/bin/env python3
"""
MiniMax M3-style Sparse Attention POC

Two-stage approach:
1. Index Stage: Lightweight attention to identify relevant blocks via BlockMaxPool + TopK
2. Sparse Stage: Main queries attend only to selected KV blocks

Key differences from our previous SubQ approach:
- Block-level aggregation (not token-level)
- Separate Index Q/K for routing
- TopK selection on block scores, not per-query token selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MiniMaxSparseAttention(nn.Module):
    """
    MiniMax M3-style sparse attention with two-stage routing.

    Stage 1 (Index): Idx Q, Idx KV → Index Attention → BlockMaxPool → TopK blocks
    Stage 2 (Sparse): Main Q → attend only to selected KV blocks
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,      # Tokens per block
        num_blocks: int = 256,     # Max blocks in sequence
        top_k_blocks: int = 16,    # Selected blocks per head
        index_q_dim: int = 64,     # Index query dimension
        index_kv_dim: int = 64,    # Index KV dimension
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.top_k_blocks = top_k_blocks
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        # Main projections (standard)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Index projections (for routing)
        self.index_q_proj = nn.Linear(hidden_size, num_heads * index_q_dim, bias=False)
        self.index_k_proj = nn.Linear(hidden_size, num_kv_heads * index_kv_dim, bias=False)
        self.index_v_proj = nn.Linear(hidden_size, num_kv_heads * index_kv_dim, bias=False)

        self.index_q_dim = index_q_dim
        self.index_kv_dim = index_kv_dim
        self.index_scale = index_kv_dim ** -0.5

        # Block pooling
        self.block_size = block_size

    def _compute_blocks(self, seq_len: int, device: torch.device) -> Tuple[int, int]:
        """Compute number of blocks and padding."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        return num_blocks, seq_len

    def _index_attention_block_scores(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute index attention scores at block level.
        idx_q: [batch, seq_len, num_heads, index_q_dim]
        idx_k: [batch, seq_len, num_kv_heads, index_kv_dim]
        Returns block-level scores after max pooling.
        """
        batch_size, seq_len, num_heads, _ = idx_q.shape
        num_kv_heads = idx_k.shape[2]

        # Reshape for block-level attention
        num_blocks, padded_len = self._compute_blocks(seq_len, idx_q.device)
        block_size = self.block_size

        # Pad sequences
        q_padded = torch.zeros(
            batch_size, padded_len, num_heads, self.index_q_dim,
            device=idx_q.device, dtype=idx_q.dtype
        )
        q_padded[:, :seq_len] = idx_q

        k_padded = torch.zeros(
            batch_size, padded_len, num_kv_heads, self.index_kv_dim,
            device=idx_k.device, dtype=idx_k.dtype
        )
        k_padded[:, :seq_len] = idx_k

        v_padded = torch.zeros(
            batch_size, padded_len, num_kv_heads, self.index_kv_dim,
            device=idx_v.device, dtype=idx_v.dtype
        )
        v_padded[:, :seq_len] = idx_v

        # Reshape to blocks: [batch, num_blocks, block_size, heads, dim]
        q_blocks = q_padded.view(batch_size, num_blocks, block_size, num_heads, self.index_q_dim)
        k_blocks = k_padded.view(batch_size, num_blocks, block_size, num_kv_heads, self.index_kv_dim)
        v_blocks = v_padded.view(batch_size, num_blocks, block_size, num_kv_heads, self.index_kv_dim)

        # Index attention: compute scores between all block pairs
        # q_blocks: [batch, num_blocks, block_size, num_heads, index_q_dim]
        # k_blocks: [batch, num_blocks, block_size, num_kv_heads, index_kv_dim]

        # We need to compute attention between query blocks and key blocks
        # For GQA: each query head group shares a KV head

        # Compute block-level attention scores
        # q_blocks: [B, nb, bs, H, d_q] -> transpose to [B, H, nb, bs, d_q]
        # k_blocks: [B, nb, bs, h, d_k] -> transpose to [B, h, nb, bs, d_k]

        q_b = q_blocks.permute(0, 3, 1, 2, 4)  # [B, H, nb, bs, d_q]
        k_b = k_blocks.permute(0, 3, 1, 2, 4)  # [B, h, nb, bs, d_k]

        # Repeat KV heads for query head groups
        k_b = k_b.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, H, nb, bs, d_k]

        # Compute attention scores per block pair
        # q_b[:, :, bi, :, :] @ k_b[:, :, bj, :, :].transpose(-2, -1)
        # Shape: [B, H, nb, bs, bs]

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.index_scale

        # Block Max Pool: take max over key block positions
        # scores: [B, H, nb_q, bs_q, nb_k, bs_k]
        # After max pool over bs_k: [B, H, nb_q, bs_q, nb_k]
        block_scores = scores.amax(dim=-1)  # Max over key block size

        # Then average over query block size
        # block_scores: [B, H, nb_q, bs_q, nb_k] -> mean over bs_q
        block_scores = block_scores.mean(dim=3)  # [B, H, nb_q, nb_k]

        return block_scores

    def _select_topk_blocks(
        self,
        block_scores: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Select top-k blocks for each head/query.
        block_scores: [batch, num_heads, num_query_blocks, num_key_blocks]
        Returns indices of selected blocks.
        """
        batch_size, num_heads, nb_q, nb_k = block_scores.shape
        actual_k = min(top_k, nb_k)

        # TopK selection per head per query block
        # We select blocks globally (same blocks for all query blocks)
        # Take top-k over the key dimension
        topk_scores, topk_indices = block_scores.topk(actual_k, dim=-1)
        return topk_indices  # [B, H, nb_q, k]

    def _sparse_attention_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Perform attention only on selected blocks - simplified gather/scatter approach.
        q: [batch, heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
        selected_blocks: [batch, heads, top_k] - block indices selected
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_kv_heads = k.shape[1]
        block_size = self.block_size
        actual_k = selected_blocks.shape[-1]

        # Repeat KV heads for GQA
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert block indices to position indices
        block_offsets = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size
        position_indices = block_base + block_offsets  # [B, H, top_k, bs]
        pos_flat = position_indices.view(batch_size, num_heads, actual_k * block_size)

        # Gather KV: [B, H, top_k*bs, d]
        k_selected = torch.gather(k_rep, 2, pos_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        v_selected = torch.gather(v_rep, 2, pos_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        # Compute attention on flattened selected positions
        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        # Create causal mask efficiently using vectorized operations
        # We need: query i can only attend to original positions <= i
        # position_indices[b, h, ki, bs] = original position
        # For query at position i, valid if any selected block*bs+offset <= i

        # Build causal mask using broadcasting
        # Create query positions [seq_len] and key positions [top_k * bs]
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
        k_pos = k_pos.expand(-1, -1, -1, seq_len).transpose(-2, -1)  # [B, H, seq, top_k*bs]
        # Actually simpler: compare query position to original key position
        # k_pos[b, h, j, i] = original position of key at flattened index j
        # We want mask[b, h, i, j] = True if k_pos[b, h, j] <= i

        k_pos_expanded = position_indices.permute(0, 1, 3, 2).reshape(batch_size, num_heads, block_size * actual_k, 1)
        q_pos_expanded = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)

        causal_mask = (k_pos_expanded <= q_pos_expanded)  # [B, H, top_k*bs, seq_q]
        causal_mask = causal_mask.transpose(-2, -1)  # [B, H, seq_q, top_k*bs]

        scores = scores.masked_fill(~causal_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_selected)

        # Now scatter back to full sequence - for each query position use the computed output
        # But we need to expand: query i attends to ALL selected positions, not just the right ones

        # Wait - our sparse attention means each query attends to all selected KV positions
        # The output needs to be scattered based on query position

        # Actually the attention output is per-query (seq_len outputs), not per-key
        # We already have out[b, h, i, :] which is the attention output for query i
        # This is dense in the sense every query position got attended to selected keys

        # We just need to map from [B, H, seq_q, top_k*bs] -> [B, H, seq_q, d]
        # But we already have the full seq_q dimension, so no scatter needed!

        return out

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard dense attention."""
        # Repeat KV heads for GQA
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        seq_len = q.shape[2]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # ===== Stage 1: Index Attention & Block Selection =====
        # Project to index dimensions
        idx_q = self.index_q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.index_q_dim
        )
        idx_k = self.index_k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.index_kv_dim
        )
        idx_v = self.index_v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.index_kv_dim
        )

        # Compute block-level attention scores
        block_scores = self._index_attention_block_scores(idx_q, idx_k, idx_v)

        # Select top-k blocks
        selected_blocks = self._select_topk_blocks(block_scores, self.top_k_blocks)

        # ===== Stage 2: Main Attention on Selected Blocks =====
        # Project main Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # For now, fallback to dense attention with collected block selection info
        # TODO: Implement proper sparse attention with selected blocks
        out = self._dense_attention(q, k, v, attention_mask)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class MiniMaxSparseAttentionSimplified(nn.Module):
    """
    Simplified MiniMax-style attention that actually works on MPS.

    Key insight: Use block-level routing, but implement sparse gather efficiently.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 256,
        top_k_blocks: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.top_k_blocks = top_k_blocks
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        # Main Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Index projections for routing
        self.index_q = nn.Linear(hidden_size, num_heads * 32, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * 32, bias=False)

    def _compute_block_scores(self, q_idx, k_idx):
        """
        Compute block-level attention scores.
        q_idx: [batch, seq_len, num_heads, idx_dim]
        k_idx: [batch, seq_len, num_kv_heads, idx_dim]
        """
        batch_size, seq_len, num_heads, idx_dim = q_idx.shape
        num_kv_heads = k_idx.shape[2]

        # Number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Pad sequences
        pad_len = num_blocks * self.block_size

        # Pad q_idx
        q_padded = torch.zeros(
            batch_size, pad_len, num_heads, idx_dim,
            device=q_idx.device, dtype=torch.float32
        )
        q_padded[:, :seq_len] = q_idx.float()

        # Pad k_idx
        k_padded = torch.zeros(
            batch_size, pad_len, num_kv_heads, idx_dim,
            device=k_idx.device, dtype=torch.float32
        )
        k_padded[:, :seq_len] = k_idx.float()

        # Reshape to blocks: [batch, num_blocks, block_size, heads, dim]
        q_blocks = q_padded.view(batch_size, num_blocks, self.block_size, num_heads, idx_dim)
        k_blocks = k_padded.view(batch_size, num_blocks, self.block_size, num_kv_heads, idx_dim)

        # Block attention: each query block attends to each key block
        # Use max pooling within blocks

        # Compute per-block scores via attention
        # q_blocks: [B, nb, bs, H, d]
        # k_blocks: [B, nb, bs, h, d]

        # Average over block size for query and key
        q_avg = q_blocks.mean(dim=2)  # [B, nb, H, d]
        k_avg = k_blocks.mean(dim=2)  # [B, nb, h, d]

        # For GQA, repeat KV heads
        k_avg_rep = k_avg.repeat_interleave(self.num_key_value_groups, dim=2)  # [B, nb, H, d]

        # Compute block-level scores: [B, H, nb_q, nb_k]
        scores = torch.matmul(q_avg.permute(0, 2, 1, 3), k_avg_rep.permute(0, 2, 1, 3).transpose(-2, -1)) * (idx_dim ** -0.5)

        # Apply causal mask at block level
        causal = torch.tril(torch.ones(num_blocks, num_blocks, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, num_blocks, num_blocks), -1e9)

        return scores

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
    ):
        """
        Apply attention only to selected blocks.
        q: [batch, heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
        selected_blocks: [batch, heads, num_blocks] boolean or indices
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_kv_heads = k.shape[1]
        block_size = self.block_size
        num_blocks = (seq_len + block_size - 1) // block_size

        # Get actual top_k
        actual_k = min(self.top_k_blocks, selected_blocks.shape[-1])

        # Use selected blocks to gather KV
        # selected_blocks: [B, H, k] - indices of selected blocks

        # Reshape Q to blocks
        q_padded = q.permute(0, 1, 2, 3)  # [B, H, seq, d]
        pad_len = num_blocks * block_size
        q_padded = torch.zeros(batch_size, num_heads, pad_len, head_dim, device=q.device, dtype=q.dtype)
        q_padded[:, :, :seq_len] = q

        # Reshape Q to [B, H, nb, bs, d]
        q_blocks = q_padded.view(batch_size, num_heads, num_blocks, block_size, head_dim)

        # Expand selected blocks to positions
        # selected_blocks: [B, H, k] -> [B, H, k, 1] -> [B, H, k, bs]
        block_pos = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size)
        selected_pos = selected_blocks.unsqueeze(-1) * block_size + torch.arange(block_size, device=q.device)

        # For each head, compute attention to selected blocks
        # For simplicity, gather selected KV and compute dense attention

        # Flatten selected positions: [B, H, k*bs]
        selected_pos_flat = selected_pos.view(batch_size, num_heads, actual_k * block_size)

        # Gather K, V
        # k: [B, num_kv_heads, seq, d] -> need to select positions
        # Expand for key value groups
        k_expanded = k.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, H, seq, d]
        v_expanded = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Gather: for each head, gather selected positions
        k_selected_list = []
        v_selected_list = []
        for b in range(batch_size):
            k_head_list = []
            v_head_list = []
            for h in range(num_heads):
                pos = selected_pos_flat[b, h]  # [k*bs]
                k_sel = k_expanded[b, h, pos]  # [k*bs, d]
                v_sel = v_expanded[b, h, pos]
                k_head_list.append(k_sel)
                v_head_list.append(v_sel)
            k_selected_list.append(torch.stack(k_head_list, dim=0))  # [H, k*bs, d]
            v_selected_list.append(torch.stack(v_head_list, dim=0))

        k_selected = torch.stack(k_selected_list, dim=0)  # [B, H, k*bs, d]
        v_selected = torch.stack(v_selected_list, dim=0)

        # Reshape Q for attention: [B, H, nb, bs, d] -> [B, H, nb*bs, d]
        q_flat = q_blocks.reshape(batch_size, num_heads, num_blocks * block_size, head_dim)

        # Compute attention between Q (all positions) and selected K,V
        # Q: [B, H, seq_len, d], K: [B, H, k*bs, d], V: [B, H, k*bs, d]
        # But we need to respect causal - only attend to past selected positions

        # For each query position i, only attend to selected positions <= i
        # Simplify: do dense attention over selected, then scatter back

        # Actually, let's just do dense over selected and scatter
        # This is approximate but captures the block selection idea

        seq_len_actual = seq_len
        k_s = k_selected[:, :, :actual_k * block_size, :]
        v_s = v_selected[:, :, :actual_k * block_size, :]

        # Compute full scores then mask by causal + block selection
        scores = torch.matmul(q_flat.float(), k_s.float().transpose(-2, -1)) * self.scale

        # Create mask: only attend to positions in selected blocks
        # selected_pos: [B, H, k, bs] -> can convert to mask
        mask = torch.zeros(batch_size, num_heads, seq_len_actual, actual_k * block_size, device=q.device)
        for b in range(batch_size):
            for h in range(num_heads):
                for ki in range(actual_k):
                    start_pos = ki * block_size
                    end_pos = start_pos + block_size
                    mask[b, h, :, start_pos:end_pos] = 1.0

        scores = scores.masked_fill(mask == 0, -1e9)

        # Causal mask on original positions
        # We need to mask based on original position, not selected position
        # This is complex - fallback to dense for now

        attn = F.softmax(scores[:, :, :seq_len_actual, :actual_k * block_size], dim=-1)
        out = torch.matmul(attn, v_s[:, :, :seq_len_actual, :actual_k * block_size])

        # Reshape output back
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Stage 1: Compute index and select blocks (for logging/speed comparison)
        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, -1)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, -1)

        block_scores = self._compute_block_scores(idx_q, idx_k)

        # TopK selection per head - same selection for all query blocks
        actual_k = min(self.top_k_blocks, block_scores.shape[-1])
        _, topk_blocks = block_scores.topk(actual_k, dim=-1)  # [B, H, nb, k]
        # Use first block's selection for simplicity (global selection)
        topk_blocks = topk_blocks[:, :, 0, :]  # [B, H, k]

        # Stage 2: Main Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # For now, use dense attention (sparse gather is still too slow on MPS)
        # The block selection is computed but not used for KV selection
        out = self._dense_fallback(q, k, v, attention_mask)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None

    def _dense_fallback(self, q, k, v, attention_mask):
        """Fallback to dense attention."""
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        seq_len = q.shape[2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)
        return out


def replace_attention_with_minimax(model, attention_class=MiniMaxSparseAttentionSimplified, **kwargs):
    """Replace Qwen2Attention with MiniMax-style sparse attention."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types

    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config

            attn = attention_class(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=module.head_dim,
                **kwargs,
            ).to(device)

            # Copy pretrained weights
            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            attn_modules.append(attn)

            def new_forward(
                self,
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                position_ids=None,
                **kw,
            ):
                return attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    **kw,
                )

            module.forward = types.MethodType(new_forward, module)

    model.apply(replace)
    return count[0], attn_modules


def benchmark_minimax_attention():
    """Benchmark MiniMax-style attention against standard attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    print("\n" + "="*60)
    print("MiniMax M3-style Sparse Attention Benchmark")
    print("="*60)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with standard attention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Replace with MiniMax-style attention
    num_replaced, attn_modules = replace_attention_with_minimax(
        model,
        MiniMaxSparseAttentionSimplified,
        block_size=16,
        top_k_blocks=16,
    )
    print(f"Replaced {num_replaced} attention layers with MiniMax-style")

    # Test prompt
    prompt = "The theory of relativity describes how"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)

    # Benchmark
    num_tokens = 30
    with torch.no_grad():
        start = time.time()
        _ = model.generate(**inputs, max_new_tokens=num_tokens)
        elapsed = time.time() - start

    speed = num_tokens / elapsed
    print(f"Speed: {speed:.1f} tokens/sec")

    # Test with longer context
    print("\n--- Context Scaling ---")
    for multiplier in [1, 5, 10]:
        long_prompt = ("Quantum mechanics " * multiplier)[:100]
        inputs = tokenizer(long_prompt, return_tensors="pt").to(next(model.parameters()).device)

        with torch.no_grad():
            start = time.time()
            try:
                _ = model.generate(**inputs, max_new_tokens=10)
                elapsed = time.time() - start
                tokens = len(tokenizer.encode(long_prompt))
                print(f"  {tokens} tokens: {10/elapsed:.1f} t/s")
            except Exception as e:
                print(f"  {multiplier}x: OOM or error")

    print("\n" + "="*60)
    print("Note: Full sparse attention implementation pending.")
    print("Current fallback uses block-level scoring but still does dense attention.")
    print("="*60)


if __name__ == "__main__":
    benchmark_minimax_attention()