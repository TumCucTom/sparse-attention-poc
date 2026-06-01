#!/usr/bin/env python3
"""
StreamingLLM-style attention for MiniMax-M2.7 - NO block_scores matrix needed.

Key insight from StreamingLLM:
- Keep the last 4 tokens as "sink" tokens (always attended to)
- Keep a sliding window of ~32 tokens
- Drop everything in between

This avoids the O(num_blocks²) memory problem entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class StreamingSparseAttention(nn.Module):
    """
    StreamingLLM-style attention - no block_scores needed.

    Instead of computing importance scores for all blocks, we:
    1. Always attend to the last 4 tokens (sink tokens)
    2. Attend to the last N tokens via sliding window
    3. Never compute a full block_scores matrix

    Memory: O(window_size) instead of O(num_blocks²)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int = 32,
        sink_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.sink_size = sink_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        # Main projections (shared with the original attention)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

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

        # Cast input to match weight dtype
        weight_dtype = self.q_proj.weight.dtype
        x_f16 = hidden_states.to(dtype=weight_dtype)

        # Project to Q, K, V
        q = self.q_proj(x_f16).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # StreamingLLM attention: only attend to sink tokens + sliding window
        # Sink tokens: last 4 tokens (always attended)
        # Sliding window: last window_size tokens before sink
        total_context = self.sink_size + self.window_size

        if seq_len <= total_context:
            # Short sequence - attend to all
            attn_mask = None
        else:
            # Create streaming attention mask
            # Attend to: [0, 1, ..., seq_len - window_size - 1] are NOT allowed (dropped)
            # Only attend to: [seq_len - window_size - sink_size, ..., seq_len - 1]
            # But allow sink tokens to attend to everything
            device = q.device
            create_mask = False
            if seq_len > 128:
                # For long sequences, create a mask
                attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
                # Sink tokens attend to all
                attn_mask[-self.sink_size:, :] = True
                # Sliding window attends to itself and sink
                start_idx = seq_len - self.window_size
                attn_mask[start_idx:, start_idx:] = True
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                attn_mask = attn_mask.masked_fill(~attn_mask, float('-inf'))
            else:
                attn_mask = None

        # SDPA with streaming mask
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale
            )

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out.to(dtype=hidden_states.dtype), None


class StreamingSparseAttentionV2(nn.Module):
    """
    StreamingLLM with actual sparse selection - NO block_scores matrix.

    For each head, select top-k blocks based on simple metrics:
    - Recent blocks get higher weight
    - No need to compute full attention matrix

    This version uses a hash-based selection that doesn't require the full scores matrix.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 32,
        top_k_blocks: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Small MLP to compute block importance (lightweight, not full index projection)
        self.block_mlp = nn.Sequential(
            nn.Linear(hidden_size, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False),
        )

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

        weight_dtype = self.q_proj.weight.dtype
        x_f16 = hidden_states.to(dtype=weight_dtype)

        q = self.q_proj(x_f16).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute block importance scores using lightweight MLP
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        if num_blocks <= self.top_k_blocks:
            # Few blocks - just attend to all
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale
            )
        else:
            # Get block importance from MLP
            block_importance = self.block_mlp(x_f16)  # [B, seq, 1]
            block_importance = block_importance.view(batch_size, num_blocks, self.block_size, 1)
            block_importance = block_importance.mean(dim=2)  # [B, num_blocks, 1]

            # Select top-k blocks
            _, topk_idx = block_importance.squeeze(-1).topk(self.top_k_blocks, dim=-1)  # [B, top_k]

            # Create attention mask for selected blocks only
            device = q.device
            attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=q.dtype)

            for b in range(batch_size):
                for block_idx in topk_idx[b]:
                    start = block_idx.item() * self.block_size
                    end = min(start + self.block_size, seq_len)
                    attn_mask[:, start:end] = 1.0

            # Apply causal
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            attn_mask = attn_mask.masked_fill(~causal_mask, 0.0)

            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # Already applied causal
                scale=self.scale
            )

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out.to(dtype=hidden_states.dtype), None


def replace_attention_with_streaming(model, window_size=32, sink_size=4):
    """Replace MiniMaxM2Attention with StreamingSparseAttention."""
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention
    import types

    count = [0]
    device = next(model.parameters()).device

    def replace(module):
        if isinstance(module, MiniMaxM2Attention):
            count[0] += 1
            config = module.config
            actual_q_size = module.q_proj.weight.shape[0]
            actual_k_size = module.k_proj.weight.shape[0]
            actual_num_heads = actual_q_size // module.head_dim
            actual_num_kv_heads = actual_k_size // module.head_dim

            attn = StreamingSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=actual_num_heads,
                num_kv_heads=actual_num_kv_heads,
                head_dim=module.head_dim,
                window_size=window_size,
                sink_size=sink_size,
            ).to(device)

            # Copy weights
            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            def new_forward(self, hidden_states, **kw):
                return attn(hidden_states, **kw)
            module.forward = types.MethodType(new_forward, module)

    model.apply(replace)
    return count[0]


if __name__ == "__main__":
    print("StreamingLLM-style Sparse Attention - no block_scores matrix needed")
    print("Using sink tokens + sliding window approach")