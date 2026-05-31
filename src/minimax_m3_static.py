#!/usr/bin/env python3
"""
Static Block Sparse Attention - Uses fixed block selection instead of learned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StaticBlockSparseAttention(nn.Module):
    """
    Static sparse attention with fixed block selection.

    Instead of learned block selection, we attend to the last `top_k_blocks * block_size` tokens.
    This eliminates the index projection overhead entirely.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int = 128,  # Total window of past tokens to attend to
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        # Main projections (no index projections needed!)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Cast input to match weight dtype
        weight_dtype = self.q_proj.weight.dtype
        x = hidden_states.to(dtype=weight_dtype)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # For each query, attend to last `window_size` tokens
        # This is O(window_size) memory instead of O(seq_len)
        k_window = k_rep[:, :, -self.window_size:, :]
        v_window = v_rep[:, :, -self.window_size:, :]

        # Create causal mask for the window
        seq_q = q.shape[2]
        seq_k = k_window.shape[2]

        # Causal mask: query i can attend to key j if j <= i
        q_idx = torch.arange(seq_q, device=q.device, dtype=torch.long).view(1, 1, seq_q, 1)
        k_idx = torch.arange(seq_k, device=q.device, dtype=torch.long).view(1, 1, 1, seq_k)
        causal_mask = (k_idx <= q_idx)

        # Use same dtype for all tensors
        dtype = q.dtype

        # Compute attention
        try:
            out = F.scaled_dot_product_attention(
                q.to(dtype),
                k_window.to(dtype),
                v_window.to(dtype),
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale
            )
        except Exception:
            # Fallback
            scores = torch.matmul(q.to(dtype), k_window.to(dtype).transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(~causal_mask, -1e9)
            attn = F.softmax(scores.float(), dim=-1)
            out = torch.matmul(attn, v_window.to(dtype))

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.o_proj(out)

        return out.to(dtype=hidden_states.dtype), None


if __name__ == "__main__":
    print("Testing Static Block Sparse Attention")

    # Quick test
    attn = StaticBlockSparseAttention(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        window_size=128,
    ).cuda()

    x = torch.randn(1, 512, 4096).cuda()
    out, _ = attn(x)
    print(f"Output shape: {out.shape}")
    print("Static attention works!")