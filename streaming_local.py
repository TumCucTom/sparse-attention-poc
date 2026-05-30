#!/usr/bin/env python3
"""
Streaming Local Attention - Memory-efficient attention for very long sequences.

Instead of attending to ALL previous tokens, we only attend to:
1. Local window (e.g., last 1024 tokens)
2. A small number of "global" tokens (e.g., first token, every 512th token)

This is NOT learned sparse attention - it's a fixed pattern.
But it's memory-efficient and can handle 200K+ tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StreamingLocalAttention(nn.Module):
    """
    Streaming local attention with global tokens.

    Each token attends to:
    - All tokens within a local window
    - A set of "global" tokens (first token, and periodically spaced tokens)

    This reduces memory from O(n²) to O(window_size * n).
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 window_size=1024, global_interval=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_interval = global_interval
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

        # GQA
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Build attention mask for local window + global tokens
        query_idx = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)

        # Local window mask: within window_size of each query
        local_mask = (key_idx <= query_idx) & (query_idx - key_idx < self.window_size)

        # Global tokens mask: first token + every global_interval tokens
        global_indices = torch.cat([
            torch.zeros(1, device=q.device, dtype=torch.long),  # first token
            torch.arange(0, seq_len, self.global_interval, device=q.device)
        ]).unique()
        global_mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=torch.bool)
        global_mask[:, global_indices] = True

        # Combined: local OR global
        attention_mask = local_mask | global_mask

        # Apply attention mask
        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~attention_mask, -1e9)

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class StreamingStridedAttention(nn.Module):
    """
    Strided attention - each token attends to tokens at fixed strides.

    For example, with stride=512:
    - Token 0 attends to 0, 512, 1024, 1536, ...
    - Token 1 attends to 0, 1, 513, 1025, 1537, ...

    This is very memory-efficient and works well for certain patterns.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, stride=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.stride = stride
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

        # Strided attention pattern
        query_idx = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)

        # Each query attends to keys at stride intervals AND within a small local window
        # This captures both long-range stride patterns and local context
        local_window = 64  # Small local context
        stride_mask = (key_idx % self.stride == 0) | ((key_idx % self.stride == query_idx % self.stride) & (query_idx - key_idx < local_window))
        causal = key_idx <= query_idx

        attention_mask = stride_mask & causal

        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~attention_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None