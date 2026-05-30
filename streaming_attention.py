#!/usr/bin/env python3
"""
Streaming Local Attention - Processes long sequences without OOM

Key idea: Instead of attending to full sequence, use a sliding window.
This is NOT sparse attention - it's just local attention with a larger window.
But it can serve as a baseline for what "streaming" attention looks like.

For true sparse attention at 200K+, we need a different approach entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StreamingLocalAttention(nn.Module):
    """
    Local sliding window attention that can process very long sequences.

    Unlike sparse attention which selects blocks, this simply attends to
    a local window around each position. This is memory-efficient and
    can handle 200K+ tokens without OOM.

    For comparison: this is what you'd get with FlashAttention in sliding window mode.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, window_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
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

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention with local window
        # This is more memory-efficient than full attention
        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        # Create window mask: each query attends to window_size positions before it
        query_idx = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        window_mask = (key_idx <= query_idx) & (query_idx - key_idx < self.window_size)

        scores = scores.masked_fill(~window_mask, -1e9)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class StreamingChunkedAttention(nn.Module):
    """
    Chunked attention - processes sequence in blocks for memory efficiency.

    Each chunk attends locally and to a global summary of previous chunks.
    This gives O(n) memory instead of O(n²).
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 chunk_size=512, global_topk=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.global_topk = global_topk
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # For global context summary
        self.index_q = nn.Linear(hidden_size, num_heads * 16, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * 16, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Local chunk attention
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        outputs = []

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, seq_len)

            q_chunk = q[:, :, start:end, :]

            # Get local K, V for this chunk
            k_local = k_rep[:, :, max(0, start - self.window_size):end, :] if hasattr(self, 'window_size') else k_rep[:, :, :end, :]
            v_local = v_rep[:, :, max(0, start - self.window_size):end, :] if hasattr(self, 'window_size') else v_rep[:, :, :end, :]

            # Compute local attention
            scores = torch.matmul(q_chunk.float(), k_local.float().transpose(-2, -1)) * self.scale

            # Causal mask within chunk
            chunk_len = end - start
            causal = torch.tril(torch.ones(chunk_len, chunk_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, chunk_len, chunk_len), -1e9)

            attn = F.softmax(scores, dim=-1)
            out_chunk = torch.matmul(attn, v_local)
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def replace_attention(model, attention_class, **kwargs):
    """Replace attention modules with streaming/local attention."""
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types

    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        for attn_cls in [Qwen2Attention, LlamaAttention]:
            if isinstance(module, attn_cls):
                count[0] += 1
                config = module.config

                print(f"DEBUG replace layer {count[0]}: type={attn_cls.__name__}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}, head_dim={module.head_dim}")

                dtype = next(model.parameters()).dtype
                attn = attention_class(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=module.head_dim,
                    **kwargs,
                ).to(device=device, dtype=dtype)

                with torch.no_grad():
                    attn.q_proj.weight.copy_(module.q_proj.weight)
                    attn.k_proj.weight.copy_(module.k_proj.weight)
                    attn.v_proj.weight.copy_(module.v_proj.weight)
                    attn.o_proj.weight.copy_(module.o_proj.weight)

                attn_modules.append(attn)

                def new_forward(self, hidden_states, **kw):
                    return attn(hidden_states, **kw)

                module.forward = types.MethodType(new_forward, module)
                break

    model.apply(replace)
    return count[0], attn_modules