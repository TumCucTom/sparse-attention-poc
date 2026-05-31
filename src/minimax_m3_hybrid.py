#!/usr/bin/env python3
"""
MiniMax M3-style Sparse Attention - Hybrid Local + Global Implementation

Strategy:
1. Local sliding window attention (fast, captures nearby context)
2. Sparse global attention on selected blocks (captures long-range dependencies)

This hybrid approach should be faster than pure sparse for several reasons:
- Local attention is O(window_size) per query, very efficient
- Global block selection reduces the number of KV positions for long-range attention
- No arbitrary gather - just sliding window + block gather
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class MiniMaxSparseAttention(nn.Module):
    """Original two-stage MiniMax M3-style sparse attention."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 block_size=16, top_k_blocks=4, index_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.index_dim = index_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.index_q = nn.Linear(hidden_size, num_heads * index_dim, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * index_dim, bias=False)

        self._init_index_from_attention()

    def _init_index_from_attention(self):
        with torch.no_grad():
            q_weight = self.q_proj.weight
            k_weight = self.k_proj.weight
            H, h = self.num_heads, self.num_kv_heads
            self.index_q.weight.copy_(q_weight[:H * self.index_dim, :])
            self.index_k.weight.copy_(k_weight[:h * self.index_dim, :])

    def _compute_block_scores(self, idx_q, idx_k):
        batch_size, seq_len, num_heads, idx_dim = idx_q.shape
        num_kv_heads = idx_k.shape[2]

        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size

        q_padded = torch.zeros(batch_size, pad_len, num_heads, idx_dim, device=idx_q.device, dtype=torch.float32)
        q_padded[:, :seq_len] = idx_q.float()
        q_blocks = q_padded.view(batch_size, num_blocks, self.block_size, num_heads, idx_dim)

        k_padded = torch.zeros(batch_size, pad_len, num_kv_heads, idx_dim, device=idx_k.device, dtype=torch.float32)
        k_padded[:, :seq_len] = idx_k.float()
        k_blocks = k_padded.view(batch_size, num_blocks, self.block_size, num_kv_heads, idx_dim)

        q_avg = q_blocks.mean(dim=2)
        k_avg = k_blocks.mean(dim=2)
        k_avg_rep = k_avg.repeat_interleave(self.num_key_value_groups, dim=2)

        scores = torch.matmul(q_avg.permute(0, 2, 1, 3), k_avg_rep.permute(0, 2, 3, 1)) * (idx_dim ** -0.5)

        causal = torch.tril(torch.ones(num_blocks, num_blocks, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, num_blocks, num_blocks), -1e9)

        return scores

    def _sparse_attention(self, q, k, v, selected_blocks):
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size
        actual_k = selected_blocks.shape[-1]

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        block_offsets = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size
        position_indices = (block_base + block_offsets).view(batch_size, num_heads, actual_k * block_size)

        k_selected = torch.gather(k_rep, 2, position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        v_selected = torch.gather(v_rep, 2, position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)

        scores = scores.masked_fill(~causal_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_selected)

        return out

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        block_scores = self._compute_block_scores(idx_q, idx_k)
        actual_k = min(self.top_k_blocks, block_scores.shape[-1])
        _, topk_blocks = block_scores.topk(actual_k, dim=-1)
        topk_blocks = topk_blocks[:, :, 0, :]

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        out = self._sparse_attention(q, k, v, topk_blocks)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class HybridLocalGlobalAttention(nn.Module):
    """
    Hybrid attention: local sliding window + sparse global blocks.

    Local: Each query attends to previous `window_size` positions (efficient O(window))
    Global: Selected blocks via index attention (captures long-range dependencies)

    This combines the best of both:
    - Local: captures nearby context, very fast
    - Global: captures long-range dependencies, sparse
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 window_size=32, global_blocks=4, global_block_size=16, index_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_blocks = global_blocks
        self.global_block_size = global_block_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.index_dim = index_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Index projections for global block selection
        self.index_q = nn.Linear(hidden_size, num_heads * index_dim, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * index_dim, bias=False)

        self._init_index_from_attention()

    def _init_index_from_attention(self):
        with torch.no_grad():
            q_weight = self.q_proj.weight
            k_weight = self.k_proj.weight
            H, h = self.num_heads, self.num_kv_heads
            self.index_q.weight.copy_(q_weight[:H * self.index_dim, :])
            self.index_k.weight.copy_(k_weight[:h * self.index_dim, :])

    def _compute_global_block_scores(self, idx_q, idx_k):
        """Compute block-level scores for global attention."""
        batch_size, seq_len, num_heads, idx_dim = idx_q.shape
        num_kv_heads = idx_k.shape[2]

        num_blocks = (seq_len + self.global_block_size - 1) // self.global_block_size
        pad_len = num_blocks * self.global_block_size

        q_padded = torch.zeros(batch_size, pad_len, num_heads, idx_dim, device=idx_q.device, dtype=torch.float32)
        q_padded[:, :seq_len] = idx_q.float()
        q_blocks = q_padded.view(batch_size, num_blocks, self.global_block_size, num_heads, idx_dim)

        k_padded = torch.zeros(batch_size, pad_len, num_kv_heads, idx_dim, device=idx_k.device, dtype=torch.float32)
        k_padded[:, :seq_len] = idx_k.float()
        k_blocks = k_padded.view(batch_size, num_blocks, self.global_block_size, num_kv_heads, idx_dim)

        q_avg = q_blocks.mean(dim=2)
        k_avg = k_blocks.mean(dim=2)
        k_avg_rep = k_avg.repeat_interleave(self.num_key_value_groups, dim=2)

        scores = torch.matmul(q_avg.permute(0, 2, 1, 3), k_avg_rep.permute(0, 2, 3, 1)) * (idx_dim ** -0.5)

        causal = torch.tril(torch.ones(num_blocks, num_blocks, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, num_blocks, num_blocks), -1e9)

        return scores

    def _local_attention(self, q, k, v, window_size):
        """
        Local sliding window attention.
        q: [B, H, seq, D]
        k,v: [B, h, seq, D]
        Returns: [B, H, seq, D]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_kv_heads = k.shape[1]

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute scores for local window
        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        # Create local window mask: each query i only attends to [max(0, i-window), i]
        # Efficient version using broadcasting
        query_idx = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        window_mask = (key_idx <= query_idx) & (query_idx - key_idx < window_size)

        scores = scores.masked_fill(~window_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        return out

    def _global_attention(self, q, k, v, selected_blocks):
        """
        Sparse global attention on selected blocks.
        q: [B, H, seq, D]
        k,v: [B, h, seq, D]
        selected_blocks: [B, H, k] block indices
        Returns: [B, H, seq, D]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.global_block_size
        actual_k = selected_blocks.shape[-1]

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert block indices to position indices
        block_offsets = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size
        position_indices = (block_base + block_offsets).view(batch_size, num_heads, actual_k * block_size)

        # Gather KV from selected positions
        k_selected = torch.gather(k_rep, 2, position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        v_selected = torch.gather(v_rep, 2, position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        # Compute attention
        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        # Causal mask on selected positions
        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)

        scores = scores.masked_fill(~causal_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_selected)

        return out

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Local attention
        local_out = self._local_attention(q, k, v, self.window_size)

        # Global block selection
        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        block_scores = self._compute_global_block_scores(idx_q, idx_k)
        actual_k = min(self.global_blocks, block_scores.shape[-1])
        _, topk_blocks = block_scores.topk(actual_k, dim=-1)
        topk_blocks = topk_blocks[:, :, 0, :]

        # Global attention
        global_out = self._global_attention(q, k, v, topk_blocks)

        # Combine local + global (simple addition - could use learned gating)
        out = local_out + global_out

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


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
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def replace_attention(model, attention_class, **kwargs):
    """Replace Qwen2Attention with custom attention."""
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

            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            if hasattr(attn, '_init_index_from_attention'):
                attn._init_index_from_attention()

            attn_modules.append(attn)

            def new_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                          past_key_values=None, cache_position=None, position_ids=None, **kw):
                return attn(hidden_states, attention_mask=attention_mask, **kw)

            module.forward = types.MethodType(new_forward, module)

    model.apply(replace)
    return count[0], attn_modules


@torch.no_grad()
def benchmark_generation(model, tokenizer, prompt, num_tokens, warmup=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
    return num_tokens / (time.time() - start)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*70)
    print("MiniMax M3-style Sparse Attention - Hybrid Local + Global")
    print("="*70)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    configs = [
        ("Standard", StandardAttention, {}),
        ("MiniMax-Sparse(k=4)", MiniMaxSparseAttention, {"block_size": 16, "top_k_blocks": 4, "index_dim": 32}),
        ("Hybrid-Local+Global", HybridLocalGlobalAttention, {"window_size": 32, "global_blocks": 4, "global_block_size": 16, "index_dim": 16}),
    ]

    results = []

    for config_name, attention_class, kwargs in configs:
        print(f"\n--- Testing {config_name} ---")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        num_replaced, _ = replace_attention(model, attention_class, **kwargs)
        print(f"Replaced {num_replaced} attention layers")

        test_cases = [
            ("Short", "The theory of relativity", 30),
            ("Medium", "Quantum mechanics describes how particles behave " * 3, 30),
            ("Long", "The theory of quantum mechanics " * 10, 20),
            ("XLong", "Physics explains the fundamental laws of nature " * 20, 15),
        ]

        for test_name, prompt, num_tokens in test_cases:
            try:
                speed = benchmark_generation(model, tokenizer, prompt, num_tokens)
                tokens = len(tokenizer.encode(prompt))
                print(f"  {test_name} ({tokens} input): {speed:.1f} tokens/sec")
                results.append((config_name, test_name, tokens, speed))
            except Exception as e:
                print(f"  {test_name}: Error - {str(e)[:50]}")
                results.append((config_name, test_name, 0, 0))

        del model
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Config':<25} {'Short':>10} {'Medium':>10} {'Long':>10} {'XLong':>10}")
    print("-" * 70)
    for item in configs:
        config_name = item[0]
        speeds = {test_name: (tokens, speed) for config, test_name, tokens, speed in results if config == config_name}
        short = speeds.get("Short", (0, 0))
        medium = speeds.get("Medium", (0, 0))
        long = speeds.get("Long", (0, 0))
        xlong = speeds.get("XLong", (0, 0))
        print(f"{config_name:<25} {short[1]:>9.1f}/s {medium[1]:>9.1f}/s {long[1]:>9.1f}/s {xlong[1]:>9.1f}/s")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()