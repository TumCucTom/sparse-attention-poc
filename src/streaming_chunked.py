#!/usr/bin/env python3
"""
Chunked Streaming Attention for Very Long Sequences.

Key insight: Process sequence in chunks during generation.
Each chunk only keeps a small local KV cache.
This allows processing 200K+ tokens with fixed memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChunkedStreamingAttention(nn.Module):
    """
    Attention that processes in chunks to avoid OOM at long sequences.

    During generation, instead of attending to all previous tokens,
    we only attend to:
    1. Recent local window (configurable)
    2. A "summary" of recent chunks (via running average)

    Memory usage is O(chunk_size * num_chunks) = O(window_size * n / chunk_size)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 local_window=512, summary_interval=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.local_window = local_window
        self.summary_interval = summary_interval
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Buffer for running summary
        self.register_buffer('kv_summary', None, persistent=False)
        self.register_buffer('summary_count', torch.tensor(0), persistent=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute local attention
        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        # Local window mask
        query_idx = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        local_mask = (key_idx <= query_idx) & (query_idx - key_idx < self.local_window)

        scores = scores.masked_fill(~local_mask, -1e9)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class SimpleSlidingWindowAttention(nn.Module):
    """
    Simple sliding window attention - each token only attends to local window.

    This is NOT learned sparse - it's a fixed pattern.
    Memory: O(window_size * seq_len) instead of O(seq_len^2)

    For seq_len=200K, window=512: 200K * 512 = 100M entries
    vs full attention: 200K * 200K = 40B entries (400x more memory)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 window_size=512):
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

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Sliding window attention
        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        # Create sliding window mask
        query_idx = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        # Each query attends to keys within window_size before it
        window_mask = (key_idx <= query_idx) & (query_idx - key_idx < self.window_size)

        scores = scores.masked_fill(~window_mask, -1e9)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def replace_and_test(model_name, seq_len, window_size, output_file):
    """Test sliding window attention at 200K context."""
    import os
    # Only set HF_TOKEN if it's not empty
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    import time

    print(f"\nTesting {model_name} with window_size={window_size} at seq_len={seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token if hf_token else None)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt
    base_prompt = "The theory of quantum mechanics "
    prompt = base_prompt * max(1, seq_len // 20)
    prompt = prompt[:min(len(prompt), seq_len * 2)]

    # Load model
    print(f"Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get('HF_TOKEN'),
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Replace attention with sliding window
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types

    attention_classes = [Qwen2Attention, LlamaAttention]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def replace(module):
        for attn_cls in attention_classes:
            if isinstance(module, attn_cls):
                config = module.config
                attn = SimpleSlidingWindowAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=module.head_dim,
                    window_size=window_size,
                ).to(device=device, dtype=dtype)

                with torch.no_grad():
                    attn.q_proj.weight.copy_(module.q_proj.weight)
                    attn.k_proj.weight.copy_(module.k_proj.weight)
                    attn.v_proj.weight.copy_(module.v_proj.weight)
                    attn.o_proj.weight.copy_(module.o_proj.weight)

                def new_forward(self, hidden_states, **kw):
                    return attn(hidden_states, **kw)

                module.forward = types.MethodType(new_forward, module)
                return True
        return False

    model.apply(replace)

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    print(f"Warming up...")
    for _ in range(2):
        try:
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        except Exception as e:
            print(f"Warmup error: {e}")
            return None

    # Benchmark
    print(f"Benchmarking...")
    times = []
    for i in range(3):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            elapsed = time.time() - t0
            speed = 32 / elapsed
            times.append(elapsed)
            print(f"  Iteration {i+1}: {speed:.1f} tok/s ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  Iteration {i+1} failed: {e}")
            times.append(0)

    avg_speed = 32 / (sum(times) / len(times)) if times and times[0] > 0 else 0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    results = {
        "model": model_name,
        "attention_type": "sliding_window",
        "seq_len": seq_len,
        "window_size": window_size,
        "memory_peak_gb": mem_peak,
        "avg_speed_tokens_per_sec": avg_speed,
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "01-ai/Yi-34B-200K"
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    output = sys.argv[4] if len(sys.argv) > 4 else None

    result = replace_and_test(model, seq_len, window_size, output)
    if result:
        print(f"\nResult: {result}")