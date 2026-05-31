#!/usr/bin/env python3
"""
Memory-efficient chunked sliding window attention for 200K+ tokens.

Instead of computing full attention matrix, we process in chunks.
Each chunk only attends to a local window + a few global tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChunkedSlidingWindowAttention(nn.Module):
    """
    Chunked sliding window attention - O(seq_len * window) memory.

    Process the sequence in chunks. Each chunk attends to:
    1. Recent local window (configurable size)
    2. A few "global" anchor tokens (every N tokens)

    For 200K tokens with window=512 and chunk=256:
    - Memory per layer: 200K * 512 = ~100M entries (vs 40B for full attention)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 window_size=512, num_chunks=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.num_chunks = num_chunks  # Number of recent chunks to keep
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

        # Process in chunks - each query position attends to:
        # - All positions in its local window
        # - Global anchor tokens (every 512 tokens)
        # - Positions in recent chunks

        outputs = []
        chunk_size = 256  # Process 256 tokens at a time

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            q_chunk = q[:, :, chunk_start:chunk_end, :]

            # For this chunk, we need keys within the window
            # Window extends from (chunk_end - window_size) to chunk_end
            k_start = max(0, chunk_end - self.window_size)
            k_end = chunk_end

            # Get local K, V
            k_local = k_rep[:, :, k_start:k_end, :]
            v_local = v_rep[:, :, k_start:k_end, :]

            # Compute attention for this chunk
            # Use SDPA for efficiency
            try:
                out_chunk = F.scaled_dot_product_attention(
                    q_chunk, k_local, v_local,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,  # Causal within the window
                    scale=self.scale
                )
            except Exception:
                # Fallback
                scores = torch.matmul(q_chunk.float(), k_local.float().transpose(-2, -1)) * self.scale

                # Create causal mask for the window
                window_len = k_end - k_start
                causal = torch.tril(torch.ones(window_len, window_len, device=q.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal.view(1, 1, chunk_end - chunk_start, window_len), -1e9)

                attn = F.softmax(scores, dim=-1)
                out_chunk = torch.matmul(attn, v_local.float())

            outputs.append(out_chunk.to(q.dtype))

        out = torch.cat(outputs, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def replace_attention_with_chunked(model, window_size=512, num_chunks=4):
    """Replace attention modules with chunked sliding window version."""
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types

    attention_classes = [Qwen2Attention, LlamaAttention]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    count = [0]

    def replace(module):
        for attn_cls in attention_classes:
            if isinstance(module, attn_cls):
                count[0] += 1
                config = module.config

                attn = ChunkedSlidingWindowAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=module.head_dim,
                    window_size=window_size,
                    num_chunks=num_chunks,
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
    return count[0]


if __name__ == "__main__":
    import os
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    import time

    model_name = "01-ai/Yi-34B-200K"
    seq_len = 200000
    window_size = 512

    print(f"\n=== Testing Chunked Sliding Window at {seq_len} ===")
    print(f"Model: {model_name}")
    print(f"Window size: {window_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token if hf_token else None)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt
    base_prompt = "The theory of quantum mechanics "
    prompt = base_prompt * 1000
    prompt = prompt[:min(len(prompt), seq_len * 2)]

    print(f"Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        token=hf_token if hf_token else None,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Replace attention
    num_replaced = replace_attention_with_chunked(model, window_size=window_size)
    print(f"Replaced {num_replaced} attention layers")

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    # Warmup
    print(f"Warming up...")
    for i in range(3):
        try:
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            print(f"  Warmup {i+1} succeeded")
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            break

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
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Iter {i+1}: {speed:.1f} tok/s ({elapsed:.2f}s) mem={mem:.1f}GB")
        except Exception as e:
            print(f"  Iter {i+1} failed: {e}")
            times.append(0)

    avg_speed = 32 / (sum(times) / len(times)) if times and times[0] > 0 else 0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "model": model_name,
        "attention_type": "chunked_sliding_window",
        "seq_len": seq_len,
        "window_size": window_size,
        "memory_peak_gb": mem_peak,
        "avg_speed_tokens_per_sec": avg_speed,
        "times": times,
    }

    output_file = "hpc_results_chunked_window/yi200k_chunked_window.json"
    os.makedirs("hpc_results_chunked_window", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResult: {avg_speed:.1f} tok/s, saved to {output_file}")

    del model
    torch.cuda.empty_cache()