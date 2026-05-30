#!/usr/bin/env python3
"""Benchmark sparse on Qwen2.5-14B with different top-k and block sizes at 128K."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from minimax_m3_sparse_attention import MiniMaxSparseAttention
import types

model_name = "Qwen/Qwen2.5-14B"
seq_len = 131072  # 128K
num_tokens = 32

hf_token = os.environ.get('HF_TOKEN', '')
token = hf_token if hf_token else None

print(f"Loading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
    token=token,
)
print(f"Model loaded in {time.time()-t0:.1f}s")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
tokenizer.pad_token = tokenizer.eos_token

# Create 128K token prompt
base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
prompt = base_prompt * 30000
prompt = prompt[:seq_len * 2]

print(f"Prompt length: {len(prompt)} chars")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
actual_tokens = inputs['input_ids'].shape[1]
print(f"Input tokens: {actual_tokens}")

# Replace attention with sparse
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

# Test different configurations
configs = [
    (4, 16),   # top_k=4, block=16 (baseline)
    (8, 16),   # top_k=8, block=16
    (4, 32),   # top_k=4, block=32
    (8, 32),   # top_k=8, block=32
    (16, 16),  # top_k=16, block=16
]

os.makedirs("hpc_results_qwen14b_sweep", exist_ok=True)

for top_k, block_size in configs:
    print(f"\n=== Testing top_k={top_k}, block_size={block_size} ===")

    # Re-load model fresh for each config
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        token=token,
    )

    count = [0]
    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config
            attn = MiniMaxSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=module.head_dim,
                block_size=block_size,
                top_k_blocks=top_k,
                index_dim=32,
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
    print(f"Replaced {count[0]} attention layers")

    print("Warming up...")
    for i in range(2):
        try:
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            print(f"  Warmup {i+1} succeeded")
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
            break

    print("Benchmarking...")
    times = []
    for i in range(3):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
            elapsed = time.time() - t0
            speed = num_tokens / elapsed
            times.append(elapsed)
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Iter {i+1}: {speed:.1f} tok/s ({elapsed:.2f}s) mem={mem:.1f}GB")
        except Exception as e:
            print(f"  Iter {i+1} failed: {e}")
            times.append(0)

    avg_speed = num_tokens / (sum(times)/len(times)) if times and times[0] > 0 else 0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3

    result = {
        "model": model_name,
        "attention_type": "sparse",
        "seq_len": seq_len,
        "input_tokens": actual_tokens,
        "num_tokens": num_tokens,
        "top_k": top_k,
        "block_size": block_size,
        "memory_peak_gb": mem_peak,
        "avg_speed_tokens_per_sec": avg_speed,
        "times": times,
    }

    with open(f"hpc_results_qwen14b_sweep/sparse_k{top_k}_b{block_size}_128k.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  => {avg_speed:.1f} tok/s")

del model
torch.cuda.empty_cache()