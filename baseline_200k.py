#!/usr/bin/env python3
"""Simple dense baseline benchmark at 200K context."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json

model_name = "01-ai/Yi-34B-200K"
seq_len = 200000

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

# Create prompt
base_prompt = "The theory of quantum mechanics "
prompt = base_prompt * 1000
prompt = prompt[:min(len(prompt), seq_len * 2)]

print(f"Prompt length: {len(prompt)} chars")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

print("Warming up...")
for i in range(3):
    try:
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        print(f"  Warmup {i+1} succeeded")
    except Exception as e:
        print(f"  Warmup {i+1} failed: {e}")
        import traceback
        traceback.print_exc()
        break

print("Benchmarking...")
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

avg_speed = 32 / (sum(times)/len(times)) if times and times[0] > 0 else 0
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

result = {
    "model": model_name,
    "attention_type": "dense_baseline",
    "seq_len": seq_len,
    "memory_peak_gb": mem_peak,
    "avg_speed_tokens_per_sec": avg_speed,
    "times": times,
}

os.makedirs("hpc_results_baseline", exist_ok=True)
with open("hpc_results_baseline/dense_200k_baseline.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResult: {avg_speed:.1f} tok/s")
print(f"Saved to hpc_results_baseline/dense_200k_baseline.json")

del model
torch.cuda.empty_cache()