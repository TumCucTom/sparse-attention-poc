#!/usr/bin/env python3
"""
Test StreamingLLM with fixed dtype handling
"""
import os
# Only set HF_TOKEN if it's not empty
hf_token = os.environ.get('HF_TOKEN', '')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from minimax_m3_exact import StreamingLLMAttention
import types
import time
import json

model_name = "01-ai/Yi-34B-200K"
sink_tokens = 4
local_window = 512
seq_len = 200000
output_file = "hpc_results_streamingllm_fixed/yi200k_sink4_local512_fixed.json"

print(f'Testing StreamingLLM with fixed dtype handling')
print(f'sink={sink_tokens} local_window={local_window} seq_len={seq_len}')

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=os.environ.get('HF_TOKEN'))
tokenizer.pad_token = tokenizer.eos_token

# Create prompt
base_prompt = 'The theory of quantum mechanics '
prompt = base_prompt * 1000
prompt = prompt[:min(len(prompt), seq_len * 2)]

print(f'Loading model...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={'0': 0},
    trust_remote_code=True,
    token=os.environ.get('HF_TOKEN'),
)
print(f'Model loaded in {time.time()-t0:.1f}s')

# Replace attention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

attention_classes = [Qwen2Attention, LlamaAttention]
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

print(f'Using device={device}, dtype={dtype}')

def replace(module):
    for attn_cls in attention_classes:
        if isinstance(module, attn_cls):
            config = module.config
            attn = StreamingLLMAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=module.head_dim,
                sink_tokens=sink_tokens,
                local_window=local_window,
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
print('Replaced attention layers')

# Prepare input
inputs = tokenizer(prompt, return_tensors='pt').to(device)

# Warmup
print('Warming up...')
for i in range(3):
    try:
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        print(f'  Warmup {i+1} succeeded')
    except Exception as e:
        print(f'  Warmup {i+1} error: {e}')
        import traceback
        traceback.print_exc()
        break

# Benchmark
print('Benchmarking...')
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
        print(f'  Iter {i+1}: {speed:.1f} tok/s ({elapsed:.2f}s) mem={mem:.1f}GB')
    except Exception as e:
        print(f'  Iter {i+1} failed: {e}')
        times.append(0)

avg_speed = 32 / (sum(times)/len(times)) if times and times[0] > 0 else 0
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

result = {
    'attention_type': 'streamingllm_fixed',
    'model': model_name,
    'sink_tokens': sink_tokens,
    'local_window': local_window,
    'memory_gb': mem_peak,
    'avg_speed_tokens_per_sec': avg_speed,
    'times': times,
}

import os
os.makedirs("hpc_results_streamingllm_fixed", exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)
print(f'Reached {avg_speed:.1f} tok/s, saved to {output_file}')

del model
torch.cuda.empty_cache()