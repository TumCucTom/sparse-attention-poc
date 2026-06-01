#!/usr/bin/env python3
"""Benchmark MiniMax-M2.7 with streaming (no block_scores) attention at 128K.

This tests the hypothesis: streaming attention without block_scores can work at 128K.
"""
import os
import subprocess
import importlib.util

# Patch rope_utils before importing transformers
_spec = importlib.util.find_spec("transformers.modeling_rope_utils")
_rope_file = _spec.origin if _spec else None
if _rope_file:
    subprocess.run(['sed', '-i', 's/if "dynamic" in rope_type:/if rope_type is not None and "dynamic" in rope_type:/', _rope_file], check=True)

class OutputRecorder:
    def __init__(self, module_class, index=0):
        self.module_class = module_class
        self.index = index
    def __repr__(self):
        return f"OutputRecorder({self.module_class.__name__}, index={self.index})"

import transformers.utils.generic
transformers.utils.generic.OutputRecorder = OutputRecorder

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if 'default' not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS['default'] = ROPE_INIT_FUNCTIONS['proportional']
ROPE_INIT_FUNCTIONS[None] = ROPE_INIT_FUNCTIONS['proportional']

from transformers.masking_utils import create_causal_mask as _orig_ccm
def create_causal_mask_patched(config, inputs_embeds, attention_mask=None,
 past_key_values=None, position_ids=None,
                                or_mask_function=None, and_mask_function=None,
                                block_sequence_ids=None, **kwargs):
    kwargs.pop('cache_position', None)
    return _orig_ccm(config, inputs_embeds, attention_mask, past_key_values,
                     position_ids, or_mask_function, and_mask_function, block_sequence_ids)
import transformers.masking_utils
transformers.masking_utils.create_causal_mask = create_causal_mask_patched
import transformers.models.minimax_m2.modeling_minimax_m2 as minimax_m2
minimax_m2.create_causal_mask = create_causal_mask_patched

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import time
import json
import types
from streaming_llm_sparse import StreamingSparseAttention

model_name = "MiniMaxAI/MiniMax-M2.7"

context_size = int(os.environ.get('CONTEXT_SIZE', '131072'))
num_tokens = int(os.environ.get('NUM_TOKENS', '32'))
attn_type = os.environ.get('ATTN_TYPE', 'dense')  # 'dense', 'streaming', 'streaming-v2'

hf_token = os.environ.get('HF_TOKEN', '')
token = hf_token if hf_token else None

print(f"Context size: {context_size} tokens ({context_size/1024:.0f}K)")
print(f"Attention type: {attn_type}")

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=token)
config.rope_parameters = {}
config.quantization_config = {"quant_method": "none"}

print(f"Loading model...")
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    token=token,
)
print(f"Model loaded in {time.time()-t0:.1f}s")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
tokenizer.pad_token = tokenizer.eos_token

prompt_char_count = context_size * 2
base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
full_prompt = (base_prompt * (prompt_char_count // len(base_prompt) + 1))[:prompt_char_count]

print(f"Prompt length: {len(full_prompt)} chars")
inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
actual_tokens = inputs['input_ids'].shape[1]
print(f"Input tokens: {actual_tokens}")

from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

count = [0]
replace_success = [False]

def replace_with_streaming(module):
    if isinstance(module, MiniMaxM2Attention):
        count[0] += 1
        config = module.config
        actual_q_size = module.q_proj.weight.shape[0]
        actual_k_size = module.k_proj.weight.shape[0]
        actual_num_heads = actual_q_size // module.head_dim
        actual_num_kv_heads = actual_k_size // module.head_dim
        if count[0] == 1:
            print(f"  Layer1: q_proj={module.q_proj.weight.shape}, head_dim={module.head_dim}")

        if attn_type == 'streaming':
            attn = StreamingSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=actual_num_heads,
                num_kv_heads=actual_num_kv_heads,
                head_dim=module.head_dim,
                window_size=32,
                sink_size=4,
            )
        else:
            print(f"Unknown attention type: {attn_type}")
            return False

        layer_device = next(module.parameters()).device if len(list(module.parameters())) > 0 else device
        torch.cuda.empty_cache()
        attn = attn.to(device=layer_device, dtype=dtype)
        torch.cuda.empty_cache()

        def new_forward(self, hidden_states, **kw):
            return attn(hidden_states, **kw)
        module.forward = types.MethodType(new_forward, module)
        return True
    return False

DO_REPLACE = (attn_type != 'dense')
if DO_REPLACE:
    try:
        model.apply(replace_with_streaming)
        print(f"Replaced {count[0]} attention layers with {attn_type}")
        replace_success[0] = True
    except Exception as e:
        print(f"Skipped replacement due to error: {e}")
        import traceback
        traceback.print_exc()
        replace_success[0] = False
else:
    print("TESTING DENSE ATTENTION")
    replace_success[0] = False

print("Warming up...")
for i in range(2):
    try:
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        print(f"  Warmup {i+1} succeeded")
    except Exception as e:
        print(f"  Warmup {i+1} failed: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"Benchmarking MiniMax-M2.7 {attn_type} at {context_size}...")
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
        import traceback
        traceback.print_exc()
        times.append(0)

avg_speed = num_tokens / (sum(times)/len(times)) if times and times[0] > 0 else 0
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

result = {
    "model": model_name,
    "attention_type": attn_type,
    "seq_len": context_size,
    "input_tokens": actual_tokens,
    "num_tokens": num_tokens,
    "memory_peak_gb": mem_peak,
    "avg_speed_tokens_per_sec": avg_speed,
    "times": times,
}

os.makedirs("hpc_results_minimax", exist_ok=True)
with open(f"hpc_results_minimax/{attn_type}_{context_size}.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nMiniMax-M2.7 {attn_type} at {context_size}: {avg_speed:.1f} tok/s")

del model
torch.cuda.empty_cache()