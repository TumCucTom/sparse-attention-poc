#!/usr/bin/env python3
"""Benchmark MiniMax-M2.7 with sparse attention."""
import os
import sys
import subprocess
import importlib.util

# CRITICAL: Patch the rope_utils file BEFORE importing transformers
# The rope_type can be None causing "if 'dynamic' in rope_type" to fail
# Find the module spec without importing it
_spec = importlib.util.find_spec("transformers.modeling_rope_utils")
_rope_file = _spec.origin if _spec else None
if _rope_file:
    subprocess.run(['sed', '-i', 's/if "dynamic" in rope_type:/if rope_type is not None and "dynamic" in rope_type:/', _rope_file], check=True)

# Patch OutputRecorder and rope init BEFORE importing the model
# Our venv has transformers 5.9.0 but MiniMax-M2.7 needs newer features

# Patch 1: OutputRecorder stub (not in transformers 5.9.0)
class OutputRecorder:
    """Stub for OutputRecorder - records a specific output index from a module."""
    def __init__(self, module_class, index=0):
        self.module_class = module_class
        self.index = index

    def __repr__(self):
        return f"OutputRecorder({self.module_class.__name__}, index={self.index})"

import transformers.utils.generic
transformers.utils.generic.OutputRecorder = OutputRecorder

# Patch 2: Add 'default' and None rope types
# The MiniMax M2 model:
#   - Sets rope_type='default' when config.rope_scaling is None
#   - Sets rope_type=None when config.rope_scaling is a dict but lacks 'rope_type'/'type' keys
# ROPE_INIT_FUNCTIONS['linear'] requires a 'factor' key - use 'proportional' instead
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if 'default' not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS['default'] = ROPE_INIT_FUNCTIONS['proportional']
ROPE_INIT_FUNCTIONS[None] = ROPE_INIT_FUNCTIONS['proportional']

# Patch 3: create_causal_mask in MiniMax M2 passes cache_position which newer
# transformers versions don't accept. Wrap it to strip that kwarg.
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
# Also patch where MiniMax M2 imports it from
import transformers.models.minimax_m2.modeling_minimax_m2 as minimax_m2
minimax_m2.create_causal_mask = create_causal_mask_patched

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import time
import json
from minimax_m3_sparse_attention import MiniMaxSparseAttention
import types

model_name = "MiniMaxAI/MiniMax-M2.7"
seq_len = 8192  # 8K - working config
num_tokens = 32

hf_token = os.environ.get('HF_TOKEN', '')
token = hf_token if hf_token else None

print(f"Loading model...")
t0 = time.time()

# Load config first and patch rope_parameters to prevent AttributeError
config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=token,
)
# rope_scaling property tries to access config.rope_parameters which doesn't exist
config.rope_parameters = {}
# Disable FP8 quantization that breaks MoE experts
# Set a fake quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING to skip quantization
config.quantization_config = {"quant_method": "none"}

# Try without offload_folder - let accelerate handle it
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

# Create 128K token prompt
base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
prompt = base_prompt * 30000
prompt = prompt[:seq_len * 2]

print(f"Prompt length: {len(prompt)} chars")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
actual_tokens = inputs['input_ids'].shape[1]
print(f"Input tokens: {actual_tokens}")

# Find attention modules in MiniMax-M2.7
from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Attention

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

# Try to replace attention modules with sparse attention (memory permitting)
# If OOM, we'll skip replacement and test base model
count = [0]
replace_success = [False]

def replace(module):
    attn_cls = type(module)
    module_name = type(module).__name__

    # MiniMax M2 uses MiniMaxM2Attention
    if module_name == 'MiniMaxM2Attention':
        count[0] += 1
        config = module.config
        # Read actual sizes from module weights (not config, which may differ)
        actual_q_size = module.q_proj.weight.shape[0]
        actual_k_size = module.k_proj.weight.shape[0]
        actual_num_heads = actual_q_size // module.head_dim
        actual_num_kv_heads = actual_k_size // module.head_dim
        if count[0] == 1:
            print(f"  Layer1: q_proj={module.q_proj.weight.shape}, k_proj={module.k_proj.weight.shape}, head_dim={module.head_dim}")
        attn = MiniMaxSparseAttention(
            hidden_size=config.hidden_size,
            num_heads=actual_num_heads,
            num_kv_heads=actual_num_kv_heads,
            head_dim=module.head_dim,
            block_size=16,
            top_k_blocks=4,
            index_dim=8,
        )
        # Move to the SAME device as the layer being replaced (not global device).
        # This avoids device mismatch when layers are distributed across GPUs.
        layer_device = next(module.parameters()).device if len(list(module.parameters())) > 0 else device
        torch.cuda.empty_cache()
        attn = attn.to(device=layer_device, dtype=dtype)
        torch.cuda.empty_cache()

        def new_forward(self, hidden_states, **kw):
            return attn(hidden_states, **kw)
        module.forward = types.MethodType(new_forward, module)
        return True
    return False

try:
    model.apply(replace)
    print(f"Replaced {count[0]} attention layers")
    replace_success[0] = True
except torch.OutOfMemoryError as e:
    print(f"Skipped attention replacement due to OOM: {e}")
    replace_success[0] = False

print("Warming up...")
for i in range(2):
    try:
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        print(f"  Warmup {i+1} succeeded")
    except Exception as e:
        import traceback
        print(f"  Warmup {i+1} failed: {e}")
        traceback.print_exc()
        break

print("Benchmarking MiniMax-M2.7 sparse at 128K...")
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
        import traceback
        print(f"  Iter {i+1} failed: {e}")
        traceback.print_exc()
        times.append(0)

avg_speed = num_tokens / (sum(times)/len(times)) if times and times[0] > 0 else 0
mem_peak = torch.cuda.max_memory_allocated() / 1024**3

result = {
    "model": model_name,
    "attention_type": "sparse",
    "seq_len": seq_len,
    "input_tokens": actual_tokens,
    "num_tokens": num_tokens,
    "memory_peak_gb": mem_peak,
    "avg_speed_tokens_per_sec": avg_speed,
    "times": times,
}

os.makedirs("hpc_results_minimax", exist_ok=True)
with open(f"hpc_results_minimax/sparse_{seq_len}.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nMiniMax-M2.7 sparse at {seq_len}: {avg_speed:.1f} tok/s")

del model
torch.cuda.empty_cache()