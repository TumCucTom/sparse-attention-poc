#!/bin/bash
#SBATCH --job-name=gpu-check-model
#SBATCH --output=logs/gpu-check-model-%j.out
#SBATCH --error=logs/gpu-check-model-%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Checking model layer devices"
echo "=========================================="

PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

$PYTHON << 'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, '.')

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
)

print("\n=== Model device check ===")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Check attention layers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
attention_layers = []
for name, module in model.named_modules():
    if isinstance(module, Qwen2Attention):
        attention_layers.append((name, next(module.parameters()).device))

print(f"\nTotal Qwen2Attention layers: {len(attention_layers)}")
print("First 5 attention layers:")
for name, dev in attention_layers[:5]:
    print(f"  {name}: {dev}")
print("Last 5 attention layers:")
for name, dev in attention_layers[-5:]:
    print(f"  {name}: {dev}")

# Check tokenizer and prompt
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
prompt = base_prompt * max(1, 4096 // len(tokenizer.encode(base_prompt)))
prompt = prompt[:min(len(prompt), 4096 * 2)]

inputs = tokenizer(prompt, return_tensors="pt")
print(f"\nPrompt length: {len(tokenizer.encode(prompt))} tokens")
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input device: {inputs['input_ids'].device}")

# Try simple forward pass
print("\n=== Testing simple forward pass ===")
try:
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to('cuda'), output_hidden_states=True)
        print(f"Forward pass successful! Output logits shape: {outputs.logits.shape}")
        print(f"All hidden states have same device: {all(t.device == torch.device('cuda:0') for t in outputs.hidden_states)}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Try generate with just 5 tokens
print("\n=== Testing generate (5 tokens) ===")
try:
    with torch.no_grad():
        output = model.generate(inputs['input_ids'].to('cuda'), max_new_tokens=5, do_sample=False)
    print(f"Generate successful! Output shape: {output.shape}")
except Exception as e:
    print(f"Generate failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
PYEOF

echo "Job completed: $(date)"