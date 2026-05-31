#!/bin/bash
#SBATCH --job-name=gpu-sparse-4096
#SBATCH --output=logs/gpu-sparse-4096-%j.out
#SBATCH --error=logs/gpu-sparse-4096-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Testing M3 Sparse with 4096 seq_len"
echo "=========================================="

PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

$PYTHON << 'PYEOF'
import torch
import sys
sys.path.insert(0, '.')
import os
os.environ['DEBUG_SPARSE'] = '1'

from minimax_m3_sparse_attention import MiniMaxSparseAttention

print("Creating attention module for Qwen2.5-7B config...")
hidden_size = 4096
num_heads = 32
num_kv_heads = 8
head_dim = 128
block_size = 16
top_k_blocks = 4
index_dim = 32

attn = MiniMaxSparseAttention(
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    block_size=block_size,
    top_k_blocks=top_k_blocks,
    index_dim=index_dim
).to(device='cuda', dtype=torch.float16)

print(f"Attention: dtype={next(attn.parameters()).dtype}, device={next(attn.parameters()).device}")

# Test with seq_len = 4096 exactly (no padding needed)
seq_len = 4096
batch_size = 1
x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
print(f"\nInput: {x.shape}")

try:
    output, _ = attn(x)
    print(f"SUCCESS! Output: {output.shape}, dtype: {output.dtype}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

# Also test with different seq_len values to pinpoint the issue
print("\n--- Testing different seq_lens ---")
for test_seq_len in [256, 512, 1024, 2048, 4096]:
    try:
        x_test = torch.randn(1, test_seq_len, hidden_size, device='cuda', dtype=torch.float16)
        output, _ = attn(x_test)
        print(f"seq_len={test_seq_len}: SUCCESS, output shape={output.shape}")
    except Exception as e:
        print(f"seq_len={test_seq_len}: FAILED - {e}")

print("\nDone!")
PYEOF

echo "Job completed: $(date)"