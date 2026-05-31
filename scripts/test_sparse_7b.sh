#!/bin/bash
#SBATCH --job-name=sparse-test
#SBATCH --output=sparse-test-%j.out
#SBATCH --error=sparse-test-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Testing Fixed Sparse Attention - Qwen 7B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"
python3 --version

pip install --user accelerate --quiet 2>/dev/null

nvidia-smi --query-gpu=name,memory.total --format=csv

python3 << 'TEST_SCRIPT'
import os
os.environ['PATH'] = '/home/b6ar/trvbale.b6ar/miniforge3/bin:' + os.environ.get('PATH', '')

import torch
from minimax_m3_sparse_attention import MiniMaxSparseAttention

print("Testing MiniMaxSparseAttention with GQA...")

# Create attention module for Qwen2.5-7B config
# 28 heads, 4 KV heads, head_dim=128, groups=7
attn = MiniMaxSparseAttention(
    hidden_size=4096,
    num_heads=28,
    num_kv_heads=4,
    head_dim=128,
    block_size=16,
    top_k_blocks=4,
    index_dim=32,
)

print(f"num_key_value_groups: {attn.num_key_value_groups}")
print(f"Expected: 7 (28/4)")

# Create dummy inputs
batch_size = 1
seq_len = 2048
hidden_states = torch.randn(batch_size, seq_len, 4096)

print(f"\nInput shape: {hidden_states.shape}")

# Test forward pass
try:
    out, _ = attn(hidden_states)
    print(f"Forward pass succeeded! Output shape: {out.shape}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete")
TEST_SCRIPT

echo ""
echo "Job completed: $(date)"
echo "=========================================="