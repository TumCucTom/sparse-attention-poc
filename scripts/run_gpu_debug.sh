#!/bin/bash
#SBATCH --job-name=gpu-debug-sparse
#SBATCH --output=logs/gpu-debug-%j.out
#SBATCH --error=logs/gpu-debug-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Debugging Sparse Attention on GPU"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

$PYTHON << 'PYEOF'
import torch
import sys
sys.path.insert(0, '.')

from minimax_m3_sparse_attention import MiniMaxSparseAttention

print("Creating attention module...")
# Typical Qwen2.5-7B config
hidden_size = 4096
num_heads = 32
num_kv_heads = 8
head_dim = 128

attn = MiniMaxSparseAttention(
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    block_size=16,
    top_k_blocks=4,
    index_dim=32
).to(device='cuda', dtype=torch.float16)

print(f"Attention created - dtype: {next(attn.parameters()).dtype}")

# Create test input
batch_size = 1
seq_len = 256
x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
print(f"Input: {x.shape}, {x.dtype}")

# Test each stage
print("\n--- Stage 1: index projection ---")
try:
    idx_q = attn.index_q(x)
    print(f"idx_q shape: {idx_q.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Stage 2: block scores ---")
try:
    idx_q_reshaped = idx_q.view(batch_size, seq_len, num_heads, attn.index_dim)
    idx_k = attn.index_k(x).view(batch_size, seq_len, num_kv_heads, attn.index_dim)
    block_scores = attn._compute_block_scores(idx_q_reshaped, idx_k.float())
    print(f"block_scores shape: {block_scores.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Stage 3: topk selection ---")
try:
    actual_k = min(attn.top_k_blocks, block_scores.shape[-1])
    _, topk_blocks = block_scores.topk(actual_k, dim=-1)
    topk_blocks = topk_blocks[:, :, 0, :]
    print(f"topk_blocks shape: {topk_blocks.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Stage 4: QKV projection ---")
try:
    weight_dtype = attn.q_proj.weight.dtype
    x_f16 = x.to(dtype=weight_dtype)
    q = attn.q_proj(x_f16).view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k = attn.k_proj(x_f16).view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    v = attn.v_proj(x_f16).view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Stage 5: sparse attention ---")
try:
    out = attn._sparse_attention(q, k, v, topk_blocks)
    print(f"sparse attn output: {out.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Full forward pass ---")
try:
    output, _ = attn(x)
    print(f"SUCCESS! Output: {output.shape}, dtype: {output.dtype}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nDone at", torch.cuda.current_device())
PYEOF

echo "Job completed: $(date)"