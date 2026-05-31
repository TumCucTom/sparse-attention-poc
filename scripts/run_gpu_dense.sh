#!/bin/bash
#SBATCH --job-name=gpu-dense-only
#SBATCH --output=logs/gpu-dense-%j.out
#SBATCH --error=logs/gpu-dense-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "GPU Dense Attention Benchmark (Baseline)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

echo "=== Testing GPU and PyTorch ==="
$PYTHON << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Test matrix multiply
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f"Matrix multiply test: {y.shape} on {y.device}")
    print("GPU test PASSED!")
PYEOF

mkdir -p hpc_results_gpu

echo ""
echo "=== Running Dense Attention Benchmark ==="
MODEL="Qwen/Qwen2.5-7B"
SEQ_LEN=4096

$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/dense_7b_gpu.json" 2>&1

echo ""
echo "Job completed: $(date)"