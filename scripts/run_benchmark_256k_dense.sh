#!/bin/bash
#SBATCH --job-name=minimax-dense-256k
#SBATCH --output=logs/benchmark_dense_256k_%j.out
#SBATCH --error=logs/benchmark_dense_256k_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "MiniMax-M2.7 DENSE Attention at 256K"
echo "Using 16 nodes (64 GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "=== Running MiniMax-M2.7 DENSE at 256K ==="

CONTEXT_SIZE=262144 USE_SPARSE=0 NUM_TOKENS=32 \
PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}" \
$PYTHON benchmarks/benchmark_minimax_very_long.py 2>&1

echo ""
echo "Job completed: $(date)"