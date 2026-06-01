#!/bin/bash
#SBATCH --job-name=minimax-256k-64n
#SBATCH --output=logs/benchmark_256k_64n_%j.out
#SBATCH --error=logs/benchmark_256k_64n_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "MiniMax-M2.7 DENSE at 256K on 64 nodes"
echo "Using 64 nodes (256 GPUs)"
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
echo "=== Running MiniMax-M2.7 DENSE at 256K on 64 nodes (256 GPUs) ==="

CONTEXT_SIZE=262144 NUM_TOKENS=32 \
PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}" \
$PYTHON benchmarks/benchmark_minimax_final.py 2>&1

echo ""
echo "Job completed: $(date)"