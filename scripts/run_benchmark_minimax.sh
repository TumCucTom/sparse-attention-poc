#!/bin/bash
#SBATCH --job-name=minimax-m2.7-sparse
#SBATCH --output=logs/benchmark_minimax-%j.out
#SBATCH --error=logs/benchmark_minimax-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "MiniMax-M2.7 with Sparse Attention at 128K"
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
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "=== Running MiniMax-M2.7 Sparse Benchmark ==="

PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}" $PYTHON benchmarks/benchmark_minimax.py 2>&1

echo ""
echo "Job completed: $(date)"