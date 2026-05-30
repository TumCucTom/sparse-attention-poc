#!/bin/bash
#SBATCH --job-name=qwen14b-256k
#SBATCH --output=logs/qwen14b-256k-%j.out
#SBATCH --error=logs/qwen14b-256k-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Qwen2.5-14B Sparse at 256K Context"
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
echo "=== Running Qwen2.5-14B Sparse at 256K ==="

$PYTHON benchmark_256k.py 2>&1

echo ""
echo "Job completed: $(date)"