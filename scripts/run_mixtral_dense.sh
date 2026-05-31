#!/bin/bash
#SBATCH --job-name=mixtral-dense
#SBATCH --output=logs/mixtral-dense-%j.out
#SBATCH --error=logs/mixtral-dense-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Mixtral Dense Attention at 64K Context"
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
echo "=== Running Mixtral Dense ==="

$PYTHON benchmark_mixtral_dense.py 2>&1

echo ""
echo "Job completed: $(date)"