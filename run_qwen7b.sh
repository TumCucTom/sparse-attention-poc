#!/bin/bash
#SBATCH --job-name=qwen7b-128k
#SBATCH --output=logs/qwen7b-128k-%j.out
#SBATCH --error=logs/qwen7b-128k-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Qwen2.5-7B Sparse vs Dense at 128K Context"
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
echo "=== Running Qwen7B Benchmarks ==="

$PYTHON benchmark_qwen7b_sparse.py 2>&1
$PYTHON benchmark_qwen7b_dense.py 2>&1

echo ""
echo "Job completed: $(date)"