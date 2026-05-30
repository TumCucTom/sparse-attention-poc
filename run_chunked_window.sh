#!/bin/bash
#SBATCH --job-name=chunked-window
#SBATCH --output=logs/chunked-window-%j.out
#SBATCH --error=logs/chunked-window-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Chunked Sliding Window at 200K"
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

OUTPUT_DIR="hpc_results_chunked_window"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Testing Chunked Sliding Window ==="

$PYTHON chunked_window.py 2>&1

echo ""
echo "Job completed: $(date)"