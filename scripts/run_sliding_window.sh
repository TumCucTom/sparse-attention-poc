#!/bin/bash
#SBATCH --job-name=sliding-win
#SBATCH --output=logs/sliding-win-%j.out
#SBATCH --error=logs/sliding-win-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sliding Window Attention at 200K Context"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_sliding_200k"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Testing Sliding Window Attention at 200K ==="

# Test at 131K with window=512
$PYTHON streaming_chunked.py \
    "$MODEL" \
    131072 \
    512 \
    "${OUTPUT_DIR}/yi131k_win512.json" 2>&1

# Test at 200K with window=512
$PYTHON streaming_chunked.py \
    "$MODEL" \
    200000 \
    512 \
    "${OUTPUT_DIR}/yi200k_win512.json" 2>&1

echo ""
echo "Job completed: $(date)"