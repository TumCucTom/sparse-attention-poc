#!/bin/bash
#SBATCH --job-name=sparse-200k-sdpa
#SBATCH --output=logs/sparse-200k-sdpa-%j.out
#SBATCH --error=logs/sparse-200k-sdpa-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention with SDPA at 200K Context"
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

OUTPUT_DIR="hpc_results_sparse_200k_sdpa"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Testing Sparse with SDPA at 200K ==="

# Test sparse at 131K first
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 131072 \
    --attention-type sparse \
    --top-k 4 \
    --block-size 16 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/sparse_seq131072_k4.json" 2>&1 || echo "Failed sparse 131K"

echo ""
echo "--- Testing sparse at 200K ---"
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 200000 \
    --attention-type sparse \
    --top-k 4 \
    --block-size 16 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/sparse_seq200000_k4.json" 2>&1 || echo "Failed sparse 200K"

echo ""
echo "Job completed: $(date)"