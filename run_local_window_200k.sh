#!/bin/bash
#SBATCH --job-name=local-win-200k
#SBATCH --output=logs/local-win-200k-%j.out
#SBATCH --error=logs/local-win-200k-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Local Window Attention at 200K Context"
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

OUTPUT_DIR="hpc_results_local_200k"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Testing Local Window Attention at 200K ==="

# Run dense at 131K first
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 131072 \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/dense_seq131072.json" 2>&1

# Run dense at 200K
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 200000 \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/dense_seq200000.json" 2>&1

echo ""
echo "Job completed: $(date)"