#!/bin/bash
#SBATCH --job-name=hybrid-yi200k
#SBATCH --output=logs/hybrid-yi200k-%j.out
#SBATCH --error=logs/hybrid-yi200k-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Hybrid Local+Global on Yi-34B-200K at 200K"
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

OUTPUT_DIR="hpc_results_hybrid_yi200k"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Testing Hybrid on Yi-34B-200K at 200K context ==="

# Test dense first
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 131072 \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/dense_seq131072.json" 2>&1 || echo "Failed dense"

$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 200000 \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/dense_seq200000.json" 2>&1 || echo "Failed dense 200K"

echo ""
echo "Job completed: $(date)"