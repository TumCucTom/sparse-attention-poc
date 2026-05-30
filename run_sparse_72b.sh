#!/bin/bash
#SBATCH --job-name=sparse-72b-test
#SBATCH --output=logs/sparse-72b-%j.out
#SBATCH --error=logs/sparse-72b-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention - Qwen2.5-72B Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_72b_sparse"
mkdir -p "$OUTPUT_DIR"

MODEL="Qwen/Qwen2.5-72B"

echo ""
echo "=== Testing 72B Dense at seq_len=8192 ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 8192 \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/dense_seq8192.json" 2>&1 || echo "Failed"

echo ""
echo "=== Testing 72B Sparse at seq_len=8192 ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 8192 \
    --attention-type sparse \
    --top-k 4 \
    --block-size 16 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/sparse_k4_seq8192.json" 2>&1 || echo "Failed"

echo ""
echo "=== Testing 72B Dense at seq_len=16384 ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 16384 \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/dense_seq16384.json" 2>&1 || echo "Failed"

echo ""
echo "=== Testing 72B Sparse at seq_len=16384 ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len 16384 \
    --attention-type sparse \
    --top-k 4 \
    --block-size 16 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 32 \
    --output "${OUTPUT_DIR}/sparse_k4_seq16384.json" 2>&1 || echo "Failed"

echo ""
echo "Job completed: $(date)"