#!/bin/bash
#SBATCH --job-name=sparse-train-7b
#SBATCH --output=logs/sparse-train-7b-%j.out
#SBATCH --error=logs/sparse-train-7b-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention Training Test - Qwen 7B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"
python3 --version

pip install --user accelerate --quiet 2>/dev/null

echo ""
echo "=== Running Training Stability Test ==="

python3 train_sparse_cuda.py \
    --model "Qwen/Qwen2.5-7B" \
    --max-seq-len 2048 \
    --batch-size 1 \
    --top-k 8 \
    --block-size 16 \
    --index-dim 32 \
    --lr 1e-4 \
    --epochs 1 \
    --gradient-checkpointing \
    --use-bf16 \
    --warmup-steps 50 \
    --save-dir "train_results_7b" \
    --log-interval 10 \
    2>&1 || echo "Training script failed"

echo ""
echo "Job completed: $(date)"
echo "=========================================="