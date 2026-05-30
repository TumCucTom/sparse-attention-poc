#!/bin/bash
#SBATCH --job-name=gpu-dsa-bench
#SBATCH --output=logs/gpu-dsa-bench-%j.out
#SBATCH --error=logs/gpu-dsa-bench-%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Testing DSA Sparse Attention - Full Model"
echo "=========================================="

PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

MODEL="Qwen/Qwen2.5-7B"
SEQ_LEN=4096

echo ""
echo "=== Running DSA benchmark ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type dsa \
    --top-k 4 \
    --block-size 16 \
    --index-dim 32 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/dsa_k4.json" 2>&1

echo ""
echo "Job completed: $(date)"