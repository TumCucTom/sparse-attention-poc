#!/bin/bash
#SBATCH --job-name=minimax-test
#SBATCH --output=logs/minimax-test-%j.out
#SBATCH --error=logs/minimax-test-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "MiniMax M2.7 Production Model Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Use venv Python in scratch
PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version

# Set HF_HOME to scratch to avoid quota issues
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "GPU info unavailable"

OUTPUT_DIR="hpc_results_minimax"
mkdir -p "$OUTPUT_DIR"

# Test MiniMax M2.7 with dense attention first
MODEL="MiniMaxAI/MiniMax-M2.7"

echo ""
echo "=== Testing MiniMax M2.7 Dense ==="
for seq_len in 2048 4096; do
    echo "M2.7 Dense seq_len=$seq_len at $(date)"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "${OUTPUT_DIR}/dense_m2.7_seq${seq_len}.json" 2>&1 || echo "Failed M2.7 dense seq_len=$seq_len"
done

echo ""
echo "=== Testing MiniMax M2.7 Sparse ==="
for seq_len in 2048 4096; do
    for top_k in 4 8; do
        echo "M2.7 Sparse seq_len=$seq_len top_k=$top_k at $(date)"
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k "$top_k" \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "${OUTPUT_DIR}/sparse_m2.7_seq${seq_len}_k${top_k}.json" 2>&1 || echo "Failed M2.7 sparse seq_len=$seq_len top_k=$top_k"
    done
done

# Also test M2.5
MODEL="MiniMaxAI/MiniMax-M2.5"

echo ""
echo "=== Testing MiniMax M2.5 Dense ==="
for seq_len in 2048; do
    echo "M2.5 Dense seq_len=$seq_len at $(date)"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "${OUTPUT_DIR}/dense_m2.5_seq${seq_len}.json" 2>&1 || echo "Failed M2.5 dense seq_len=$seq_len"
done

echo ""
echo "Job completed: $(date)"