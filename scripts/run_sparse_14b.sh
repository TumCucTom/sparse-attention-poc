#!/bin/bash
#SBATCH --job-name=sparse-14b-test
#SBATCH --output=logs/sparse-14b-%j.out
#SBATCH --error=logs/sparse-14b-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention Test - Qwen2.5-14B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Use venv Python in scratch
PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version

# Set HF_HOME to scratch to avoid quota issues
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"

nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_14b_sparse"
mkdir -p "$OUTPUT_DIR"

MODEL="Qwen/Qwen2.5-14B"

echo ""
echo "=== Testing DENSE attention on 14B ==="
for seq_len in 2048 4096 8192; do
    output_file="${OUTPUT_DIR}/dense_seq${seq_len}.json"

    if [ -f "$output_file" ]; then
        echo "Skipping dense seq_len=$seq_len (already exists)"
        continue
    fi

    echo "Dense seq_len=$seq_len at $(date)"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "$output_file" 2>&1 || echo "Failed dense seq_len=$seq_len"
done

echo ""
echo "=== Testing SPARSE attention on 14B ==="
for seq_len in 2048 4096 8192; do
    for top_k in 4 8; do
        output_file="${OUTPUT_DIR}/sparse_seq${seq_len}_k${top_k}.json"

        if [ -f "$output_file" ]; then
            echo "Skipping sparse seq_len=$seq_len top_k=$top_k (already exists)"
            continue
        fi

        echo "Sparse seq_len=$seq_len top_k=$top_k at $(date)"
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k "$top_k" \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "$output_file" 2>&1 || echo "Failed sparse seq_len=$seq_len top_k=$top_k"
    done
done

echo ""
echo "Job completed: $(date)"