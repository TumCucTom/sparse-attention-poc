#!/bin/bash
#SBATCH --job-name=yi34b-200k-final
#SBATCH --output=logs/yi34b-200k-final-%j.out
#SBATCH --error=logs/yi34b-200k-final-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Yi-34B-200K Full Context Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_yi34b_200k_fixed"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Testing Yi-34B-200K at Full 200K Context ==="

# Test at 200K context
seq_len=200000

for attn_type in dense sparse; do
    if [ "$attn_type" == "sparse" ]; then
        top_k=4
        output_file="${OUTPUT_DIR}/${attn_type}_seq${seq_len}_k${top_k}.json"
    else
        output_file="${OUTPUT_DIR}/${attn_type}_seq${seq_len}.json"
    fi

    if [ -f "$output_file" ]; then
        echo "Skipping $attn_type seq_len=$seq_len (exists)"
        continue
    fi

    echo ""
    echo "--- Yi-34B $attn_type seq_len=$seq_len ---"
    echo "Time: $(date)"

    if [ "$attn_type" == "sparse" ]; then
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k "$top_k" \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "$output_file" 2>&1 || echo "Failed: $attn_type seq_len=$seq_len"
    else
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type dense \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "$output_file" 2>&1 || echo "Failed: $attn_type seq_len=$seq_len"
    fi
done

echo ""
echo "Job completed: $(date)"