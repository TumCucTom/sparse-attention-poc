#!/bin/bash
#SBATCH --job-name=sparse-32b-long
#SBATCH --output=logs/sparse-32b-long-%j.out
#SBATCH --error=logs/sparse-32b-long-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention - 32B and Long Context"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_32b_long"
mkdir -p "$OUTPUT_DIR"

# Test 1: Qwen2.5-14B at longer contexts (16384, 32768)
MODEL14="Qwen/Qwen2.5-14B"
echo ""
echo "=== 14B at Long Contexts ==="
for seq_len in 16384 32768; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/14b_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            top_k=4
            output_file="${OUTPUT_DIR}/14b_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "14B $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL14" \
                --seq-len "$seq_len" \
                --attention-type sparse \
                --top-k "$top_k" \
                --block-size 16 \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed"
        else
            $PYTHON hpc_benchmark.py \
                --model "$MODEL14" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed"
        fi
    done
done

# Test 2: Qwen2.5-32B at medium contexts (8192, 16384)
MODEL32="Qwen/Qwen2.5-32B"
echo ""
echo "=== 32B Tests ==="
for seq_len in 8192 16384; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/32b_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            top_k=4
            output_file="${OUTPUT_DIR}/32b_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "32B $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL32" \
                --seq-len "$seq_len" \
                --attention-type sparse \
                --top-k "$top_k" \
                --block-size 16 \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed"
        else
            $PYTHON hpc_benchmark.py \
                --model "$MODEL32" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed"
        fi
    done
done

echo ""
echo "Job completed: $(date)"