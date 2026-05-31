#!/bin/bash
#SBATCH --job-name=public-long-context
#SBATCH --output=logs/public-long-context-%j.out
#SBATCH --error=logs/public-long-context-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Public Long Context Models"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_long_models"
mkdir -p "$OUTPUT_DIR"

# Test Qwen2.5-14B with extended RoPE (configure for longer context)
MODEL_QWEN14="Qwen/Qwen2.5-14B"
echo ""
echo "=== Testing Qwen2.5-14B with Extended Context (via RoPE scaling) ==="
for seq_len in 32768 65536 131072; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/qwen14_ext_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            output_file="${OUTPUT_DIR}/qwen14_ext_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "Qwen2.5-14B $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL_QWEN14" \
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
                --model "$MODEL_QWEN14" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed: $attn_type seq_len=$seq_len"
        fi
    done
done

# Test Phi-3-medium (4K native, but we can test scaling behavior)
MODEL_PHI="microsoft/Phi-3-medium-4k-instruct"
echo ""
echo "=== Testing Phi-3-medium ==="
for seq_len in 16384 32768; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/phi3_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            output_file="${OUTPUT_DIR}/phi3_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "Phi-3-medium $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL_PHI" \
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
                --model "$MODEL_PHI" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed: $attn_type seq_len=$seq_len"
        fi
    done
done

echo ""
echo "Job completed: $(date)"