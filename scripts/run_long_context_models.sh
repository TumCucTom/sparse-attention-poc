#!/bin/bash
#SBATCH --job-name=long-context-models
#SBATCH --output=logs/long-context-models-%j.out
#SBATCH --error=logs/long-context-models-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Testing Sparse Attention on Long Context Models"
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

# Test 1: Yi-1.5-34B-200K (200K context, 34B params)
MODEL_YI="01-ai/Yi-1.5-34B-200K"
echo ""
echo "=== Testing Yi-1.5-34B-200K ==="
for seq_len in 32768 65536 131072 200000; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/yi34b_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            output_file="${OUTPUT_DIR}/yi34b_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "Yi-1.5-34B $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL_YI" \
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
                --model "$MODEL_YI" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed: $attn_type seq_len=$seq_len"
        fi
    done
done

# Test 2: DeepSeek V2 (200K context)
MODEL_DEEPSEEK="deepseek-ai/DeepSeek-V2"
echo ""
echo "=== Testing DeepSeek V2 ==="
for seq_len in 32768 65536 131072; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/deepseek_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            output_file="${OUTPUT_DIR}/deepseek_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "DeepSeek V2 $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL_DEEPSEEK" \
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
                --model "$MODEL_DEEPSEEK" \
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