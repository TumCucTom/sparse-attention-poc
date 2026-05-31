#!/bin/bash
#SBATCH --job-name=long-context-quant
#SBATCH --output=logs/long-context-quant-%j.out
#SBATCH --error=logs/long-context-quant-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Long Context with Quantization"
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

# Test Qwen2.5-72B with int4 quantization to fit in GPU memory
MODEL_QWEN72="Qwen/Qwen2.5-72B-Instruct-GPTQ"
echo ""
echo "=== Testing Qwen2.5-72B-GPTQ (Quantized) at Long Contexts ==="
for seq_len in 32768 65536 131072; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/qwen72gptq_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            output_file="${OUTPUT_DIR}/qwen72gptq_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "Qwen2.5-72B-GPTQ $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL_QWEN72" \
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
                --model "$MODEL_QWEN72" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed: $attn_type seq_len=$seq_len"
        fi
    done
done

# Test Yi-1.5-6B (smaller model with 200K context)
MODEL_YI6="01-ai/Yi-1.5-6B-200K"
echo ""
echo "=== Testing Yi-1.5-6B-200K (6B params, 200K context) ==="
for seq_len in 65536 131072 200000; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/yi6b_${attn_type}_seq${seq_len}_k${top_k}.json"
        else
            output_file="${OUTPUT_DIR}/yi6b_${attn_type}_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (exists)"
            continue
        fi

        echo "Yi-1.5-6B $attn_type seq_len=$seq_len at $(date)"
        if [ "$attn_type" == "sparse" ]; then
            $PYTHON hpc_benchmark.py \
                --model "$MODEL_YI6" \
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
                --model "$MODEL_YI6" \
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