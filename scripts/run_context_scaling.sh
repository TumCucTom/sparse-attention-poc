#!/bin/bash
#SBATCH --job-name=context-scaling
#SBATCH --output=logs/context-scaling-%j.out
#SBATCH --error=logs/context-scaling-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Context Scaling Test - Sparse Attention"
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

OUTPUT_DIR="hpc_results_context_scaling"
mkdir -p "$OUTPUT_DIR"

# Test Qwen2.5-7B with dense vs sparse at increasing context lengths
MODEL="Qwen/Qwen2.5-7B"

echo ""
echo "=== Testing Context Scaling ==="
for seq_len in 2048 4096 8192 16384; do
    for attn_type in dense sparse; do
        if [ "$attn_type" == "sparse" ]; then
            top_k=4
            output_file="${OUTPUT_DIR}/sparse_7b_seq${seq_len}_k${top_k}.json"
        else
            top_k=4
            output_file="${OUTPUT_DIR}/dense_7b_seq${seq_len}.json"
        fi

        if [ -f "$output_file" ]; then
            echo "Skipping $attn_type seq_len=$seq_len (already exists)"
            continue
        fi

        echo "${attn_type^} seq_len=$seq_len at $(date)"

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
                --output "$output_file" 2>&1 || echo "Failed sparse seq_len=$seq_len"
        else
            $PYTHON hpc_benchmark.py \
                --model "$MODEL" \
                --seq-len "$seq_len" \
                --attention-type dense \
                --iterations 3 \
                --warmup 2 \
                --num-tokens 32 \
                --output "$output_file" 2>&1 || echo "Failed dense seq_len=$seq_len"
        fi
    done
done

echo ""
echo "=== Testing Hybrid Attention at Long Context ==="
for seq_len in 8192 16384; do
    output_file="${OUTPUT_DIR}/hybrid_7b_seq${seq_len}.json"

    if [ -f "$output_file" ]; then
        echo "Skipping hybrid seq_len=$seq_len (already exists)"
        continue
    fi

    echo "Hybrid seq_len=$seq_len at $(date)"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type hybrid \
        --top-k 4 \
        --block-size 16 \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "$output_file" 2>&1 || echo "Failed hybrid seq_len=$seq_len"
done

echo ""
echo "Job completed: $(date)"