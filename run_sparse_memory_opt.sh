#!/bin/bash
#SBATCH --job-name=sparse-mem-opt
#SBATCH --output=logs/sparse-mem-opt-%j.out
#SBATCH --error=logs/sparse-mem-opt-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention Memory Optimization Tests"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_mem_opt"
mkdir -p "$OUTPUT_DIR"

MODEL="01-ai/Yi-34B-200K"

echo ""
echo "=== Memory-Optimized Sparse Configurations ==="

# Config 1: Smaller block_size=8, top_k=2
seq_len=65536
top_k=2
block_size=8
output_file="${OUTPUT_DIR}/sparse_k${top_k}_b${block_size}_seq${seq_len}.json"

if [ ! -f "$output_file" ]; then
    echo ""
    echo "--- Config: block_size=$block_size, top_k=$top_k, seq_len=$seq_len ---"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "$output_file" 2>&1 || echo "Failed"
fi

# Config 2: block_size=8, top_k=4
top_k=4
output_file="${OUTPUT_DIR}/sparse_k${top_k}_b${block_size}_seq${seq_len}.json"

if [ ! -f "$output_file" ]; then
    echo ""
    echo "--- Config: block_size=$block_size, top_k=$top_k, seq_len=$seq_len ---"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "$output_file" 2>&1 || echo "Failed"
fi

# Config 3: Try at 131072 with smaller config
seq_len=131072
top_k=2
block_size=8
output_file="${OUTPUT_DIR}/sparse_k${top_k}_b${block_size}_seq${seq_len}.json"

if [ ! -f "$output_file" ]; then
    echo ""
    echo "--- Config: block_size=$block_size, top_k=$top_k, seq_len=$seq_len ---"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "$output_file" 2>&1 || echo "Failed"
fi

# Config 4: Try at 200K with very small config
seq_len=200000
top_k=2
block_size=8
output_file="${OUTPUT_DIR}/sparse_k${top_k}_b${block_size}_seq${seq_len}.json"

if [ ! -f "$output_file" ]; then
    echo ""
    echo "--- Config: block_size=$block_size, top_k=$top_k, seq_len=$seq_len ---"
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "$output_file" 2>&1 || echo "Failed"
fi

echo ""
echo "Job completed: $(date)"