#!/bin/bash
#SBATCH --job-name=long-context-sweep
#SBATCH --output=logs/long-context-sweep-%j.out
#SBATCH --error=logs/long-context-sweep-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Long Context Sweep - Find Crossover Point"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="${SLURM_SUBMIT_DIR}/venv/bin/python"
$PYTHON --version
export HF_HOME="/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache"
nvidia-smi --query-gpu=name,memory.total --format=csv

OUTPUT_DIR="hpc_results_long_sweep"
mkdir -p "$OUTPUT_DIR"

MODEL="Qwen/Qwen2.5-14B"

# Test sequence lengths from 16K to 128K to find where sparse wins
for seq_len in 16384 32768 65536 131072; do
    echo ""
    echo "=== Testing seq_len=$seq_len ==="

    # Dense
    output_file="${OUTPUT_DIR}/dense_seq${seq_len}.json"
    if [ ! -f "$output_file" ]; then
        echo "Dense seq_len=$seq_len at $(date)"
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type dense \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "$output_file" 2>&1 || echo "Failed dense seq_len=$seq_len"
    else
        echo "Skipping dense seq_len=$seq_len (exists)"
    fi

    # Sparse with top_k=4
    output_file="${OUTPUT_DIR}/sparse_k4_seq${seq_len}.json"
    if [ ! -f "$output_file" ]; then
        echo "Sparse k=4 seq_len=$seq_len at $(date)"
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k 4 \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "$output_file" 2>&1 || echo "Failed sparse seq_len=$seq_len"
    else
        echo "Skipping sparse seq_len=$seq_len (exists)"
    fi

    # Sparse with top_k=8
    output_file="${OUTPUT_DIR}/sparse_k8_seq${seq_len}.json"
    if [ ! -f "$output_file" ]; then
        echo "Sparse k=8 seq_len=$seq_len at $(date)"
        $PYTHON hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k 8 \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "$output_file" 2>&1 || echo "Failed sparse seq_len=$seq_len"
    else
        echo "Skipping sparse k8 seq_len=$seq_len (exists)"
    fi
done

echo ""
echo "Job completed: $(date)"