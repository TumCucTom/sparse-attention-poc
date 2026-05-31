#!/bin/bash
#SBATCH --job-name=sparse-bench
#SBATCH --output=sparse-bench-%j.out
#SBATCH --error=sparse-bench-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Environment setup (adjust for your HPC cluster)
# module load cuda/12.1
# module load python/3.10
# source ~/venvs/subq/bin/activate

echo "=========================================="
echo "HPC Sparse Attention Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_JOB_GPUS"
echo "=========================================="

python3 --version
nvidia-smi || echo "No NVIDIA GPU available"

# Model sizes to test
MODELS=("Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-3B")

# Sequence lengths
SEQ_LENS=(8192 16384 32768 65536)

# Attention types
ATTN_TYPES=("sparse" "hybrid" "dense")

# Top-K values for sparse attention
TOP_KS=(4 8 16)
BLOCK_SIZE=16

# Output directory
OUTPUT_DIR="hpc_results"
mkdir -p "$OUTPUT_DIR"

# Function to run benchmark
run_benchmark() {
    local model=$1
    local seq_len=$2
    local attn_type=$3
    local top_k=$4
    local block_size=${5:-16}
    local output_file="${OUTPUT_DIR}/${model##*/}_${attn_type}_seq${seq_len}_k${top_k}_b${block_size}.json"

    echo ""
    echo "----------------------------------------"
    echo "Running: $model | seq=$seq_len | type=$attn_type | top_k=$top_k"
    echo "Output: $output_file"
    echo "----------------------------------------"

    # Extract model size for naming
    model_name="${model##*/}"

    python3 hpc_benchmark.py \
        --model "$model" \
        --seq-len "$seq_len" \
        --attention-type "$attn_type" \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --iterations 5 \
        --warmup 2 \
        --num-tokens 64 \
        --output "$output_file"

    echo "Completed: $output_file"
}

# Run dense baseline for each model at each sequence length
echo ""
echo "=== Running Dense Baseline ==="
for model in "${MODELS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        run_benchmark "$model" "$seq_len" "dense" 0 16
    done
done

# Run sparse attention with different top_k
echo ""
echo "=== Running Sparse Attention ==="
for model in "${MODELS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        for top_k in "${TOP_KS[@]}"; do
            run_benchmark "$model" "$seq_len" "sparse" "$top_k" "$BLOCK_SIZE"
        done
    done
done

# Run hybrid attention
echo ""
echo "=== Running Hybrid Attention ==="
for model in "${MODELS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        run_benchmark "$model" "$seq_len" "hybrid" 8 "$BLOCK_SIZE"
    done
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

# Generate summary
python3 -c "
import json
import glob
import os

results = []
for f in glob.glob('${OUTPUT_DIR}/*.json'):
    with open(f) as fp:
        results.append(json.load(fp))

# Compute speedup ratios
by_model = {}
for r in results:
    model = r['model']
    if model not in by_model:
        by_model[model] = {}
    key = f\"{r['attention_type']}_{r['top_k']}_{r['seq_len']}\"
    by_model[model][key] = r

print('\\n' + '='*80)
print('SPEEDUP RATIOS (sparse vs dense)')
print('='*80)
for model, runs in by_model.items():
    print(f'\\n{model}:')
    dense_runs = {k.replace('dense_', ''): v for k, v in runs.items() if k.startswith('dense_')}
    for k, dense in dense_runs.items():
        sparse_runs = [v for vk, v in runs.items() if vk.startswith('sparse_') and k in vk]
        for sparse in sparse_runs:
            speedup = sparse['avg_speed_tokens_per_sec'] / dense['avg_speed_tokens_per_sec']
            print(f\"  seq={sparse['seq_len']:5d} top_k={sparse['top_k']:2d}: {speedup:.2f}x \" +
                  f\"(dense={dense['avg_speed_tokens_per_sec']:.1f}/s sparse={sparse['avg_speed_tokens_per_sec']:.1f}/s)\")
"