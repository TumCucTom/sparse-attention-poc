#!/bin/bash
#SBATCH --job-name=sparse-sweep
#SBATCH --output=logs/sparse-sweep-%j.out
#SBATCH --error=logs/sparse-sweep-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention Parameter Sweep - 7B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/home/b6ar/trvbale.b6ar/scratch/miniforge3/bin:$PATH"
python3 --version

OUTPUT_DIR="hpc_results_sweep"
mkdir -p "$OUTPUT_DIR"

MODEL="Qwen/Qwen2.5-7B"
SEQ_LEN=4096

echo ""
echo "=== Parameter Sweep on 7B model ==="

# Configurations to test: block_size, top_k, index_dim
# Focus on most promising configs based on prior 7B results
CONFIGS=(
    "16,4,32"   # baseline
    "16,4,16"   # smaller index
    "16,8,32"   # higher top_k
    "32,4,32"   # larger block
)

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r block_size top_k index_dim <<< "$config"
    echo "Config: block_size=$block_size, top_k=$top_k, index_dim=$index_dim at $(date)"
    python3 hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$SEQ_LEN" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --index-dim "$index_dim" \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 64 \
        --output "${OUTPUT_DIR}/sparse_bs${block_size}_k${top_k}_idx${index_dim}.json" 2>&1 || echo "Failed"
done

# Dense baseline
echo ""
echo "=== Dense baseline ==="
python3 hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "${OUTPUT_DIR}/dense_baseline.json" 2>&1 || echo "Dense failed"

echo ""
echo "=== Hybrid attention ==="
python3 hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type hybrid \
    --top-k 4 \
    --block-size 16 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "${OUTPUT_DIR}/hybrid_k4.json" 2>&1 || echo "Hybrid failed"

# Summary
echo ""
echo "=== Summary ==="
python3 << 'SUMMARY'
import json
import glob
import os

os.chdir('/home/b6ar/trvbale.b6ar/scratch/sparse-attention-poc')

results = []
for f in sorted(glob.glob('hpc_results_sweep/*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

print("\n" + "="*70)
print("PARAMETER SWEEP RESULTS - Qwen 7B @ seq_len=4096")
print("="*70)

if not results:
    print("No results found!")
else:
    dense = next((r for r in results if r['attention_type'] == 'dense'), None)
    baseline = dense['avg_speed_tokens_per_sec'] if dense else 0
    print(f"\nDense baseline: {baseline:.2f} tok/s")

    dense_hybrid = next((r for r in results if r['attention_type'] == 'hybrid'), None)
    if dense_hybrid:
        print(f"Hybrid speedup: {dense_hybrid['avg_speed_tokens_per_sec'] / baseline:.2f}x")

    print("\n" + "-"*70)
    print(f"{'Config':<30} {'Speed':>10} {'Speedup':>8}")
    print("-"*70)

    sparse = [r for r in results if r['attention_type'] == 'sparse']
    for r in sorted(sparse, key=lambda x: -x['avg_speed_tokens_per_sec']):
        speed = r['avg_speed_tokens_per_sec']
        cfg = f"bs={r['block_size']}, k={r['top_k']}, idx={r['index_dim']}"
        speedup = speed / baseline if baseline > 0 else 0
        print(f"{cfg:<30} {speed:>10.2f} {speedup:>8.2f}x")

with open('hpc_results_sweep/summary.json', 'w') as f:
    json.dump({'results': results}, f, indent=2)
print("\nSummary saved to hpc_results_sweep/summary.json")
SUMMARY

echo ""
echo "Job completed: $(date)"