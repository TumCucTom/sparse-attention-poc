#!/bin/bash
#SBATCH --job-name=sparse-bench-7b
#SBATCH --output=sparse-bench-7b-%j.out
#SBATCH --error=sparse-bench-7b-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "HPC Sparse Attention Benchmark - Qwen 7B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"
python3 --version

pip install --user accelerate --quiet 2>/dev/null

nvidia-smi --query-gpu=name,memory.total --format=csv

# Output directory
OUTPUT_DIR="hpc_results_7b_v2"
mkdir -p "$OUTPUT_DIR"

MODEL="Qwen/Qwen2.5-7B"
SEQ_LENS=(2048 4096 8192)

echo ""
echo "=== Running Dense Baseline (7B) ==="
for seq_len in "${SEQ_LENS[@]}"; do
    echo "Dense seq_len=$seq_len at $(date)"
    python3 hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 64 \
        --output "${OUTPUT_DIR}/dense_7b_seq${seq_len}.json" 2>&1 || echo "Failed dense seq_len=$seq_len"
done

echo "Dense benchmarks done: $(date)"

echo ""
echo "=== Running Sparse Attention (7B) - FIXED ==="
for seq_len in "${SEQ_LENS[@]}"; do
    echo "Sparse seq_len=$seq_len at $(date)"
    python3 hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type sparse \
        --top-k 4 \
        --block-size 16 \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 64 \
        --output "${OUTPUT_DIR}/sparse_7b_seq${seq_len}_k4.json" 2>&1 || echo "Failed sparse seq_len=$seq_len k=4"
done

echo "Sparse benchmarks done: $(date)"

echo ""
echo "=== Running Hybrid Attention (7B) ==="
for seq_len in "${SEQ_LENS[@]}"; do
    echo "Hybrid seq_len=$seq_len at $(date)"
    python3 hpc_hybrid_bench.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --mode hybrid \
        --window-size 128 \
        --global-size 64 \
        --chunk-size 64 \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 64 \
        --output "${OUTPUT_DIR}/hybrid_7b_seq${seq_len}.json" 2>&1 || echo "Failed hybrid seq_len=$seq_len"
done

echo "Hybrid benchmarks done: $(date)"

# Generate summary
python3 << 'SUMMARY_SCRIPT'
import json
import glob
import os

os.chdir('/home/b6ar/trvbale.b6ar/scratch/sparse-attention-poc')

results = []
for f in sorted(glob.glob('hpc_results_7b_v2/*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

print("\n" + "="*70)
print("BENCHMARK RESULTS - Qwen 7B (Fixed Sparse)")
print("="*70)

if not results:
    print("No results found!")
else:
    dense = {r['seq_len']: r['avg_speed_tokens_per_sec'] for r in results if r['attention_type'] == 'dense'}

    print(f"\n{'Seq Len':>10} {'Attention':>12} {'TopK':>6} {'Speed':>10} {'Baseline':>10} {'Speedup':>8}")
    print("-"*70)

    for seq_len in sorted(set(r['seq_len'] for r in results)):
        baseline = dense.get(seq_len, 0)
        print(f"\nSequence Length: {seq_len} (Dense: {baseline:.1f} tok/s)")
        for r in sorted([x for x in results if x['seq_len'] == seq_len], key=lambda x: (x['attention_type'], x.get('top_k', 0))):
            speed = r['avg_speed_tokens_per_sec']
            speedup = speed / baseline if baseline > 0 else 0
            top_k = r.get('top_k', '-')
            print(f"  {r['attention_type']:>12} top_k={top_k:>4} {speed:>10.1f} {baseline:>10.1f} {speedup:>7.2f}x")

    with open('hpc_results_7b_v2/summary.json', 'w') as f:
        json.dump({'model': 'Qwen/Qwen2.5-7B', 'results': results, 'dense_baselines': dense}, f, indent=2)
    print("\nSummary saved to hpc_results_7b_v2/summary.json")
SUMMARY_SCRIPT

echo ""
echo "Job completed: $(date)"
echo "=========================================="