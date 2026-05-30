#!/bin/bash
#SBATCH --job-name=bench-14b
#SBATCH --output=logs/bench-14b-%j.out
#SBATCH --error=logs/bench-14b-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "HPC Benchmark - Qwen 14B and 32B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"
python3 --version

pip install --user accelerate --quiet 2>/dev/null

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "GPU info unavailable"

OUTPUT_DIR="hpc_results_14b"
mkdir -p "$OUTPUT_DIR"

# Test Qwen 14B
MODEL_14B="Qwen/Qwen2.5-14B"
SEQ_LENS=(2048 4096)
TOP_KS=(4 8)

echo ""
echo "=== Testing Qwen 14B ==="
for seq_len in "${SEQ_LENS[@]}"; do
    for top_k in "${TOP_KS[@]}"; do
        echo "14B Sparse seq_len=$seq_len top_k=$top_k at $(date)"
        python3 hpc_benchmark.py \
            --model "$MODEL_14B" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k "$top_k" \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "${OUTPUT_DIR}/sparse_14b_seq${seq_len}_k${top_k}.json" 2>&1 || echo "Failed 14B sparse seq_len=$seq_len top_k=$top_k"
    done
done

echo ""
echo "=== Qwen 14B Dense Baseline ==="
for seq_len in "${SEQ_LENS[@]}"; do
    echo "14B Dense seq_len=$seq_len at $(date)"
    python3 hpc_benchmark.py \
        --model "$MODEL_14B" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "${OUTPUT_DIR}/dense_14b_seq${seq_len}.json" 2>&1 || echo "Failed 14B dense seq_len=$seq_len"
done

# Test Qwen 32B
MODEL_32B="Qwen/Qwen2.5-32B"
SEQ_LENS=(2048 4096)
TOP_KS=(4 8)

echo ""
echo "=== Testing Qwen 32B ==="
for seq_len in "${SEQ_LENS[@]}"; do
    for top_k in "${TOP_KS[@]}"; do
        echo "32B Sparse seq_len=$seq_len top_k=$top_k at $(date)"
        python3 hpc_benchmark.py \
            --model "$MODEL_32B" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k "$top_k" \
            --block-size 16 \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 32 \
            --output "${OUTPUT_DIR}/sparse_32b_seq${seq_len}_k${top_k}.json" 2>&1 || echo "Failed 32B sparse seq_len=$seq_len top_k=$top_k"
    done
done

echo ""
echo "=== Qwen 32B Dense Baseline ==="
for seq_len in "${SEQ_LENS[@]}"; do
    echo "32B Dense seq_len=$seq_len at $(date)"
    python3 hpc_benchmark.py \
        --model "$MODEL_32B" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 32 \
        --output "${OUTPUT_DIR}/dense_32b_seq${seq_len}.json" 2>&1 || echo "Failed 32B dense seq_len=$seq_len"
done

# Generate summary
python3 << 'SUMMARY'
import json
import glob
import os

os.chdir('/home/b6ar/trvbale.b6ar/scratch/sparse-attention-poc')

results = []
for f in sorted(glob.glob('hpc_results_14b/*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

print("\n" + "="*70)
print("BENCHMARK RESULTS - Qwen 14B & 32B")
print("="*70)

if not results:
    print("No results found!")
else:
    # Group by model and seq_len
    by_model = {}
    for r in results:
        model_key = r['model'].split('/')[-1]
        if model_key not in by_model:
            by_model[model_key] = {}
        seq_len = r['seq_len']
        if seq_len not in by_model[model_key]:
            by_model[model_key][seq_len] = []
        by_model[model_key][seq_len].append(r)

    for model_key in sorted(by_model.keys()):
        print(f"\n{model_key}:")
        for seq_len in sorted(by_model[model_key].keys()):
            runs = by_model[model_key][seq_len]
            dense = next((r for r in runs if r['attention_type'] == 'dense'), None)
            baseline = dense['avg_speed_tokens_per_sec'] if dense else 0
            print(f"  Seq {seq_len}: baseline={baseline:.2f} tok/s")

            sparse_runs = sorted([r for r in runs if r['attention_type'] == 'sparse'], key=lambda x: x.get('top_k', 0))
            for r in sparse_runs:
                speed = r['avg_speed_tokens_per_sec']
                top_k = r.get('top_k', '-')
                speedup = speed / baseline if baseline > 0 else 0
                print(f"    Sparse top_k={top_k}: {speed:.2f} tok/s ({speedup:.2f}x)")

    with open('hpc_results_14b/summary.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print("\nSummary saved to hpc_results_14b/summary.json")
SUMMARY

echo ""
echo "Job completed: $(date)"
echo "=========================================="