#!/bin/bash
#SBATCH --job-name=sparse-bench-7b
#SBATCH --output=sparse-bench-7b-%j.out
#SBATCH --error=sparse-bench-7b-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "HPC Sparse Attention Benchmark - Qwen 7B"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Use miniforge Python
export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"

python3 --version

# Install required packages
echo "Installing required packages..."
pip install --user accelerate --quiet 2>/dev/null

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 << 'CHECK'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
CHECK

# Output directory
OUTPUT_DIR="hpc_results_7b"
mkdir -p "$OUTPUT_DIR"

# Models to test
MODEL="Qwen/Qwen2.5-7B"
SEQ_LENS=(2048 4096 8192)
TOP_KS=(4 8 16)
BLOCK_SIZE=16

# Run dense baseline first
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
        --output "${OUTPUT_DIR}/dense_7b_seq${seq_len}.json" 2>&1 | tee -a "${OUTPUT_DIR}/dense_${seq_len}.log" || echo "Failed dense seq_len=$seq_len"
done

echo "Dense benchmarks done: $(date)"

# Run sparse attention benchmarks
echo ""
echo "=== Running Sparse Attention (7B) ==="
for seq_len in "${SEQ_LENS[@]}"; do
    for top_k in "${TOP_KS[@]}"; do
        echo "Sparse seq_len=$seq_len top_k=$top_k at $(date)"
        python3 hpc_benchmark.py \
            --model "$MODEL" \
            --seq-len "$seq_len" \
            --attention-type sparse \
            --top-k "$top_k" \
            --block-size "$BLOCK_SIZE" \
            --iterations 3 \
            --warmup 2 \
            --num-tokens 64 \
            --output "${OUTPUT_DIR}/sparse_7b_seq${seq_len}_k${top_k}.json" 2>&1 | tee -a "${OUTPUT_DIR}/sparse_${seq_len}_k${top_k}.log" || echo "Failed sparse seq_len=$seq_len top_k=$top_k"
    done
done

echo "Sparse benchmarks done: $(date)"

# Run hybrid attention benchmarks
echo ""
echo "=== Running Hybrid Attention (7B) ==="
for seq_len in "${SEQ_LENS[@]}"; do
    echo "Hybrid seq_len=$seq_len at $(date)"
    python3 hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type hybrid \
        --top-k 8 \
        --block-size "$BLOCK_SIZE" \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 64 \
        --output "${OUTPUT_DIR}/hybrid_7b_seq${seq_len}.json" 2>&1 | tee -a "${OUTPUT_DIR}/hybrid_${seq_len}.log" || echo "Failed hybrid seq_len=$seq_len"
done

echo "Hybrid benchmarks done: $(date)"

# Generate summary
python3 << 'SUMMARY_SCRIPT'
import json
import glob
import os

os.chdir('/home/b6ar/trvbale.b6ar/scratch/sparse-attention-poc')

results = []
for f in sorted(glob.glob('hpc_results_7b/*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

print("\n" + "="*70)
print("BENCHMARK RESULTS - Qwen 7B")
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

    with open('hpc_results_7b/summary.json', 'w') as f:
        json.dump({'model': 'Qwen/Qwen2.5-7B', 'results': results, 'dense_baselines': dense}, f, indent=2)
    print("\nSummary saved to hpc_results_7b/summary.json")
SUMMARY_SCRIPT

echo ""
echo "Job completed: $(date)"
echo "=========================================="