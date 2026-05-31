#!/bin/bash
#SBATCH --job-name=gpu-sparse-bench
#SBATCH --output=logs/gpu-bench-%j.out
#SBATCH --error=logs/gpu-bench-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "GPU Sparse Attention Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Use the pytorch-cu128 environment that has working CUDA
PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

echo "PyTorch and CUDA info:"
$PYTHON << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
PYEOF

mkdir -p hpc_results_gpu

# Benchmark 7B model with different attention types
MODEL="Qwen/Qwen2.5-7B"
SEQ_LEN=4096

echo ""
echo "=== Dense Baseline (GPU) ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type dense \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/dense_7b.json" 2>&1 || echo "Dense failed"

echo ""
echo "=== Sparse M3 Attention (GPU) ==="
for top_k in 4 8; do
    $PYTHON hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$SEQ_LEN" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size 16 \
        --index-dim 32 \
        --iterations 3 \
        --warmup 2 \
        --num-tokens 64 \
        --output "hpc_results_gpu/sparse_m3_k${top_k}.json" 2>&1 || echo "Sparse M3 k=$top_k failed"
done

echo ""
echo "=== Hybrid Attention (GPU) ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type hybrid \
    --top-k 4 \
    --block-size 16 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/hybrid_k4.json" 2>&1 || echo "Hybrid failed"

echo ""
echo "=== DSA (DeepSeek) Attention (GPU) ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type dsa \
    --top-k 4 \
    --block-size 16 \
    --index-dim 64 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/dsa_k4.json" 2>&1 || echo "DSA failed"

# Summary
echo ""
echo "=== Summary ==="
$PYTHON << 'SUMMARY'
import json
import glob
import os

os.chdir('/home/b6ar/trvbale.b6ar/scratch/sparse-attention-poc')

results = []
for f in sorted(glob.glob('hpc_results_gpu/*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

print("\n" + "="*70)
print("GPU BENCHMARK RESULTS - Qwen 7B @ seq_len=4096")
print("="*70)

if not results:
    print("No results found!")
else:
    dense = next((r for r in results if r['attention_type'] == 'dense'), None)
    baseline = dense['avg_speed_tokens_per_sec'] if dense else 0
    print(f"\nDense baseline: {baseline:.2f} tok/s")

    print("\n" + "-"*70)
    print(f"{'Attention':<15} {'Speed':>10} {'Speedup':>8} {'Memory':>10}")
    print("-"*70)

    for r in sorted(results, key=lambda x: -x['avg_speed_tokens_per_sec']):
        speed = r['avg_speed_tokens_per_sec']
        attn_type = r['attention_type']
        speedup = speed / baseline if baseline > 0 else 0
        mem = r.get('memory_peak_gb', 0)
        print(f"{attn_type:<15} {speed:>10.2f} {speedup:>8.2f}x {mem:>10.1f}GB")

with open('hpc_results_gpu/summary.json', 'w') as f:
    json.dump({'results': results}, f, indent=2)
print("\nSummary saved to hpc_results_gpu/summary.json")
SUMMARY

echo ""
echo "Job completed: $(date)"