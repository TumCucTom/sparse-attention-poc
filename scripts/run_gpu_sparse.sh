#!/bin/bash
#SBATCH --job-name=gpu-sparse-m3
#SBATCH --output=logs/gpu-sparse-%j.out
#SBATCH --error=logs/gpu-sparse-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p workq

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "GPU Sparse Attention Benchmark - M3"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

PYTHON="/home/b6ar/trvbale.b6ar/scratch/miniforge3/envs/pytorch-cu128/bin/python3"

$PYTHON << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
PYEOF

mkdir -p hpc_results_gpu

MODEL="Qwen/Qwen2.5-7B"
SEQ_LEN=4096

echo ""
echo "=== Testing M3 Sparse (top_k=4) ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type sparse \
    --top-k 4 \
    --block-size 16 \
    --index-dim 32 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/sparse_m3_k4.json" 2>&1

echo ""
echo "=== Testing M3 Sparse (top_k=8) ==="
$PYTHON hpc_benchmark.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --attention-type sparse \
    --top-k 8 \
    --block-size 16 \
    --index-dim 32 \
    --iterations 3 \
    --warmup 2 \
    --num-tokens 64 \
    --output "hpc_results_gpu/sparse_m3_k8.json" 2>&1

# Summary
echo ""
echo "=== Summary ==="
$PYTHON << 'SUMMARY'
import json
import glob

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
    print(f"\nDense baseline: {baseline:.1f} tok/s")

    print("\n" + "-"*70)
    print(f"{'Type':<15} {'Speed':>10} {'Speedup':>8} {'Memory':>10}")
    print("-"*70)

    for r in sorted(results, key=lambda x: -x['avg_speed_tokens_per_sec']):
        speed = r['avg_speed_tokens_per_sec']
        attn_type = r['attention_type']
        speedup = speed / baseline if baseline > 0 else 0
        mem = r.get('memory_peak_gb', 0)
        print(f"{attn_type:<15} {speed:>10.1f} {speedup:>8.2f}x {mem:>9.1f}GB")

with open('hpc_results_gpu/summary.json', 'w') as f:
    json.dump({'results': results}, f, indent=2)
SUMMARY

echo ""
echo "Job completed: $(date)"