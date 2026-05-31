#!/bin/bash
#SBATCH --job-name=sparse-bench-7b
#SBATCH --output=logs/sparse-bench-7b-%j.out
#SBATCH --error=logs/sparse-bench-7b-%j.err
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

export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"
python3 --version

pip install --user accelerate --quiet 2>/dev/null

nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "GPU info unavailable"

OUTPUT_DIR="hpc_results_7b"
mkdir -p "$OUTPUT_DIR"

MODEL="Qwen/Qwen2.5-7B"
SEQ_LENS=(2048 4096 8192)
TOP_KS=(4 8 16)
BLOCK_SIZE=16

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
            --output "${OUTPUT_DIR}/sparse_7b_seq${seq_len}_k${top_k}.json" 2>&1 || echo "Failed sparse seq_len=$seq_len top_k=$top_k"
    done
done

echo "Sparse benchmarks done: $(date)"

python3 << 'SUMMARY_SCRIPT'
import json
import glob
import os

os.chdir('/home/b6ar/trvbale.b6ar/scratch/sparse-attention-poc')

results = []
for f in sorted(glob.glob('hpc_results_7b/sparse_*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

print("\n" + "="*70)
print("SPARSE BENCHMARK RESULTS - Qwen 7B")
print("="*70)

if not results:
    print("No results found!")
else:
    print(f"\n{'Seq Len':>10} {'TopK':>6} {'Speed':>10} {'Memory GB':>10}")
    print("-"*50)

    for seq_len in sorted(set(r['seq_len'] for r in results)):
        for r in sorted([x for x in results if x['seq_len'] == seq_len], key=lambda x: x.get('top_k', 0)):
            speed = r.get('avg_speed_tokens_per_sec', 0)
            mem = r.get('memory_peak_gb', 0)
            top_k = r.get('top_k', '-')
            print(f"{seq_len:>10} {top_k:>6} {speed:>10.1f} {mem:>10.2f}")

    with open('hpc_results_7b/sparse_summary.json', 'w') as f:
        json.dump({'model': 'Qwen/Qwen2.5-7B', 'results': results}, f, indent=2)
    print("\nSparse summary saved to hpc_results_7b/sparse_summary.json")
SUMMARY_SCRIPT

echo ""
echo "Job completed: $(date)"
echo "=========================================="