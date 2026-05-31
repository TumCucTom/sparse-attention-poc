#!/bin/bash
#SBATCH --job-name=param-sweep
#SBATCH --output=param-sweep-%j.out
#SBATCH --error=param-sweep-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Parameter Sweep: top_k and block_size"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Model
MODEL="Qwen/Qwen2.5-3B"

# Sequence lengths to test
SEQ_LENS=(8192 16384 32768)

# Top-K values
TOP_KS=(2 4 8 16)

# Block sizes
BLOCK_SIZES=(16 32)

# Output
OUTPUT_DIR="sweep_results"
mkdir -p "$OUTPUT_DIR"

run_benchmark() {
    local seq_len=$1
    local top_k=$2
    local block_size=$3
    local output_file="${OUTPUT_DIR}/seq${seq_len}_k${top_k}_b${block_size}.json"

    echo "Running: seq=$seq_len top_k=$top_k block_size=$block_size"

    python3 hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type sparse \
        --top-k "$top_k" \
        --block-size "$block_size" \
        --iterations 5 \
        --warmup 2 \
        --num-tokens 64 \
        --output "$output_file"

    echo "  -> Saved to $output_file"
}

# Dense baseline
echo ""
echo "=== Dense Baseline ==="
for seq_len in "${SEQ_LENS[@]}"; do
    python3 hpc_benchmark.py \
        --model "$MODEL" \
        --seq-len "$seq_len" \
        --attention-type dense \
        --iterations 5 \
        --warmup 2 \
        --num-tokens 64 \
        --output "${OUTPUT_DIR}/dense_seq${seq_len}.json"
done

# Sparse sweep
echo ""
echo "=== Sparse Parameter Sweep ==="
for seq_len in "${SEQ_LENS[@]}"; do
    for top_k in "${TOP_KS[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            run_benchmark "$seq_len" "$top_k" "$block_size"
        done
    done
done

# Generate summary
echo ""
echo "=========================================="
echo "Generating Summary"
echo "=========================================="

python3 -c "
import json
import glob

results = []
for f in sorted(glob.glob('${OUTPUT_DIR}/*.json')):
    try:
        with open(f) as fp:
            results.append(json.load(fp))
    except:
        pass

# Group by config
configs = {}
for r in results:
    key = f\"top_k={r.get('top_k', 0)}_block={r.get('block_size', 0)}_seq={r.get('seq_len', 0)}\"
    configs[key] = r

# Find dense baseline for each seq_len
dense = {r['seq_len']: r for r in results if r['attention_type'] == 'dense'}

print()
print('='*80)
print('PARAMETER SWEEP RESULTS')
print('='*80)
print()
print(f'{'SEQ_LEN':>8} {'TOP_K':>6} {'BLOCK':>6} {'SPEED':>10} {'BASELINE':>10} {'SPEEDUP':>8}')
print('-'*80)

for seq_len in sorted(set(r['seq_len'] for r in results)):
    baseline = dense.get(seq_len, {}).get('avg_speed_tokens_per_sec', 0)
    print(f\"Sequence length: {seq_len} (baseline: {baseline:.1f} tok/s)\")

    sparse_runs = [r for r in results if r['attention_type'] == 'sparse' and r['seq_len'] == seq_len]
    for r in sorted(sparse_runs, key=lambda x: (x['top_k'], x['block_size'])):
        speed = r['avg_speed_tokens_per_sec']
        speedup = speed / baseline if baseline > 0 else 0
        print(f\"  {r['top_k']:>6} {r['block_size']:>6} {speed:>10.1f} {baseline:>10.1f} {speedup:>7.2f}x\")

print()
print('='*80)
"