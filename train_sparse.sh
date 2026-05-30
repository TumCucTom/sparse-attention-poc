#!/bin/bash
#SBATCH --job-name=sparse-train
#SBATCH --output=sparse-train-%j.out
#SBATCH --error=sparse-train-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Sparse Attention Training on HPC/CUDA"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Environment setup (adjust for your HPC cluster)
# module load cuda/12.1
# module load python/3.10
# source ~/venvs/subq/bin/activate

python3 --version
nvidia-smi

# Model sizes to test
MODELS=("Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-3B")

# Sequence lengths
SEQ_LENS=(4096 8192 16384)

# Top-K values
TOP_KS=(4 8 16)

# Output directory
OUTPUT_DIR="train_results"
mkdir -p "$OUTPUT_DIR"

for model in "${MODELS[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        for top_k in "${TOP_KS[@]}"; do
            model_name="${model##*/}"
            output_file="${OUTPUT_DIR}/${model_name}_seq${seq_len}_k${top_k}_results.json"

            echo ""
            echo "----------------------------------------"
            echo "Training: $model | seq=$seq_len | top_k=$top_k"
            echo "Output: $output_file"
            echo "----------------------------------------"

            python3 train_sparse_cuda.py \
                --model "$model" \
                --max-seq-len "$seq_len" \
                --batch-size 1 \
                --top-k "$top_k" \
                --block-size 16 \
                --index-dim 32 \
                --lr 1e-4 \
                --epochs 1 \
                --gradient-checkpointing \
                --use-bf16 \
                --warmup-steps 100 \
                --save-dir "$OUTPUT_DIR" \
                --log-interval 20

            echo "Completed: $model | seq=$seq_len | top_k=$top_k"
            echo ""

            # Clear GPU memory between runs
            python3 -c "import torch; torch.cuda.empty_cache()"
        done
    done
done

echo "=========================================="
echo "All training complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="