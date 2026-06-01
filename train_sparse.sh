#!/bin/bash
#SBATCH --job-name=sparse_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT

# HPC Training Script for Sparse Attention
# Usage: sbatch train_sparse.sh [MODEL] [STEPS]

set -e

# Module loading
module load CUDA/12.1
module load Python/3.11
module load cuDNN/8.9

# Virtual environment
source ~/.venv/bin/activate

# Arguments
MODEL=${1:-"Qwen/Qwen2.5-1.5B"}
STEPS=${2:-1000}
SEQ_LEN=${3:-512}
BATCH_SIZE=${4:-1}
LR=${5:-1e-4}
TOP_K=${6:-4}
BLOCK_SIZE=${7:-16}
OUTPUT_DIR=${8:-"./hpc_results"}

cd /Users/tom/talos-vs-macbook/sparse-attention-poc

# Create output directory
mkdir -p $OUTPUT_DIR

# Print config
echo "=============================================="
echo "HPC Sparse Attention Training"
echo "=============================================="
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "Seq Len: $SEQ_LEN"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Top-K: $TOP_K"
echo "Block Size: $BLOCK_SIZE"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Run training
python3 train_sparse.py \
    --model $MODEL \
    --steps $STEPS \
    --seq-len $SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --top-k $TOP_K \
    --block-size $BLOCK_SIZE \
    --checkpoint-dir $OUTPUT_DIR/checkpoints \
    --save-every 100 \
    --output $OUTPUT_DIR/results.json

echo "Training complete! Results in $OUTPUT_DIR"