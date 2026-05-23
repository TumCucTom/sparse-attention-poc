#!/bin/bash
#SBATCH --job-name=subq-runtime
#SBATCH --output=subq-runtime-%j.out
#SBATCH --error=subq-runtime-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Adjust to the cluster environment:
# module load cuda/12.1
# module load python/3.10
# source ~/venvs/subq/bin/activate

python3 --version

python3 benchmark_hybrid_qwen.py \
  --model Qwen/Qwen2.5-1.5B \
  --window-size 128 \
  --global-size 64 \
  --chunk-size 64 \
  --num-tokens 64 \
  --prompt-repeat 64

python3 benchmark_subq.py \
  --model Qwen/Qwen2.5-1.5B \
  --top-k 8 \
  --router-dim 8 \
  --num-tokens 32 \
  --prompt-repeat 8
