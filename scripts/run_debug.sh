#!/bin/bash
#SBATCH --job-name=sparse-debug
#SBATCH --output=logs/sparse-debug-%j.out
#SBATCH --error=logs/sparse-debug-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="/home/b6ar/trvbale.b6ar/miniforge3/bin:$PATH"
export DEBUG_ATTN=1

echo "=========================================="
echo "Sparse Attention Debug Test"
echo "=========================================="

python3 test_model_sparse.py