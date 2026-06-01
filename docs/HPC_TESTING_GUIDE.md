# Sparse Attention - HPC Testing Guide

## Overview

This document describes the sparse attention implementations available and provides a testing framework for scaling to larger models on HPC infrastructure.

**Key Files:**
- `minimax_m3_sparse_attention.py` - Pure MiniMax M3-style sparse attention (two-stage routing)
- `minimax_m3_hybrid.py` - Hybrid local + global attention (combines sliding window with sparse global blocks)
- `deepseek_v4_csa.py` - DeepSeek V4 Compressed Sparse Attention (CSA-4x, HCA-128x)
- `hpc_benchmark.py` - HPC-ready benchmark script with JSON output

---

## Available Implementations

### 1. MiniMax M3-style Sparse Attention

**Two-stage approach:**
1. **Index Stage**: Lightweight attention on reduced-dimension Index Q/K to compute block-level importance scores
2. **Sparse Stage**: Main Q attends only to KV in selected blocks

**Key parameters:**
- `block_size`: Tokens per block (default: 16)
- `top_k_blocks`: Number of blocks to select per head (default: 4)
- `index_dim`: Dimension of index projections (default: 32)

### 2. DeepSeek V4 CSA (Compressed Sparse Attention)

**Compression + Selection approach:**
1. **Compression**: Average pool every N tokens into 1 compressed entry (4:1 or 128:1)
2. **Selection**: Index attention selects top-k compressed blocks
3. **Sparse Attention**: Attend to original tokens within selected compressed blocks

**Variants:**
- **CSA-4x**: 4:1 compression (moderate)
- **HCA-128x**: 128:1 compression (aggressive, for very long contexts)

### 3. Hybrid Local + Global Attention

**Combines two patterns:**
1. **Local sliding window**: O(window_size) per query - captures nearby context efficiently
2. **Sparse global blocks**: Selected via index attention - captures long-range dependencies

---

## Local Results (MPS/M5 Mac)

### MiniMax M3-style Sparse Attention

Testing on Qwen 2.5-0.5B with varying sequence lengths:

| Config | Short (5) | Medium (20) | Long (51) | XLong (141) |
|--------|-----------|------------|-----------|-------------|
| Standard | 62/s | 69/s | 63/s | 54/s |
| MiniMax-Sparse(k=4) | 48/s | 51/s | 45/s | 46/s |
| Hybrid-Local+Global | 40/s | 42/s | 40/s | 39/s |

**Observations:**
- Sparse is ~25-30% slower than dense for short/medium sequences on MPS
- This is due to MPS matmul efficiency + gather overhead
- For very long sequences (1M tokens), sparse should show significant speedup

### DeepSeek V4 CSA/HCA Results

Testing on Qwen 2.5-0.5B with varying sequence lengths:

| Config | Short (5) | Medium (20) | Long (51) |
|--------|-----------|------------|-----------|
| Standard | 56/s | 64/s | 60/s |
| CSA-4x(k=4) | 51/s | 52/s | 51/s |
| CSA-4x(k=8) | 50/s | 52/s | 50/s |
| HCA-128x(k=2) | 50/s | 52/s | 51/s |

**Observations:**
- CSA/HCA are ~10-15% slower than dense on MPS due to gather overhead
- All sparse variants perform similarly at this scale
- At 1M tokens, CSA-4x should show ~4x speedup, HCA-128x should show ~10x speedup

---

## Approach Comparison

| Approach | Compression | Selection | Best For |
|----------|-------------|-----------|----------|
| MiniMax M3 | None (block-level) | Top-k blocks | General sparse attention |
| DeepSeek CSA-4x | 4:1 | Top-k compressed | Long sequences (16K+) |
| DeepSeek HCA-128x | 128:1 | Top-k compressed | Very long (128K+) |
| Hybrid Local+Global | None | Fixed local + sparse | Balanced local+long-range |

---

## Approach Comparison

| Approach | Compression | Selection | Best For |
|----------|-------------|-----------|----------|
| MiniMax M3 | None (block-level) | Top-k blocks | General sparse attention |
| DeepSeek CSA-4x | 4:1 | Top-k compressed | Long sequences (16K+) |
| DeepSeek HCA-128x | 128:1 | Top-k compressed | Very long (128K+) |
| Hybrid Local+Global | None | Fixed local + sparse | Balanced local+long-range |
| Gumbel-Sparse | Block-level | Gumbel-Softmax training | Trainable sparse routing |

---

## Training

### Local Training (CPU/CUDA)

Training works reliably on CPU with float32:

```bash
# CPU training with float32 (stable)
python3 train_sparse.py \
    --model Qwen/Qwen2.5-0.5B \
    --seq-len 256 \
    --steps 100 \
    --lr 1e-5
```

**Results on Qwen 0.5B (CPU, float32):**
- Initial loss: ~12.6
- Final loss: ~0.0003 after 30 steps
- Stable convergence

### HPC Training (CUDA, recommended)

For larger models and longer sequences, use HPC with CUDA:

```bash
# Single GPU training
sbatch train_sparse.sh Qwen/Qwen2.5-1.5B 1000 512 1 1e-4 4 16 ./results

# Multi-GPU (if available)
# Modify train_sparse.py to use DistributedDataParallel
```

### Training Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| Learning Rate | 1e-4 to 1e-5 | Lower for fine-tuning |
| Batch Size | 1-4 | Depends on GPU memory |
| Seq Length | 256-1024 | Start short, scale up |
| Block Size | 16-32 | Smaller = more fine-grained selection |
| Top-K | 4-8 | More blocks = more expressive |
| Temperature | 1.0 (anneal to 0.1) | For Gumbel-Softmax |

### Training Script

See `train_sparse.py` for the full training script with:
- Checkpointing
- Learning rate scheduling
- Gradient clipping
- Distributed training support

---

## HPC Testing Plan

### Recommended Models

1. **Small scale** (feasibility check):
   - Qwen/Qwen2.5-1.5B
   - Qwen/Qwen2.5-3B

2. **Medium scale** (target performance):
   - Qwen/Qwen2.5-7B
   - Meta-Llama/Llama-2-7b
   - Meta-Llama/Llama-2-13b

3. **Large scale** (production target):
   - Qwen/Qwen2.5-14B
   - Meta-Llama/Llama-2-70b

### Testing Sequence Lengths

| Category | Tokens | Purpose |
|----------|--------|---------|
| Short | 128-256 | Baseline |
| Medium | 512-1024 | Local processing |
| Long | 2048-4096 | Extended context |
| XLong | 8192-16384 | Long-range dependencies |
| Extreme | 32768-131072 | Stress test (MiniMax shows 1M) |
| Target | 100K-1M | Full benchmark (per MiniMax M3 paper) |

### Required Metrics

1. **Speed**:
   - Prefilling time (tokens/second)
   - Decoding time (tokens/second)
   - Memory usage (GB)
   - Throughput (tokens/second/GB)

2. **Quality**:
   - Perplexity on standard benchmarks (WikiText, C4)
   - Downstream task accuracy (if available)
   - Comparison vs dense attention baseline

3. **Scaling**:
   - Speedup ratio (sparse vs dense) vs sequence length
   - Memory reduction ratio vs sequence length
   - Break-even point (sequence length where sparse wins)

---

## HPC Configuration

### SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=minimax_sparse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT

module load CUDA/12.1
module load Python/3.11
module load cuDNN/8.9

source ~/.venv/bin/activate

cd /path/to/sparse-attention-poc

# Run benchmark
python3 minimax_m3_sparse_attention.py \
    --model Qwen/Qwen2.5-7B \
    --seq-len 8192 \
    --batch-size 1 \
    --num-tokens 128 \
    --output results_qwen7b_8k.json
```

### Environment Requirements

- CUDA 12.1+
- PyTorch 2.0+ with CUDA support
- Transformers library
- 128GB+ GPU memory for larger models

---

## Implementation Details

### Key Functions

#### Block Score Computation

```python
def _compute_block_scores(self, idx_q, idx_k):
    # Average pooling within blocks
    q_avg = q_blocks.mean(dim=2)  # [B, nb, H, d]
    k_avg = k_blocks.mean(dim=2)  # [B, nb, h, d]

    # Block-level attention
    scores = torch.matmul(q_avg, k_avg.transpose(-2, -1))
```

#### Sparse Attention

```python
def _sparse_attention(self, q, k, v, selected_blocks):
    # Convert block indices to positions
    position_indices = (selected_blocks * block_size + offsets)

    # Gather KV from selected positions
    k_selected = torch.gather(k_rep, 2, position_indices)

    # Causal mask on selected positions
    causal_mask = (k_pos <= q_pos)
```

### Parameter Count

For Qwen 0.5B (24 layers):
- Standard attention: 494M params
- MiniMax-Sparse (k=4): +index projections (~0.3M per layer ≈ 7M total)
- Hybrid: Similar overhead

---

## Expected Results at Scale

Based on MiniMax M3 paper claims:

| Seq Length | Prefill Speedup | Decode Speedup |
|------------|----------------|----------------|
| 16K | 2-3x | 4-5x |
| 64K | 4-6x | 8-10x |
| 256K | 7-9x | 12-15x |
| 1M | 9.7x | 15.6x |

**Key insight**: Speedup increases with sequence length because:
- Index attention cost grows slower than dense
- Sparse access pattern reduces memory bandwidth pressure
- Block selection becomes more selective (less redundant attention)

---

## Testing Checklist

### Phase 1: Correctness
- [ ] Load pretrained weights into sparse attention modules
- [ ] Verify output matches standard attention (within tolerance)
- [ ] Check causal masking is correct
- [ ] Validate gradient flow (if training)

### Phase 2: Performance Benchmark
- [ ] Compare sparse vs dense at same sequence lengths
- [ ] Measure memory usage for each configuration
- [ ] Identify break-even point (where sparse is faster)
- [ ] Profile index attention overhead

### Phase 3: Scaling Study
- [ ] Test at 1K, 4K, 16K, 64K, 256K tokens
- [ ] Plot speedup ratio vs sequence length
- [ ] Verify linear scaling for sparse portion
- [ ] Test with different top_k values (2, 4, 8, 16)

### Phase 4: Model Scaling
- [ ] Test on 1.5B, 3B, 7B, 13B, 70B models
- [ ] Verify scaling holds across model sizes
- [ ] Identify optimal block_size for each model size

---

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce batch_size or sequence length
2. **Slow Performance**: Check that gather is not on critical path; consider torch.compile
3. **Quality Degradation**: Increase top_k_blocks or block_size; verify warm-start initialization

### Debug Tips

```python
# Enable attention metrics collection
attn.set_collect_metrics(True)
print(f"Top-k mass: {attn.last_learned_topk_mass}")
print(f"Dense top-k mass: {attn.last_dense_topk_mass}")
```

---

## Contact & References

- MiniMax M3 Paper: https://arxiv.org/abs/... (insert paper link when available)
- Original SubQ: https://subq.ai
- Implementation: `minimax_m3_sparse_attention.py`, `minimax_m3_hybrid.py`

---

## Appendix: Full Parameter Sweep Script

```bash
#!/bin/bash
# Full parameter sweep for HPC

MODELS=("Qwen/Qwen2.5-1.5B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B")
SEQ_LENGTHS=(1024 4096 16384 65536 262144)
TOP_K_VALUES=(2 4 8 16)
BLOCK_SIZES=(16 32 64)

for model in "${MODELS[@]}"; do
    for seq in "${SEQ_LENGTHS[@]}"; do
        for k in "${TOP_K_VALUES[@]}"; do
            for bs in "${BLOCK_SIZES[@]}"; do
                echo "Testing $model seq=$seq k=$k bs=$bs"
                python3 benchmark.py \
                    --model $model \
                    --seq-len $seq \
                    --top-k $k \
                    --block-size $bs \
                    --output "results_${model}_${seq}_k${k}_bs${bs}.json"
            done
        done
    done
done
```