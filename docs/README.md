# Sparse Attention for Long Context LLMs

## Overview

This repository contains experiments with sparse attention mechanisms for LLMs with 128K-1M token context windows. Implements MiniMax M3-style two-stage sparse attention with learned block selection, plus StreamingLLM-style alternatives.

## Architecture

```
src/
├── minimax_m3_sparse_attention.py   # Main sparse attention (index + sparse attention)
├── minimax_m3_static.py             # Static window attention (no learning)
├── minimax_m3_exact.py              # Exact attention for comparison
├── minimax_m3_hybrid.py             # Hybrid attention variants
├── streaming_llm_sparse.py          # StreamingLLM-style (no block_scores)
└── streaming_*.py                   # Other StreamingLLM variants
```

## Key Results

### MiniMax-M2.7 (456B MoE) Results (June 2026)

| Context | Nodes | GPUs | Dense | Notes |
|---------|-------|------|-------|-------|
| 128K | 16 | 64 | **1.4 tok/s** | Works! 90.3GB |
| 256K | 64 | 256 | OOM | Still memory-bound |

**Key finding:** MiniMax-M2.7 at 128K context works with dense attention on 64 GPUs (1.4 tok/s). Even 256 GPUs (64 nodes) is not enough for 256K context - the model is memory-bound at this scale.

**Why streaming attention didn't help:** Replacing attention modules on a loaded model requires allocating new modules before old ones can be freed, causing OOM during the replacement phase.

### Smaller Models (Earlier Results)

| Model | Context | Sparse | Dense | Speedup | Memory |
|-------|---------|--------|-------|---------|--------|
| Qwen2.5-14B | 512K | **2.3 tok/s** | 0.5 tok/s | **4.6x** | 70.7 GB |
| Qwen2.5-14B | 200K | **6.2 tok/s** | 2.5 tok/s | **2.5x** | 47.1 GB |
| Qwen2.5-14B | 128K | 9.5 tok/s | 5.2 tok/s | 1.8x | 41.3 GB |
| Yi34B | 128K | 3.2 tok/s | 2.6 tok/s | 1.2x | 78.8 GB |
| **Static window** | 128K | **12.0 tok/s** | 5.2 tok/s | **2.3x** | 38.2 GB |

**Key insight**: Static window attention (last 128 tokens) outperforms learned sparse at shorter contexts due to zero overhead.

## Why Sparse Attention Fails on MiniMax-M2.7 at Large Contexts

The block_scores matrix computation is the bottleneck:

```
num_blocks = seq_len / block_size = 131072 / 512 = 256 blocks
block_scores = num_heads × num_blocks² = 48 × 256² ≈ 3M entries per head
With GQA (8x KV replication): O(num_heads × num_blocks²) becomes memory-prohibitive
```

**Streaming attention limitation:** Even without block_scores, replacing attention modules on a loaded model requires allocating new modules before freeing old ones - causing OOM at replacement time.

**Solution approach:** Would need to either:
1. Load model with streaming attention from the start (no replacement)
2. Use CPU offloading for attention modules during replacement
3. Try with significantly more GPUs (64+ nodes = 256+ GPUs)

## Sparse Attention Mechanism

Two-stage approach inspired by MiniMax M3:

1. **Index attention**: Small MLP computes block importance scores
2. **Sparse attention**: Attend only to top-k blocks

```python
attn = MiniMaxSparseAttention(
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    head_dim=128,
    block_size=32,      # Tokens per block
    top_k_blocks=4,     # Blocks to attend to
    index_dim=16,       # Index projection dimension
)
```

## Repository Structure

```
sparse-attention-poc/
├── src/                    # Core sparse attention implementations
├── benchmarks/             # Benchmark scripts
│   ├── benchmark_minimax.py         # MiniMax-M2.7 benchmarks
│   ├── benchmark_minimax_very_long.py  # 128K-256K benchmarks
│   ├── benchmark_streaming_128k.py  # Streaming attention test
│   └── benchmark_static.py          # Static window attention
├── scripts/                # HPC submission scripts
├── hpc_results_minimax/    # Benchmark results (JSON)
├── training/               # Training scripts (exploratory)
├── experiments/            # Experimental variants
└── docs/
    ├── README.md
    └── BENCHMARK_RESULTS.md
```

## Usage

### HPC Benchmark (MiniMax-M2.7 at 128K)
```bash
cd scripts
sbatch run_benchmark_128k_16nodes.sh
```

### HPC Benchmark (Streaming Attention)
```bash
cd scripts
sbatch run_benchmark_streaming_16nodes.sh
```

### Local Test
```bash
source venv/bin/activate
python benchmarks/benchmark_static.py
```

## GPU Hours Used

Total: ~170 GPU-hours (budget was 200 hours)

## Conclusions

1. **MiniMax-M2.7 at 128K works** with dense attention on 64 GPUs (1.4 tok/s)
2. **256K context needs more memory** than 256 GPUs provide - would need CPU offloading or model parallelism improvements
3. **Streaming attention replacement fails** because replacement requires allocating new modules before old ones are freed
4. **For very large contexts**, would need to load model with streaming attention from the start, not replace after loading

## References

- [MiniMax M3 Paper](https://arxiv.org/abs/2501.12599) - Two-stage sparse attention
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - Sink tokens + local window
- [SubQ](https://subq.ai) - Learned sparse routing