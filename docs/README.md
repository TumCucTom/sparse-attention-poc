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
| 128K | 16 | 64 | **1.3 tok/s** | Works! 90.3GB |
| 256K | 16 | 64 | OOM | Needs more GPUs |

**Key finding:** MiniMax-M2.7 at 128K context works with dense attention on 64 GPUs. Sparse attention with block_scores matrix fails at 128K because O(num_blocks² × num_heads) exceeds memory.

### Smaller Models (Earlier Results)

| Model | Context | Sparse | Dense | Speedup | Memory |
|-------|---------|--------|-------|---------|--------|
| Qwen2.5-14B | 512K | **2.3 tok/s** | 0.5 tok/s | **4.6x** | 70.7 GB |
| Qwen2.5-14B | 200K | **6.2 tok/s** | 2.5 tok/s | **2.5x** | 47.1 GB |
| Qwen2.5-14B | 128K | 9.5 tok/s | 5.2 tok/s | 1.8x | 41.3 GB |
| Yi34B | 128K | 3.2 tok/s | 2.6 tok/s | 1.2x | 78.8 GB |
| **Static window** | 128K | **12.0 tok/s** | 5.2 tok/s | **2.3x** | 38.2 GB |

**Key insight**: Static window attention (last 128 tokens) outperforms learned sparse at shorter contexts due to zero overhead.

## Why Sparse Attention Fails on MiniMax-M2.7 at 128K

The block_scores matrix computation is the bottleneck:

```
num_blocks = seq_len / block_size = 131072 / 512 = 256 blocks
block_scores = num_heads × num_blocks² = 48 × 256² ≈ 3M entries per head
With GQA (8x KV replication): O(num_heads × num_blocks²) becomes memory-prohibitive
```

**Solution approach:** StreamingLLM-style attention that avoids the block_scores matrix entirely.

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
│   ├── run_benchmark_minimax.sh
│   ├── run_benchmark_128k_16nodes.sh
│   └── run_benchmark_streaming_16nodes.sh
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

Total: ~50 GPU-hours (budget was 200 hours)

| Job | Context | Nodes | Result | GPU-hours |
|-----|---------|-------|--------|-----------|
| 4955729 | Sparse 128K | 16 | OOM | 2.6 |
| 4955807 | Dense 128K | 16 | ✓ 1.3 tok/s | 16 |
| 4956083 | Streaming 128K | 16 | ✓ 1.3 tok/s* | 15.7 |
| 4956181 | Dense 256K | 16 | OOM | 16 |

*Streaming replacement didn't properly apply - still using dense attention

## References

- [MiniMax M3 Paper](https://arxiv.org/abs/2501.12599) - Two-stage sparse attention
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - Sink tokens + local window
- [SubQ](https://subq.ai) - Learned sparse routing