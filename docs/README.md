# Sparse Attention for Long Context LLMs

## Overview

This repository contains experiments with sparse attention mechanisms for LLMs with 200K-1M token context windows. Implements MiniMax M3-style two-stage sparse attention with learned block selection.

## Architecture

```
src/
├── minimax_m3_sparse_attention.py   # Main sparse attention (index + sparse attention)
├── minimax_m3_static.py             # Static window attention (no learning)
├── minimax_m3_exact.py              # Exact attention for comparison
├── minimax_m3_hybrid.py             # Hybrid attention variants
└── streaming_*.py                   # StreamingLLM variants
```

## Key Results (GH200 120GB GPU)

| Model | Context | Sparse | Dense | Speedup | Memory |
|-------|---------|--------|-------|---------|--------|
| Qwen2.5-14B | 512K | **2.3 tok/s** | 0.5 tok/s | **4.6x** | 70.7 GB |
| Qwen2.5-14B | 200K | **6.2 tok/s** | 2.5 tok/s | **2.5x** | 47.1 GB |
| Qwen2.5-14B | 128K | 9.5 tok/s | 5.2 tok/s | 1.8x | 41.3 GB |
| Yi34B | 128K | 3.2 tok/s | 2.6 tok/s | 1.2x | 78.8 GB |
| **Static window** | 128K | **12.0 tok/s** | 5.2 tok/s | **2.3x** | 38.2 GB |

**Key insight**: Static window attention (last 128 tokens) outperforms learned sparse at shorter contexts due to zero overhead.

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
├── benchmarks/              # Benchmark scripts
│   ├── benchmark_minimax.py      # MiniMax-M2.7 with sparse attention
│   ├── benchmark_static.py       # Static window attention
│   └── ...
├── scripts/                # HPC submission scripts
│   ├── run_benchmark_minimax.sh
│   └── ...
├── results/                # Benchmark results (JSON)
├── training/               # Training scripts (exploratory)
├── experiments/            # Experimental variants
└── docs/
    ├── README.md
    └── BENCHMARK_RESULTS.md
```

## Usage

### HPC Benchmark
```bash
cd scripts
sbatch run_benchmark_minimax.sh
```

### Local Test
```bash
source venv/bin/activate
python benchmarks/benchmark_static.py
```

## MiniMax-M2.7 Status

**Working:** Sparse attention inference at 8K context on 4xGH200 GPUs.

| Metric | Value |
|--------|-------|
| Model | MiniMax-M2.7 (456B MoE) |
| GPUs | 4x GH200 120GB |
| Context | 8K tokens |
| Speed | 2.3 tok/s |
| Attention layers replaced | 62/62 |
| Load time | ~90s |

**What works:**
- Model loads with `device_map="auto"` across 4 GPUs
- All 62 attention layers successfully replaced with `MiniMaxSparseAttention`
- Inference runs at 8K context

**What doesn't work (tested extensively):**
- 32K+: OOM even with chunked SDPA. Root cause: sparse attention requires full-sequence Q/K/V projections + index projections + block scores, exceeding available memory
- 128K+ with 4 nodes (16 GPUs): All 62 layers replaced but inference still OOMs at the attention matmul
- Chunked SDPA was implemented but GPU is still at 94-95GB utilization before chunking even starts

**Key insight:** Sparse attention on MiniMax-M2.7 at these context lengths actually uses MORE memory than dense attention because:
1. Extra index projections (MLP) on full sequence
2. Block scores matrix (O(num_heads × num_blocks²))
3. No Flash Attention optimization for sparse patterns
4. Dense attention benefits from highly optimized Flash Attention

**Compatibility fixes applied:**
- `create_causal_mask` patch: strips `cache_position` kwarg for older transformers
- `rope_type=None` patch: prevents crash in `modeling_rope_utils.py`
- Per-layer device placement: avoids device mismatch with `device_map="auto"`
- Weight shape inference: reads `num_heads` from actual weight shapes (not config)
- Chunked SDPA fallback: for sequences > 8192 tokens

## Limitations

- **MiniMax-M2.7 at 32K+**: Sparse attention uses more memory than optimized dense attention
- **Single GPU**: Too large even with sparse attention — needs 4 GPUs
- **Large models (32B+)**: Diminishing returns from sparse attention (compute-bound)
- **Learned vs Static**: Static wins at short contexts; learned may win at 1M+ tokens
