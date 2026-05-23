# Sparse Attention POC

## What is this?

A proof-of-concept exploring sparse attention for LLMs, with two tracks:

1. **Practical runtime track**
   Fixed hybrid local+global attention that is fast enough to benchmark locally on Apple Silicon.

2. **Research track**
   Learned sparse routing inspired by SubQ / SSA, including dense-teacher distillation and sparse fine-tuning experiments.

## The Problem

Standard attention is O(T²) in sequence length. Even with chunked/memory-efficient approaches that limit context, we saw:
- **4096 tokens on MPS**: MemEff (chunked) = 104ms, Standard = 142ms

## Current Best Local Runtime

The most practical local runtime in this repo today is fixed hybrid attention:

1. **Local sliding window** (128 tokens) - captures nearby context efficiently
2. **Global prefix attention** (64 tokens) - cheap attention to first N tokens as a "summary"

This is currently the strongest path for local speed experiments.

## Benchmarks

**Device**: Apple M5 Mac MPS (PyTorch 2.8.0)  
**Config**: dim=256, heads=4, head_dim=64, layers=2

```
Seq Len         SDPA       MemEff     Standard        Local     Hybrid-1            Best
----------------------------------------------------------------------------------------------
   256         2.3ms       1.9ms        2.1ms        1.9ms        2.4ms  MemEff (chunked)
   512         6.2ms       4.2ms        5.1ms        4.1ms        3.7ms  Hybrid (w=128,g=64)
  1024        21.3ms      10.5ms       11.7ms       21.3ms        6.3ms  Hybrid (w=128,g=64)
  2048        60.2ms      31.9ms       33.9ms       61.4ms       12.6ms  Hybrid (w=128,g=64)
  4096       248.2ms     103.9ms      151.7ms      121.3ms       23.5ms  Hybrid (w=128,g=64)
```

**Key result at 4096 tokens**:
| Method | Time | Speedup vs MemEff |
|--------|------|-------------------|
| SDPA (FlashAttn) | 248ms | - |
| MemEff (chunked) | 104ms | 1x (baseline) |
| Hybrid (w=128,g=64) | **24ms** | **4.4x** |

## Winner Summary

| Method | Wins | Avg Time |
|--------|------|----------|
| SDPA | 0/5 | 67.6ms |
| MemEff | 1/5 | 30.5ms |
| Standard | 0/5 | 40.9ms |
| Local (window) | 0/5 | 42.0ms |
| **Hybrid (w=128,g=64)** | **4/5** | **9.7ms** |

## Files

- `benchmark.py` - Synthetic attention benchmark
- `benchmark_hybrid_qwen.py` - Safe local benchmark for dense vs fixed hybrid attention on Qwen
- `benchmark_subq.py` - Corrected sparse runtime benchmark variants on Qwen
- `benchmark_retrieval_qwen.py` - Small synthetic long-context retrieval benchmark
- `subq_runtime.py` - Shared HF-compatible sparse/hybrid runtimes
- `train_subq_hybrid.py` - Dense-teacher -> sparse fine-tune training experiments
- `subq_poc.py` - Original SubQ-inspired POC
- `docs/subq_ssa_reproduction_plan.md` - Gap-to-SSA reproduction plan
- `docs/LOCAL_SAFE_USAGE.md` - Safe local usage guidance

## Why This Works on MPS

Apple Metal has efficient matmul kernels but limited memory bandwidth. Our hybrid approach:
- Reduces total matmul operations by ~75% vs full attention
- Avoids the Python loop overhead that kills performance on CUDA
- Window size stays constant regardless of sequence length

## Current Status

- We have a **credible sparse-attention POC**.
- We do **not** yet have a full reproduction of SubQ / SSA’s learned sub-quadratic claims.
- The best **runtime** path locally is fixed hybrid local+global attention.
- The best **training** path locally is dense-teacher supervised sparse routing.

## Limitations & Next Steps

1. **Runtime kernels** - The learned sparse routing variants are still too gather-heavy to be the best runtime path on MPS.

2. **Long-context proof** - We do not yet have credible 128K+ scaling evidence locally.

3. **Retrieval quality** - We still need stronger long-context retrieval evaluation.

4. **Scaled training** - The training recipe needs larger models, longer contexts, and likely HPC resources to move beyond toy LM scale.

## References

- [karpathy/microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) - Base implementation
- [SubQ](https://subq.ai) - Sparse attention via learned routing
- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO-aware exact attention
- [Longformer](https://arxiv.org/abs/2004.05150) - Local + global attention pattern

## Usage

```bash
python3 benchmark.py
```

For safe local HF runs, see:

```bash
cat docs/LOCAL_SAFE_USAGE.md
```
