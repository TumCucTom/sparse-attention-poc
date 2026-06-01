# Sparse Attention for Long Context LLMs

## Motivation

Standard attention is O(T²) in sequence length. For long-context LLMs (128K-1M tokens), this becomes prohibitive — both in compute and memory. The goal of this project was to find a sparse attention approach that:

1. Scales to long contexts (128K+) without OOM
2. Maintains quality comparable to dense attention
3. Is practical to train and deploy

We worked through several sparse attention approaches, starting small and moving to HPC.

---

## Approach Progression

### 1. Initial Exploration: Synthetic Benchmarks (Local)

Before touching any model, we benchmarked attention primitives in isolation on synthetic tensors:
- **SDPA** (PyTorch FlashAttention)
- **MemEff** (memory-efficient chunked attention)
- **Standard** (naive matmul)
- **Local** (sliding window)
- **Hybrid** (local + global prefix)

**Key local finding at 4096 tokens (MPS/M5):**

| Method | Time | Speedup vs MemEff |
|--------|------|-------------------|
| SDPA | 248ms | - |
| MemEff | 104ms | 1x (baseline) |
| **Hybrid (w=128, g=64)** | **24ms** | **4.4x** |

**Insight:** Fixed hybrid (local sliding window + global prefix) is the fastest path locally. This informed our direction: real LLM workloads care about a few important tokens (local + global), not all-to-all attention.

### 2. SubQ / SSA-Inspired Sparse Routing (Local, Qwen 0.5B)

Tried learned sparse routing inspired by [SubQ](https://subq.ai) and SSA:

- A trainable router picks which tokens to attend to
- Implemented with various tricks: STE (straight-through estimator), gumbel-softmax, dense-teacher distillation

**Result:** Gradient blackout problem — the hard top-k selection kills gradients to the router. Multiple attempts (`train_subq_*.py`) failed to produce a working trained sparse model on Qwen 0.5B. The router either didn't learn or diverged.

**Lesson:** Pure learned sub-quadratic routing is hard. Two-stage approaches (learned index + sparse attention) are more stable.

### 3. MiniMax M3 Two-Stage Sparse Attention (Local → HPC)

Studied the [MiniMax M3](https://arxiv.org/abs/2412.12215) approach and implemented it from scratch:

**Stage 1 (Index):** Lightweight attention on reduced-dim Index Q/K to compute block importance
**Stage 2 (Sparse):** Main Q attends only to KV in selected top-k blocks

Key files:
- `src/minimax_m3_sparse_attention.py` — Pure M3 two-stage
- `src/minimax_m3_hybrid.py` — Hybrid local + global (sliding window + sparse global blocks)

**Local results (Qwen 0.5B, MPS):**

| Config | Short (5) | Medium (20) | Long (51) | XLong (141) |
|--------|-----------|-------------|-----------|-------------|
| Standard | 62/s | 69/s | 63/s | 54/s |
| MiniMax-Sparse(k=4) | 48/s | 51/s | 45/s | 46/s |
| Hybrid-Local+Global | 40/s | 42/s | 40/s | 39/s |

**Local finding:** Sparse is ~25-30% slower than dense for short/medium sequences on MPS — gather overhead + small problem size. The win only materializes at much longer contexts.

### 4. DeepSeek V4 CSA / HCA (Local, Qwen 0.5B)

Inspired by [DeepSeek's Compressed Sparse Attention](https://digg.com/ai/78gnmbpg), implemented CSA with 4:1 compression and HCA with 128:1 compression:

- **CSA-4x:** Compress every 4 tokens → 1 summary entry, then top-k selection
- **HCA-128x:** Compress every 128 tokens → 1 entry (extreme, for very long contexts)

**Local results (Qwen 0.5B, MPS):**

| Config | Short (5) | Medium (20) | Long (51) |
|--------|-----------|-------------|-----------|
| Standard | 56/s | 64/s | 60/s |
| CSA-4x(k=4) | 51/s | 52/s | 51/s |
| CSA-4x(k=8) | 50/s | 52/s | 50/s |
| HCA-128x(k=2) | 50/s | 52/s | 51/s |

**Insight:** CSA/HCA are similar to M3 in cost at this scale (gather dominates). HCA's 128x compression pays off only at very long sequences (1M+ tokens) where it reduces selection space dramatically.

### 5. Training Pipeline (Local)

Built a training pipeline using Gumbel-Softmax for differentiable block selection:
- `src/gumbel_sparse_attention.py` — Gumbel-Softmax based trainable sparse attention
- `src/trainable_sparse_attention.py` — Earlier attempt with custom router
- `train_sparse.py` — Full training script with checkpointing, LR scheduling
- `train_sparse.sh` — SLURM script for HPC

**Local finding (CPU, float32):** 30 steps: loss 12.64 → 0.0003 — stable convergence with standard attention. Sparse-path training hit NaN on MPS/CUDA with float16 (numerical issues with sparse ops). Needs float32 or mixed precision with careful scaling for HPC.

### 6. HPC Validation (HPC Cluster)

The big test: does sparse actually win at scale? Ran on HPC with MiniMax-M2.7 (456B MoE):

| Context | Nodes | GPUs | Dense | Sparse | Notes |
|---------|-------|------|-------|--------|-------|
| 128K | 16 | 64 | 1.4 tok/s | OOM (block_scores) | Sparse block_scores OOM |
| 256K | 64 | 256 | OOM | OOM | Memory-bound |

**Findings:**
- **128K works** with dense on 64 GPUs (1.4 tok/s, 90.3GB)
- **256K needs more memory** than 256 GPUs provide
- **Sparse path OOMs** at 128K: `block_scores` matrix is 48 × 256² = 3M entries per head, and with GQA this becomes prohibitive
- **Streaming attention replacement fails** because allocating new modules before freeing old ones causes OOM during replacement

---

## Key Lessons Learned

1. **Fixed hybrid (local + global) is fast and stable** — beats learned sparse at every scale we tested locally
2. **SubQ-style learned routing has gradient issues** — hard selection kills gradient flow
3. **MiniMax M3 is more stable** than learned routing — fixed two-stage with learned block importance
4. **CSA/HCA are similar to M3 at small scale** — win at very long contexts (1M+)
5. **Memory, not compute, is the bottleneck at scale** — block_scores matrix blows up for 100K+ tokens
6. **Module replacement is OOM-prone** — must use attention-native model loading, not replace-after-load

---

## Repository Structure

```
sparse-attention-poc/
├── src/                              # Core sparse attention implementations
│   ├── minimax_m3_sparse_attention.py  # M3 two-stage
│   ├── minimax_m3_hybrid.py            # M3 + local sliding window
│   ├── deepseek_v4_csa.py              # CSA-4x, HCA-128x
│   ├── gumbel_sparse_attention.py      # Gumbel-Softmax for training
│   └── trainable_sparse_attention.py   # Trainable router version
├── benchmarks/                       # Local + HPC benchmarks
├── scripts/                          # HPC submission scripts
├── training/                         # Training experiments
├── experiments/                      # Experimental variants
├── hpc_results_minimax/              # HPC benchmark results
├── hpc_benchmark.py                  # HPC benchmark script (sparse/hybrid/dense)
├── train_sparse.py                   # Training script
├── train_sparse.sh                   # SLURM training script
└── docs/
    ├── README.md                     # Detailed docs
    ├── HPC_TESTING_GUIDE.md          # HPC testing guide
    └── BENCHMARK_RESULTS.md          # Detailed results
```

---

## Usage

### Local benchmark (Qwen 0.5B on M5 Mac)
```bash
python src/minimax_m3_sparse_attention.py
python deepseek_v4_csa.py
```

### HPC benchmark (Qwen 1.5B-7B, 8K-64K context)
```bash
sbatch --gres=gpu:1 --mem=128G --time=04:00:00 \
  python3 hpc_benchmark.py \
    --model Qwen/Qwen2.5-7B \
    --seq-len 16384 \
    --attention-type sparse \
    --top-k 8 \
    --output results_qwen7b_16k.json
```

### Training (HPC, recommended float32)
```bash
sbatch train_sparse.sh Qwen/Qwen2.5-1.5B 1000 512 1 1e-5 4 16 ./results
```

### Local training (CPU, float32 — stable)
```bash
python3 train_sparse.py \
    --model Qwen/Qwen2.5-0.5B \
    --seq-len 256 \
    --steps 100 \
    --lr 1e-5
```

---

## GPU Hours Used

Total: ~170 GH200 hours

## Conclusions

1. **Local small-model work is essential** — synthetic benchmarks + Qwen 0.5B revealed which approaches are worth scaling up
2. **Fixed hybrid is the safest win** — 4.4x speedup locally, no training required
3. **M3/CSA work but need careful memory management** — block_scores OOM at 100K+ tokens on real models
4. **For MiniMax-M2.7 (456B MoE) at 128K, dense works** — sparse path OOMs
5. **HPC streaming-attention replacement is fragile** — would need to load model with attention from the start
