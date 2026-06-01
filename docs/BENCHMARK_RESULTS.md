# Sparse Attention Benchmark Results

**Hardware:** NVIDIA GH200 120GB GPU Cluster | **Date:** June 2026

---

## MiniMax-M2.7 Large Context Results (June 2026)

Testing MiniMax-M2.7 (456B MoE) at very large contexts using HPC cluster.

| Context | Nodes | GPUs | Dense | Notes |
|---------|-------|------|-------|-------|
| 128K | 16 | 64 | **1.4 tok/s** | Works! 90.3GB |
| 256K | 64 | 256 | OOM | Still memory-bound |

**Key findings:**
1. MiniMax-M2.7 at 128K works with dense attention on 64 GPUs (1.4 tok/s, 90.3GB peak memory)
2. Even 256 GPUs (64 nodes) cannot run 256K context - model is memory-bound
3. Sparse attention's block_scores matrix (O(num_blocks² × num_heads)) is the bottleneck
4. Streaming attention replacement fails because replacement requires allocating new modules before freeing old ones

**Why sparse fails on MiniMax-M2.7:**
- With block_size=512, 128K tokens = 256 blocks
- block_scores = 48 heads × 256² = ~3M entries per head
- With GQA (48 Q heads, 8 KV heads), this exceeds GPU memory

---

## Key Findings (Earlier Results)

### 1. Sparse Attention shines at very long contexts (200K+ tokens)

| Model | Context | Sparse | Dense | Speedup | Memory (sparse) |
|-------|---------|--------|-------|---------|-----------------|
| Qwen2.5-14B | 512K | **2.3 tok/s** | 0.5 tok/s | **4.6x** | 70.7 GB |
| Qwen2.5-14B | 200K | **6.2 tok/s** | 2.5 tok/s | **2.5x** | 47.1 GB |
| Qwen2.5-14B | 128K | 9.5 tok/s | 5.2 tok/s | 1.8x | 41.3 GB |
| Yi34B | 128K | 3.2 tok/s | 2.6 tok/s | 1.2x | 78.8 GB |

### 2. Static window attention beats learned sparse at shorter contexts

| Config | Speed | Notes |
|--------|-------|-------|
| **Static window (128 tokens)** | **12.0 tok/s** | Best overall at 128K |
| Learned sparse (top-k=4, block=32) | 9.5 tok/s | Index projection overhead hurts |
| Learned sparse (top-k=4, block=16) | 9.2 tok/s | Lower memory but slower |
| Dense baseline | 5.2 tok/s | Memory-bound at 128K |

### 3. Sparse attention is memory-bound at large contexts

At 200K+ tokens, sparse attention's O(top_k × block_size) memory footprint becomes a major advantage:
- Dense attention: O(seq_len) memory grows to ~80GB at 512K
- Sparse attention: O(top_k × block_size) stays ~70GB regardless of seq_len

### 4. Larger models (32B+) show diminishing sparse benefits

- **Qwen2.5-32B dense at 128K:** 3.0 tok/s — already compute-bound
- **Yi34B sparse vs dense at 128K:** 3.2 vs 2.6 tok/s (1.2x only)

Large models spend proportionally less time on attention vs compute-heavy FFN layers, so sparse attention's O(1) memory advantage doesn't translate to proportional speedups.

---

## Hyperparameter Sweep (Qwen2.5-14B @ 128K)

| Config | Memory | Speed |
|--------|--------|-------|
| k=4, block=32 | 66.3 GB | 9.5 tok/s |
| k=4, block=16 | 41.3 GB | 9.2 tok/s |
| k=8, block=16 | 55.2 GB | 8.7 tok/s |
| k=8, block=32 | 46.2 GB | 8.6 tok/s |
| k=16, block=16 | 59.0 GB | 8.1 tok/s |

**Optimal: k=4, block=32** for best speed; **k=4, block=16** for best memory.

---

## MiniMax M3-Style Sparse Attention

Implementation: [minimax_m3_sparse_attention.py](../src/minimax_m3_sparse_attention.py)

Two-stage approach:
1. **Index attention:** Small MLP (hidden_dim → index_dim) computes block importance scores
2. **Sparse attention:** Attend only to top-k blocks using standard SDPA

Key parameters:
- `block_size=32`: Number of tokens per block
- `top_k_blocks=4`: Number of blocks to attend to per head
- `index_dim=16`: Compressed representation dimension

---

## StreamingLLM-Style Alternative

Implementation: [streaming_llm_sparse.py](../src/streaming_llm_sparse.py)

This approach avoids the block_scores matrix entirely by using:
- **Sink tokens**: Last 4 tokens (always attended)
- **Sliding window**: Last N tokens (configurable, default 32)
- No block importance scoring needed

Memory: O(window_size) instead of O(num_blocks²)

---

## Limitations

1. **MiniMax-M2.7 at 128K+**: Dense attention works on 64 GPUs but sparse attention's block_scores matrix exceeds memory. Streaming attention replacement fails because it requires allocating new modules before freeing old ones.

2. **256K context**: Even 256 GPUs cannot handle MiniMax-M2.7 at 256K with dense attention. Would need model parallelism or CPU offloading strategies.

3. **Model-specific attention patterns:** Results vary significantly by model architecture (GQA vs MHA, FFN/compute ratio, etc.)

4. **Learned vs Static**: Static wins at short contexts; learned may win at 1M+ tokens

5. **MiniMax-M2.7 MoE size:** 456B params needs 4+ GPUs for fp16 inference — offloading strategies required for larger models

---

## GPU Hours Used

Total: ~170 GPU-hours (budget was 200 hours)

## Conclusions

1. **MiniMax-M2.7 at 128K works** with dense attention on 64 GPUs (1.4 tok/s)
2. **256K context needs more memory** than 256 GPUs provide - would need CPU offloading or model parallelism improvements
3. **Streaming attention replacement fails** because replacement requires allocating new modules before old ones are freed
4. **For very large contexts**, would need to load model with streaming attention from the start, not replace after loading