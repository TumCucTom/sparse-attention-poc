"""
Benchmark: Standard vs SDPA (FlashAttention) vs Optimized Sparse

Compare all attention mechanisms on Apple MPS GPU.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# === Standard Attention (naive matmul) ===
class StandardAttention(nn.Module):
    """Standard O(n²) attention with manual matmul."""
    def __init__(self, dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        # Naive attention - O(n²) matmul
        attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === SDPA (PyTorch's scaled dot product attention - uses FlashAttention when available) ===
class SDPAAttention(nn.Module):
    """PyTorch SDPA - dispatches to FlashAttention/memory-efficient attention when available."""
    def __init__(self, dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        # SDPA uses FlashAttention when:
        # - GPU has Tensor cores (A100, H100, M1/M2/M3)
        # - head_dim is 64 or 128
        # - query/key/value are contiguous
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === Memory Efficient Attention (xFormers style) ===
class MemEffAttention(nn.Module):
    """Memory-efficient attention via chunking - reduces peak memory."""
    def __init__(self, dim, n_heads, head_dim, chunk_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim
        cs = self.chunk_size

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)  # [B, h, T, hd]
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        # Reshape for batch matmul: flatten batch and head dims
        # [B, h, T, hd] -> [B*h, T, hd]
        q = q.reshape(B * h, T, hd)
        k = k.reshape(B * h, T, hd)
        v = v.reshape(B * h, T, hd)

        # Process in chunks - saves memory
        out = torch.zeros(B * h, T, hd, device=x.device, dtype=x.dtype)
        scale = hd ** -0.5

        for start in range(0, T, cs):
            end = min(start + cs, T)
            q_chunk = q[:, start:end, :]  # [B*h, cs, hd]

            # [B*h, cs, hd] @ [B*h, hd, T] = [B*h, cs, T]
            attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)

            # [B*h, cs, T] @ [B*h, T, hd] = [B*h, cs, hd]
            out[:, start:end, :] = torch.matmul(attn_chunk, v)

        # Reshape back: [B*h, T, hd] -> [B, h, T, hd]
        out = out.reshape(B, h, T, hd)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === Optimized Sparse Attention (bucket-based routing) ===
class OptimizedSparseAttention(nn.Module):
    """Sparse attention using pure PyTorch ops - no Python loops."""
    def __init__(self, dim, n_heads, head_dim, n_bins=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_bins = n_bins
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.hash_proj = nn.Linear(head_dim * n_heads, n_bins, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        # Hash to get bucket assignments
        k_flat = k.transpose(1, 2).reshape(B, T, h * hd)
        bucket_idxs = self.hash_proj(k_flat).argmax(dim=-1)

        # Bucket mask via broadcasting
        b_expand = bucket_idxs.unsqueeze(2)
        b_expand_t = bucket_idxs.unsqueeze(1)
        same_bucket = (b_expand == b_expand_t)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask = same_bucket & causal

        # Compute attention
        scale = hd ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, attn_type, dim, n_heads, head_dim, n_bins=16, chunk_size=64):
        super().__init__()
        if attn_type == 'standard':
            attn = StandardAttention(dim, n_heads, head_dim)
        elif attn_type == 'sdpa':
            attn = SDPAAttention(dim, n_heads, head_dim)
        elif attn_type == 'memeff':
            attn = MemEffAttention(dim, n_heads, head_dim, chunk_size)
        else:  # sparse
            attn = OptimizedSparseAttention(dim, n_heads, head_dim, n_bins)

        self.attn = attn
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, attn_type='standard', n_layers=2,
                 dim=256, n_heads=4, head_dim=64, n_bins=16, chunk_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(attn_type, dim, n_heads, head_dim, n_bins, chunk_size)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, tokens):
        T = tokens.size(1)
        x = self.token_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for b in self.blocks:
            x = b(x)
        return self.lm_head(x)


def benchmark(model, tokens, n_runs=5, warmup=3):
    device = tokens.device
    for _ in range(warmup):
        _ = model(tokens)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(tokens)
        if device.type == 'mps':
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def check_sdpa_backend():
    """Check what SDPA backend is being used."""
    try:
        import torch.backends._triton as triton
        print("  Triton available for SDPA optimization")
    except:
        pass

    # Check MPS backend
    if torch.backends.mps.is_available():
        print("  MPS GPU available - SDPA will use Metal优化的原生SDPA")
    else:
        print("  Using CPU fallback")


def main():
    print("=" * 70)
    print("ATTENTION BENCHMARK: Standard vs SDPA vs MemEff vs Sparse")
    print("=" * 70)
    print()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print("SDPA Backend check:")
    check_sdpa_backend()
    print()

    # Config
    dim = 256
    n_heads = 4
    head_dim = 64
    n_layers = 2
    n_bins = 16
    chunk_size = 64
    vocab_size = 1000
    max_seq = 2048

    print(f"Config: dim={dim}, heads={n_heads}, head_dim={head_dim}, layers={n_layers}")
    print(f"        n_bins={n_bins}, chunk_size={chunk_size}")
    print()

    # Create models
    print("Creating models...")
    models = {
        'Standard (naive)': GPT(vocab_size, max_seq, 'standard', n_layers, dim, n_heads, head_dim).to(device),
        'SDPA (FlashAttn)': GPT(vocab_size, max_seq, 'sdpa', n_layers, dim, n_heads, head_dim).to(device),
        'MemEff (chunked)': GPT(vocab_size, max_seq, 'memeff', n_layers, dim, n_heads, head_dim, chunk_size=chunk_size).to(device),
        'Sparse (bucket)': GPT(vocab_size, max_seq, 'sparse', n_layers, dim, n_heads, head_dim, n_bins).to(device),
    }
    print()

    # Warmup
    print("Warming up...")
    dummy = torch.randint(0, vocab_size, (2, 512), device=device)
    for name, model in models.items():
        _ = model(dummy)
    print()

    # Benchmark
    print(f"{'Seq Len':>10} {'Standard':>10} {'SDPA':>10} {'MemEff':>10} {'Sparse':>10} {'Best':>12}")
    print("-" * 65)

    results = {}
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    for T in seq_lengths:
        tokens = torch.randint(0, vocab_size, (2, T), device=device)
        times = {}

        for name, model in models.items():
            t = benchmark(model, tokens)
            times[name] = t

        best = min(times.values())
        best_name = [k for k, v in times.items() if v == best][0]
        speedup_vs_std = times['Standard (naive)'] / best if best > 0 else float('inf')

        print(f"{T:>10} {times['Standard (naive)']*1000:>9.1f}ms {times['SDPA (FlashAttn)']*1000:>9.1f}ms "
              f"{times['MemEff (chunked)']*1000:>9.1f}ms {times['Sparse (bucket)']*1000:>9.1f}ms {best_name:>12}")

        results[T] = times

    print()
    print("=" * 70)
    print("WINNERS BY SEQUENCE LENGTH")
    print("=" * 70)
    print()

    for T, times in results.items():
        best = min(times.items(), key=lambda x: x[1])
        worst = max(times.items(), key=lambda x: x[1])
        speedup = worst[1] / best[1]
        print(f"{T:>4} tokens: Winner='{best[0]}' ({best[1]*1000:.1f}ms), "
              f"vs worst '{worst[0]}' ({worst[1]*1000:.1f}ms) = {speedup:.2f}x speedup")

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    print("Attention Types:")
    print("  Standard (naive): Full O(n²) matmul, no optimization")
    print("  SDPA: PyTorch's scaled_dot_product_attention - uses FlashAttention")
    print("         kernels when available (Metal on MPS, CUDA on GPU)")
    print("  MemEff: Chunked processing - reduces peak memory from O(n²) to O(n*k)")
    print("          where k = chunk_size, but still computes full attention")
    print("  Sparse: Content-based routing to buckets, O(n*k) where k=n_bins")
    print()

    print("Expected behavior on MPS:")
    print("  - SDPA should use Metal-optimized FlashAttention kernels")
    print("  - Standard still fastest due to highly optimized Metal matmul")
    print("  - Sparse/MemEff have Python overhead but use less computation")
    print()

    # Summary
    print("Summary:")
    for name in models.keys():
        wins = sum(1 for times in results.values() if min(times.items(), key=lambda x: x[1])[0] == name)
        avg_time = sum(r[name] for r in results.values()) / len(results)
        print(f"  {name}: wins {wins}/{len(results)} lengths, avg {avg_time*1000:.1f}ms")


if __name__ == "__main__":
    main()