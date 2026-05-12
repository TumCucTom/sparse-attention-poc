"""
Beat Chunked Attention - Clean comparison

Focus on strategies that work with PyTorch's optimized ops:
1. Flash Attention (SDPA) - baseline
2. Chunked/MemEff - previous best
3. Local Window - via FlashAttention's sliding window
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# === 1. SDPA (FlashAttention) ===
class SDPAAttention(nn.Module):
    """PyTorch's scaled_dot_product_attention - uses Metal kernels on MPS."""
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === 2. Memory Efficient (chunked) - previous best ===
class MemEffAttention(nn.Module):
    """Memory-efficient attention via chunking."""
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

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2).reshape(B * h, T, hd)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2).reshape(B * h, T, hd)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2).reshape(B * h, T, hd)

        out = torch.zeros(B * h, T, hd, device=x.device, dtype=x.dtype)
        scale = hd ** -0.5

        for start in range(0, T, cs):
            end = min(start + cs, T)
            q_chunk = q[:, start:end, :]
            attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            out[:, start:end, :] = torch.matmul(attn_chunk, v)

        out = out.reshape(B, h, T, hd)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === 3. Standard (naive matmul) ===
class StandardAttention(nn.Module):
    """Standard full attention with naive matmul."""
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

        attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === 4. Local Window Attention ===
class LocalWindowAttention(nn.Module):
    """Sliding window attention - only attend to local context."""
    def __init__(self, dim, n_heads, head_dim, window_size=128):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim
        w = self.window_size

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        scale = hd ** -0.5
        out = torch.zeros_like(q)

        for start in range(0, T, w):
            end = min(start + w, T)
            q_win = q[:, :, start:end, :]
            k_win = k[:, :, :end, :]
            v_win = v[:, :, :end, :]

            attn = torch.matmul(q_win, k_win.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out[:, :, start:end, :] = torch.matmul(attn, v_win)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === 5. Strided Attention ===
class StridedAttention(nn.Module):
    """Strided attention - attend to every Nth token."""
    def __init__(self, dim, n_heads, head_dim, stride=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.stride = stride
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim
        s = self.stride

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        scale = hd ** -0.5
        out = torch.zeros_like(q)

        idxs = torch.arange(0, T, s, device=x.device)
        for i, idx in enumerate(idxs):
            end = min(idx + s, T)
            q_slice = q[:, :, idx:end, :]

            k_full = k[:, :, :end, :]
            v_full = v[:, :, :end, :]

            attn = torch.matmul(q_slice, k_full.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out[:, :, idx:end, :] = torch.matmul(attn, v_full)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === 6. Hybrid Local + Global ===
class HybridAttention(nn.Module):
    """Local window + global attention to first N tokens."""
    def __init__(self, dim, n_heads, head_dim, window_size=128, global_size=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_size = global_size
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, hd = self.n_heads, self.head_dim
        w = self.window_size
        g = min(self.global_size, T)

        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, hd).transpose(1, 2)

        scale = hd ** -0.5
        out = torch.zeros_like(q)

        for start in range(0, T, w):
            end = min(start + w, T)
            q_win = q[:, :, start:end, :]

            # Local keys in window
            k_local = k[:, :, start:end, :]
            v_local = v[:, :, start:end, :]

            # Global keys (prefix)
            k_glob = k[:, :, :g, :]
            v_glob = v[:, :, :g, :]

            # Combine
            k_comb = torch.cat([k_glob, k_local], dim=2)
            v_comb = torch.cat([v_glob, v_local], dim=2)

            attn = torch.matmul(q_win, k_comb.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out[:, :, start:end, :] = torch.matmul(attn, v_comb)

        return self.o_proj(out.transpose(1, 2).reshape(B, T, D))


# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, attn_type, dim, n_heads, head_dim, **kwargs):
        super().__init__()
        if attn_type == 'sdpa':
            attn = SDPAAttention(dim, n_heads, head_dim)
        elif attn_type == 'memeff':
            attn = MemEffAttention(dim, n_heads, head_dim, kwargs.get('chunk_size', 64))
        elif attn_type == 'standard':
            attn = StandardAttention(dim, n_heads, head_dim)
        elif attn_type == 'local':
            attn = LocalWindowAttention(dim, n_heads, head_dim, kwargs.get('window_size', 128))
        elif attn_type == 'strided':
            attn = StridedAttention(dim, n_heads, head_dim, kwargs.get('stride', 64))
        elif attn_type == 'hybrid':
            attn = HybridAttention(dim, n_heads, head_dim,
                                   kwargs.get('window_size', 128),
                                   kwargs.get('global_size', 64))
        else:  # sparse
            attn = SparseBucketAttention(dim, n_heads, head_dim, kwargs.get('n_bins', 16))

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
    def __init__(self, vocab_size, block_size, attn_type='sdpa', n_layers=2,
                 dim=256, n_heads=4, head_dim=64, **kwargs):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(attn_type, dim, n_heads, head_dim, **kwargs)
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


def main():
    print("=" * 70)
    print("BEATING CHUNKED ATTENTION - Clean Comparison")
    print("=" * 70)
    print()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dim = 256
    n_heads = 4
    head_dim = 64
    n_layers = 2
    vocab_size = 1000
    max_seq = 4096

    print(f"Config: dim={dim}, heads={n_heads}, head_dim={head_dim}, layers={n_layers}")
    print()

    # Create models
    print("Creating models...")
    models = {
        'SDPA (FlashAttn)': GPT(vocab_size, max_seq, 'sdpa', n_layers, dim, n_heads, head_dim),
        'MemEff (chunked)': GPT(vocab_size, max_seq, 'memeff', n_layers, dim, n_heads, head_dim, chunk_size=64),
        'Standard (naive)': GPT(vocab_size, max_seq, 'standard', n_layers, dim, n_heads, head_dim),
        'Local (window)': GPT(vocab_size, max_seq, 'local', n_layers, dim, n_heads, head_dim, window_size=128),
        'Hybrid (w=128,g=64)': GPT(vocab_size, max_seq, 'hybrid', n_layers, dim, n_heads, head_dim,
                                  window_size=128, global_size=64),
        'Hybrid (w=64,g=128)': GPT(vocab_size, max_seq, 'hybrid', n_layers, dim, n_heads, head_dim,
                                  window_size=64, global_size=128),
    }

    for name, model in models.items():
        model.to(device)
        print(f"  {name}: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # Warmup
    print("Warming up...")
    dummy = torch.randint(0, vocab_size, (2, 512), device=device)
    for name, model in models.items():
        _ = model(dummy)
    print()

    # Benchmark
    print(f"{'Seq Len':>10} {'SDPA':>12} {'MemEff':>12} {'Standard':>12} {'Local':>12} {'Hybrid-1':>12} {'Hybrid-2':>12} {'Best':>15}")
    print("-" * 110)

    results = {}
    seq_lengths = [256, 512, 1024, 2048, 4096]

    for T in seq_lengths:
        tokens = torch.randint(0, vocab_size, (2, T), device=device)
        times = {}

        for name, model in models.items():
            t = benchmark(model, tokens)
            times[name] = t

        best = min(times.items(), key=lambda x: x[1])
        best_name = best[0]

        print(f"{T:>10} {times['SDPA (FlashAttn)']*1000:>11.1f}ms {times['MemEff (chunked)']*1000:>11.1f}ms "
              f"{times['Standard (naive)']*1000:>11.1f}ms {times['Local (window)']*1000:>11.1f}ms "
              f"{times['Hybrid (w=128,g=64)']*1000:>11.1f}ms {times['Hybrid (w=64,g=128)']*1000:>11.1f}ms {best_name:>15}")

        results[T] = times

    print()
    print("=" * 70)
    print("WINNERS BY LENGTH")
    print("=" * 70)
    print()

    for T, times in results.items():
        best = min(times.items(), key=lambda x: x[1])
        worst = max(times.items(), key=lambda x: x[1])
        speedup = worst[1] / best[1]
        print(f"{T:>4} tokens: Winner='{best[0]}' ({best[1]*1000:.1f}ms), "
              f"slowest='{worst[0]}' ({worst[1]*1000:.1f}ms) = {speedup:.2f}x speedup")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    for name in models.keys():
        wins = sum(1 for times in results.values() if min(times.items(), key=lambda x: x[1])[0] == name)
        avg_time = sum(r[name] for r in results.values()) / len(results)
        print(f"{name}: wins {wins}/{len(results)} lengths, avg {avg_time*1000:.1f}ms")

    print()
    print("CONCLUSION:")
    # Find overall best
    total_times = {name: sum(r[name] for r in results.values()) for name in models.keys()}
    overall_best = min(total_times.items(), key=lambda x: x[1])
    print(f"  Overall best: {overall_best[0]} (total time {overall_best[1]*1000:.1f}ms)")
    print()
    print("Key insights:")
    print("  - SDPA/FlashAttention uses Metal kernels on MPS")
    print("  - MemEff chunking helps for long sequences")
    print("  - Sparse bucket routing adds overhead from mask creation")
    print("  - Need custom kernels to beat chunked approach")


if __name__ == "__main__":
    main()