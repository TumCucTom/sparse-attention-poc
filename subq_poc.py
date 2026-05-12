"""
SubQ-inspired Sparse Attention using PyTorch

Based on Subquadratic's SubQ: instead of O(n²) attention over all tokens,
route each token to buckets based on content, then only attend within the same
bucket(s). Complexity becomes O(n*k) where k = avg bucket size << n.

Extension of karpathy/microgpt: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class SparseAttention(nn.Module):
    """
    Content-based routing inspired by SubQ.

    Instead of O(n²) attention over all tokens, we:
    1. Hash each token's key into buckets using a learned projection
    2. For each query, attend only to tokens in same bucket

    Complexity: O(n) to hash, O(n*k) for attention where k = avg bucket size << n
    """
    def __init__(self, n_heads: int, head_dim: int, n_bins: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_bins = n_bins

        # Learned hash projection
        self.hash_proj = nn.Linear(n_heads * head_dim, n_bins, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q, k, v: [batch, n_heads, T, head_dim]
        Returns: [batch, n_heads, T, head_dim]
        """
        batch_size, n_heads, T, head_dim = q.shape

        # Combine heads for hashing: [batch, T, n_heads * head_dim]
        k_reshaped = k.transpose(1, 2).reshape(batch_size, T, n_heads * head_dim)

        # Hash keys to get bucket assignments: [batch, T]
        hash_logits = self.hash_proj(k_reshaped)  # [batch, T, n_bins]
        bucket_idxs = hash_logits.argmax(dim=-1)  # [batch, T]

        outputs = []
        for b in range(batch_size):
            bucket_expanded = bucket_idxs[b].unsqueeze(1)  # [T, 1]
            same_bucket = (bucket_expanded == bucket_expanded.T)  # [T, T]

            # Causal mask
            pos_indices = torch.arange(T, device=q.device).unsqueeze(0)
            causal_mask = pos_indices <= pos_indices.T

            mask = same_bucket & causal_mask

            head_outputs = []
            for h in range(n_heads):
                q_b_h = q[b, h]
                k_b_h = k[b, h]
                v_b_h = v[b, h]

                scores = torch.matmul(q_b_h, k_b_h.T) / (head_dim ** 0.5)
                scores = scores.masked_fill(~mask, float('-inf'))

                attn_weights = F.softmax(scores, dim=-1)
                out_b_h = torch.matmul(attn_weights, v_b_h)
                head_outputs.append(out_b_h)

            outputs.append(torch.stack(head_outputs, dim=0))

        return torch.stack(outputs, dim=0)


class MultiHeadSparseAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, n_bins: int = 16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.attn = SparseAttention(n_heads, head_dim, n_bins=n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, T, dim = x.shape

        q = self.q_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, T, dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, mlp_ratio: int = 4, n_bins: int = 16):
        super().__init__()
        self.attn = MultiHeadSparseAttention(dim, n_heads, head_dim, n_bins)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layers: int = 4,
                 dim: int = 256, n_heads: int = 4, head_dim: int = 64, n_bins: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, n_bins=n_bins)
            for _ in range(n_layers)
        ])

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        T = tokens.size(1)
        x = self.token_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new: int = 50, temperature: float = 1.0):
        for _ in range(max_new):
            idx_cond = tokens[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == self.vocab_size - 1).all():
                break

        return tokens


class Tokenizer:
    def __init__(self, text: str):
        self.uchars = sorted(set(text))
        self.BOS = len(self.uchars)
        self.vocab_size = len(self.uchars) + 1

    def encode(self, s: str):
        return [self.uchars.index(c) for c in s]

    def decode(self, seq):
        return ''.join([self.uchars[i] for i in seq if i < self.BOS])


# Expanded training data - more names and variations
TRAINING_TEXT = """
Alice Bob Charlie Diana Edward Frank Grace Henry Isabel Julia Karl Laura Maria Nathan Olivia Paul Quincy Ruby Sophia Thomas Uma Victor Wendy Xavier Yvonne Zachary
Emma Noah Liam Olivia Ava William James Sophia Isabella Benjamin Elijah Charlotte Noah Samuel Mia Evelyn Henry Alexander Theodore
Harper Evelyn Abigail Sofia Scarlett Elizabeth Rosemary Daisy Penelope Violet Aurora Fiona Phoebe Rose Lily Claire Audrey Ruby Jasmine
Luna Claire Magnolia Pearl Penelope Ivy Aurora Diana Valentina Sofia Camila Gabriela Elena Paula Camila Isabella
James Robert Michael William David Joseph Charles Thomas Andrew Benjamin Nicholas Christopher Daniel Matthew Anthony Mark Steven
Maria Ana Paula Carla Sofia Isabella Carmen Rosa Pilar Lucia Elena Carmen Margarita Teresa Cristina
Thomas Christopher William Robert Michael James David Andrew Paul Mark Joseph Steven Brian Kevin Timothy
Alexander Andrew Christopher James Matthew Robert William David Michael Joseph Daniel Thomas Benjamin Kevin
"""


def train_model():
    print("=== SubQ-Inspired Sparse Attention (PyTorch) ===")
    print("Content-based routing to buckets, attention only within bucket")
    print("O(n*k) complexity where k = avg bucket size << n")
    print()

    enc = Tokenizer(TRAINING_TEXT)
    seq = enc.encode(TRAINING_TEXT)
    block_size = max(64, len(seq))

    print(f"Vocab: {enc.vocab_size}, Block: {block_size}")
    print(f"Training text length: {len(seq)} tokens")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = GPT(
        vocab_size=enc.vocab_size,
        block_size=block_size,
        n_layers=4,
        dim=256,
        n_heads=4,
        head_dim=64,
        n_bins=16
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Prepare training data - create overlapping sequences
    seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    model.train()
    batch_size = 4
    seq_len = min(block_size, len(seq) - 1)

    for epoch in range(1000):
        optimizer.zero_grad()

        # Random starting positions - ensure we have enough room for seq_len
        max_start = max(0, len(seq) - seq_len - 1)
        start_indices = [random.randint(0, max_start) if max_start > 0 else 0 for _ in range(batch_size)]

        tokens_list = []
        targets_list = []
        for s in start_indices:
            tokens_list.append(seq_tensor[s:s + seq_len])
            targets_list.append(seq_tensor[s + 1:s + seq_len + 1])

        tokens = torch.stack(tokens_list)
        targets = torch.stack(targets_list)

        logits, loss = model(tokens, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: loss = {loss.item():.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")

    print()
    print("=== Generation ===")

    model.eval()
    test_prompts = ["Alice", "Bob", "Emma", "James", "Maria", "Thomas", "Harper", "Luna"]

    for prompt in test_prompts:
        tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(tokens, max_new=30, temperature=0.8)
        result = enc.decode(generated[0].tolist())
        print(f"  '{prompt}' -> '{result}'")

    print()
    print("Note: This is a POC. Real SubQ uses sophisticated learned routing.")


if __name__ == "__main__":
    import random
    random.seed(42)
    torch.manual_seed(42)

    train_model()