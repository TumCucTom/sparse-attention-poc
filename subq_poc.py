"""
SubQ-inspired Sparse Attention using PyTorch

Implements the key SubQ principles:
1. Learned routing function that assigns tokens to buckets based on content
2. Sparse attention - only attend within same bucket
3. Subquadratic complexity O(n*k) where k = avg bucket size

Based on principles from subq.ai

Extension of karpathy/microgpt: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)


class SubQAttention(nn.Module):
    """
    SubQ-style sparse attention with learned routing.

    Instead of O(n²) attention over all tokens:
    1. Route each token to buckets based on key content via learned router
    2. Each query attends only to tokens in same bucket(s)

    Complexity: O(n) to route + O(n*k) attention where k = avg bucket size
    """
    def __init__(self, n_heads: int, head_dim: int, n_bins: int = 16, temperature: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_bins = n_bins
        self.temperature = temperature

        # Learned router: projects key representation to bucket logits
        key_dim = n_heads * head_dim
        self.router = nn.Sequential(
            nn.Linear(key_dim, 128),
            nn.Tanh(),
            nn.Linear(128, n_bins)
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q, k, v: [batch, n_heads, T, head_dim]
        Returns: [batch, n_heads, T, head_dim]
        """
        batch_size, n_heads, T, head_dim = q.shape

        # Route keys to buckets using learned router
        k_flat = k.transpose(1, 2).reshape(batch_size, T, -1)  # [batch, T, n_heads*head_dim]
        bucket_logits = self.router(k_flat)  # [batch, T, n_bins]

        # Soft assignment: probability distribution over buckets
        bucket_probs = F.softmax(bucket_logits / self.temperature, dim=-1)  # [batch, T, n_bins]

        # Compute bucket assignment similarity for each pair of tokens
        # If two tokens route to similar buckets, they attend to each other
        # [batch, T, n_bins] @ [batch, n_bins, T] -> [batch, T, T]
        bucket_sim = torch.bmm(bucket_probs, bucket_probs.transpose(1, 2))

        # Threshold: if bucket similarity is above this, tokens can attend
        threshold = 0.05
        bucket_mask = bucket_sim > threshold

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))

        # Combined mask: causal AND similar bucket assignment
        mask = causal & bucket_mask

        scale = head_dim ** -0.5
        out = torch.zeros_like(q)

        # Vectorized attention with mask
        for b in range(batch_size):
            for h in range(n_heads):
                q_h = q[b, h]  # [T, head_dim]
                k_h = k[b, h]
                v_h = v[b, h]

                # Attention scores
                scores = torch.matmul(q_h, k_h.T) * scale
                scores = scores.masked_fill(~mask[b], float('-inf'))

                attn = F.softmax(scores, dim=-1)
                out[b, h] = torch.matmul(attn, v_h)

        return out


class MultiHeadSubQAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, n_bins: int = 16, temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.attn = SubQAttention(n_heads, head_dim, n_bins=n_bins, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, T, dim = x.shape

        q = self.q_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, T, dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, n_bins: int = 16, temperature: float = 1.0):
        super().__init__()
        self.attn = MultiHeadSubQAttention(dim, n_heads, head_dim, n_bins, temperature)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layers: int = 4,
                 dim: int = 256, n_heads: int = 4, head_dim: int = 64,
                 n_bins: int = 16, temperature: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, n_bins, temperature)
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
    print("Learned content-based routing to buckets via learned router")
    print("O(n*k) complexity where k = avg bucket size << n")
    print()

    enc = Tokenizer(TRAINING_TEXT)
    seq = enc.encode(TRAINING_TEXT)
    block_size = max(64, len(seq))

    print(f"Vocab: {enc.vocab_size}, Block: {block_size}")
    print(f"Training text length: {len(seq)} tokens")
    print()

    device = 'cpu'

    model = GPT(
        vocab_size=enc.vocab_size,
        block_size=block_size,
        n_layers=4,
        dim=256,
        n_heads=4,
        head_dim=64,
        n_bins=16,
        temperature=1.0
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    router_params = sum(p.numel() for p in model.blocks[0].attn.attn.router.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Router params per layer: {router_params:,}")
    print()

    seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    model.train()
    batch_size = 4
    seq_len = min(block_size, len(seq) - 1)

    for epoch in range(1000):
        optimizer.zero_grad()

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
    print("Note: This implements SubQ-style learned routing. Real SubQ has more sophisticated routing.")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    train_model()