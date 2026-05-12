"""
SubQ-style Sparse Attention using PyTorch

Implements SubQ's key principles:
1. Each attention head has a learnable router
2. Router selects top-k tokens to attend to (content-dependent selection)
3. Attention computed only on selected subset with proper causal masking
4. Complexity O(k² + T) where k << T

Based on MoSA (Mixture of Sparse Attention) using Expert-Choice Routing.

Extension of karpathy/microgpt: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)


class SubQRouter(nn.Module):
    """
    Learnable router that scores each token for attention selection.
    Each head has its own router weights - selects top-k tokens to attend to.
    """
    def __init__(self, dim: int, n_heads: int, top_k: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.top_k = top_k
        self.router = nn.Linear(dim, n_heads, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, T, dim]
        returns: top_k indices [batch, n_heads, top_k]
        """
        batch_size, T, dim = x.shape

        # Router projects to per-head scores: [batch, T, n_heads]
        router_scores = self.router(x)

        # Permute to [batch, n_heads, T] for top-k selection per head
        router_scores = router_scores.permute(0, 2, 1)

        # Handle case where T < top_k
        actual_k = min(self.top_k, T)
        topk_scores, topk_indices = router_scores.topk(actual_k, dim=-1)

        return topk_indices, topk_scores


class SubQAttention(nn.Module):
    """
    SubQ-style sparse attention using top-k token selection.

    Key insight: each head selects which tokens to attend to based on learned routing.
    Attention is computed only on the selected subset.

    The causal mask must respect ORIGINAL token positions, not subset positions.
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int, top_k: int = 32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.top_k = top_k

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.router = SubQRouter(dim, n_heads, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, T, dim]
        returns: [batch, T, dim]
        """
        batch_size, T, dim = x.shape

        # Get selected indices per head
        topk_indices, _ = self.router(x)
        # topk_indices: [batch, n_heads, actual_k]

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Process each head
        out = torch.zeros_like(q)

        for b in range(batch_size):
            for h in range(self.n_heads):
                # Get selected token indices for this head: [actual_k]
                idxs = topk_indices[b, h]  # original positions
                actual_k = idxs.size(0)

                # Get Q,K,V for selected positions
                q_h = q[b, h, idxs]  # [actual_k, head_dim]
                k_h = k[b, h, idxs]
                v_h = v[b, h, idxs]

                # Compute attention scores
                scale = self.head_dim ** -0.5
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

                # Build causal mask based on ORIGINAL positions
                # token i can attend to token j if original_position_i >= original_position_j
                # We need to check which pairs satisfy this
                # Create matrix of original indices for selected positions
                orig_idx = idxs.unsqueeze(1)  # [actual_k, 1]
                orig_idx_t = idxs.unsqueeze(0)  # [1, actual_k]
                can_attend = orig_idx >= orig_idx_t  # [actual_k, actual_k]

                scores = scores.masked_fill(~can_attend, float('-inf'))

                # Softmax
                attn = F.softmax(scores, dim=-1)

                # Compute output
                out_h = torch.matmul(attn, v_h)

                # Place output back at selected positions
                out[b, h, idxs] = out_h

        out = out.transpose(1, 2).contiguous().view(batch_size, T, dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, top_k: int = 32):
        super().__init__()
        self.attn = SubQAttention(dim, n_heads, head_dim, top_k)
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
                 dim: int = 256, n_heads: int = 4, head_dim: int = 64, top_k: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim = dim

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, top_k)
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
    print("=== SubQ-Style Sparse Attention (PyTorch) ===")
    print("Top-k token selection per head via learned router")
    print("O(k² + T) complexity where k = top_k << T")
    print()

    enc = Tokenizer(TRAINING_TEXT)
    seq = enc.encode(TRAINING_TEXT)
    block_size = max(64, len(seq))

    print(f"Vocab: {enc.vocab_size}, Block: {block_size}")
    print(f"Training text length: {len(seq)} tokens")
    print()

    device = 'cpu'

    top_k = 32
    model = GPT(
        vocab_size=enc.vocab_size,
        block_size=block_size,
        n_layers=4,
        dim=256,
        n_heads=4,
        head_dim=64,
        top_k=top_k
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    router_params = sum(p.numel() for p in model.blocks[0].attn.router.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Router params per layer: {router_params:,}")
    print(f"Top-k per head: {top_k}")
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
    print("Note: This implements SubQ-style top-k routing. Real SubQ has more sophisticated routing.")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    train_model()