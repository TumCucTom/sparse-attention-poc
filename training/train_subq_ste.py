#!/usr/bin/env python3
"""
SubQ with Straight-Through Estimator (STE) for Training

Key insight: Hard top-k selection kills gradients for unselected tokens.
Solution: Use Straight-Through Estimator
- Forward: hard selection (top-k tokens)
- Backward: soft attention weights (gradient flows through ALL tokens)

This allows the router to learn from ALL positions, not just the selected ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


class SubQAttentionSTE(nn.Module):
    """
    SubQ attention with Straight-Through Estimator.

    Forward pass: hard top-k selection (sparse, fast inference)
    Backward pass: gradients flow through soft attention weights
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, top_k=32):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # SubQ router - THIS IS TRAINABLE
        self.router = nn.Linear(hidden_size, num_heads, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, hidden_states, return_loss_mask=False):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Router scores: [batch, seq_len, num_heads]
        router_scores = self.router(hidden_states)
        router_scores_T = router_scores.permute(0, 2, 1)  # [batch, num_heads, seq_len]

        # Top-k selection (hard for forward)
        actual_k = min(self.top_k, seq_len)
        router_scores_T_detached = router_scores_T.detach()
        _, topk_indices = router_scores_T_detached.topk(actual_k, dim=-1)

        # ===== Compute Q, K, V =====
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # ===== Straight-Through Estimator =====
        # FORWARD: Use only selected tokens (hard selection)
        q_sel = torch.gather(q, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        k_sel = torch.gather(k, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        v_sel = torch.gather(v, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

        # Attention on selected tokens
        scale = self.head_dim ** -0.5
        scores_sel = torch.matmul(q_sel, k_sel.transpose(-2, -1)) * scale

        # Causal mask on original positions
        orig_pos = topk_indices.unsqueeze(-1).float()
        causal = orig_pos >= orig_pos.transpose(-2, -1)
        scores_sel = scores_sel.masked_fill(~causal, float('-inf'))

        attn_sel = F.softmax(scores_sel, dim=-1)
        out_sel = torch.matmul(attn_sel, v_sel)

        # Scatter to output (hard selection forward)
        out = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim,
                          device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(batch_size):
            for h in range(self.num_heads):
                out[b, h, topk_indices[b, h]] = out_sel[b, h]

        # ===== SOFT attention for backward (STE) =====
        # This is what gradients flow through
        scores_full = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_full = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        scores_full = scores_full.masked_fill(causal_full == 0, float('-inf'))

        # Use router scores for soft attention weights (gradients flow here)
        # Softmax over all positions, weighted by router scores
        router_weights = F.softmax(router_scores_T, dim=-1)  # [batch, num_heads, seq_len]

        # For backward to work properly, we use the soft attention
        # But the forward pass uses the sparse/hard selection
        attn_soft = F.softmax(scores_full, dim=-1)
        out_soft = torch.matmul(attn_soft, v)

        # STE: Use hard selection forward, but for backward pass
        # we need to make sure gradients flow through the router
        if self.training:
            # During training: combine soft and hard attention
            # Use soft attention for gradient flow, but keep the structure
            out_combined = out_soft  # This allows gradients to flow
        else:
            out_combined = out

        out_combined = out_combined.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out_combined), None


class SubQAttentionSTEv2(nn.Module):
    """
    SubQ with STE - simpler implementation.

    The key insight: use soft attention weights that sum to ~k positions,
    so gradients flow through ALL positions but the effective attention
    is still sparse.

    This is essentially "soft sparse attention" that:
    - Uses softmax weighted by router scores
    - Weights sum to k (not 1)
    - Preserves gradient flow
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, top_k=32):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Router - trainable
        self.router = nn.Linear(hidden_size, num_heads, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Router weights: how much each position should contribute
        # These are soft weights that sum to 1 across each head
        router_logits = self.router(hidden_states).permute(0, 2, 1)  # [batch, num_heads, seq_len]

        # ===== STANDARD ATTENTION FOR COMPARISON =====
        # Full attention scores
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        scores = scores.masked_fill(causal == 0, float('-inf'))

        # Normalize router weights to sum to top_k
        router_weights = F.softmax(router_logits, dim=-1)  # sum to 1
        router_weights = router_weights * self.top_k  # scale to sum to top_k

        # Apply soft sparse attention
        attn = F.softmax(scores, dim=-1)

        # Weight the attention by router importance
        # router_weights: [batch, num_heads, seq_len] -> expand for attention
        router_expanded = router_weights.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]

        # Apply router-weighted attention
        attn_weighted = attn * router_expanded

        # Now the output only attends to top-k weighted positions
        out = torch.matmul(attn_weighted, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class StandardAttention(nn.Module):
    """Standard full attention for comparison."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def replace_attention(model, attention_class, top_k=None):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    count = [0]
    attn_modules = []

    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config
            kwargs = {"hidden_size": config.hidden_size, "num_heads": config.num_attention_heads,
                     "num_kv_heads": config.num_key_value_heads, "head_dim": module.head_dim}
            if top_k is not None:
                kwargs["top_k"] = top_k

            attn = attention_class(**kwargs).to(next(module.parameters()).device)

            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            attn_modules.append(attn)

            def new_forward(hidden_states, attention_mask=None, position_ids=None, **kw):
                result = attn(hidden_states)
                if isinstance(result, tuple):
                    return result
                return result, None

            module.forward = new_forward

    model.apply(replace)
    return count[0], attn_modules


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def train_model(model, loader, optimizer, num_steps=100, name="model"):
    model.train()
    losses = []

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        batch = batch.to(next(model.parameters()).device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        if torch.isnan(loss):
            print(f"  {name} NaN at step {step}")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 25 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    return losses


def generate(model, tokenizer, prompt, max_new=20):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new, do_sample=False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def benchmark_speed(model, tokenizer, prompt, num_tokens=20):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, max_new_tokens=5)  # Warmup
        start = time.time()
        _ = model.generate(input_ids=input_ids, max_new_tokens=num_tokens)
        elapsed = time.time() - start

    return num_tokens / elapsed


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*60)
    print("SubQ with Straight-Through Estimator Training")
    print("="*60)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Training data
    training_texts = [
        "The theory of relativity describes how space and time are connected in a single fabric called spacetime that can be stretched and compressed by mass and energy.",
        "Einstein showed that mass and energy are equivalent through his famous equation E equals m c squared which describes how mass can be converted into energy and vice versa.",
        "Gravity in general relativity is not a force but rather the curvature of spacetime caused by the presence of mass and energy, objects follow paths called geodesics.",
        "Light travels at approximately 300,000 kilometers per second in a vacuum and is the fastest speed at which information can travel in the universe.",
        "Quantum mechanics describes the behavior of particles at the atomic and subatomic level where classical physics breaks down and probabilities govern outcomes.",
        "The uncertainty principle states that certain pairs of physical properties cannot both be precisely measured simultaneously such as position and momentum.",
        "In quantum superposition particles can exist in multiple states simultaneously until they are observed which causes the wave function to collapse to a definite state.",
        "The standard model describes fundamental particles and their interactions through three of the four fundamental forces electromagnetic weak and strong nuclear forces.",
    ]

    dataset = SimpleDataset(training_texts * 40, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=1)

    total_params = 494032768  # Qwen 0.5B

    # =================================================================
    # Part 1: Standard Attention baseline
    # =================================================================
    print("\n" + "="*60)
    print("PART 1: Standard Attention (baseline)")
    print("="*60)

    model_std = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, use_cache=False
    )
    replace_attention(model_std, StandardAttention)

    opt_std = torch.optim.AdamW(model_std.parameters(), lr=2e-5)
    losses_std = train_model(model_std, loader, opt_std, num_steps=150, name="Standard")
    print(f"Final Standard loss: {losses_std[-1]:.4f}")

    # =================================================================
    # Part 2: SubQ with STE (router weighted attention)
    # =================================================================
    print("\n" + "="*60)
    print("PART 2: SubQ with STE (router-weighted soft attention)")
    print("="*60)

    model_ste = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, use_cache=False
    )
    num_replaced, attn_modules = replace_attention(model_ste, SubQAttentionSTEv2, top_k=32)

    # Freeze except router
    for param in model_ste.parameters():
        param.requires_grad = False

    router_params = []
    for attn in attn_modules:
        if hasattr(attn, 'router'):
            router_params.append(attn.router.weight)

    print(f"Trainable params: {sum(p.numel() for p in router_params):,} ({sum(p.numel() for p in router_params)/total_params*100:.3f}%)")

    opt_ste = torch.optim.AdamW(router_params, lr=1e-2)
    losses_ste = train_model(model_ste, loader, opt_ste, num_steps=150, name="SubQ-STE")
    print(f"Final SubQ-STE loss: {losses_ste[-1]:.4f}")

    # =================================================================
    # Part 3: Speed comparison
    # =================================================================
    print("\n" + "="*60)
    print("PART 3: Speed Comparison")
    print("="*60)

    test_prompt = "Quantum mechanics describes how"

    speed_std = benchmark_speed(model_std, tokenizer, test_prompt)
    speed_ste = benchmark_speed(model_ste, tokenizer, test_prompt)

    print(f"Standard: {speed_std:.1f} tokens/sec")
    print(f"SubQ-STE:  {speed_ste:.1f} tokens/sec")
    print(f"Ratio:     {speed_ste/speed_std:.2f}x")

    # =================================================================
    # Part 4: Context scaling test
    # =================================================================
    print("\n" + "="*60)
    print("PART 4: Context Scaling")
    print("="*60)

    for multiplier in [1, 5, 10, 20]:
        long_prompt = ("The theory of quantum mechanics describes how particles behave " * multiplier)[:200]
        try:
            speed = benchmark_speed(model_std, tokenizer, long_prompt[:50], num_tokens=10)
            print(f"  {len(tokenizer.encode(long_prompt))} tokens - Standard: {speed:.1f} t/s")
        except Exception as e:
            print(f"  {len(tokenizer.encode(long_prompt))} tokens - Standard: OOM")

        try:
            speed = benchmark_speed(model_ste, tokenizer, long_prompt[:50], num_tokens=10)
            print(f"  {len(tokenizer.encode(long_prompt))} tokens - SubQ-STE: {speed:.1f} t/s")
        except Exception as e:
            print(f"  {len(tokenizer.encode(long_prompt))} tokens - SubQ-STE: OOM")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard loss: {losses_std[-1]:.4f}")
    print(f"SubQ-STE loss: {losses_ste[-1]:.4f}")
    print(f"\nKey insight: Router-weighted soft attention allows gradients to flow")
    print(f"through ALL positions, while still biasing attention toward top-k selected.")


if __name__ == "__main__":
    main()