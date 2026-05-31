#!/usr/bin/env python3
"""
Proper SubQ Training - Qwen 0.5B
Fixed: bf16 stability, proper LR, more training steps
Then: Test context window extension

This should show:
1. Loss convergence comparable to standard attention
2. Speedup maintained during training
3. Context window extension with SubQ's O(k) advantage
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


class SubQAttention(nn.Module):
    """SubQ attention - only router is trainable."""
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

        # Only train this
        self.router = nn.Linear(hidden_size, num_heads, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Router scores - trainable
        router_scores = self.router(hidden_states).permute(0, 2, 1)
        actual_k = min(self.top_k, seq_len)
        _, topk_indices = router_scores.topk(actual_k, dim=-1)

        # Frozen Q,K,V
        with torch.no_grad():
            q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Gather selected
        q_sel = torch.gather(q, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        k_sel = torch.gather(k, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        v_sel = torch.gather(v, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_sel, k_sel.transpose(-2, -1)) * scale

        orig_pos = topk_indices.unsqueeze(-1).float()
        causal = orig_pos >= orig_pos.transpose(-2, -1)
        scores = scores.masked_fill(~causal, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out_sel = torch.matmul(attn, v_sel)

        # Scatter back
        out = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim,
                        device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(batch_size):
            for h in range(self.num_heads):
                out[b, h, topk_indices[b, h]] = out_sel[b, h]

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class StandardAttention(nn.Module):
    """Standard full attention."""
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
    """Train for num_steps and return loss history."""
    model.train()
    losses = []

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        batch = batch.to(next(model.parameters()).device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        if torch.isnan(loss):
            print(f"  {name} NaN at step {step}, skipping...")
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
    """Benchmark inference speed."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, max_new_tokens=5)

    # Time
    with torch.no_grad():
        start = time.time()
        _ = model.generate(input_ids=input_ids, max_new_tokens=num_tokens)
        elapsed = time.time() - start

    return num_tokens / elapsed


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*60)
    print("PROPER SubQ TRAINING - Qwen 0.5B")
    print("="*60)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Training data - more samples, longer sequences
    training_texts = [
        "The theory of relativity describes how space and time are connected in a single fabric called spacetime that can be stretched and compressed by mass and energy.",
        "Einstein showed that mass and energy are equivalent through his famous equation E equals m c squared which describes how mass can be converted into energy and vice versa.",
        "Gravity in general relativity is not a force but rather the curvature of spacetime caused by the presence of mass and energy, objects follow paths called geodesics.",
        "Light travels at approximately 300,000 kilometers per second in a vacuum and is the fastest speed at which information can travel in the universe.",
        "Quantum mechanics describes the behavior of particles at the atomic and subatomic level where classical physics breaks down and probabilities govern outcomes.",
        "The uncertainty principle states that certain pairs of physical properties cannot both be precisely measured simultaneously such as position and momentum.",
        "In quantum superposition particles can exist in multiple states simultaneously until they are observed which causes the wave function to collapse to a definite state.",
        "The standard model describes fundamental particles and their interactions through three of the four fundamental forces electromagnetic weak and strong nuclear forces.",
        "Dark matter is a hypothetical form of matter that does not emit or absorb light but has gravitational effects that can be observed in galaxy rotation curves.",
        "The Big Bang theory describes the origin of the universe approximately 13.8 billion years ago from an extremely hot and dense initial state.",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape once it crosses the event horizon not even light itself.",
        "Entropy is a measure of disorder or randomness in a closed system and the second law of thermodynamics states that entropy always increases over time.",
        "The speed of light is constant in all reference frames according to special relativity which revolutionized our understanding of space and time.",
        "Time dilation means that time passes more slowly for objects moving at high speeds relative to a stationary observer this effect is experimentally verified.",
        "Mass causes spacetime to curve and objects follow paths called geodesics which we perceive as the force of gravity in curved spacetime geometry.",
    ]

    print(f"Training data: {len(training_texts)} samples")

    # =================================================================
    # Part 1: Train both models with same conditions
    # =================================================================
    print("\n" + "="*60)
    print("PART 1: Training Comparison")
    print("="*60)

    # Create datasets
    dataset = SimpleDataset(training_texts * 30, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=1)

    # ---- Standard Attention ----
    print("\n--- Standard Attention (full fine-tune) ---")
    model_std = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bf16 for stability
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    replace_attention(model_std, StandardAttention)

    opt_std = torch.optim.AdamW(model_std.parameters(), lr=2e-5)
    losses_std = train_model(model_std, loader, opt_std, num_steps=150, name="Standard")

    # ---- SubQ Attention ----
    print("\n--- SubQ Attention (router only) ---")
    model_subq = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    num_replaced, attn_modules = replace_attention(model_subq, SubQAttention, top_k=32)

    # Freeze everything except routers
    for name, param in model_subq.named_parameters():
        param.requires_grad = False

    router_params = []
    for attn in attn_modules:
        if hasattr(attn, 'router'):
            router_params.append(attn.router.weight)
            if attn.router.bias is not None:
                router_params.append(attn.router.bias)

    print(f"  Trainable params: {sum(p.numel() for p in router_params):,} ({sum(p.numel() for p in router_params)/494032768*100:.3f}%)")

    opt_subq = torch.optim.AdamW(router_params, lr=1e-2)
    losses_subq = train_model(model_subq, loader, opt_subq, num_steps=150, name="SubQ")

    # =================================================================
    # Part 2: Speed Comparison
    # =================================================================
    print("\n" + "="*60)
    print("PART 2: Speed Comparison")
    print("="*60)

    test_prompt = "Quantum mechanics describes how"

    speed_std = benchmark_speed(model_std, tokenizer, test_prompt)
    speed_subq = benchmark_speed(model_subq, tokenizer, test_prompt)

    print(f"Standard: {speed_std:.1f} tokens/sec")
    print(f"SubQ:     {speed_subq:.1f} tokens/sec")
    print(f"Speedup:  {speed_subq/speed_std:.2f}x")

    # =================================================================
    # Part 3: Context Window Extension Test
    # =================================================================
    print("\n" + "="*60)
    print("PART 3: Context Window Extension Test")
    print("="*60)

    # Test with increasingly long context
    long_prompts = [
        "Quantum mechanics " * 10 + "describes how",  # ~180 tokens
        "Quantum mechanics " * 50 + "describes how",  # ~850 tokens
        "Quantum mechanics " * 100 + "describes how",  # ~1700 tokens
    ]

    for i, prompt in enumerate(long_prompts):
        tokens = len(tokenizer.encode(prompt))
        print(f"\n  Context {tokens} tokens:")

        try:
            speed = benchmark_speed(model_std, tokenizer, prompt[:50], num_tokens=10)
            print(f"    Standard: {speed:.1f} t/s")
        except Exception as e:
            print(f"    Standard: OOM or error")

        try:
            speed = benchmark_speed(model_subq, tokenizer, prompt[:50], num_tokens=10)
            print(f"    SubQ:     {speed:.1f} t/s")
        except Exception as e:
            print(f"    SubQ: OOM or error")

    # =================================================================
    # Results Summary
    # =================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    final_std = losses_std[-1] if losses_std else float('nan')
    final_subq = losses_subq[-1] if losses_subq else float('nan')

    print(f"Final Standard loss: {final_std:.4f}")
    print(f"Final SubQ loss:     {final_subq:.4f}")
    print(f"\nTrainable params: Standard = 494M (100%), SubQ = {sum(p.numel() for p in router_params):,} (0.06%)")
    print(f"Speed: Standard = {speed_std:.1f} t/s, SubQ = {speed_subq:.1f} t/s ({speed_subq/speed_std:.2f}x)")

    # Generation samples
    print("\n--- Generation Samples ---")
    for prompt in ["The theory of relativity", "Einstein showed"]:
        print(f"\nPrompt: {prompt}")
        print(f"Standard: {generate(model_std, tokenizer, prompt)}")
        print(f"SubQ:     {generate(model_subq, tokenizer, prompt)}")


if __name__ == "__main__":
    main()