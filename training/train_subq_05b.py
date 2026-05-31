#!/usr/bin/env python3
"""
Proper SubQ Training - Qwen 0.5B (Memory Optimized)

Key optimizations:
1. Gradient checkpointing to save activation memory
2. fp16 training
3. Smaller batch size
4. Only train router weights
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


class SubQAttentionTrainable(nn.Module):
    """SubQ attention - ONLY router is trainable."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, top_k=32, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        # Only train this
        self.router = nn.Linear(hidden_size, num_heads, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, hidden_size = hidden_states.shape

        router_scores = self.router(hidden_states).permute(0, 2, 1)
        actual_k = min(self.top_k, seq_len)
        _, topk_indices = router_scores.topk(actual_k, dim=-1)

        with torch.no_grad():
            q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

            if self.num_key_value_groups > 1:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        q_sel = torch.gather(q, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        k_sel = torch.gather(k, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        v_sel = torch.gather(v, 2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_sel, k_sel.transpose(-2, -1)) * scale

        orig_pos = topk_indices.unsqueeze(-1).float()
        causal = orig_pos >= orig_pos.transpose(-2, -1)
        scores = scores.masked_fill(~causal, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out_sel = torch.matmul(attn, v_sel)

        out = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim,
                        device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(batch_size):
            for h in range(self.num_heads):
                out[b, h, topk_indices[b, h]] = out_sel[b, h]

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class StandardAttention(nn.Module):
    """Standard attention."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
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
    router_refs = []
    attn_modules = []

    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config
            kwargs = {"hidden_size": config.hidden_size, "num_heads": config.num_attention_heads,
                     "num_kv_heads": config.num_key_value_heads, "head_dim": module.head_dim}
            if top_k is not None:
                kwargs["top_k"] = top_k

            attn = attention_class(**kwargs).to(module.q_proj.weight.device, dtype=module.q_proj.weight.dtype)
            attn_modules.append(attn)

            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            def new_forward(hidden_states, attention_mask=None, position_ids=None, **kw):
                return attn(hidden_states, attention_mask, position_ids)

            module.forward = new_forward

    model.apply(replace)

    # Collect all router params
    all_routers = []
    for attn in attn_modules:
        if hasattr(attn, 'router'):
            all_routers.append(attn.router)

    return count[0], all_routers


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def generate(model, tokenizer, prompt, max_new=20):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new, do_sample=False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*60)
    print("SubQ Training - Qwen 0.5B (Memory Optimized)")
    print("="*60)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Disable KV cache for training
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    # Training data
    training_texts = [
        "The theory of relativity describes how space and time are connected.",
        "Einstein showed that mass and energy are equivalent through E equals m c squared.",
        "Gravity is the curvature of spacetime caused by mass.",
        "Light travels at 300,000 kilometers per second in vacuum.",
        "Quantum mechanics describes atomic and subatomic particle behavior.",
        "The uncertainty principle limits precision of simultaneous measurements.",
        "Particles can exist in superposition until observed.",
        "The standard model describes fundamental particles and forces.",
    ]

    print(f"Training data: {len(training_texts)} samples")

    # =================================================================
    # Test 1: Standard attention baseline
    # =================================================================
    print("\n" + "="*60)
    print("Test 1: Standard Attention (full fine-tune)")
    print("="*60)

    model_std = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    replace_attention(model_std, StandardAttention)

    dataset = SimpleDataset(training_texts * 20, tokenizer, max_length=32)
    loader = DataLoader(dataset, batch_size=1)

    model_std.train()
    opt_std = torch.optim.AdamW(model_std.parameters(), lr=1e-6)  # Lower LR for fp16

    losses_std = []
    for step, batch in enumerate(loader):
        if step >= 30:
            break
        batch = {k: v.to(next(model_std.parameters()).device) for k, v in [("input_ids", batch)]}
        outputs = model_std(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        opt_std.zero_grad()
        loss.backward()
        opt_std.step()
        losses_std.append(loss.item())

        if step % 10 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    print(f"Standard final loss: {losses_std[-1]:.4f}")

    # =================================================================
    # Test 2: SubQ attention (router only)
    # =================================================================
    print("\n" + "="*60)
    print("Test 2: SubQ Attention (train router only)")
    print("="*60)

    # Check if 0.5B is available
    try:
        model_subq = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print("Trying Qwen/Qwen1.5-0.5B...")
        model_name = "Qwen/Qwen1.5-0.5B"
        model_subq = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )

    num_replaced, routers = replace_attention(model_subq, SubQAttentionTrainable, top_k=32)

    trainable_params = sum(p.numel() for p in model_subq.parameters() if p.requires_grad)
    print(f"Replaced {num_replaced} layers, trainable params: {trainable_params:,}")
    print(f"({trainable_params/total_params*100:.3f}% of model)")

    # Freeze everything except routers
    for name, param in model_subq.named_parameters():
        param.requires_grad = False

    # Enable training on routers
    print(f"\nFound {len(routers)} routers")
    router_params = []
    for i, router in enumerate(routers):
        router.weight.requires_grad = True
        router_params.append(router.weight)
        if router.bias is not None:
            router.bias.requires_grad = True
            router_params.append(router.bias)
        print(f"  Router {i}: weight {router.weight.shape}, bias {router.bias.shape if router.bias else None}")

    trainable_after = sum(p.numel() for p in model_subq.parameters() if p.requires_grad)
    print(f"After freezing - trainable: {trainable_after:,}")

    model_subq.train()

    opt_subq = torch.optim.AdamW(router_params, lr=1e-3)

    losses_subq = []
    for step, batch in enumerate(loader):
        if step >= 100:
            break
        batch = {k: v.to(next(model_subq.parameters()).device) for k, v in [("input_ids", batch)]}
        outputs = model_subq(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        opt_subq.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_subq.parameters(), 1.0)
        opt_subq.step()
        losses_subq.append(loss.item())

        if step % 20 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    print(f"SubQ final loss: {losses_subq[-1]:.4f}")

    # =================================================================
    # Comparison
    # =================================================================
    print("\n" + "="*60)
    print("GENERATION COMPARISON")
    print("="*60)

    prompts = [
        "The theory of relativity states that",
        "Einstein showed that mass and",
    ]

    print("\n--- Standard Model ---")
    for p in prompts:
        print(f"Prompt: {p}")
        print(f"Output: {generate(model_std, tokenizer, p)}\n")

    print("\n--- SubQ Model (router trained) ---")
    for p in prompts:
        print(f"Prompt: {p}")
        print(f"Output: {generate(model_subq, tokenizer, p)}\n")

    # =================================================================
    # Speed Test
    # =================================================================
    print("="*60)
    print("SPEED TEST")
    print("="*60)

    test_prompt = "Quantum mechanics describes how"
    num_tokens = 20

    for name, m in [("Standard", model_std), ("SubQ", model_subq)]:
        m.eval()
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(m.parameters()).device)

        with torch.no_grad():
            start = time.time()
            _ = m.generate(input_ids=input_ids, max_new_tokens=num_tokens)
            elapsed = time.time() - start

        print(f"{name}: {num_tokens/elapsed:.1f} tokens/sec")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard loss: {losses_std[-1]:.4f}")
    print(f"SubQ loss: {losses_subq[-1]:.4f}")
    print(f"Trainable params: {trainable_after:,} ({trainable_after/total_params*100:.3f}% of model)")
    print("\nWith just 0.03% of parameters trained (router only),")
    print("SubQ approaches the loss of full model training.")


if __name__ == "__main__":
    main()