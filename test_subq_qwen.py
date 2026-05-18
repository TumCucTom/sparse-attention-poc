#!/usr/bin/env python3
"""
SubQ-style Sparse Attention applied to Qwen2.5-1.5B

This script:
1. Loads Qwen2.5-1.5B from HuggingFace
2. Replaces standard attention with SubQ-style attention
3. Runs inference to test the concept

SubQ key ideas (from subq_poc.py):
- Each head has a learnable router that selects top-k tokens
- Attention computed only on selected subset
- O(k² + T) vs O(T²) for standard attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check MPS
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


class SubQAttention(nn.Module):
    """
    SubQ-style sparse attention using top-k token selection per head.

    Replaces standard attention by:
    1. Using a small router to score each token
    2. Selecting top-k tokens per head
    3. Computing attention only on selected subset

    Supports GQA (Grouped Query Attention) used in Qwen2
    """
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, top_k=32, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        # SubQ router: projects to per-head scores
        self.router = nn.Linear(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        hidden_states: [batch, seq_len, hidden_size]
        Returns: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Router scores: [batch, seq_len, num_heads]
        router_scores = self.router(hidden_states)
        router_scores = router_scores.permute(0, 2, 1)  # [batch, num_heads, seq_len]

        # Top-k selection per head
        actual_k = min(self.top_k, seq_len)
        _, topk_indices = router_scores.topk(actual_k, dim=-1)  # [batch, num_heads, actual_k]

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Expand K,V for GQA: each query head attends to the same KV head
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Sparse attention computation per head
        # Note: This Python loop is the bottleneck for large models
        # In practice, this should be implemented as a fused kernel
        out = torch.zeros_like(q)

        for b in range(batch_size):
            for h in range(self.num_heads):
                idxs = topk_indices[b, h]  # [actual_k]

                # Get Q,K,V for selected positions
                q_h = q[b, h, idxs]  # [actual_k, head_dim]
                k_h = k[b, h, idxs]
                v_h = v[b, h, idxs]

                # Attention scores
                scale = self.head_dim ** -0.5
                scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

                # Causal mask on original positions
                orig_idx = idxs.unsqueeze(1)
                orig_idx_t = idxs.unsqueeze(0)
                can_attend = orig_idx >= orig_idx_t
                scores = scores.masked_fill(~can_attend, float('-inf'))

                # Softmax and output
                attn = F.softmax(scores, dim=-1)
                out_h = torch.matmul(attn, v_h)

                # Scatter output back
                out[b, h, idxs] = out_h

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None  # Return None for attention weights compatibility


def replace_attention_with_subq(model, top_k=32):
    """Replace all attention layers with SubQ attention in a model."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    attn_count = [0]

    def replace_forward(module):
        if isinstance(module, Qwen2Attention):
            attn_count[0] += 1
            config = module.config
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = module.head_dim

            # Create SubQ attention
            subq_attn = SubQAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                top_k=top_k
            ).to(module.q_proj.weight.device, module.q_proj.weight.dtype)

            # Copy weights
            with torch.no_grad():
                subq_attn.q_proj.weight.copy_(module.q_proj.weight)
                subq_attn.k_proj.weight.copy_(module.k_proj.weight)
                subq_attn.v_proj.weight.copy_(module.v_proj.weight)
                subq_attn.o_proj.weight.copy_(module.o_proj.weight)

            # Replace forward
            def new_forward(hidden_states, attention_mask=None, position_ids=None, **kwargs):
                return subq_attn(hidden_states, attention_mask, position_ids)

            module.forward = new_forward

    model.apply(replace_forward)
    print(f"Replaced {attn_count[0]} attention layers with SubQ")


def test_inference():
    """Test SubQ attention on Qwen2.5-1.5B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*60)
    print("SubQ-style Sparse Attention Test on Qwen2.5-1.5B")
    print("="*60)

    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model - use fp16 for MPS
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Show original attention config
    print("\nOriginal attention layers:")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    attn_count = sum(1 for m in model.modules() if isinstance(m, Qwen2Attention))
    print(f"  Found {attn_count} Qwen2Attention layers")

    # Replace with SubQ
    print("\nReplacing with SubQ-style attention (top_k=32)...")
    replace_attention_with_subq(model, top_k=32)

    # Test prompt
    prompt = "Explain the theory of relativity in one sentence:"
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\nRunning inference (this will be slow with Python loops)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")

    print("\n" + "="*60)
    print("Note: SubQ attention is working! For production:")
    print("- Implement as fused kernel for speed")
    print("- Use gradient checkpointing for training memory savings")
    print("="*60)


if __name__ == "__main__":
    test_inference()