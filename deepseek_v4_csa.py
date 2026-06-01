#!/usr/bin/env python3
"""
DeepSeek V4-style Compressed Sparse Attention (CSA)

Key ideas from DeepSeek V4:
1. Token-level compression: compress every N tokens into 1 KV entry (4:1 or 128:1)
2. Lightning indexer: top-k selection on compressed blocks
3. Sparse attention on selected compressed entries

This is complementary to MiniMax M3's approach:
- MiniMax M3: block-level selection without compression
- DeepSeek V4: compression + then top-k selection

Two compression modes:
- CSA-4x: compress 4 tokens -> 1 entry (moderate compression)
- CSA-128x: compress 128 tokens -> 1 entry (aggressive, like HCA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class DeepSeekV4CSA(nn.Module):
    """
    DeepSeek V4-style Compressed Sparse Attention.

    Compression ratios:
    - compression_ratio=4: CSA-4x (moderate)
    - compression_ratio=128: CSA-128x (aggressive, HCA-like)

    The compression is done via average pooling within compressed chunks.
    Then a lightweight indexer selects top-k compressed entries.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compression_ratio: int = 4,
        top_k_blocks: int = 4,
        index_dim: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.top_k_blocks = top_k_blocks
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.index_dim = index_dim

        # Main projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Index projections for routing (lower dimension)
        self.index_q = nn.Linear(hidden_size, num_heads * index_dim, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * index_dim, bias=False)

        # Projection to map index_dim to head_dim for block scoring
        self.index_to_head = nn.Linear(index_dim, head_dim, bias=False)

    def _init_index_from_attention(self):
        """Initialize index projections from attention projections for warm start."""
        with torch.no_grad():
            q_weight = self.q_proj.weight
            k_weight = self.k_proj.weight
            H = self.num_heads
            h = self.num_kv_heads
            self.index_q.weight.copy_(q_weight[:H * self.index_dim, :])
            self.index_k.weight.copy_(k_weight[:h * self.index_dim, :])

    def _compress_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV by average pooling within chunks.

        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]

        Returns:
            k_comp: [batch, num_kv_heads, num_comp_blocks, head_dim]
            v_comp: [batch, num_kv_heads, num_comp_blocks, head_dim]
            position_map: [batch, seq_len] - maps original position to compressed block index
        """
        batch_size, num_kv_heads, seq_len, head_dim = k.shape
        compression = self.compression_ratio

        num_comp_blocks = (seq_len + compression - 1) // compression
        pad_len = num_comp_blocks * compression

        # Create position mapping: position i maps to block i // compression
        position_map = torch.arange(seq_len, device=k.device).unsqueeze(0).unsqueeze(0)
        position_map = position_map // compression  # [1, 1, seq_len]

        # Pad k and v
        k_padded = torch.zeros(batch_size, num_kv_heads, pad_len, head_dim,
                               device=k.device, dtype=k.dtype)
        k_padded[:, :, :seq_len] = k

        v_padded = torch.zeros(batch_size, num_kv_heads, pad_len, head_dim,
                               device=v.device, dtype=v.dtype)
        v_padded[:, :, :seq_len] = v

        # Reshape to [B, h, num_comp_blocks, compression, head_dim]
        k_reshaped = k_padded.view(batch_size, num_kv_heads, num_comp_blocks, compression, head_dim)
        v_reshaped = v_padded.view(batch_size, num_kv_heads, num_comp_blocks, compression, head_dim)

        # Average pool within each chunk
        k_comp = k_reshaped.mean(dim=3)  # [B, h, num_comp_blocks, head_dim]
        v_comp = v_reshaped.mean(dim=3)  # [B, h, num_comp_blocks, head_dim]

        return k_comp, v_comp, position_map

    def _compute_block_scores(self, idx_q: torch.Tensor, k_comp: torch.Tensor) -> torch.Tensor:
        """
        Compute compressed block-level attention scores.

        idx_q: [batch, seq_len, num_heads, index_dim]
        k_comp: [batch, num_kv_heads, num_comp_blocks, head_dim]
        Returns: [batch, num_heads, num_comp_blocks] scores for each compressed block
        """
        batch_size, seq_len, num_heads, idx_dim = idx_q.shape
        num_kv_heads = k_comp.shape[1]
        num_comp_blocks = k_comp.shape[2]

        # Average pool idx_q across the sequence to get a single query representation
        # This gives us a "summary" query for selecting blocks
        idx_q_avg = idx_q.mean(dim=1)  # [B, H, d_idx]

        # Project from index_dim to head_dim for compatibility with k_comp
        idx_q_proj = self.index_to_head(idx_q_avg)  # [B, H, head_dim]

        # k_comp: [B, h, nb, d] -> repeat for GQA to [B, H, nb, d]
        k_comp_rep = k_comp.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, H, nb, d]

        # For index attention, we compute scores between avg query and all compressed K
        # idx_q_proj: [B, H, d], k_comp_rep: [B, H, nb, d]
        # Shape: [B, H, 1, nb] after matmul
        idx_q_expanded = idx_q_proj.unsqueeze(2)  # [B, H, 1, d]

        # Block-level scores: [B, H, 1, nb] after matmul
        scores = torch.matmul(idx_q_expanded.float(), k_comp_rep.float().transpose(-2, -1))
        scores = scores.squeeze(2) * (self.head_dim ** -0.5)  # [B, H, nb]

        # No causal mask needed for block selection - the summary query attends to all blocks
        # The actual causal masking happens in _sparse_attention with per-position indices

        return scores

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_comp: torch.Tensor,
        v_comp: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse attention using selected compressed KV blocks.

        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
        k_comp: [batch, num_kv_heads, num_comp_blocks, head_dim] - compressed K
        v_comp: [batch, num_kv_heads, num_comp_blocks, head_dim] - compressed V
        selected_blocks: [batch, num_heads, top_k] - compressed block indices
        Returns: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        compression = self.compression_ratio
        actual_k = selected_blocks.shape[-1]

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Create position-to-compressed-block mapping
        positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len)
        block_indices = positions // compression  # [1, 1, seq_len] -> which compressed block each position belongs to

        # For each selected compressed block, we need to attend to ALL original tokens within that block
        # So we expand selected block indices to position indices

        # selected_blocks: [B, H, top_k] block indices
        # block_base: [B, H, top_k, 1] base position
        block_base = selected_blocks.unsqueeze(-1) * compression  # [B, H, top_k, 1]
        # offset positions within each block: [1, 1, 1, compression]
        offsets = torch.arange(compression, device=q.device).view(1, 1, 1, compression)
        # position_indices: [B, H, top_k * compression]
        position_indices = (block_base + offsets).view(batch_size, num_heads, actual_k * compression)

        # Gather KV from selected positions (from ORIGINAL uncompressed KV)
        k_selected = torch.gather(
            k_rep, 2,
            position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )  # [B, H, top_k*comp, d]
        v_selected = torch.gather(
            v_rep, 2,
            position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )  # [B, H, top_k*comp, d]

        # Compute attention scores
        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        # Causal mask on selected positions
        k_pos = position_indices.view(batch_size, num_heads, actual_k * compression, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)

        scores = scores.masked_fill(~causal_mask, -1e9)

        # Softmax and compute output
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_selected)

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute main QKV
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Index attention for block selection
        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        # Compress KV
        k_comp, v_comp, _ = self._compress_kv(k, v)

        # Compute block scores and get top-k
        block_scores = self._compute_block_scores(idx_q, k_comp)
        actual_k = min(self.top_k_blocks, block_scores.shape[-1])
        _, topk_blocks = block_scores.topk(actual_k, dim=-1)
        # topk_blocks is [B, H, actual_k] - no need to slice

        # Sparse attention on selected blocks
        out = self._sparse_attention(q, k, v, k_comp, v_comp, topk_blocks)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


class DeepSeekV4HCA(nn.Module):
    """
    DeepSeek V4's Heavily Compressed Attention (HCA).

    Uses 128x compression - each compressed entry represents 128 tokens.
    This is extremely aggressive and is designed for very long contexts.

    The intuition is that for extremely long sequences:
    - Most information is local
    - Long-range dependencies can be captured with a few compressed "summary" tokens
    - 128:1 compression means 1M tokens -> 8K compressed entries

    This is essentially CSA-128x with more aggressive filtering.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        top_k_blocks: int = 2,
        index_dim: int = 32,
    ):
        super().__init__()
        # HCA always uses 128:1 compression
        self.compression_ratio = 128
        self.top_k_blocks = top_k_blocks
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.index_dim = index_dim

        # Main projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Index projections - heavier compression needs more expressive index
        self.index_q = nn.Linear(hidden_size, num_heads * index_dim, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * index_dim, bias=False)

        # Projection to map index_dim to head_dim
        self.index_to_head = nn.Linear(index_dim, head_dim, bias=False)

        # For HCA, we also need a compression index to help select which 128-token chunks matter
        self.compression_index = nn.Linear(hidden_size, num_heads * index_dim, bias=False)

    def _init_index_from_attention(self):
        """Initialize index projections from attention projections."""
        with torch.no_grad():
            self.index_q.weight.copy_(self.q_proj.weight[:self.num_heads * self.index_dim, :])
            self.index_k.weight.copy_(self.k_proj.weight[:self.num_kv_heads * self.index_dim, :])

    def _compress_kv(self, k, v):
        """128:1 compression via average pooling."""
        batch_size, num_kv_heads, seq_len, head_dim = k.shape
        compression = 128

        num_comp_blocks = (seq_len + compression - 1) // compression
        pad_len = num_comp_blocks * compression

        # Pad
        k_padded = torch.zeros(batch_size, num_kv_heads, pad_len, head_dim,
                               device=k.device, dtype=torch.float32)
        k_padded[:, :, :seq_len] = k.float()

        v_padded = torch.zeros(batch_size, num_kv_heads, pad_len, head_dim,
                               device=v.device, dtype=torch.float32)
        v_padded[:, :, :seq_len] = v.float()

        # Reshape and pool
        k_reshaped = k_padded.view(batch_size, num_kv_heads, num_comp_blocks, compression, head_dim)
        v_reshaped = v_padded.view(batch_size, num_kv_heads, num_comp_blocks, compression, head_dim)

        k_comp = k_reshaped.mean(dim=3)
        v_comp = v_reshaped.mean(dim=3)

        return k_comp, v_comp

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """HCA forward - very similar to CSA but with 128x compression."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Main QKV
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Index attention
        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        # Compress
        k_comp, v_comp = self._compress_kv(k, v)
        num_comp_blocks = k_comp.shape[2]

        # Block scores using averaged query
        idx_q_avg = idx_q.mean(dim=1)  # [B, H, idx_dim]
        idx_q_proj = self.index_to_head(idx_q_avg)  # [B, H, head_dim]
        k_comp_rep = k_comp.repeat_interleave(self.num_key_value_groups, dim=1)

        # No causal mask needed for block selection - the summary query attends to all blocks
        # Actual causal masking happens in _sparse_attention with per-position indices
        scores = torch.matmul(idx_q_proj.unsqueeze(2).float(), k_comp_rep.float().transpose(-2, -1)).squeeze(2)

        # Top-k blocks (keep very small for HCA)
        actual_k = min(self.top_k_blocks, num_comp_blocks)
        _, topk_blocks = scores.topk(actual_k, dim=-1)
        # topk_blocks shape: [B, H, actual_k]

        # Sparse attention on selected blocks (using original KV, not compressed)
        batch_size, num_heads, seq_len, head_dim = q.shape
        compression = 128

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        block_base = topk_blocks.unsqueeze(-1) * compression
        offsets = torch.arange(compression, device=q.device).view(1, 1, 1, compression)
        position_indices = (block_base + offsets).view(batch_size, num_heads, actual_k * compression)

        k_selected = torch.gather(k_rep, 2, position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        v_selected = torch.gather(v_rep, 2, position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        k_pos = position_indices.view(batch_size, num_heads, actual_k * compression, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)
        scores = scores.masked_fill(~causal_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_selected)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def replace_attention(model, attention_class, **kwargs):
    """Replace Qwen2Attention with custom attention."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types

    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config

            attn = attention_class(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=module.head_dim,
                **kwargs,
            ).to(device)

            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            if hasattr(attn, '_init_index_from_attention'):
                attn._init_index_from_attention()

            attn_modules.append(attn)

            def new_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                          past_key_values=None, cache_position=None, position_ids=None, **kw):
                return attn(hidden_states, attention_mask=attention_mask, **kw)

            module.forward = types.MethodType(new_forward, module)

    model.apply(replace)
    return count[0], attn_modules


@torch.no_grad()
def benchmark_generation(model, tokenizer, prompt, num_tokens, warmup=3):
    """Benchmark text generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
    elapsed = time.time() - start

    return num_tokens / elapsed


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*70)
    print("DeepSeek V4-style CSA/HCA Attention")
    print("="*70)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Test configurations
    configs = [
        ("Standard", None, {}),
        ("CSA-4x(k=4)", DeepSeekV4CSA, {"compression_ratio": 4, "top_k_blocks": 4, "index_dim": 32}),
        ("CSA-4x(k=8)", DeepSeekV4CSA, {"compression_ratio": 4, "top_k_blocks": 8, "index_dim": 32}),
        ("HCA-128x(k=2)", DeepSeekV4HCA, {"top_k_blocks": 2, "index_dim": 32}),
    ]

    results = []

    for config_name, attention_class, kwargs in configs:
        print(f"\n--- Testing {config_name} ---")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if attention_class is not None:
            num_replaced, _ = replace_attention(model, attention_class, **kwargs)
            print(f"Replaced {num_replaced} attention layers")
        else:
            print("Using standard attention")

        # Benchmark
        test_cases = [
            ("Short", "The theory of relativity", 30),
            ("Medium", "Quantum mechanics describes how particles behave " * 3, 30),
            ("Long", "The theory of quantum mechanics " * 10, 20),
        ]

        for test_name, prompt, num_tokens in test_cases:
            speed = benchmark_generation(model, tokenizer, prompt, num_tokens)
            tokens_in_prompt = len(tokenizer.encode(prompt))
            print(f"  {test_name} ({tokens_in_prompt} input tokens): {speed:.1f} tokens/sec")
            results.append((config_name, test_name, speed))

        del model
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Config':<20} {'Short':>12} {'Medium':>12} {'Long':>12}")
    print("-" * 60)
    for config_name, _, _ in configs:
        speeds = {test_name: speed for config, test_name, speed in results if config == config_name}
        short = speeds.get("Short", 0)
        medium = speeds.get("Medium", 0)
        long = speeds.get("Long", 0)
        print(f"{config_name:<20} {short:>11.1f}/s {medium:>11.1f}/s {long:>11.1f}/s")

    print("\n" + "="*70)
    print("KEY INSIGHT: DeepSeek V4 CSA vs MiniMax M3")
    print("="*70)
    print("""
DeepSeek V4 CSA:
- Compresses every N tokens into 1 summary entry (4:1 or 128:1)
- Then does top-k selection on compressed entries
- HCA (128:1) is extreme compression for very long contexts

MiniMax M3:
- Block-level selection WITHOUT compression
- Each block keeps original resolution

CSA with 4:1 compression:
- For 1M token sequence -> 250K compressed entries
- Then top-k selects ~32 entries per head
- Each selected entry represents 4 original tokens

HCA with 128:1 compression:
- For 1M token sequence -> ~8K compressed entries
- Then top-k selects ~4 entries per head
- Each selected entry represents 128 original tokens
- Extreme compression - may lose detail but very fast
""")


if __name__ == "__main__":
    main()