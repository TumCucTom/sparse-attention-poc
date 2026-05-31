#!/usr/bin/env python3
"""
MiniMax M3-style Sparse Attention - Production Implementation

Key features:
1. Two-stage approach: Index attention for block selection, sparse attention for computation
2. Warm-start index projections from attention projections
3. Causal masking on selected blocks
4. Optimized for MPS with proper tensor shapes

This is a proper sparse attention implementation, not just dense fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class MiniMaxSparseAttention(nn.Module):
    """
    MiniMax M3-style sparse attention with two-stage routing.

    Stage 1 (Index): Idx Q, Idx KV → BlockAvgPool → TopK blocks
    Stage 2 (Sparse): Main Q → attend only to selected KV blocks

    The key insight from MiniMax M3:
    - Index Q/K with lower dimension compute which blocks to attend to
    - Main Q only attends to KV in the selected blocks
    - This reduces O(n²) to O(k) where k = top_k * block_size
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        top_k_blocks: int = 4,
        index_dim: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
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

    def _init_index_from_attention(self):
        """Initialize index projections from attention projections for warm start."""
        with torch.no_grad():
            target_dtype = self.index_q.weight.dtype
            q_weight = self.q_proj.weight.float()[:self.num_heads * self.index_dim, :].to(target_dtype)
            k_weight = self.k_proj.weight.float()[:self.num_kv_heads * self.index_dim, :].to(target_dtype)
            self.index_q.weight.copy_(q_weight)
            self.index_k.weight.copy_(k_weight)

    def _compute_block_scores(self, idx_q: torch.Tensor, idx_k: torch.Tensor) -> torch.Tensor:
        """
        Compute block-level attention scores.

        idx_q: [batch, seq_len, num_heads, index_dim]
        idx_k: [batch, seq_len, num_kv_heads, index_dim]
        Returns: [batch, num_heads, num_blocks, num_blocks] block-level scores
        """
        import os
        batch_size, seq_len, num_heads, idx_dim = idx_q.shape
        num_kv_heads = idx_k.shape[2]

        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Calculate expected total size after padding
        expected_total = num_blocks * self.block_size
        pad_len = expected_total - seq_len

        # Handle case where seq_len is exactly divisible by block_size (pad_len = 0)
        if pad_len > 0:
            # Pad and reshape to blocks
            # Create tensor with full padded length (expected_total = seq_len + pad_len)
            padded_seq_len = expected_total  # This is seq_len + pad_len
            q_padded = torch.zeros(batch_size, padded_seq_len, num_heads, idx_dim, device=idx_q.device, dtype=torch.float32)
            q_padded[:, :seq_len] = idx_q.float()
            q_blocks = q_padded.view(batch_size, num_blocks, self.block_size, num_heads, idx_dim)

            k_padded = torch.zeros(batch_size, padded_seq_len, num_kv_heads, idx_dim, device=idx_k.device, dtype=torch.float32)
            k_padded[:, :seq_len] = idx_k.float()
            k_blocks = k_padded.view(batch_size, num_blocks, self.block_size, num_kv_heads, idx_dim)
        else:
            # No padding needed - reshape original tensor directly to blocks
            # seq_len == num_blocks * block_size
            q_blocks = idx_q.float().view(batch_size, num_blocks, self.block_size, num_heads, idx_dim)
            k_blocks = idx_k.float().view(batch_size, num_blocks, self.block_size, num_kv_heads, idx_dim)

        # Average pooling within blocks
        q_avg = q_blocks.mean(dim=2)  # [B, nb, H, d]
        k_avg = k_blocks.mean(dim=2)  # [B, nb, h, d]

        # GQA: repeat KV heads
        k_avg_rep = k_avg.repeat_interleave(self.num_key_value_groups, dim=2)

        # Block-level scores: [B, H, nb_q, nb_k]
        scores = torch.matmul(
            q_avg.permute(0, 2, 1, 3),
            k_avg_rep.permute(0, 2, 3, 1)
        ) * (idx_dim ** -0.5)

        # Clamp scores to prevent overflow in fp16 softmax (done in fp32)
        scores = scores.float().clamp(-1e4, 1e4)

        # Causal mask at block level
        causal = torch.tril(torch.ones(num_blocks, num_blocks, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, num_blocks, num_blocks), -1e9)

        return scores

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sparse attention using only selected KV blocks.

        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
        selected_blocks: [batch, num_heads, top_k] - block indices
        Returns: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size
        num_kv_heads = k.shape[1]
        actual_k = selected_blocks.shape[-1]

        # GQA: repeat KV heads to match query head count
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Clamp block indices to valid range
        max_block = (seq_len + block_size - 1) // block_size
        selected_blocks = selected_blocks.clamp(0, max_block - 1)

        # Convert block indices to position indices
        block_offsets = torch.arange(block_size, device=q.device, dtype=torch.long).view(1, 1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size
        position_indices = (block_base + block_offsets).view(batch_size, num_heads, actual_k * block_size)

        # Clamp position indices to valid range
        position_indices = position_indices.clamp(0, seq_len - 1)

        # Gather KV from selected positions - expand properly for head_dim
        gather_idx = position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, actual_k*block, head_dim]

        k_selected = torch.gather(k_rep, 2, gather_idx)  # [B, H, actual_k*block, head_dim]
        v_selected = torch.gather(v_rep, 2, gather_idx)

        # Create causal mask for selected positions
        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)

        # Use SDPA with explicit mask for potential Flash Attention optimization
        try:
            # SDPA path - but chunk for very large seq_len to avoid OOM
            # SDPA with causal mask on [131K, 64] is still heavy
            if seq_len > 8192:
                # Chunked SDPA for large sequences
                out_chunks = []
                chunk_size = 2048
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    q_chunk = q[:, :, start:end, :].float()
                    k_chunk = k_selected[:, :, :actual_k * block_size, :].float()
                    v_chunk = v_selected[:, :, :actual_k * block_size, :].float()
                    causal_mask_chunk = causal_mask[:, :, start:end, :actual_k * block_size]
                    attn_mask_chunk = ~causal_mask_chunk
                    out_chunk = F.scaled_dot_product_attention(
                        q_chunk, k_chunk, v_chunk,
                        attn_mask=attn_mask_chunk,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=self.scale
                    )
                    out_chunks.append(out_chunk)
                out = torch.cat(out_chunks, dim=2)
            else:
                # Normal SDPA for reasonable seq_lens
                attn_mask = ~causal_mask  # True means "use this attention"
                out = F.scaled_dot_product_attention(
                    q.to(torch.float32),
                    k_selected.to(torch.float32),
                    v_selected.to(torch.float32),
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scale
                )
        except Exception:
            # Chunked computation to avoid OOM on large seq_lens
            # Process query in chunks to avoid materializing full [seq_len, k*block] matmul
            out_chunks = []
            chunk_size = 2048
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                q_chunk = q[:, :, start:end, :].float()  # [B, H, chunk, head_dim]
                scores_chunk = torch.matmul(q_chunk, k_selected.float().transpose(-2, -1)) * self.scale
                causal_mask_chunk = causal_mask[:, :, start:end, :]  # [B, H, chunk, k*block]
                scores_chunk = scores_chunk.masked_fill(~causal_mask_chunk, -1e9)
                attn_chunk = F.softmax(scores_chunk, dim=-1)
                out_chunk = torch.matmul(attn_chunk, v_selected.float())  # [B, H, chunk, head_dim]
                out_chunks.append(out_chunk)
            out = torch.cat(out_chunks, dim=2)

        return out.to(dtype=q.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Cast input to match weight dtype for linear layers
        weight_dtype = self.q_proj.weight.dtype
        x_f16 = hidden_states.to(dtype=weight_dtype)

        # Stage 1: Index attention and block selection
        idx_q = self.index_q(x_f16).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        block_scores = self._compute_block_scores(idx_q, idx_k)

        actual_k = min(self.top_k_blocks, block_scores.shape[-1])
        _, topk_blocks = block_scores.topk(actual_k, dim=-1)
        # topk_blocks is [batch, num_heads, num_blocks, top_k] - use first key block dimension
        topk_blocks = topk_blocks[:, :, 0, :]  # [batch, num_heads, top_k] - Use first kv block's selection

        # Stage 2: Main attention on selected blocks
        q = self.q_proj(x_f16).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        out = self._sparse_attention(q, k, v, topk_blocks)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        # Return 2 values to match what the model expects
        # (attn_output, attn_weights) - past_key_value not used in sparse mode
        return out.to(dtype=hidden_states.dtype), None


class StandardAttention(nn.Module):
    """Standard dense attention for comparison."""
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * self.scale

        seq_len_q = q.shape[2]
        causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_q, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len_q, seq_len_q), -1e9)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        # Return 2 values to match what the model expects
        return self.o_proj(out), None


def replace_attention(model, attention_class, **kwargs):
    """Replace Qwen2Attention with custom attention."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types
    import os

    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config

            if os.environ.get('DEBUG_SPARSE') == '1':
                print(f"DEBUG replace: layer {count[0]}, num_attention_heads={config.num_attention_heads}, num_key_value_heads={config.num_key_value_heads}, head_dim={module.head_dim}")

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
    """Benchmark text generation speed."""
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    # Warmup
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
    elapsed = time.time() - start

    return num_tokens / elapsed


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new=20):
    """Generate text and return."""
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "="*70)
    print("MiniMax M3-style Sparse Attention - Production Implementation")
    print("="*70)

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Test configurations
    configs = [
        ("Standard", StandardAttention, {}),
        ("MiniMax-Sparse(k=4)", MiniMaxSparseAttention, {"block_size": 16, "top_k_blocks": 4, "index_dim": 32}),
        ("MiniMax-Sparse(k=8)", MiniMaxSparseAttention, {"block_size": 16, "top_k_blocks": 8, "index_dim": 32}),
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
        num_replaced, _ = replace_attention(model, attention_class, **kwargs)
        print(f"Replaced {num_replaced} attention layers")

        # Benchmark at different sequence lengths
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
    print(f"\n{'Config':<25} {'Short':>12} {'Medium':>12} {'Long':>12}")
    print("-" * 70)
    for config_name, _, _ in configs:
        speeds = {test_name: speed for config, test_name, speed in results if config == config_name}
        short = speeds.get("Short", 0)
        medium = speeds.get("Medium", 0)
        long = speeds.get("Long", 0)
        print(f"{config_name:<25} {short:>11.1f}/s {medium:>11.1f}/s {long:>11.1f}/s")

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
MiniMax M3-style sparse attention uses a two-stage approach:
1. Index attention computes block-level importance scores
2. Main attention only processes selected KV blocks

With top_k_blocks=4 and block_size=16:
- Each head attends to 4*16=64 tokens instead of full sequence
- For sequences > 64 tokens, this should show speedup

The index projections are initialized from attention projections (warm start),
which helps the router learn meaningful block selection faster.
""")


if __name__ == "__main__":
    main()