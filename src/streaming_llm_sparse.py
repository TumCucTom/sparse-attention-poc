#!/usr/bin/env python3
"""
StreamingLLM-style attention for MiniMax-M2.7 - NO block_scores matrix needed.

Key insight from StreamingLLM:
- Keep the last 4 tokens as "sink" tokens (always attended to)
- Keep a sliding window of ~32 tokens  
- Drop everything in between

This avoids the O(num_blocks²) memory problem entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StreamingSparseAttention(nn.Module):
    """
    StreamingLLM-style attention - no block_scores needed.

    Instead of computing importance scores for all blocks, we:
    1. Always attend to the last 4 tokens (sink tokens)
    2. Attend to the last N tokens via sliding window
    3. Never compute a full block_scores matrix

    Memory: O(window_size) instead of O(num_blocks²)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int = 32,
        sink_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.sink_size = sink_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        # Main projections (shared with the original attention)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

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

        # Cast input to match weight dtype
        weight_dtype = self.q_proj.weight.dtype
        x_f16 = hidden_states.to(dtype=weight_dtype)

        # Project to Q, K, V
        q = self.q_proj(x_f16).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_f16).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # StreamingLLM attention: only attend to sink tokens + sliding window
        # Sink tokens: last 4 tokens (always attended)
        # Sliding window: last window_size tokens before sink
        total_context = self.sink_size + self.window_size

        if seq_len <= total_context:
            # Short sequence - attend to all
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale
            )
        else:
            # Create streaming attention mask
            # Only attend to last (sink_size + window_size) tokens
            device = q.device
            seq_len_q = q.shape[2]
            
            # Create causal mask first
            causal_mask = torch.tril(torch.ones(seq_len_q, seq_len_q, device=device, dtype=torch.bool))
            
            # Now create streaming-specific mask: only attend to last total_context tokens
            streaming_mask = torch.zeros(seq_len_q, seq_len_q, device=device, dtype=torch.bool)
            start_idx = seq_len_q - total_context
            streaming_mask[start_idx:, start_idx:] = True
            
            # Combine: causal AND streaming (both must be True)
            attn_mask = causal_mask & streaming_mask
            
            # Convert to attention mask for SDPA (True = ignore, False = attend)
            ignore_mask = ~attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=ignore_mask.float().masked_fill(ignore_mask, float('-inf')),
                dropout_p=0.0,
                is_causal=False,  # Already applied via mask
                scale=self.scale
            )

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out.to(dtype=hidden_states.dtype), None


def replace_attention_with_streaming(model, window_size=32, sink_size=4):
    """Replace attention modules with StreamingSparseAttention."""
    import types

    count = [0]
    device = next(model.parameters()).device

    def replace_fn(module):
        # Get module class name to avoid importing MiniMaxM2Attention
        module_name = type(module).__name__
        module_full_name = f"{type(module).__module__}.{module_name}"
        
        # Match MiniMaxM2Attention from transformers
        if 'MiniMaxM2Attention' in module_name or 'MiniMaxM2Attention' in module_full_name:
            count[0] += 1
            config = module.config
            
            # Get actual sizes from weights
            actual_q_size = module.q_proj.weight.shape[0]
            actual_k_size = module.k_proj.weight.shape[0]
            actual_num_heads = actual_q_size // module.head_dim
            actual_num_kv_heads = actual_k_size // module.head_dim
            
            if count[0] == 1:
                print(f"  Replacing attention layer {count[0]}: {module_name}")
                print(f"    hidden_size={config.hidden_size}, num_heads={actual_num_heads}, head_dim={module.head_dim}")

            attn = StreamingSparseAttention(
                hidden_size=config.hidden_size,
                num_heads=actual_num_heads,
                num_kv_heads=actual_num_kv_heads,
                head_dim=module.head_dim,
                window_size=window_size,
                sink_size=sink_size,
            ).to(device)

            # Copy weights
            with torch.no_grad():
                attn.q_proj.weight.copy_(module.q_proj.weight)
                attn.k_proj.weight.copy_(module.k_proj.weight)
                attn.v_proj.weight.copy_(module.v_proj.weight)
                attn.o_proj.weight.copy_(module.o_proj.weight)

            def new_forward(self, hidden_states, **kw):
                return attn(hidden_states, **kw)
            module.forward = types.MethodType(new_forward, module)
            return True
        return False

    model.apply(replace_fn)
    return count[0]
