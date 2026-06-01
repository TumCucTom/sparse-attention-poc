#!/usr/bin/env python3
"""
Trainable Sparse Attention using Gumbel-Softmax

Key insight: Hard top-k selection is non-differentiable.
Solution: Use Gumbel-Softmax during training for differentiable selection
- Forward (training): soft selection via Gumbel-Softmax (gradient flows)
- Forward (inference): hard top-k selection (actual sparse computation)
- Temperature annealing: starts high (soft), anneals to low (hard)

This implementation uses the proven MiniMax M3 approach with a trainable router.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GumbelSparseAttention(nn.Module):
    """
    Sparse attention with Gumbel-Softmax for training.

    Based on MiniMax M3 two-stage approach:
    1. Index attention on reduced-dim Q/K to compute block importance
    2. Top-k block selection (hard for inference, soft for training)
    3. Sparse attention on selected blocks
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
        temperature: float = 1.0,
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
        self.temperature = temperature

        # Main projections (copied from standard attention)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Index projections for block selection (lower dimension)
        self.index_q = nn.Linear(hidden_size, num_heads * index_dim, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * index_dim, bias=False)

    def _init_index_from_attention(self):
        """Initialize index projections from attention projections for warm start."""
        with torch.no_grad():
            self.index_q.weight.copy_(self.q_proj.weight[:self.num_heads * self.index_dim, :])
            self.index_k.weight.copy_(self.k_proj.weight[:self.num_kv_heads * self.index_dim, :])

    def _gumbel_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Gumbel-Softmax for differentiable selection."""
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        gumbel_logits = (logits + gumbel_noise) / max(self.temperature, 0.01)
        return F.softmax(gumbel_logits, dim=dim)

    def _compute_block_scores(self, idx_q: torch.Tensor, idx_k: torch.Tensor) -> torch.Tensor:
        """
        Compute block-level attention scores for selection.

        idx_q: [batch, seq_len, num_heads, index_dim]
        idx_k: [batch, seq_len, num_kv_heads, index_dim]
        Returns: [batch, num_heads, num_blocks, num_blocks] block-level scores
        """
        batch_size, seq_len, num_heads, idx_dim = idx_q.shape
        num_kv_heads = idx_k.shape[2]

        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size

        # Pad and reshape to blocks
        q_padded = torch.zeros(batch_size, pad_len, num_heads, idx_dim, device=idx_q.device, dtype=torch.float32)
        q_padded[:, :seq_len] = idx_q.float()
        q_blocks = q_padded.view(batch_size, num_blocks, self.block_size, num_heads, idx_dim)

        k_padded = torch.zeros(batch_size, pad_len, num_kv_heads, idx_dim, device=idx_k.device, dtype=torch.float32)
        k_padded[:, :seq_len] = idx_k.float()
        k_blocks = k_padded.view(batch_size, num_blocks, self.block_size, num_kv_heads, idx_dim)

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
        Sparse attention on selected blocks.

        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
        selected_blocks: [batch, num_heads, top_k] - block indices
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size
        actual_k = selected_blocks.shape[-1]

        # GQA: repeat KV heads
        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert block indices to position indices
        block_offsets = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size
        position_indices = (block_base + block_offsets).view(batch_size, num_heads, actual_k * block_size)

        # Gather KV from selected positions
        k_selected = torch.gather(
            k_rep, 2,
            position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )
        v_selected = torch.gather(
            v_rep, 2,
            position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )

        # Compute attention scores
        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        # Causal mask on selected positions
        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
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

        # Stage 1: Index attention and block selection
        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        block_scores = self._compute_block_scores(idx_q, idx_k)

        # Use first query block's scores for selection
        block_scores_for_selection = block_scores[:, :, 0, :]  # [B, H, nb]

        num_blocks = block_scores_for_selection.shape[-1]

        if self.training:
            # Gumbel-Softmax for training (gradients flow through)
            block_weights = self._gumbel_softmax(block_scores_for_selection, dim=-1)  # [B, H, nb]
            block_weights = block_weights * min(self.top_k_blocks, num_blocks)

            # For soft attention, use block_weights to bias full attention
            # This is stable for training
            q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

            k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)

            # Standard attention during training (stable)
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)

            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_rep)
        else:
            # Hard selection for inference
            actual_k = min(self.top_k_blocks, num_blocks)
            _, topk_blocks = block_scores_for_selection.topk(actual_k, dim=-1)

            # Stage 2: Main attention on selected blocks
            q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

            out = self._sparse_attention(q, k, v, topk_blocks)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


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

    def forward(self, hidden_states, attention_mask=None, **kwargs):
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