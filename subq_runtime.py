#!/usr/bin/env python3
"""
Reusable SubQ runtime components.

This module provides a per-query sparse attention path that keeps all query
positions active and sparsifies only key/value access. It supports:
- dense mode
- sparse learned mode via a low-rank router
- sparse oracle mode for upper-bound quality analysis
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class PerQuerySparseAttention(nn.Module):
    """Dense or sparse attention with per-query key selection."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        top_k=32,
        router_dim=8,
        bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.router_dim = router_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.router_scale = router_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.router_q = nn.Linear(hidden_size, num_heads * router_dim, bias=False)
        self.router_k = nn.Linear(hidden_size, num_heads * router_dim, bias=False)
        nn.init.normal_(self.router_q.weight, std=0.02)
        nn.init.normal_(self.router_k.weight, std=0.02)

        self.mode = "sparse_learned"
        self.collect_metrics = False
        self.last_dense_topk_mass = None
        self.last_learned_topk_mass = None

    def set_mode(self, mode):
        self.mode = mode

    def set_collect_metrics(self, collect_metrics):
        self.collect_metrics = collect_metrics

    def init_router_from_attention_projections(self):
        """
        Warm-start the low-rank router from the trained Q/K projections.

        This is not a substitute for distillation, but it gives inference-time
        benchmarks a deterministic, non-random sparse selector.
        """
        with torch.no_grad():
            q_weight = self.q_proj.weight.view(self.num_heads, self.head_dim, self.hidden_size)
            k_weight = self.k_proj.weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)

            router_q = torch.zeros_like(self.router_q.weight.view(self.num_heads, self.router_dim, self.hidden_size))
            router_k = torch.zeros_like(self.router_k.weight.view(self.num_heads, self.router_dim, self.hidden_size))

            dims = min(self.router_dim, self.head_dim)
            router_q[:, :dims] = q_weight[:, :dims]

            for head in range(self.num_heads):
                kv_head = head // self.num_key_value_groups
                router_k[head, :dims] = k_weight[kv_head, :dims]

            self.router_q.weight.copy_(router_q.view_as(self.router_q.weight))
            self.router_k.weight.copy_(router_k.view_as(self.router_k.weight))

    def _causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def _project_qkv(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        return q, k, v

    def _dense_scores(self, q, k, attention_mask=None):
        key_states = repeat_kv(k, self.num_key_value_groups)
        scores = torch.matmul(q.float(), key_states.float().transpose(-2, -1)) * self.scale
        if attention_mask is None:
            seq_len = q.shape[2]
            causal_mask = self._causal_mask(seq_len, q.device)
            scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)
        else:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        return scores

    def _dense_attention(self, q, k, v, attention_mask=None):
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        scores = self._dense_scores(q, k, attention_mask=attention_mask)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn.to(value_states.dtype), value_states)
        return out, attn

    def _router_scores(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        rq = self.router_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.router_dim)
        rk = self.router_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.router_dim)

        rq = rq.permute(0, 2, 1, 3)
        rk = rk.permute(0, 2, 1, 3)

        scores = torch.matmul(rq.float(), rk.float().transpose(-2, -1)) * self.router_scale
        if attention_mask is None:
            causal_mask = self._causal_mask(seq_len, hidden_states.device)
            scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)
        else:
            scores = scores + attention_mask[:, :, :, : seq_len]
        return scores

    def _sparse_attention_from_indices(self, q, k, v, topk_indices, attention_mask=None):
        _, _, seq_len, _ = q.shape
        query_positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)

        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        k_expanded = key_states.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        v_expanded = value_states.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)

        k_selected = torch.gather(k_expanded, 3, gather_index)
        v_selected = torch.gather(v_expanded, 3, gather_index)

        scores = (q.unsqueeze(-2).float() * k_selected.float()).sum(dim=-1) * self.scale
        if attention_mask is None:
            valid = topk_indices <= query_positions
            scores = scores.masked_fill(~valid, -1e9)
        else:
            gathered_mask = torch.gather(
                attention_mask.expand(-1, self.num_heads, seq_len, -1),
                -1,
                topk_indices,
            )
            scores = scores + gathered_mask

        attn = F.softmax(scores, dim=-1).to(v.dtype)
        return (attn.unsqueeze(-1) * v_selected).sum(dim=-2)

    def _topk_mass(self, dense_attn, indices):
        if indices.dim() == 3:
            indices = indices.unsqueeze(2).expand(-1, -1, dense_attn.shape[2], -1)
        return torch.gather(dense_attn, -1, indices).sum(dim=-1).mean().detach()

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, position_ids=None, **kwargs):
        q, k, v = self._project_qkv(hidden_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        seq_len = q.shape[2]
        actual_k = min(self.top_k, seq_len)

        dense_out = None
        dense_attn = None
        if self.mode in {"dense", "sparse_oracle"} or self.collect_metrics:
            dense_out, dense_attn = self._dense_attention(q, k, v, attention_mask=attention_mask)

        if self.mode == "dense":
            out = dense_out
        elif self.mode == "sparse_oracle":
            oracle_indices = dense_attn.topk(actual_k, dim=-1).indices
            out = self._sparse_attention_from_indices(q, k, v, oracle_indices, attention_mask=attention_mask)
        elif self.mode == "sparse_learned":
            router_scores = self._router_scores(hidden_states, attention_mask=attention_mask)
            learned_indices = router_scores.topk(actual_k, dim=-1).indices
            out = self._sparse_attention_from_indices(q, k, v, learned_indices, attention_mask=attention_mask)
        else:
            raise ValueError(f"Unknown attention mode: {self.mode}")

        if self.collect_metrics and dense_attn is not None:
            if self.mode == "sparse_learned":
                indices = learned_indices
            else:
                router_scores = self._router_scores(hidden_states, attention_mask=attention_mask)
                indices = router_scores.topk(actual_k, dim=-1).indices
            oracle_indices = dense_attn.topk(actual_k, dim=-1).indices
            self.last_learned_topk_mass = self._topk_mass(dense_attn, indices)
            self.last_dense_topk_mass = self._topk_mass(dense_attn, oracle_indices)
        else:
            self.last_learned_topk_mass = None
            self.last_dense_topk_mass = None

        batch_size = hidden_states.shape[0]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out), None


def replace_attention(
    model,
    attention_class=PerQuerySparseAttention,
    init_router_from_attention=True,
    **kwargs,
):
    """Replace Qwen2 attention with a registered custom attention module."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    count = 0
    attn_modules = []

    def patch(module):
        nonlocal count
        if not isinstance(module, Qwen2Attention):
            return

        count += 1
        config = module.config
        custom_attn = attention_class(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=module.head_dim,
            **kwargs,
        ).to(module.q_proj.weight.device, module.q_proj.weight.dtype)

        with torch.no_grad():
            custom_attn.q_proj.weight.copy_(module.q_proj.weight)
            custom_attn.k_proj.weight.copy_(module.k_proj.weight)
            custom_attn.v_proj.weight.copy_(module.v_proj.weight)
            custom_attn.o_proj.weight.copy_(module.o_proj.weight)

        if init_router_from_attention and hasattr(custom_attn, "init_router_from_attention_projections"):
            custom_attn.init_router_from_attention_projections()

        if hasattr(module, "layer_idx"):
            custom_attn.layer_idx = module.layer_idx

        module.subq_attn = custom_attn

        def new_forward(
            self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            position_ids=None,
            **kw,
        ):
            result = self.subq_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            if isinstance(result, tuple):
                return result
            return result, None

        module.forward = types.MethodType(new_forward, module)
        attn_modules.append(custom_attn)

    model.apply(patch)
    return count, attn_modules


def set_attention_mode(attn_modules, mode, collect_metrics=False):
    for attn in attn_modules:
        attn.set_mode(mode)
        attn.set_collect_metrics(collect_metrics)


class StructuredSparseAttention(nn.Module):
    """
    Faster structured sparse attention for inference on MPS.

    Pattern:
    - every query attends to a fixed local window
    - plus a small learned global prefix/pool shared per head

    This avoids per-query arbitrary gathers over the entire sequence.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        top_k=8,
        router_dim=8,
        window_size=128,
        global_size=32,
        bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.router_dim = router_dim
        self.window_size = window_size
        self.global_size = global_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.router_scale = router_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.router_q = nn.Linear(hidden_size, num_heads * router_dim, bias=False)
        self.router_k = nn.Linear(hidden_size, num_heads * router_dim, bias=False)
        nn.init.normal_(self.router_q.weight, std=0.02)
        nn.init.normal_(self.router_k.weight, std=0.02)

        self.mode = "structured_learned"
        self.collect_metrics = False
        self.last_dense_topk_mass = None
        self.last_learned_topk_mass = None

    def set_mode(self, mode):
        self.mode = mode

    def set_collect_metrics(self, collect_metrics):
        self.collect_metrics = collect_metrics

    def init_router_from_attention_projections(self):
        with torch.no_grad():
            q_weight = self.q_proj.weight.view(self.num_heads, self.head_dim, self.hidden_size)
            k_weight = self.k_proj.weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)

            router_q = torch.zeros_like(self.router_q.weight.view(self.num_heads, self.router_dim, self.hidden_size))
            router_k = torch.zeros_like(self.router_k.weight.view(self.num_heads, self.router_dim, self.hidden_size))

            dims = min(self.router_dim, self.head_dim)
            router_q[:, :dims] = q_weight[:, :dims]
            for head in range(self.num_heads):
                kv_head = head // self.num_key_value_groups
                router_k[head, :dims] = k_weight[kv_head, :dims]

            self.router_q.weight.copy_(router_q.view_as(self.router_q.weight))
            self.router_k.weight.copy_(router_k.view_as(self.router_k.weight))

    def _project_qkv(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _router_scores(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        rq = self.router_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.router_dim).transpose(1, 2)
        rk = self.router_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.router_dim).transpose(1, 2)
        scores = torch.matmul(rq.float(), rk.float().transpose(-2, -1)) * self.router_scale
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : seq_len]
        else:
            causal = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -1e9)
        return scores

    def _dense_attention(self, q, k, v, attention_mask=None):
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q.float(), key_states.float().transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            seq_len = q.shape[2]
            causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn.to(value_states.dtype), value_states)
        return out, attn

    def _build_indices(self, hidden_states, seq_len, attention_mask=None, dense_attn=None):
        local_size = min(self.window_size, seq_len)
        global_size = min(self.global_size, seq_len)
        total_k = min(local_size + global_size, seq_len)

        base = torch.arange(seq_len, device=hidden_states.device)
        local_offsets = torch.arange(local_size, device=hidden_states.device)
        local_start = (base.unsqueeze(-1) - (local_size - 1)).clamp_min(0)
        local_indices = local_start + local_offsets
        local_indices = torch.minimum(local_indices, base.unsqueeze(-1))
        local_indices = local_indices.view(1, 1, seq_len, local_size).expand(hidden_states.shape[0], self.num_heads, -1, -1)

        if self.mode == "structured_oracle" and dense_attn is not None:
            global_indices = dense_attn.topk(global_size, dim=-1).indices
        else:
            router_scores = self._router_scores(hidden_states, attention_mask=attention_mask)
            global_indices = router_scores.topk(global_size, dim=-1).indices

        combined = torch.cat([local_indices, global_indices], dim=-1)
        combined = combined.sort(dim=-1).values
        if combined.shape[-1] > total_k:
            combined = combined[..., -total_k:]
        return combined

    def _structured_attention(self, q, k, v, hidden_states, attention_mask=None, dense_attn=None):
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        indices = self._build_indices(hidden_states, q.shape[2], attention_mask=attention_mask, dense_attn=dense_attn)

        seq_len = q.shape[2]
        k_expanded = key_states.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        v_expanded = value_states.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)

        k_selected = torch.gather(k_expanded, 3, gather_index)
        v_selected = torch.gather(v_expanded, 3, gather_index)

        scores = (q.unsqueeze(-2).float() * k_selected.float()).sum(dim=-1) * self.scale
        if attention_mask is not None:
            gathered_mask = torch.gather(
                attention_mask.expand(-1, self.num_heads, seq_len, -1),
                -1,
                indices,
            )
            scores = scores + gathered_mask
        else:
            query_positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
            scores = scores.masked_fill(indices > query_positions, -1e9)

        attn = F.softmax(scores, dim=-1).to(value_states.dtype)
        out = (attn.unsqueeze(-1) * v_selected).sum(dim=-2)
        return out, indices

    def _topk_mass(self, dense_attn, indices):
        if indices.dim() == 3:
            indices = indices.unsqueeze(2).expand(-1, -1, dense_attn.shape[2], -1)
        return torch.gather(dense_attn, -1, indices).sum(dim=-1).mean().detach()

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, position_ids=None, **kwargs):
        q, k, v = self._project_qkv(hidden_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        dense_out = None
        dense_attn = None
        if self.mode == "dense" or self.collect_metrics or self.mode == "structured_oracle":
            dense_out, dense_attn = self._dense_attention(q, k, v, attention_mask=attention_mask)

        if self.mode == "dense":
            out = dense_out
            indices = None
        elif self.mode in {"structured_learned", "structured_oracle"}:
            out, indices = self._structured_attention(
                q,
                k,
                v,
                hidden_states,
                attention_mask=attention_mask,
                dense_attn=dense_attn,
            )
        else:
            raise ValueError(f"Unknown attention mode: {self.mode}")

        if self.collect_metrics and dense_attn is not None and indices is not None:
            self.last_learned_topk_mass = self._topk_mass(dense_attn, indices)
            oracle_idx = dense_attn.topk(min(indices.shape[-1], dense_attn.shape[-1]), dim=-1).indices
            self.last_dense_topk_mass = self._topk_mass(dense_attn, oracle_idx)
        else:
            self.last_learned_topk_mass = None
            self.last_dense_topk_mass = None

        batch_size, _, seq_len, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out), None


class LocalGlobalTokenSparseAttention(nn.Module):
    """
    Local window attention plus a small learned set of global tokens per head.

    Unlike arbitrary per-query top-k routing, the global token set is shared
    across all queries in a head, which makes the runtime much cheaper.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        top_k=8,
        router_dim=8,
        window_size=128,
        chunk_size=64,
        bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.router_dim = router_dim
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.global_router = nn.Linear(hidden_size, num_heads, bias=False)
        nn.init.normal_(self.global_router.weight, std=0.02)

        self.mode = "local_global_learned"
        self.collect_metrics = False
        self.last_dense_topk_mass = None
        self.last_learned_topk_mass = None

    def set_mode(self, mode):
        self.mode = mode

    def set_collect_metrics(self, collect_metrics):
        self.collect_metrics = collect_metrics

    def init_router_from_attention_projections(self):
        with torch.no_grad():
            q_weight = self.q_proj.weight.view(self.num_heads, self.head_dim, self.hidden_size)
            summary = q_weight[:, : min(self.head_dim, 4)].mean(dim=1)
            self.global_router.weight.copy_(summary)

    def _project_qkv(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _dense_attention(self, q, k, v, attention_mask=None):
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q.float(), key_states.float().transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            seq_len = q.shape[2]
            causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn.to(value_states.dtype), value_states)
        return out, attn

    def _select_global_indices(self, hidden_states, dense_attn=None):
        seq_len = hidden_states.shape[1]
        global_k = min(self.top_k, seq_len)
        if self.mode == "local_global_oracle" and dense_attn is not None:
            importance = dense_attn.sum(dim=2)
        else:
            importance = self.global_router(hidden_states).permute(0, 2, 1).float()
        return importance.topk(global_k, dim=-1).indices.sort(dim=-1).values

    def _compute_local_global_attention(self, q, k, v, hidden_states, attention_mask=None, dense_attn=None):
        batch_size, _, seq_len, _ = q.shape
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        global_indices = self._select_global_indices(hidden_states, dense_attn=dense_attn)
        global_gather = global_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_global = torch.gather(key_states, 2, global_gather)
        v_global = torch.gather(value_states, 2, global_gather)

        out = torch.zeros_like(q)
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            local_start = max(0, start - self.window_size + 1)
            q_chunk = q[:, :, start:end, :]
            k_local = key_states[:, :, local_start:end, :]
            v_local = value_states[:, :, local_start:end, :]

            k_combined = torch.cat([k_global, k_local], dim=2)
            v_combined = torch.cat([v_global, v_local], dim=2)

            scores = torch.matmul(q_chunk.float(), k_combined.float().transpose(-2, -1)) * self.scale

            q_pos = torch.arange(start, end, device=q.device).view(1, 1, end - start, 1)
            global_pos = global_indices.unsqueeze(2).expand(-1, -1, end - start, -1)
            local_pos = torch.arange(local_start, end, device=q.device).view(1, 1, 1, end - local_start)

            global_valid = global_pos <= q_pos
            local_valid = (local_pos <= q_pos) & (local_pos >= (q_pos - self.window_size + 1))
            valid = torch.cat([global_valid, local_valid.expand(batch_size, self.num_heads, -1, -1)], dim=-1)
            scores = scores.masked_fill(~valid, -1e9)

            attn = F.softmax(scores, dim=-1).to(v_combined.dtype)
            out[:, :, start:end, :] = torch.matmul(attn, v_combined)

        return out, global_indices

    def _topk_mass(self, dense_attn, indices):
        return torch.gather(dense_attn, -1, indices).sum(dim=-1).mean().detach()

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, position_ids=None, **kwargs):
        q, k, v = self._project_qkv(hidden_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        dense_out = None
        dense_attn = None
        if self.mode == "dense" or self.collect_metrics or self.mode == "local_global_oracle":
            dense_out, dense_attn = self._dense_attention(q, k, v, attention_mask=attention_mask)

        if self.mode == "dense":
            out = dense_out
            global_indices = None
        elif self.mode in {"local_global_learned", "local_global_oracle"}:
            out, global_indices = self._compute_local_global_attention(
                q,
                k,
                v,
                hidden_states,
                attention_mask=attention_mask,
                dense_attn=dense_attn,
            )
        else:
            raise ValueError(f"Unknown attention mode: {self.mode}")

        if self.collect_metrics and dense_attn is not None and global_indices is not None:
            self.last_learned_topk_mass = self._topk_mass(dense_attn, global_indices)
            oracle_idx = dense_attn.sum(dim=2).topk(global_indices.shape[-1], dim=-1).indices
            self.last_dense_topk_mass = self._topk_mass(dense_attn, oracle_idx)
        else:
            self.last_learned_topk_mass = None
            self.last_dense_topk_mass = None

        batch_size, _, seq_len, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out), None


class FixedHybridAttention(nn.Module):
    """
    Fast HF-compatible local + prefix-global attention.

    This is the current pragmatic path for speed + longer context:
    - fixed local window per query
    - fixed global prefix tokens
    - no learned routing overhead
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        window_size=128,
        global_size=64,
        chunk_size=64,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_size = global_size
        self.chunk_size = chunk_size
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

        self.mode = "hybrid"
        self.collect_metrics = False
        self.last_dense_topk_mass = None
        self.last_learned_topk_mass = None
        self.layer_idx = None

    def set_mode(self, mode):
        self.mode = mode

    def set_collect_metrics(self, collect_metrics):
        self.collect_metrics = collect_metrics

    def init_router_from_attention_projections(self):
        return

    def _project_qkv(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _dense_attention(self, q, k, v, attention_mask=None):
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q.float(), key_states.float().transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            seq_len = q.shape[2]
            causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn.to(value_states.dtype), value_states)
        return out

    def _hybrid_attention(self, q, k, v):
        batch_size, _, seq_len, _ = q.shape
        key_states = repeat_kv(k, self.num_key_value_groups)
        value_states = repeat_kv(v, self.num_key_value_groups)
        g = min(self.global_size, seq_len)
        w = min(self.window_size, seq_len)
        cs = min(self.chunk_size, seq_len)

        k_global = key_states[:, :, :g, :]
        v_global = value_states[:, :, :g, :]
        out = torch.zeros_like(q)

        for start in range(0, seq_len, cs):
            end = min(start + cs, seq_len)
            local_start = max(0, start - w + 1)

            q_chunk = q[:, :, start:end, :]
            k_local = key_states[:, :, local_start:end, :]
            v_local = value_states[:, :, local_start:end, :]

            k_comb = torch.cat([k_global, k_local], dim=2)
            v_comb = torch.cat([v_global, v_local], dim=2)

            scores = torch.matmul(q_chunk.float(), k_comb.float().transpose(-2, -1)) * self.scale

            q_pos = torch.arange(start, end, device=q.device).view(-1, 1)
            g_pos = torch.arange(0, g, device=q.device).view(1, -1)
            l_pos = torch.arange(local_start, end, device=q.device).view(1, -1)
            g_mask = g_pos <= q_pos
            l_mask = l_pos <= q_pos
            mask = torch.cat([g_mask, l_mask], dim=1)
            scores = scores.masked_fill(~mask.view(1, 1, end - start, g + end - local_start), -1e9)

            attn = F.softmax(scores, dim=-1).to(v_comb.dtype)
            out[:, :, start:end, :] = torch.matmul(attn, v_comb)

        return out

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        q, k, v = self._project_qkv(hidden_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        if self.mode == "dense":
            out = self._dense_attention(q, k, v, attention_mask=attention_mask)
        else:
            out = self._hybrid_attention(q, k, v)

        self.last_dense_topk_mass = None
        self.last_learned_topk_mass = None

        batch_size, _, seq_len, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out), None
