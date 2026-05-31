#!/usr/bin/env python3
"""
Test script to reproduce and debug sparse attention GQA issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniMaxSparseAttention(nn.Module):
    """MiniMax M3-style sparse attention - debug version"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, block_size=16, top_k_blocks=4, index_dim=32):
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

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.index_q = nn.Linear(hidden_size, num_heads * index_dim, bias=False)
        self.index_k = nn.Linear(hidden_size, num_kv_heads * index_dim, bias=False)

    def _compute_block_scores(self, idx_q, idx_k):
        batch_size, seq_len, num_heads, idx_dim = idx_q.shape
        num_kv_heads = idx_k.shape[2]

        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size

        q_padded = torch.zeros(batch_size, pad_len, num_heads, idx_dim, device=idx_q.device, dtype=torch.float32)
        q_padded[:, :seq_len] = idx_q.float()
        q_blocks = q_padded.view(batch_size, num_blocks, self.block_size, num_heads, idx_dim)

        k_padded = torch.zeros(batch_size, pad_len, num_kv_heads, idx_dim, device=idx_k.device, dtype=torch.float32)
        k_padded[:, :seq_len] = idx_k.float()
        k_blocks = k_padded.view(batch_size, num_blocks, self.block_size, num_kv_heads, idx_dim)

        q_avg = q_blocks.mean(dim=2)
        k_avg = k_blocks.mean(dim=2)

        k_avg_rep = k_avg.repeat_interleave(self.num_key_value_groups, dim=2)

        scores = torch.matmul(
            q_avg.permute(0, 2, 1, 3),
            k_avg_rep.permute(0, 2, 3, 1)
        ) * (idx_dim ** -0.5)

        causal = torch.tril(torch.ones(num_blocks, num_blocks, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.view(1, 1, num_blocks, num_blocks), -1e9)

        return scores

    def _sparse_attention(self, q, k, v, selected_blocks):
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size
        num_kv_heads = k.shape[1]
        actual_k = selected_blocks.shape[-1]

        print(f"  _sparse_attention: batch={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
        print(f"    k shape: {k.shape}, v shape: {v.shape}")
        print(f"    num_key_value_groups: {self.num_key_value_groups}")
        print(f"    selected_blocks shape: {selected_blocks.shape}, actual_k: {actual_k}")

        k_rep = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_key_value_groups, dim=1)
        print(f"    k_rep shape: {k_rep.shape}, v_rep shape: {v_rep.shape}")

        max_block = (seq_len + block_size - 1) // block_size
        selected_blocks = selected_blocks.clamp(0, max_block - 1)

        block_offsets = torch.arange(block_size, device=q.device, dtype=torch.long).view(1, 1, 1, block_size)
        block_base = selected_blocks.unsqueeze(-1) * block_size
        position_indices = (block_base + block_offsets).view(batch_size, num_heads, actual_k * block_size)
        position_indices = position_indices.clamp(0, seq_len - 1)

        print(f"    position_indices shape: {position_indices.shape}")

        gather_idx = position_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        print(f"    gather_idx shape: {gather_idx.shape}, k_rep dim 2: {k_rep.shape[2]}")

        try:
            k_selected = torch.gather(k_rep, 2, gather_idx)
            v_selected = torch.gather(v_rep, 2, gather_idx)
            print(f"    k_selected shape: {k_selected.shape}")
        except Exception as e:
            print(f"    GATHER ERROR: {e}")
            raise

        scores = torch.matmul(q.float(), k_selected.float().transpose(-2, -1)) * self.scale

        k_pos = position_indices.view(batch_size, num_heads, actual_k * block_size, 1)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, 1, seq_len)
        causal_mask = (k_pos <= q_pos).transpose(-2, -1)
        scores = scores.masked_fill(~causal_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_selected)

        return out

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, hidden_size = hidden_states.shape
        print(f"\nforward: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")

        idx_q = self.index_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.index_dim)
        idx_k = self.index_k(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.index_dim)

        block_scores = self._compute_block_scores(idx_q, idx_k)
        print(f"block_scores shape: {block_scores.shape}")

        actual_k = min(self.top_k_blocks, block_scores.shape[-1])
        print(f"actual_k: {actual_k}")

        try:
            _, topk_blocks = block_scores.topk(actual_k, dim=-1)
            print(f"topk_blocks shape: {topk_blocks.shape}")
            topk_blocks = topk_blocks[:, :, 0, :]
            print(f"topk_blocks after slicing: {topk_blocks.shape}")
        except Exception as e:
            print(f"TOPK ERROR: {e}")
            raise

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        out = self._sparse_attention(q, k, v, topk_blocks)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(out), None


def test_sparse_attention():
    """Test with Qwen 7B-like config: 28 heads, 4 KV heads"""
    print("="*60)
    print("Testing MiniMaxSparseAttention with GQA")
    print("="*60)

    # Qwen 7B config
    hidden_size = 3584
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    top_k_blocks = 4
    index_dim = 32

    seq_len = 2048
    batch_size = 1

    print(f"\nConfig: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"GQA ratio: {num_heads / num_kv_heads}")

    attn = MiniMaxSparseAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        top_k_blocks=top_k_blocks,
        index_dim=index_dim,
    )

    # Random input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    print(f"\nRunning forward pass...")
    try:
        out, _ = attn(hidden_states)
        print(f"\nSUCCESS! Output shape: {out.shape}")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sparse_attention()