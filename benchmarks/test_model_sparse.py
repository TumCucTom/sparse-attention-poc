#!/usr/bin/env python3
"""
Test script to reproduce the actual error path in HPC benchmark
"""

import os
os.environ['DEBUG_ATTN'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from minimax_m3_sparse_attention import MiniMaxSparseAttention, StandardAttention

import types

def replace_attention(model, attention_class, **kwargs):
    """Replace Qwen2Attention with custom attention."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        if isinstance(module, Qwen2Attention):
            count[0] += 1
            config = module.config

            print(f"\nReplacing attention layer {count[0]}")
            print(f"  config.hidden_size: {config.hidden_size}")
            print(f"  config.num_attention_heads: {config.num_attention_heads}")
            print(f"  config.num_key_value_heads: {config.num_key_value_heads}")
            print(f"  module.head_dim: {module.head_dim}")
            print(f"  module.q_proj.weight.shape: {module.q_proj.weight.shape}")
            print(f"  module.k_proj.weight.shape: {module.k_proj.weight.shape}")

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
                print(f"  Initializing index projections...")
                attn._init_index_from_attention()

            attn_modules.append(attn)

            def new_forward(self, hidden_states, **kw):
                return attn(hidden_states, **kw)

            module.forward = types.MethodType(new_forward, module)

    model.apply(replace)
    return count[0], attn_modules


def test_model():
    print("="*60)
    print("Testing MiniMaxSparseAttention replacement on Qwen2.5-7B")
    print("="*60)

    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    print("Model loaded")

    # Replace with sparse attention
    print("\nReplacing attention with MiniMaxSparseAttention...")
    num_replaced, _ = replace_attention(
        model,
        MiniMaxSparseAttention,
        block_size=16,
        top_k_blocks=8,
        index_dim=32,
    )
    print(f"\nReplaced {num_replaced} attention layers")

    # Create a test input
    prompt = "The theory of quantum mechanics describes how"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Sequence length: {input_ids.shape[1]}")

    # Try generation
    print("\nTrying generation...")
    try:
        output = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=False)
        print(f"SUCCESS! Output shape: {output.shape}")
        print(f"Generated: {tokenizer.decode(output[0])}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()