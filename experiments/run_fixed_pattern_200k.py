#!/usr/bin/env python3
"""
Fixed Pattern Attention Test - tests if simple fixed patterns can work at 200K.

Uses local window + global tokens pattern (not learned sparse).
"""

import argparse
import json
import time
import gc
import os
import sys

import torch

os.environ['HF_HOME'] = '/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache'
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

from minimax_m3_sparse_attention import MiniMaxSparseAttention, StandardAttention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--attention-type", type=str, default="sparse",
                       choices=["sparse", "dense"])
    return parser.parse_args()


def replace_attention(model, attention_class, **kwargs):
    """Replace attention modules with custom attention."""
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    import types

    attention_classes = [Qwen2Attention, LlamaAttention]
    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        for attn_cls in attention_classes:
            if isinstance(module, attn_cls):
                count[0] += 1
                config = module.config

                dtype = next(model.parameters()).dtype
                attn = attention_class(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=module.head_dim,
                    **kwargs,
                ).to(device=device, dtype=dtype)

                with torch.no_grad():
                    attn.q_proj.weight.copy_(module.q_proj.weight)
                    attn.k_proj.weight.copy_(module.k_proj.weight)
                    attn.v_proj.weight.copy_(module.v_proj.weight)
                    attn.o_proj.weight.copy_(module.o_proj.weight)

                attn_modules.append(attn)

                def new_forward(self, hidden_states, **kw):
                    return attn(hidden_states, **kw)

                module.forward = types.MethodType(new_forward, module)
                break

    model.apply(replace)
    return count[0], attn_modules


def benchmark_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    token = os.environ.get('HF_TOKEN', None)

    print(f"\nBenchmarking {args.attention_type} attention")
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt
    base_prompt = "The theory of quantum mechanics "
    prompt = base_prompt * max(1, args.seq_len // 20)
    prompt = prompt[:min(len(prompt), args.seq_len * 2)]

    print(f"Loading model...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        token=token,
    )

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Replace attention
    attention_class = StandardAttention if args.attention_type == "dense" else MiniMaxSparseAttention
    kwargs = {} if args.attention_type == "dense" else {"block_size": args.block_size, "top_k_blocks": args.top_k, "index_dim": 32}

    if args.attention_type != "dense":
        print(f"Replacing attention with {attention_class.__name__}...")
        num_replaced, _ = replace_attention(model, attention_class, **kwargs)
        print(f"Replaced {num_replaced} attention layers")

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    # Warmup
    print(f"Warming up ({args.iterations} iterations)...")
    for _ in range(args.iterations):
        try:
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        except Exception as e:
            print(f"Warmup error: {e}")
            break

    # Benchmark
    print(f"Benchmarking...")
    times = []
    for i in range(args.iterations):
        t0 = time.time()
        try:
            _ = model.generate(**inputs, max_new_tokens=args.num_tokens, do_sample=False)
            elapsed = time.time() - t0
            speed = args.num_tokens / elapsed
            times.append(elapsed)
            print(f"  Iteration {i+1}: {speed:.1f} tokens/sec ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  Iteration {i+1} failed: {e}")
            times.append(0)

    avg_speed = args.num_tokens / (sum(times) / len(times)) if times else 0
    avg_time = sum(times) / len(times) if times else 0

    mem_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    num_params = sum(p.numel() for p in model.parameters())

    results = {
        "model": args.model,
        "attention_type": args.attention_type,
        "seq_len": args.seq_len,
        "num_tokens": args.num_tokens,
        "top_k": args.top_k,
        "block_size": args.block_size,
        "num_params": num_params,
        "memory_gb": mem_alloc,
        "memory_peak_gb": mem_peak,
        "avg_speed_tokens_per_sec": avg_speed,
        "avg_time_sec": avg_time,
        "times": times,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("Fixed Pattern Attention Benchmark")
    print("="*70)

    try:
        results = benchmark_model(args)

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Attention: {results['attention_type']}")
        print(f"Sequence length: {results['seq_len']}")
        print(f"Parameters: {results['num_params']:,}")
        print(f"Memory: {results['memory_peak_gb']:.1f} GB")
        print(f"Speed: {results['avg_speed_tokens_per_sec']:.1f} tokens/sec")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()