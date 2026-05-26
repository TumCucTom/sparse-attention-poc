#!/usr/bin/env python3
"""
HPC Benchmark Script for MiniMax M3-style Sparse Attention

Usage:
    python3 hpc_benchmark.py --model Qwen/Qwen2.5-7B --seq-len 16384 --top-k 8

This script is designed to be run on HPC infrastructure with GPU access.
It benchmarks sparse attention against dense attention at various sequence lengths.
"""

import argparse
import json
import time
import gc
import os
import torch

from minimax_m3_sparse_attention import MiniMaxSparseAttention, StandardAttention
from minimax_m3_hybrid import HybridLocalGlobalAttention


def parse_args():
    parser = argparse.ArgumentParser(description="HPC Benchmark for MiniMax M3 Sparse Attention")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B",
                        help="Model name (e.g., Qwen/Qwen2.5-7B)")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Input sequence length")
    parser.add_argument("--num-tokens", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Number of blocks to select")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Size of each block")
    parser.add_argument("--index-dim", type=int, default=32,
                        help="Index projection dimension")
    parser.add_argument("--window-size", type=int, default=32,
                        help="Window size for hybrid attention")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--attention-type", type=str, default="sparse",
                        choices=["sparse", "hybrid", "dense"],
                        help="Type of attention to benchmark")
    return parser.parse_args()


def get_attention_class(attention_type):
    if attention_type == "dense":
        return StandardAttention
    elif attention_type == "sparse":
        return MiniMaxSparseAttention
    elif attention_type == "hybrid":
        return HybridLocalGlobalAttention
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def get_kwargs(attention_type, args):
    if attention_type == "dense":
        return {}
    elif attention_type == "sparse":
        return {
            "block_size": args.block_size,
            "top_k_blocks": args.top_k,
            "index_dim": args.index_dim,
        }
    elif attention_type == "hybrid":
        return {
            "window_size": args.window_size,
            "global_blocks": args.top_k,
            "global_block_size": args.block_size,
            "index_dim": args.index_dim,
        }
    return {}


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


@torch.no_grad
def benchmark_generation(model, tokenizer, prompt, num_tokens, warmup=3):
    """Benchmark text generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    # Warmup
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    times = []
    for _ in range(args.iterations):
        start = time.time()
        _ = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    speed = num_tokens / avg_time

    return {
        "speed_tokens_per_sec": speed,
        "avg_time_sec": avg_time,
        "times": times,
    }


def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def benchmark_model(args):
    """Run benchmark on specified model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"Benchmarking {args.attention_type} attention")
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Top-K: {args.top_k}, Block-size: {args.block_size}")
    print(f"{'='*70}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt
    base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
    prompt = base_prompt * max(1, args.seq_len // len(tokenizer.encode(base_prompt)))
    prompt = prompt[:min(len(prompt), args.seq_len * 2)]  # Keep it reasonable

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()

    if args.attention_type == "dense":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # For sparse/hybrid, we need to load and replace
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Replace attention
    attention_class = get_attention_class(args.attention_type)
    kwargs = get_kwargs(args.attention_type, args)

    if args.attention_type != "dense":
        print(f"Replacing attention with {attention_class.__name__}...")
        num_replaced, _ = replace_attention(model, attention_class, **kwargs)
        print(f"Replaced {num_replaced} attention layers")

    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    memory_gb = get_memory_usage()

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    for _ in range(args.warmup):
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    print(f"\nBenchmarking ({args.iterations} iterations)...")
    times = []
    for i in range(args.iterations):
        t0 = time.time()
        _ = model.generate(**inputs, max_new_tokens=args.num_tokens, do_sample=False)
        elapsed = time.time() - t0
        speed = args.num_tokens / elapsed
        times.append(elapsed)
        print(f"  Iteration {i+1}: {speed:.1f} tokens/sec ({elapsed:.2f}s)")

    avg_speed = args.num_tokens / (sum(times) / len(times))
    avg_time = sum(times) / len(times)
    memory_peak = get_memory_usage()

    # Generate sample output
    outputs = model.generate(**inputs, max_new_tokens=args.num_tokens, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "model": args.model,
        "attention_type": args.attention_type,
        "seq_len": args.seq_len,
        "num_tokens": args.num_tokens,
        "top_k": args.top_k,
        "block_size": args.block_size,
        "index_dim": args.index_dim,
        "num_params": num_params,
        "memory_gb": memory_gb,
        "memory_peak_gb": memory_peak,
        "avg_speed_tokens_per_sec": avg_speed,
        "avg_time_sec": avg_time,
        "times": times,
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "load_time_sec": load_time,
    }

    return results


def main():
    global args
    args = parse_args()

    print("\n" + "="*70)
    print("HPC Benchmark for MiniMax M3-style Sparse Attention")
    print("="*70)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run benchmark
    results = benchmark_model(args)

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Model: {results['model']}")
    print(f"Attention type: {results['attention_type']}")
    print(f"Sequence length: {results['seq_len']}")
    print(f"Number of parameters: {results['num_params']:,}")
    print(f"Memory usage: {results['memory_peak_gb']:.1f} GB")
    print(f"Average speed: {results['avg_speed_tokens_per_sec']:.1f} tokens/sec")
    print(f"Average time: {results['avg_time_sec']:.2f}s")

    # Save results
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()