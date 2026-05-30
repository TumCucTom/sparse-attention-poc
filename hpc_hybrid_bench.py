#!/usr/bin/env python3
"""
HPC Benchmark - Fixed Hybrid Attention Only
Focuses on working hybrid attention implementation from subq_runtime
"""

import argparse
import json
import time
import gc
import os
import sys

os.environ['PATH'] = '/home/b6ar/trvbale.b6ar/miniforge3/bin:' + os.environ.get('PATH', '')

import torch

from subq_runtime import FixedHybridAttention, replace_attention, set_attention_mode


def parse_args():
    parser = argparse.ArgumentParser(description="HPC Hybrid Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--global-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", type=str, default="hybrid", choices=["dense", "hybrid"])
    return parser.parse_args()


def benchmark_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"Benchmarking {args.mode} attention")
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Window: {args.window_size}, Global: {args.global_size}, Chunk: {args.chunk_size}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
    prompt = base_prompt * max(1, args.seq_len // len(tokenizer.encode(base_prompt)))
    prompt = prompt[:min(len(prompt), args.seq_len * 2)]

    print(f"\nLoading model...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Replace attention with hybrid
    num_replaced, attn_modules = replace_attention(
        model,
        attention_class=FixedHybridAttention,
        init_router_from_attention=False,
        window_size=args.window_size,
        global_size=args.global_size,
        chunk_size=args.chunk_size,
    )
    print(f"Replaced {num_replaced} attention layers")

    # Set mode
    set_attention_mode(attn_modules, args.mode, collect_metrics=False)

    num_params = sum(p.numel() for p in model.parameters())
    memory_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        try:
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        except Exception as e:
            print(f"Warmup error: {e}")
            break

    # Benchmark
    print(f"\nBenchmarking ({args.iterations} iterations)...")
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
    memory_peak = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "model": args.model,
        "attention_type": args.mode,
        "seq_len": args.seq_len,
        "num_tokens": args.num_tokens,
        "window_size": args.window_size,
        "global_size": args.global_size,
        "chunk_size": args.chunk_size,
        "num_params": num_params,
        "memory_gb": memory_gb,
        "memory_peak_gb": memory_peak,
        "avg_speed_tokens_per_sec": avg_speed,
        "avg_time_sec": avg_time,
        "times": times,
        "load_time_sec": load_time,
    }

    return results


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("HPC Hybrid Attention Benchmark")
    print("="*70)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")

    try:
        results = benchmark_model(args)

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Model: {results['model']}")
        print(f"Attention: {results['attention_type']}")
        print(f"Sequence: {results['seq_len']}")
        print(f"Parameters: {results['num_params']:,}")
        print(f"Speed: {results['avg_speed_tokens_per_sec']:.1f} tokens/sec")

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()