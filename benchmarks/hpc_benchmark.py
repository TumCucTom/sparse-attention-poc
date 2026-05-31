#!/usr/bin/env python3
"""
HPC Benchmark Script for Sparse Attention
Robust version with proper CUDA/CPU handling and accelerate fallback
"""

import argparse
import json
import time
import gc
import os
import sys

import torch

# Set HF_HOME to scratch to avoid quota issues
os.environ['HF_HOME'] = '/lus/lfs1aip2/scratch/b6ar/trvbale.b6ar/cache'

# Handle CUDA driver mismatch gracefully
if torch.cuda.is_available():
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"CUDA init failed: {e}, will use CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

from minimax_m3_sparse_attention import MiniMaxSparseAttention, StandardAttention
from minimax_m3_hybrid import HybridLocalGlobalAttention
from deepseek_sparse_attention import DeepSeekSparseAttention


def parse_args():
    parser = argparse.ArgumentParser(description="HPC Benchmark for Sparse Attention")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--index-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--attention-type", type=str, default="sparse",
                       choices=["sparse", "hybrid", "dense", "dsa", "local"])
    return parser.parse_args()


def get_attention_class(attention_type):
    if attention_type == "dense":
        return StandardAttention
    elif attention_type == "sparse":
        return MiniMaxSparseAttention
    elif attention_type == "hybrid":
        return HybridLocalGlobalAttention
    elif attention_type == "dsa":
        return DeepSeekSparseAttention
    elif attention_type == "local":
        return StandardAttention
    raise ValueError(f"Unknown attention type: {attention_type}")


def get_kwargs(attention_type, args):
    if attention_type == "dense":
        return {}
    elif attention_type == "sparse":
        return {"block_size": args.block_size, "top_k_blocks": args.top_k, "index_dim": args.index_dim}
    elif attention_type == "hybrid":
        return {"window_size": 32, "global_blocks": args.top_k, "global_block_size": args.block_size, "index_dim": args.index_dim}
    elif attention_type == "dsa":
        return {"block_size": args.block_size, "top_k_blocks": args.top_k, "compression_dim": args.index_dim}
    return {}


def replace_attention(model, attention_class, **kwargs):
    """Replace attention modules with custom attention (supports Qwen2, Llama, Yi)."""
    import types

    # Try multiple attention class types
    attention_classes = []
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        attention_classes.append(Qwen2Attention)
    except ImportError:
        pass

    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
        attention_classes.append(LlamaAttention)
    except ImportError:
        pass

    try:
        from transformers.models.yi.modeling_yi import YiAttention
        attention_classes.append(YiAttention)
    except ImportError:
        pass

    if not attention_classes:
        print("ERROR: Could not import any attention class (Qwen2Attention, LlamaAttention, YiAttention)")
        return 0, []

    count = [0]
    attn_modules = []
    device = next(model.parameters()).device

    def replace(module):
        for attn_cls in attention_classes:
            if isinstance(module, attn_cls):
                count[0] += 1
                config = module.config

                print(f"DEBUG replace layer {count[0]}: type={attn_cls.__name__}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}, head_dim={module.head_dim}")

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
                break  # Only replace once per module

    model.apply(replace)
    return count[0], attn_modules


def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3
    return 0, 0


def benchmark_model(args):
    """Run benchmark on specified model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"Benchmarking {args.attention_type} attention")
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Top-K: {args.top_k}, Block-size: {args.block_size}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # Load tokenizer
    token = os.environ.get('HF_TOKEN', None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # Create prompt
    base_prompt = "The theory of quantum mechanics describes how particles behave at the atomic level. "
    prompt = base_prompt * max(1, args.seq_len // len(tokenizer.encode(base_prompt)))
    prompt = prompt[:min(len(prompt), args.seq_len * 2)]

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()

    model = None  # Initialize before try block
    try:
        # Try with device_map first (needs accelerate)
        if device == 'cuda':
            # Use explicit device placement instead of device_map to avoid layer scatter issues
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map={"": 0},  # Put all on cuda:0
                trust_remote_code=True,
                token=token,
            )
    except Exception as e:
        print(f"device_map failed ({e}), trying CPU fallback...")

    if model is None:
        # Fallback to CPU without device_map
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Replace attention for sparse/hybrid
    attention_class = get_attention_class(args.attention_type)
    kwargs = get_kwargs(args.attention_type, args)

    if args.attention_type != "dense":
        print(f"Replacing attention with {attention_class.__name__}...")
        num_replaced, _ = replace_attention(model, attention_class, **kwargs)
        print(f"Replaced {num_replaced} attention layers")

    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    mem_alloc, mem_peak = get_memory_usage()

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

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
        "memory_gb": mem_alloc,
        "memory_peak_gb": mem_peak,
        "avg_speed_tokens_per_sec": avg_speed,
        "avg_time_sec": avg_time,
        "times": times,
        "load_time_sec": load_time,
    }

    return results


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("HPC Benchmark for Sparse Attention")
    print("="*70)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        results = benchmark_model(args)

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Model: {results['model']}")
        print(f"Attention type: {results['attention_type']}")
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

    print(f"{'='*70}")


if __name__ == "__main__":
    main()