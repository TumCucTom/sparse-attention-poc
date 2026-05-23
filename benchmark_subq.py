#!/usr/bin/env python3
"""
Benchmark the corrected per-query SubQ runtime on Qwen2.5.

Compares:
- dense attention
- sparse learned attention
- sparse oracle attention

The runtime path matches the working hybrid experiment: all queries remain
active and only key/value selection is sparse.
"""

import argparse
import gc
import os
import time
import warnings

import torch

from subq_runtime import (
    LocalGlobalTokenSparseAttention,
    PerQuerySparseAttention,
    StructuredSparseAttention,
    replace_attention,
    set_attention_mode,
)

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def enforce_safe_local_limits(model_name, prompt_repeat, num_tokens):
    if os.environ.get("SUBQ_UNSAFE_OK") == "1":
        return
    if "0.5B" not in model_name:
        raise RuntimeError("Safe local mode only allows 0.5B models. Set SUBQ_UNSAFE_OK=1 to override.")
    if prompt_repeat > 2:
        raise RuntimeError("Safe local mode limits --prompt-repeat to 2. Set SUBQ_UNSAFE_OK=1 to override.")
    if num_tokens > 8:
        raise RuntimeError("Safe local mode limits --num-tokens to 8. Set SUBQ_UNSAFE_OK=1 to override.")


@torch.no_grad()
def evaluate_prompt_loss(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


@torch.no_grad()
def benchmark_generation(model, tokenizer, prompt, num_tokens, mode_name):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    _ = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=num_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - start

    generated = outputs.shape[1] - inputs.input_ids.shape[1]
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{mode_name}")
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  Speed:  {generated / elapsed:.2f} tokens/sec")
    print(f"  Output: {text[:220]}")
    return {
        "mode": mode_name,
        "elapsed": elapsed,
        "tokens_per_second": generated / elapsed,
        "text": text,
    }


def run_single_model(model_name, top_k, router_dim, prompt, loss_prompts, num_tokens, attention_class, mode_names, extra_kwargs=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 72)
    print(f"Benchmarking {model_name} with top_k={top_k}, router_dim={router_dim}")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
    )

    kwargs = {"top_k": top_k, "router_dim": router_dim}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    num_replaced, attn_modules = replace_attention(
        model,
        attention_class=attention_class,
        **kwargs,
    )
    print(f"Replaced {num_replaced} attention layers")

    results = []
    for mode in mode_names:
        set_attention_mode(attn_modules, mode, collect_metrics=True)
        losses = [evaluate_prompt_loss(model, tokenizer, item) for item in loss_prompts]
        avg_loss = sum(losses) / len(losses)
        learned_values = [attn.last_learned_topk_mass.item() for attn in attn_modules if attn.last_learned_topk_mass is not None]
        oracle_values = [attn.last_dense_topk_mass.item() for attn in attn_modules if attn.last_dense_topk_mass is not None]
        learned_mass = sum(learned_values) / len(learned_values) if learned_values else None
        oracle_mass = sum(oracle_values) / len(oracle_values) if oracle_values else None

        gen_stats = benchmark_generation(model, tokenizer, prompt, num_tokens, mode)
        gen_stats["loss"] = avg_loss
        gen_stats["learned_mass"] = learned_mass
        gen_stats["oracle_mass"] = oracle_mass
        results.append(gen_stats)

    cleanup()
    del model
    cleanup()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--router-dim", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--global-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    enforce_safe_local_limits(args.model, args.prompt_repeat, args.num_tokens)

    prompt_base = (
        "Explain how general relativity and quantum mechanics differ, "
        "then mention one reason unifying them is difficult."
    )
    prompt = " ".join([prompt_base] * args.prompt_repeat)
    loss_prompts = [
        "General relativity explains gravity as geometry of spacetime.",
        "Quantum mechanics predicts probabilities for microscopic systems.",
        "The speed of light limits how quickly information can travel.",
        "Physicists still lack a complete quantum theory of gravity.",
    ]

    results_unstructured = run_single_model(
        args.model,
        args.top_k,
        args.router_dim,
        prompt,
        loss_prompts,
        args.num_tokens,
        attention_class=PerQuerySparseAttention,
        mode_names=["dense", "sparse_learned", "sparse_oracle"],
    )

    results_structured = run_single_model(
        args.model,
        args.top_k,
        args.router_dim,
        prompt,
        loss_prompts,
        args.num_tokens,
        attention_class=StructuredSparseAttention,
        mode_names=["dense", "structured_learned", "structured_oracle"],
        extra_kwargs={"window_size": args.window_size, "global_size": args.global_size},
    )

    results_local_global = run_single_model(
        args.model,
        args.top_k,
        args.router_dim,
        prompt,
        loss_prompts,
        args.num_tokens,
        attention_class=LocalGlobalTokenSparseAttention,
        mode_names=["dense", "local_global_learned", "local_global_oracle"],
        extra_kwargs={"window_size": args.window_size, "chunk_size": args.chunk_size},
    )

    results = results_unstructured + results_structured + results_local_global

    print("\n" + "=" * 72)
    print("Benchmark Summary")
    print("=" * 72)
    print(f"{'Mode':<18} {'Loss':>8} {'Speed':>10} {'Mass':>8}")
    print("-" * 72)
    for item in results:
        mass = item["learned_mass"]
        if item["mode"] in {"sparse_oracle", "structured_oracle", "local_global_oracle"}:
            mass = item["oracle_mass"]
        mass_str = f"{mass:.3f}" if mass is not None else "-"
        print(f"{item['mode']:<18} {item['loss']:>8.4f} {item['tokens_per_second']:>10.2f} {mass_str:>8}")


if __name__ == "__main__":
    main()
