#!/usr/bin/env python3
"""
Benchmark dense vs fixed hybrid local+global attention on Qwen2.5.
"""

import argparse
import gc
import os
import time
import warnings

import torch

from subq_runtime import FixedHybridAttention, replace_attention, set_attention_mode

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
    if prompt_repeat > 8:
        raise RuntimeError("Safe local mode limits --prompt-repeat to 8. Set SUBQ_UNSAFE_OK=1 to override.")
    if num_tokens > 16:
        raise RuntimeError("Safe local mode limits --num-tokens to 16. Set SUBQ_UNSAFE_OK=1 to override.")


@torch.no_grad()
def benchmark_generation(model, tokenizer, prompt, num_tokens):
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
    return {
        "elapsed": elapsed,
        "speed": generated / elapsed,
        "text": tokenizer.decode(outputs[0], skip_special_tokens=True),
    }


@torch.no_grad()
def evaluate_loss(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--global-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--prompt-repeat", type=int, default=4)
    args = parser.parse_args()
    enforce_safe_local_limits(args.model, args.prompt_repeat, args.num_tokens)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_base = (
        "Explain how general relativity and quantum mechanics differ, "
        "then mention one reason unifying them is difficult."
    )
    prompt = " ".join([prompt_base] * args.prompt_repeat)
    eval_prompt = "General relativity explains gravity as geometry of spacetime."

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
    )
    num_replaced, attn_modules = replace_attention(
        model,
        attention_class=FixedHybridAttention,
        init_router_from_attention=False,
        window_size=args.window_size,
        global_size=args.global_size,
        chunk_size=args.chunk_size,
    )
    print(f"Replaced {num_replaced} attention layers")

    for mode in ["dense", "hybrid"]:
        set_attention_mode(attn_modules, mode, collect_metrics=False)
        loss = evaluate_loss(model, tokenizer, eval_prompt)
        stats = benchmark_generation(model, tokenizer, prompt, args.num_tokens)
        print(f"\nMode: {mode}")
        print(f"  Loss:  {loss:.4f}")
        print(f"  Time:  {stats['elapsed']:.2f}s")
        print(f"  Speed: {stats['speed']:.2f} tokens/sec")
        print(f"  Text:  {stats['text'][:220]}")

    cleanup()


if __name__ == "__main__":
    main()
