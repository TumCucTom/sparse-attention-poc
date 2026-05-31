#!/usr/bin/env python3
"""
Synthetic long-context retrieval benchmark for dense vs hybrid Qwen runtime.

This is not a full RULER/MRCR reproduction. It is a local proxy to answer:
- does the sparse runtime still recover a planted distant fact?
- how does this change as context grows?
"""

import argparse
import gc
import os
import random
import re
import string
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


def enforce_safe_local_limits(model_name, trials):
    if os.environ.get("SUBQ_UNSAFE_OK") == "1":
        return
    if "0.5B" not in model_name:
        raise RuntimeError("Safe local mode only allows 0.5B models. Set SUBQ_UNSAFE_OK=1 to override.")
    if trials > 1:
        raise RuntimeError("Safe local mode limits --trials to 1. Set SUBQ_UNSAFE_OK=1 to override.")


def random_word(rng, length=8):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def build_example(repeat_count, answer, seed):
    rng = random.Random(seed)
    filler = []
    for _ in range(repeat_count):
        filler.append(
            f"{random_word(rng)} {random_word(rng)} {random_word(rng)} {random_word(rng)}."
        )

    insert_at = repeat_count // 2
    fact = f"The access code is {answer}."
    filler.insert(insert_at, fact)
    context = " ".join(filler)
    prompt = (
        f"{context}\n\n"
        f"Question: What is the access code?\n"
        f"Answer:"
    )
    return prompt


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text, elapsed


def extract_code(text):
    match = re.search(r"access code(?: is)?\s+([A-Z0-9\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    tail = text.split("Answer:")[-1].strip().split()
    if tail:
        return re.sub(r"[^A-Z0-9\-]", "", tail[0].upper())
    return ""


def run_mode(model, attn_modules, tokenizer, mode, repeat_counts, trials):
    set_attention_mode(attn_modules, mode, collect_metrics=False)
    rows = []
    for repeat_count in repeat_counts:
        correct = 0
        total_time = 0.0
        total_prompt_tokens = 0
        for trial in range(trials):
            answer = f"ZX{trial}{repeat_count}"
            prompt = build_example(repeat_count, answer, seed=trial + repeat_count)
            total_prompt_tokens += len(tokenizer.encode(prompt))
            text, elapsed = generate_answer(model, tokenizer, prompt)
            pred = extract_code(text)
            correct += int(answer.upper() == pred)
            total_time += elapsed
        rows.append(
            {
                "repeat_count": repeat_count,
                "avg_prompt_tokens": total_prompt_tokens / trials,
                "accuracy": correct / trials,
                "avg_time": total_time / trials,
            }
        )
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--global-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    enforce_safe_local_limits(args.model, args.trials)

    repeat_counts = [32, 64, 128]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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

    dense_rows = run_mode(model, attn_modules, tokenizer, "dense", repeat_counts, args.trials)
    hybrid_rows = run_mode(model, attn_modules, tokenizer, "hybrid", repeat_counts, args.trials)

    print("\nDense")
    for row in dense_rows:
        print(
            f"  repeats={row['repeat_count']:<4} "
            f"tokens~{row['avg_prompt_tokens']:<7.0f} "
            f"acc={row['accuracy']:.2f} "
            f"time={row['avg_time']:.2f}s"
        )

    print("\nHybrid")
    for row in hybrid_rows:
        print(
            f"  repeats={row['repeat_count']:<4} "
            f"tokens~{row['avg_prompt_tokens']:<7.0f} "
            f"acc={row['accuracy']:.2f} "
            f"time={row['avg_time']:.2f}s"
        )

    cleanup()


if __name__ == "__main__":
    main()
