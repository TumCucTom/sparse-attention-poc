#!/usr/bin/env python3
"""
Sanity test for the corrected SubQ runtime on Qwen2.5.
"""

import argparse
import warnings

import torch

from subq_runtime import replace_attention, set_attention_mode

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--router-dim", type=int, default=8)
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("Per-query SubQ Runtime Test on Qwen2.5")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"top_k={args.top_k}, router_dim={args.router_dim}")

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
        top_k=args.top_k,
        router_dim=args.router_dim,
    )
    print(f"Replaced {num_replaced} attention layers")

    prompt = (
        "Summarize the relationship between general relativity and "
        "quantum mechanics in two sentences."
    )
    print(f"\nPrompt: {prompt}")

    for mode in ["dense", "sparse_learned", "sparse_oracle"]:
        set_attention_mode(attn_modules, mode, collect_metrics=True)
        text = generate(model, tokenizer, prompt)
        learned_mass = None
        oracle_mass = None
        if attn_modules and attn_modules[0].last_learned_topk_mass is not None:
            learned_mass = sum(
                attn.last_learned_topk_mass.item() for attn in attn_modules if attn.last_learned_topk_mass is not None
            ) / len(attn_modules)
            oracle_mass = sum(
                attn.last_dense_topk_mass.item() for attn in attn_modules if attn.last_dense_topk_mass is not None
            ) / len(attn_modules)
        print(f"\nMode: {mode}")
        if learned_mass is not None:
            print(f"  learned_mass={learned_mass:.3f}, oracle_mass={oracle_mass:.3f}")
        print(f"  {text}")


if __name__ == "__main__":
    main()
