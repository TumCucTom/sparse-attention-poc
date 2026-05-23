# Local Safe Usage

This repo can stress a 16 GB RAM machine if you run repeated long-context HuggingFace benchmarks.

## Safe Defaults

Use only:

- `Qwen/Qwen2.5-0.5B`
- short generations
- small prompt repeats
- single-run smoke tests

The benchmark scripts now fail closed unless you explicitly opt out.

## Safe Commands

Tiny hybrid smoke test:

```bash
python3 benchmark_hybrid_qwen.py --model Qwen/Qwen2.5-0.5B --prompt-repeat 1 --num-tokens 2
```

Small corrected sparse benchmark:

```bash
python3 benchmark_subq.py --model Qwen/Qwen2.5-0.5B --prompt-repeat 1 --num-tokens 4
```

Synthetic benchmark only:

```bash
python3 benchmark.py
```

## Unsafe Override

Only use this if you are intentionally accepting RAM/swap risk:

```bash
SUBQ_UNSAFE_OK=1 ...
```

## Avoid Locally

- repeated 1.5B model sweeps
- long prompt-repeat values
- multi-trial retrieval sweeps
- long-context generation loops

Those should move to HPC or a more controlled machine.
