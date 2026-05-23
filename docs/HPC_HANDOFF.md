# HPC Handoff

## Goal

Use HPC resources for the work that is no longer meaningful or safe on the local 16 GB RAM machine:

1. larger-model runtime benchmarking
2. longer-context prefill / decode scaling
3. synthetic retrieval sweeps at larger prompt lengths
4. scaled sparse training experiments


## What Is Already Proven Locally

- Original SubQ POC had a structural bug for causal LM.
- HF integration now preserves:
  - RoPE
  - attention mask semantics
  - cache-aware hybrid attention path
- Best local training recipe:
  - dense LM train
  - router distillation
  - sparse-path joint fine-tune
- Best local runtime path:
  - fixed hybrid local+global attention
- Best local runtime win observed:
  - `Qwen/Qwen2.5-0.5B`
  - long prompt cached generation
  - about `1.30x` speedup over dense


## Recommended HPC Priorities

### 1. Runtime Scaling

Run:

- `benchmark_hybrid_qwen.py`
- `benchmark_subq.py`

Targets:

- `Qwen/Qwen2.5-1.5B`
- larger prompt repeats
- larger generation lengths
- record memory usage and tokens/sec

Questions:

- does fixed hybrid stay faster at larger scale?
- does learned sparse routing ever beat fixed hybrid once memory pressure is removed?


### 2. Retrieval Evaluation

Run:

- `benchmark_retrieval_qwen.py`

Extend:

- larger prompt lengths
- more trials
- stronger planted-fact layouts

Questions:

- does hybrid preserve distant retrieval as context grows?
- does learned sparse preserve retrieval better than fixed hybrid?


### 3. Training Scaling

Run:

- `train_subq_hybrid.py`

Scale:

- larger dataset
- larger model
- longer sequences
- longer training horizon

Questions:

- does the dense-teacher sparse training recipe remain stable at larger scale?
- can learned sparse routing remain useful beyond toy LM conditions?


## Minimal Environment Assumptions

- CUDA-capable GPUs preferred
- enough VRAM for:
  - `Qwen/Qwen2.5-1.5B`
  - long prompts
  - repeated evaluation runs
- Python 3.9+
- PyTorch + Transformers installed


## Suggested First HPC Runs

### Runtime

```bash
python3 benchmark_hybrid_qwen.py \
  --model Qwen/Qwen2.5-1.5B \
  --window-size 128 \
  --global-size 64 \
  --chunk-size 64 \
  --num-tokens 64 \
  --prompt-repeat 64
```

```bash
python3 benchmark_subq.py \
  --model Qwen/Qwen2.5-1.5B \
  --top-k 8 \
  --router-dim 8 \
  --num-tokens 32 \
  --prompt-repeat 8
```

### Retrieval

```bash
SUBQ_UNSAFE_OK=1 python3 benchmark_retrieval_qwen.py \
  --model Qwen/Qwen2.5-1.5B \
  --window-size 128 \
  --global-size 64 \
  --chunk-size 64 \
  --trials 4
```

### Training

```bash
SUBQ_UNSAFE_OK=1 python3 train_subq_hybrid.py \
  --model Qwen/Qwen2.5-0.5B \
  --dense-steps 100 \
  --router-steps 200 \
  --sparse-steps 100 \
  --top-k 8 \
  --router-dim 8 \
  --max-length 128
```


## Success Criteria For HPC Phase

1. Show runtime scaling at larger context without local memory/swap constraints.
2. Determine whether learned sparse routing can beat fixed hybrid in actual runtime.
3. Establish whether retrieval quality survives at larger context.
4. Determine whether the training recipe scales beyond toy LM conditions.


## Current Honest Boundary

Local machine is still useful for:

- code changes
- tiny smoke tests
- synthetic benchmarks

HPC is now the right place for:

- repeated large-model evaluation
- long-context sweeps
- retrieval sweeps
- scaled sparse training
