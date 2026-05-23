# SubQ / SSA Reproduction Plan

## Goal

Reproduce the core claims implied by SubQ / SSA, not just a generic sparse-attention proof of concept.

That means demonstrating:

1. A truly sub-quadratic learned routing runtime.
2. Long-context scaling at meaningful sequence lengths.
3. Retrieval quality on long-context tasks.
4. A scalable training recipe beyond toy LM experiments.
5. Systems work needed to make the method practically fast.


## Current Status

### Achieved

- Identified that the original SubQ POC was structurally wrong for causal LM.
  It selected a global subset of query positions per head and zeroed most token outputs.

- Identified that the earlier HF replacement path did not properly register the custom attention modules for optimization.

- Built a better training recipe in `train_subq_hybrid.py`:
  - dense LM training
  - router distillation from dense attention
  - sparse-path joint fine-tuning

- Built a shared HF-compatible runtime in `subq_runtime.py`:
  - RoPE preserved
  - attention-mask contract preserved
  - cache-aware hybrid attention supported

- Demonstrated one practical runtime win locally with fixed hybrid attention:
  - `Qwen/Qwen2.5-0.5B`
  - long prompt cached generation
  - ~1.30x speedup over dense


## Gap To SSA Claims

### 1. Truly Sub-Quadratic Learned Routing Runtime

Current status:
- Learned sparse routing exists locally.
- It is trainable with dense-teacher supervision.
- It is not yet the best runtime path on MPS because arbitrary per-query sparse gather is too expensive.

Still needed:
- A learned sparse runtime whose complexity and actual kernel behavior scale sub-quadratically.
- Ideally no full per-query arbitrary gather over the sequence.
- A runtime primitive that survives integration into HF generation with cache support.

Local success criteria:
- Learned sparse runtime beats dense and fixed hybrid on long-prompt generation for small Qwen locally.
- Scaling curve is meaningfully flatter than dense as context grows.

Likely HPC / systems threshold:
- If the only remaining blocker is kernel efficiency or GPU-parallel scale rather than algorithm design, move to HPC / CUDA work.


### 2. Long-Context Scaling Proof

Current status:
- Local tests are short-context or modest long-prompt only.
- No credible 128K+ evidence yet.

Still needed:
- Prefill benchmarks at increasingly large sequence lengths.
- Cached decoding benchmarks with long prompts.
- Memory scaling measurements.

Local success criteria:
- Push synthetic and small-model benchmarks as far as local memory allows.
- Produce plots / tables showing scaling trend up to the local hardware ceiling.

HPC threshold:
- Once local MPS cannot support the next sequence-length regime needed for meaningful evidence.
- Specifically once the next milestone is 128K+ prefill or large-model long-context evaluation.


### 3. Retrieval Quality

Current status:
- Only toy LM losses and short prompt generation comparisons.
- No long-context retrieval benchmark yet.

Still needed:
- Long-context synthetic retrieval tasks.
- Ideally RULER-like or MRCR-like evaluation.
- Demonstrate that sparse retrieval still recovers distant relevant tokens.

Local success criteria:
- Build and run synthetic retrieval tests at lengths the local machine can support.
- Show failure/success modes for dense vs sparse runtimes.

HPC threshold:
- Full RULER/MRCR-scale sweeps on larger models or long contexts.


### 4. Scaled Training Recipe

Current status:
- Hybrid dense-teacher -> sparse fine-tune works on toy LM setup.
- No serious long-context dataset, no SFT at scale, no RL.

Still needed:
- Long-context distillation or SFT on a meaningful corpus.
- Larger model and longer contexts.
- Potential RL stage if matching SSA-style training claims matters.

Local success criteria:
- Validate that the training recipe still works on a somewhat larger local setup.
- Show that the router remains useful as sequence length grows.

HPC threshold:
- Once optimizer state, model size, or context length exceed local capacity.
- Real long-context SFT / RL will almost certainly require HPC.


### 5. Systems Work

Current status:
- Most sparse runtimes here are Python- and gather-heavy.
- Fixed hybrid is currently the strongest practical runtime path.

Still needed:
- Kernel optimization.
- Better cache-aware inference.
- Potential sequence parallelism or custom CUDA/Triton kernels.

Local success criteria:
- Identify the best algorithmic sparse pattern before low-level kernel work.
- Produce evidence that a given pattern is worth optimizing.

HPC threshold:
- Once the remaining bottleneck is clearly implementation/kernel throughput rather than design.


## Immediate Local Work Plan

1. Continue refining learned sparse runtime candidates that avoid arbitrary per-query gather.
2. Benchmark scaling locally for:
   - dense
   - fixed hybrid
   - learned sparse candidates
3. Add long-context synthetic retrieval tests.
4. Record the exact point where:
   - model size
   - context length
   - runtime kernel needs
   make local progress no longer meaningful.


## Exit Criteria For Local Work

Stop local-only work and move to HPC when any of these become true:

1. The best remaining path requires kernel work or GPU throughput that cannot be approximated on local MPS.
2. The next meaningful benchmark is 128K+ context and local hardware cannot run it reliably.
3. The next meaningful training experiment requires larger models, real long-context SFT, or RL beyond local memory/time.
4. Local experiments stop changing the conclusion because hardware, not algorithm design, is the limiting factor.


## Safe Local Mode

Because this machine has 16 GB RAM and can spill heavily into swap, local execution must stay conservative.

Default safe policy:

- Only run `Qwen/Qwen2.5-0.5B` locally.
- Keep long-prompt benchmark repeats small.
- Keep generation lengths short.
- Avoid multi-trial long-context sweeps by default.
- Prefer single-model / single-run comparisons over repeated reload-heavy sweeps.

Current script guardrails:

- `benchmark_hybrid_qwen.py`
  - default `--prompt-repeat 4`
  - default `--num-tokens 8`
  - blocks models larger than `0.5B` unless `SUBQ_UNSAFE_OK=1`

- `benchmark_retrieval_qwen.py`
  - default `--trials 1`
  - reduced repeat schedule
  - blocks models larger than `0.5B` unless `SUBQ_UNSAFE_OK=1`

- `benchmark_subq.py`
  - default `--num-tokens 8`
  - blocks large prompt repeats and larger-than-0.5B models unless `SUBQ_UNSAFE_OK=1`

Unsafe override:

- Set `SUBQ_UNSAFE_OK=1` only when intentionally accepting higher memory/swap risk.


## Current Working Conclusion

- We do have a credible sparse-attention research POC.
- We do not yet have a faithful reproduction of SSA/SubQ’s full claims.
- The best local runtime path today is fixed hybrid local+global attention.
- The best local training path today is dense-teacher supervised sparse routing.
- The next decisive question is whether a learned routing runtime can become both:
  - actually sub-quadratic in practice
  - actually faster than fixed hybrid and dense on long context
