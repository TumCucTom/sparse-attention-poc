#!/usr/bin/env python3
"""
Hybrid SubQ experiment: dense training, sparse inference.

This script fixes two issues in the earlier POCs:
1. All query positions remain active. We sparsify key/value selection per query.
2. Custom attention modules are registered on the model, so optimizers can see them.

Training strategy:
- Phase 1: fine-tune the model with dense attention until LM loss is reasonable.
- Phase 2: freeze the model and train only a low-rank router to match dense attention.
- Evaluation: compare dense attention, oracle sparse attention, and learned sparse attention.
"""

import argparse
import gc
import time
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


class TeacherRoutedSparseAttention(nn.Module):
    """
    Dense attention for training, sparse attention for inference.

    The router is trained to approximate dense attention weights directly.
    At inference we keep every query active and sparsify only the key/value set.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        top_k=32,
        router_dim=16,
        teacher_temperature=0.7,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.router_dim = router_dim
        self.teacher_temperature = teacher_temperature
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.router_scale = router_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.router_q = nn.Linear(hidden_size, num_heads * router_dim, bias=False)
        self.router_k = nn.Linear(hidden_size, num_heads * router_dim, bias=False)
        nn.init.normal_(self.router_q.weight, std=0.02)
        nn.init.normal_(self.router_k.weight, std=0.02)

        self.mode = "dense"
        self.collect_metrics = True
        self.last_router_loss = None
        self.last_dense_topk_mass = None
        self.last_learned_topk_mass = None

    def set_mode(self, mode):
        self.mode = mode

    def set_collect_metrics(self, collect_metrics):
        self.collect_metrics = collect_metrics

    def _project_qkv(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        return q, k, v

    def _causal_mask(self, seq_len, device_):
        return torch.tril(torch.ones(seq_len, seq_len, device=device_, dtype=torch.bool))

    def _dense_attention(self, q, k, v):
        seq_len = q.shape[2]
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        causal_mask = self._causal_mask(seq_len, q.device)
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn.to(v.dtype), v)
        return out, attn, scores

    def _router_scores(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        rq = self.router_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.router_dim)
        rk = self.router_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.router_dim)

        rq = rq.permute(0, 2, 1, 3)
        rk = rk.permute(0, 2, 1, 3)

        scores = torch.matmul(rq.float(), rk.float().transpose(-2, -1)) * self.router_scale
        causal_mask = self._causal_mask(seq_len, hidden_states.device)
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), -1e9)
        return scores

    def _teacher_probs(self, dense_scores):
        return F.softmax(dense_scores.detach() / self.teacher_temperature, dim=-1)

    def _sparse_attention_from_indices(self, q, k, v, topk_indices):
        batch_size, num_heads, seq_len, _ = q.shape
        actual_k = topk_indices.shape[-1]
        query_positions = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)

        k_expanded = k.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        v_expanded = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)

        k_selected = torch.gather(k_expanded, 3, gather_index)
        v_selected = torch.gather(v_expanded, 3, gather_index)

        scores = (q.unsqueeze(-2).float() * k_selected.float()).sum(dim=-1) * self.scale
        valid = topk_indices <= query_positions
        scores = scores.masked_fill(~valid, -1e9)

        attn = F.softmax(scores, dim=-1).to(v.dtype)
        out = (attn.unsqueeze(-1) * v_selected).sum(dim=-2)
        return out

    def _topk_mass(self, dense_attn, indices):
        mass = torch.gather(dense_attn, -1, indices).sum(dim=-1)
        return mass.mean().detach()

    def forward(self, hidden_states):
        q, k, v = self._project_qkv(hidden_states)
        seq_len = q.shape[2]
        actual_k = min(self.top_k, seq_len)

        need_dense = self.mode in {"dense", "sparse_oracle"} or self.training or self.collect_metrics
        dense_out = None
        dense_attn = None
        teacher_probs = None

        if need_dense:
            dense_out, dense_attn, dense_scores = self._dense_attention(q, k, v)
            teacher_probs = self._teacher_probs(dense_scores)
            router_scores = self._router_scores(hidden_states)
            router_log_probs = F.log_softmax(router_scores, dim=-1)

            teacher_log_probs = teacher_probs.clamp_min(1e-8).log()
            kl = teacher_probs * (teacher_log_probs - router_log_probs)
            self.last_router_loss = kl.sum(dim=-1).mean()

            learned_topk = router_scores.topk(actual_k, dim=-1).indices
            dense_topk = dense_attn.topk(actual_k, dim=-1).indices
            self.last_learned_topk_mass = self._topk_mass(dense_attn, learned_topk)
            self.last_dense_topk_mass = self._topk_mass(dense_attn, dense_topk)
        else:
            router_scores = self._router_scores(hidden_states)
            learned_topk = router_scores.topk(actual_k, dim=-1).indices
            self.last_router_loss = None
            self.last_learned_topk_mass = None
            self.last_dense_topk_mass = None

        if self.mode == "dense":
            out = dense_out
        elif self.mode == "sparse_oracle":
            oracle_indices = dense_attn.topk(actual_k, dim=-1).indices
            out = self._sparse_attention_from_indices(q, k, v, oracle_indices)
        elif self.mode == "sparse_learned":
            out = self._sparse_attention_from_indices(q, k, v, learned_topk)
        else:
            raise ValueError(f"Unknown attention mode: {self.mode}")

        batch_size = hidden_states.shape[0]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out), None


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def replace_attention(model, attention_class, **kwargs):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    count = 0
    attn_modules = []

    def patch(module):
        nonlocal count
        if not isinstance(module, Qwen2Attention):
            return

        count += 1
        config = module.config
        custom_attn = attention_class(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=module.head_dim,
            **kwargs,
        ).to(module.q_proj.weight.device, module.q_proj.weight.dtype)

        with torch.no_grad():
            custom_attn.q_proj.weight.copy_(module.q_proj.weight)
            custom_attn.k_proj.weight.copy_(module.k_proj.weight)
            custom_attn.v_proj.weight.copy_(module.v_proj.weight)
            custom_attn.o_proj.weight.copy_(module.o_proj.weight)

        module.subq_attn = custom_attn

        for param in module.parameters():
            param.requires_grad = False

        def new_forward(self, hidden_states, attention_mask=None, position_ids=None, **kw):
            result = self.subq_attn(hidden_states)
            if isinstance(result, tuple):
                return result
            return result, None

        module.forward = types.MethodType(new_forward, module)
        attn_modules.append(custom_attn)

    model.apply(patch)
    return count, attn_modules


def set_attention_mode(attn_modules, mode, collect_metrics=True):
    for attn in attn_modules:
        attn.set_mode(mode)
        attn.set_collect_metrics(collect_metrics)


def mean_router_loss(attn_modules):
    losses = [attn.last_router_loss for attn in attn_modules if attn.last_router_loss is not None]
    if not losses:
        return None
    return torch.stack(losses).mean()


def mean_topk_mass(attn_modules, attr_name):
    masses = [getattr(attn, attr_name) for attn in attn_modules if getattr(attn, attr_name) is not None]
    if not masses:
        return None
    return torch.stack(masses).mean().item()


def dense_finetune(model, attn_modules, loader, steps, lr):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "subq_attn.q_proj" in name or "subq_attn.k_proj" in name or "subq_attn.v_proj" in name or "subq_attn.o_proj" in name:
            param.requires_grad = True
        elif "subq_attn.router_" in name:
            param.requires_grad = False
        elif "self_attn." not in name:
            param.requires_grad = True

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    losses = []
    set_attention_mode(attn_modules, "dense", collect_metrics=False)
    model.train()

    for step, batch in enumerate(loader):
        if step >= steps:
            break

        batch = batch.to(next(model.parameters()).device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  Dense step {step}: lm_loss = {loss.item():.4f}")

    return losses


def router_distill(model, attn_modules, loader, steps, lr):
    for param in model.parameters():
        param.requires_grad = False

    router_params = []
    for attn in attn_modules:
        attn.router_q.weight.requires_grad = True
        attn.router_k.weight.requires_grad = True
        router_params.extend([attn.router_q.weight, attn.router_k.weight])

    optimizer = torch.optim.AdamW(router_params, lr=lr)

    losses = []
    coverages = []
    oracle_coverages = []

    set_attention_mode(attn_modules, "dense", collect_metrics=True)
    model.train()

    for step, batch in enumerate(loader):
        if step >= steps:
            break

        batch = batch.to(next(model.parameters()).device)
        _ = model(batch, labels=batch)
        router_loss = mean_router_loss(attn_modules)
        if router_loss is None:
            continue

        optimizer.zero_grad()
        router_loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        optimizer.step()

        losses.append(router_loss.item())
        learned_mass = mean_topk_mass(attn_modules, "last_learned_topk_mass")
        oracle_mass = mean_topk_mass(attn_modules, "last_dense_topk_mass")
        coverages.append(learned_mass)
        oracle_coverages.append(oracle_mass)

        if step % 20 == 0:
            print(
                f"  Router step {step}: distill_loss = {router_loss.item():.4f}, "
                f"learned_mass = {learned_mass:.3f}, oracle_mass = {oracle_mass:.3f}"
            )

    return losses, coverages, oracle_coverages


def sparse_joint_finetune(model, attn_modules, loader, steps, model_lr, router_lr, router_lambda):
    for param in model.parameters():
        param.requires_grad = False

    model_params = []
    router_params = []
    for name, param in model.named_parameters():
        if "subq_attn.router_" in name:
            param.requires_grad = True
            router_params.append(param)
        elif "subq_attn.q_proj" in name or "subq_attn.k_proj" in name or "subq_attn.v_proj" in name or "subq_attn.o_proj" in name:
            param.requires_grad = True
            model_params.append(param)
        elif "self_attn." not in name:
            param.requires_grad = True
            model_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": model_params, "lr": model_lr},
            {"params": router_params, "lr": router_lr},
        ]
    )

    lm_losses = []
    router_losses = []
    total_losses = []
    coverages = []
    oracle_coverages = []

    set_attention_mode(attn_modules, "sparse_learned", collect_metrics=True)
    model.train()

    for step, batch in enumerate(loader):
        if step >= steps:
            break

        batch = batch.to(next(model.parameters()).device)
        outputs = model(batch, labels=batch)
        lm_loss = outputs.loss
        router_loss = mean_router_loss(attn_modules)
        if router_loss is None:
            router_loss = lm_loss.new_zeros(())

        total_loss = lm_loss + router_lambda * router_loss

        optimizer.zero_grad()
        total_loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        lm_losses.append(lm_loss.item())
        router_losses.append(router_loss.item())
        total_losses.append(total_loss.item())
        learned_mass = mean_topk_mass(attn_modules, "last_learned_topk_mass")
        oracle_mass = mean_topk_mass(attn_modules, "last_dense_topk_mass")
        coverages.append(learned_mass)
        oracle_coverages.append(oracle_mass)

        if step % 20 == 0:
            print(
                f"  Sparse step {step}: lm_loss = {lm_loss.item():.4f}, "
                f"router_loss = {router_loss.item():.4f}, "
                f"learned_mass = {learned_mass:.3f}, oracle_mass = {oracle_mass:.3f}"
            )

    return {
        "lm_losses": lm_losses,
        "router_losses": router_losses,
        "total_losses": total_losses,
        "learned_coverages": coverages,
        "oracle_coverages": oracle_coverages,
    }


@torch.no_grad()
def evaluate_loss(model, attn_modules, loader, mode, max_batches=16, collect_metrics=True):
    set_attention_mode(attn_modules, mode, collect_metrics=collect_metrics)
    model.eval()

    losses = []
    learned_masses = []
    oracle_masses = []

    for step, batch in enumerate(loader):
        if step >= max_batches:
            break

        batch = batch.to(next(model.parameters()).device)
        outputs = model(batch, labels=batch)
        losses.append(outputs.loss.item())

        learned_mass = mean_topk_mass(attn_modules, "last_learned_topk_mass")
        oracle_mass = mean_topk_mass(attn_modules, "last_dense_topk_mass")
        if learned_mass is not None:
            learned_masses.append(learned_mass)
        if oracle_mass is not None:
            oracle_masses.append(oracle_mass)

    result = {"loss": sum(losses) / len(losses)}
    if learned_masses:
        result["learned_mass"] = sum(learned_masses) / len(learned_masses)
    if oracle_masses:
        result["oracle_mass"] = sum(oracle_masses) / len(oracle_masses)
    return result


@torch.no_grad()
def benchmark_forward(model, attn_modules, loader, mode, repeats=5):
    batch = next(iter(loader)).to(next(model.parameters()).device)
    set_attention_mode(attn_modules, mode, collect_metrics=False)
    model.eval()

    for _ in range(2):
        _ = model(batch)

    start = time.time()
    for _ in range(repeats):
        _ = model(batch)
    elapsed = time.time() - start
    return elapsed / repeats


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dense-steps", type=int, default=100)
    parser.add_argument("--router-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--router-dim", type=int, default=16)
    parser.add_argument("--dense-lr", type=float, default=2e-5)
    parser.add_argument("--router-lr", type=float, default=2e-3)
    parser.add_argument("--sparse-steps", type=int, default=40)
    parser.add_argument("--sparse-model-lr", type=float, default=5e-6)
    parser.add_argument("--sparse-router-lr", type=float, default=1e-3)
    parser.add_argument("--router-lambda", type=float, default=0.1)
    parser.add_argument("--eval-batches", type=int, default=16)
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("Hybrid SubQ: Dense Training, Router Distillation, Sparse Inference")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"top_k={args.top_k}, router_dim={args.router_dim}, max_length={args.max_length}")
    print(
        f"dense_steps={args.dense_steps}, router_steps={args.router_steps}, "
        f"sparse_steps={args.sparse_steps}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    training_texts = [
        "The theory of relativity describes how space and time are connected in a single fabric called spacetime that can be stretched and compressed by mass and energy.",
        "Einstein showed that mass and energy are equivalent through his famous equation E equals m c squared which describes how mass can be converted into energy and vice versa.",
        "Gravity in general relativity is not a force but rather the curvature of spacetime caused by the presence of mass and energy, objects follow paths called geodesics.",
        "Light travels at approximately 300,000 kilometers per second in a vacuum and is the fastest speed at which information can travel in the universe.",
        "Quantum mechanics describes the behavior of particles at the atomic and subatomic level where classical physics breaks down and probabilities govern outcomes.",
        "The uncertainty principle states that certain pairs of physical properties cannot both be precisely measured simultaneously such as position and momentum.",
        "In quantum superposition particles can exist in multiple states simultaneously until they are observed which causes the wave function to collapse to a definite state.",
        "The standard model describes fundamental particles and their interactions through three of the four fundamental forces electromagnetic weak and strong nuclear forces.",
    ]

    eval_texts = [
        "General relativity explains gravity as curvature of spacetime produced by matter and energy.",
        "Quantum theory uses probabilities to describe the behavior of particles at very small scales.",
        "Light speed in vacuum sets the maximum rate at which information can move through space.",
        "The standard model organizes particles and forces into a unified quantum framework.",
    ]

    train_dataset = SimpleDataset(training_texts * 40, tokenizer, max_length=args.max_length)
    eval_dataset = SimpleDataset(eval_texts * 8, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=1)
    eval_loader = DataLoader(eval_dataset, batch_size=1)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )

    num_replaced, attn_modules = replace_attention(
        model,
        TeacherRoutedSparseAttention,
        top_k=args.top_k,
        router_dim=args.router_dim,
    )
    print(f"Replaced {num_replaced} attention layers")

    print("\nInitial evaluation")
    dense_eval_initial = evaluate_loss(model, attn_modules, eval_loader, "dense", max_batches=args.eval_batches, collect_metrics=False)
    sparse_eval_initial = evaluate_loss(model, attn_modules, eval_loader, "sparse_learned", max_batches=args.eval_batches, collect_metrics=True)
    oracle_eval_initial = evaluate_loss(model, attn_modules, eval_loader, "sparse_oracle", max_batches=args.eval_batches, collect_metrics=True)
    print(f"  Dense loss:         {dense_eval_initial['loss']:.4f}")
    print(f"  Sparse learned:     {sparse_eval_initial['loss']:.4f} (mass={sparse_eval_initial.get('learned_mass', 0):.3f})")
    print(f"  Sparse oracle:      {oracle_eval_initial['loss']:.4f} (mass={oracle_eval_initial.get('oracle_mass', 0):.3f})")

    print("\nPhase 1: dense LM fine-tune")
    dense_losses = dense_finetune(model, attn_modules, train_loader, steps=args.dense_steps, lr=args.dense_lr)
    print(f"  Dense LM loss: {dense_losses[0]:.4f} -> {dense_losses[-1]:.4f}")

    cleanup()

    print("\nPost-dense evaluation")
    dense_eval = evaluate_loss(model, attn_modules, eval_loader, "dense", max_batches=args.eval_batches, collect_metrics=False)
    sparse_eval_before = evaluate_loss(model, attn_modules, eval_loader, "sparse_learned", max_batches=args.eval_batches, collect_metrics=True)
    oracle_eval = evaluate_loss(model, attn_modules, eval_loader, "sparse_oracle", max_batches=args.eval_batches, collect_metrics=True)
    print(f"  Dense loss:         {dense_eval['loss']:.4f}")
    print(f"  Sparse learned:     {sparse_eval_before['loss']:.4f} (mass={sparse_eval_before.get('learned_mass', 0):.3f})")
    print(f"  Sparse oracle:      {oracle_eval['loss']:.4f} (mass={oracle_eval.get('oracle_mass', 0):.3f})")

    print("\nPhase 2: router distillation")
    router_losses, learned_coverages, oracle_coverages = router_distill(
        model,
        attn_modules,
        train_loader,
        steps=args.router_steps,
        lr=args.router_lr,
    )
    print(
        f"  Router loss: {router_losses[0]:.4f} -> {router_losses[-1]:.4f}, "
        f"learned_mass: {learned_coverages[0]:.3f} -> {learned_coverages[-1]:.3f}"
    )
    print(f"  Oracle mass ceiling during distill: {oracle_coverages[-1]:.3f}")

    cleanup()

    print("\nFinal evaluation")
    dense_eval_final = evaluate_loss(model, attn_modules, eval_loader, "dense", max_batches=args.eval_batches, collect_metrics=False)
    sparse_eval_final = evaluate_loss(model, attn_modules, eval_loader, "sparse_learned", max_batches=args.eval_batches, collect_metrics=True)
    oracle_eval_final = evaluate_loss(model, attn_modules, eval_loader, "sparse_oracle", max_batches=args.eval_batches, collect_metrics=True)
    print(f"  Dense loss:         {dense_eval_final['loss']:.4f}")
    print(f"  Sparse learned:     {sparse_eval_final['loss']:.4f} (mass={sparse_eval_final.get('learned_mass', 0):.3f})")
    print(f"  Sparse oracle:      {oracle_eval_final['loss']:.4f} (mass={oracle_eval_final.get('oracle_mass', 0):.3f})")

    print("\nPhase 3: sparse-path joint fine-tune")
    sparse_stats = sparse_joint_finetune(
        model,
        attn_modules,
        train_loader,
        steps=args.sparse_steps,
        model_lr=args.sparse_model_lr,
        router_lr=args.sparse_router_lr,
        router_lambda=args.router_lambda,
    )
    print(
        f"  Sparse LM loss: {sparse_stats['lm_losses'][0]:.4f} -> {sparse_stats['lm_losses'][-1]:.4f}, "
        f"learned_mass: {sparse_stats['learned_coverages'][0]:.3f} -> {sparse_stats['learned_coverages'][-1]:.3f}"
    )
    print(f"  Router regularizer at end: {sparse_stats['router_losses'][-1]:.4f}")

    cleanup()

    print("\nPost-sparse-joint evaluation")
    dense_eval_joint = evaluate_loss(model, attn_modules, eval_loader, "dense", max_batches=args.eval_batches, collect_metrics=False)
    sparse_eval_joint = evaluate_loss(model, attn_modules, eval_loader, "sparse_learned", max_batches=args.eval_batches, collect_metrics=True)
    oracle_eval_joint = evaluate_loss(model, attn_modules, eval_loader, "sparse_oracle", max_batches=args.eval_batches, collect_metrics=True)
    print(f"  Dense loss:         {dense_eval_joint['loss']:.4f}")
    print(f"  Sparse learned:     {sparse_eval_joint['loss']:.4f} (mass={sparse_eval_joint.get('learned_mass', 0):.3f})")
    print(f"  Sparse oracle:      {oracle_eval_joint['loss']:.4f} (mass={oracle_eval_joint.get('oracle_mass', 0):.3f})")

    print("\nForward benchmark (single batch, unoptimized Python prototype)")
    dense_time = benchmark_forward(model, attn_modules, eval_loader, "dense")
    sparse_time = benchmark_forward(model, attn_modules, eval_loader, "sparse_learned")
    print(f"  Dense forward:      {dense_time * 1000:.2f} ms")
    print(f"  Sparse learned:     {sparse_time * 1000:.2f} ms")
    print(f"  Sparse/Dense ratio: {sparse_time / dense_time:.2f}x")

    print("\nSummary")
    print("  Dense training keeps LM optimization stable.")
    print("  Oracle sparse loss shows whether sparse key selection itself is viable.")
    print("  Router distillation measures whether the learned selector can approach the oracle.")


if __name__ == "__main__":
    main()
