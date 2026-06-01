#!/usr/bin/env python3
"""
Training Script for Trainable Sparse Attention

Supports:
1. Local training on MPS/CPU
2. HPC training with SLURM (single/multi-node)
3. Checkpointing and resuming
4. Gradient checkpointing for long sequences

Usage:
    # Local
    python3 train_sparse.py --model Qwen/Qwen2.5-0.5B --steps 100 --seq-len 256

    # HPC
    sbatch --gres=gpu:1 train_sparse.sh Qwen/Qwen2.5-1.5B 1000
"""

import argparse
import os
import time
import gc
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
from trainable_sparse_attention import TrainableSparseAttention, StandardAttention, replace_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse Attention Models")
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model name")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per GPU")

    # Training
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")

    # Sparse attention params
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size for sparse attention")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Top-k blocks to select")
    parser.add_argument("--index-dim", type=int, default=32,
                        help="Index projection dimension")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Gumbel-Softmax temperature")

    # HPC / Distributed
    parser.add_argument("--distributed", action="store_true",
                        help="Use distributed training")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank for distributed training")
    parser.add_argument("--world-size", type=int, default=1,
                        help="World size for distributed training")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    return parser.parse_args()


class TextDataset(Dataset):
    """Simple text dataset for training."""

    def __init__(self, texts, tokenizer, max_length=256):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def get_training_texts():
    """Get training texts - simple synthetic data for now."""
    texts = [
        "The theory of quantum mechanics describes how particles behave at the atomic level.",
        "Machine learning algorithms can find patterns in large datasets.",
        "The human brain contains billions of neurons that communicate through electrical signals.",
        "Climate change is affecting weather patterns around the world.",
        "The history of artificial intelligence began in the 1950s.",
        "DNA contains the genetic instructions for the development of living organisms.",
        "The universe is approximately 13.8 billion years old.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The stock market reflects the collective expectations of investors.",
        "Neural networks are inspired by the structure of the human brain.",
        "Encryption protects sensitive information transmitted over the internet.",
        "The immune system defends the body against infections and diseases.",
        "Space exploration has led to many technological advances.",
        "The law of supply and demand determines prices in a market economy.",
        "Climate models predict future temperature changes based on emissions scenarios.",
        "The human genome project mapped all human genes.",
        "Renewable energy sources are becoming increasingly important.",
        "The deep ocean remains largely unexplored by humans.",
        "Artificial intelligence is transforming many industries.",
        "The internet has revolutionized communication and commerce.",
    ]
    # Repeat to get enough samples
    return texts * 100


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_distributed():
    """Setup for distributed training."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train_step(model, batch, optimizer, device, grad_clip=1.0):
    """Single training step."""
    model.train()

    # Move batch to device
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
    else:
        input_ids = batch.to(device)

    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    # Check for NaN
    if torch.isnan(loss):
        return None

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return loss.item()


def evaluate(model, dataloader, device, num_batches=10):
    """Evaluate model loss."""
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch.to(device)

            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item()
            num_samples += 1

    return total_loss / max(num_samples, 1)


def learning_rate_schedule(step, warmup_steps, total_steps, lr):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()

    # Setup distributed if needed
    if args.distributed:
        setup_distributed()
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = get_device()

    print("\n" + "="*70)
    print("Training Sparse Attention Model")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps: {args.steps}")
    print(f"Learning rate: {args.lr}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model...")
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if not args.distributed else None,
        trust_remote_code=True,
    )

    if args.distributed:
        model = model.to(device)
        model = DDP(model)

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Replace attention with trainable sparse attention
    print(f"\nReplacing attention with TrainableSparseAttention...")
    num_replaced, _ = replace_attention(model, TrainableSparseAttention, **{
        "block_size": args.block_size,
        "top_k_blocks": args.top_k,
        "index_dim": args.index_dim,
        "temperature": args.temperature,
    })
    print(f"Replaced {num_replaced} attention layers")

    # Create dataset
    print(f"\nPreparing dataset...")
    texts = get_training_texts()
    dataset = TextDataset(texts, tokenizer, max_length=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]

    # Training loop
    print(f"\nStarting training...")
    losses = []
    best_loss = float('inf')

    for step in range(start_step, args.steps):
        # Get batch
        batch = next(iter(dataloader))

        # Update learning rate
        lr = learning_rate_schedule(step, args.warmup_steps, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training step
        loss = train_step(model, batch, optimizer, device, args.grad_clip)

        if loss is None:
            print(f"  Step {step}: NaN loss, skipping")
            continue

        losses.append(loss)

        if step % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(len(losses), 10)
            print(f"  Step {step}: loss={loss:.4f}, avg={avg_loss:.4f}, lr={lr:.2e}")

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{step+1}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict() if not args.distributed else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

            if loss < best_loss:
                best_loss = loss

    # Final evaluation
    print(f"\nEvaluating...")
    eval_loss = evaluate(model, dataloader, device, num_batches=20)
    print(f"Final evaluation loss: {eval_loss:.4f}")

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict() if not args.distributed else model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_loss": eval_loss,
    }, final_path)
    print(f"Saved final model: {final_path}")

    # Cleanup
    if args.distributed:
        destroy_process_group()

    # Save results
    results = {
        "model": args.model,
        "steps": args.steps,
        "final_loss": eval_loss,
        "best_loss": best_loss,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "losses": losses[-100:],  # Last 100 losses
        "config": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "block_size": args.block_size,
            "top_k": args.top_k,
            "index_dim": args.index_dim,
            "temperature": args.temperature,
        }
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()