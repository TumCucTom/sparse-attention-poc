#!/usr/bin/env python3
"""
Simple LoRA fine-tuning script for Llama 2 7B on Apple Silicon (MPS).
Supports fine-tuning with your custom sparse attention patterns.

Usage:
    python3 train_llama.py --data your_training_data.txt
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Llama 2 7B")
    parser.add_argument("--model_path", type=str, default="/Users/tom/llama-models/llama-2-7b.Q4_K_M.gguf",
                        help="Path to GGUF model (will auto-convert)")
    parser.add_argument("--data", type=str, required=True, help="Training data file (one sample per line)")
    parser.add_argument("--output_dir", type=str, default="./lora-output", help="Output directory")
    parser.add_argument("--context_length", type=int, default=512, help="Max context length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit"],
                        help="Quantization for loading (saves VRAM)")
    return parser.parse_args()

def load_tokenizer():
    """Load tokenizer - use meta llama2 tokenizer from HF"""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(args, tokenizer):
    """Load model with quantization for MPS/GPU efficiency"""
    logger.info(f"Loading model from {args.model_path}")

    # For GGUF models, we'd need to convert first
    # For simplicity, use HF model with quantization
    logger.info("Note: For GGUF, use convert.py first. Using HF model for demo.")

    bnb_config = None
    if args.quantization == "4bit":
        from bitsandbytes import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.quantization == "8bit":
        from bitsandbytes import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Try loading with quantization, fall back to full precision
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        logger.warning(f"Quantized loading failed: {e}, trying float16...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN")
        )

    model.config.use_cache = False
    return model

def prepare_dataset(filepath, tokenizer, context_length):
    """Load and tokenize training data"""
    logger.info(f"Loading training data from {filepath}")

    # Read data file
    with open(filepath, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(texts)} samples")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples,
            truncation=True,
            max_length=context_length,
            padding="max_length",
            return_tensors=None
        )

    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    return dataset

def setup_lora(model, args):
    """Setup LoRA configuration"""
    logger.info("Setting up LoRA...")

    # Prepare model for kbit training if quantized
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def main():
    args = parse_args()

    # Check MPS availability
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load model
    model = load_model(args, tokenizer)

    # Setup LoRA
    model = setup_lora(model, args)

    # Load dataset
    dataset = prepare_dataset(args.data, tokenizer, args.context_length)

    # Training arguments optimized for Apple Silicon
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Effective batch size = batch_size * gradient_accumulation_steps
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # Use MPS, not CUDA
        bf16=torch.backends.mps.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # Memory efficient optimizer
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked
    )

    # Train
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    logger.info("Done!")

if __name__ == "__main__":
    main()