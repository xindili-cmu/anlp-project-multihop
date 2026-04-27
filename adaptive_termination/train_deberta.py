#!/usr/bin/env python3
"""
Step 2: Fine-tune DeBERTa-base as a binary classifier for evidence sufficiency.

Input format (from construct_training_data.py):
    Each line in the JSONL file has:
    {
        "question": "...",
        "evidence": "...",
        "candidate_answer": "...",
        "label": 0 or 1
    }

The model learns: given (question, evidence, candidate_answer), predict whether
the evidence is sufficient to support the answer.

Input to DeBERTa: [CLS] question [SEP] evidence [SEP] candidate_answer [SEP]
Output: probability ∈ [0, 1] that evidence is sufficient.

Usage:
    python train_deberta.py \
        --train_file training_data_train.jsonl \
        --val_file training_data_val.jsonl \
        --output_dir deberta_sufficiency_classifier \
        --epochs 5 \
        --batch_size 8 \
        --learning_rate 2e-5 \
        --max_length 512

Requirements:
    pip install torch transformers --break-system-packages
    (or within venv: pip install torch transformers)
"""

import json
import argparse
import os
import random
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DebertaTokenizer,
    DebertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class SufficiencyDataset(Dataset):
    """Dataset for evidence sufficiency classification."""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)
        
        print(f"Loaded {len(self.samples)} samples from {file_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Construct input text:
        # Format: question [SEP] evidence [SEP] candidate_answer
        # The tokenizer will add [CLS] at the start automatically.
        #
        # We use encode_plus with text and text_pair for proper segment handling.
        # Since we have 3 parts, we concatenate evidence and answer with a separator.
        text_a = item["question"]
        text_b = item["evidence"] + " [ANSWER] " + item["candidate_answer"]
        
        encoding = self.tokenizer.encode_plus(
            text_a,
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,  # truncate evidence if too long
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # For per-label stats
    tp = 0  # true positives (predicted 1, actual 1)
    fp = 0  # false positives (predicted 1, actual 0)
    fn = 0  # false negatives (predicted 0, actual 1)
    tn = 0  # true negatives (predicted 0, actual 0)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            
            total_loss += outputs.loss.item() * len(labels)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            # Per-label stats
            for pred, label in zip(preds, labels):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1
                else:
                    tn += 1
    
    avg_loss = total_loss / total
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for evidence sufficiency")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="deberta_sufficiency_classifier")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = DebertaTokenizer.from_pretrained(args.model_name)
    model = DebertaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.to(device)
    
    # Load data
    print("Loading datasets...")
    train_dataset = SufficiencyDataset(args.train_file, tokenizer, args.max_length)
    val_dataset = SufficiencyDataset(args.val_file, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # PSC compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # larger batch for eval (no gradients)
        shuffle=False,
        num_workers=0,
    )
    
    # Optimizer and scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n{'='*60}")
    print(f"Training config:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_f1 = 0.0
    best_epoch = -1
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += outputs.loss.item() * len(labels)
            preds = torch.argmax(outputs.logits, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += len(labels)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % args.log_every == 0:
                running_loss = epoch_loss / epoch_total
                running_acc = epoch_correct / epoch_total
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss: {running_loss:.4f} | "
                    f"Acc: {running_acc:.4f}"
                )
        
        # End of epoch
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(
            f"  Val   - Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"P: {val_metrics['precision']:.4f} | "
            f"R: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )
        print(
            f"  Val confusion: TP={val_metrics['tp']} FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']} TN={val_metrics['tn']}"
        )
        
        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            
            save_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save training info
            with open(os.path.join(save_path, "training_info.json"), "w") as f:
                json.dump({
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "args": vars(args),
                }, f, indent=2)
            
            print(f"  ★ New best model saved! F1={best_val_f1:.4f}")
        
        print(f"{'='*60}\n")
    
    # Save final model too
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"Training complete!")
    print(f"Best model: epoch {best_epoch}, val F1={best_val_f1:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
