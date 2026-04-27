#!/usr/bin/env python3
"""
Step 1: Construct training data for the DeBERTa evidence sufficiency classifier.

This script reads HotpotQA raw data (which contains gold supporting fact annotations)
and constructs training samples of the form (question, evidence, candidate_answer, label).

The core idea:
- For each HotpotQA question, we know which paragraphs are gold supporting facts.
- We simulate partial evidence states (hop 1, hop 2, ...) by progressively adding
  gold paragraphs.
- At each state, we determine whether the evidence is "sufficient" to answer the question.
- Label = 1 if evidence contains ALL gold supporting paragraphs (sufficient).
- Label = 0 if evidence is missing any gold supporting paragraph (insufficient).

The candidate answer field:
- For label=1 (sufficient evidence): we use the gold answer.
- For label=0 (insufficient evidence): we also use the gold answer.
  This teaches the classifier that even when the answer TEXT is correct,
  if the evidence doesn't fully support it, confidence should be low.

Additionally, we generate "distractor-only" negatives where the evidence
consists entirely of non-supporting paragraphs.

Usage:
    python construct_training_data.py \
        --input_file /path/to/downloads/hotpotqa/dev.json \
        --output_file /path/to/training_data.jsonl \
        --split_ratio 0.85

    # For using HotpotQA train split (recommended for final training):
    python construct_training_data.py \
        --input_file /path/to/downloads/hotpotqa/train.json \
        --output_file /path/to/training_data.jsonl
"""

import json
import argparse
import random
import os
from collections import defaultdict


def extract_paragraph_text(context, title):
    """Extract the full text of a paragraph given its title from HotpotQA context."""
    for para_title, sentences in context:
        if para_title == title:
            return " ".join(sentences)
    return None


def extract_supporting_paragraphs(context, supporting_facts):
    """
    Get the unique supporting paragraph titles and their texts.
    
    Args:
        context: list of [title, [sentences...]] from HotpotQA
        supporting_facts: list of [title, sent_id] from HotpotQA
    
    Returns:
        List of (title, full_text) for each unique supporting paragraph,
        in the order they first appear in supporting_facts.
    """
    seen_titles = set()
    supporting_paras = []
    
    for title, _ in supporting_facts:
        if title not in seen_titles:
            text = extract_paragraph_text(context, title)
            if text:
                supporting_paras.append((title, text))
                seen_titles.add(title)
    
    return supporting_paras


def extract_distractor_paragraphs(context, supporting_facts):
    """Get paragraphs that are NOT supporting facts (distractors)."""
    supporting_titles = set(title for title, _ in supporting_facts)
    distractors = []
    
    for title, sentences in context:
        if title not in supporting_titles:
            distractors.append((title, " ".join(sentences)))
    
    return distractors


def construct_samples_for_question(item):
    """
    Construct training samples for a single HotpotQA question.
    
    For a typical 2-hop question with supporting paragraphs P1 and P2:
    - Sample 1: evidence=P1 only, answer=gold → label=0 (insufficient)
    - Sample 2: evidence=P1+P2, answer=gold → label=1 (sufficient)
    - Sample 3: evidence=distractors, answer=gold → label=0 (insufficient)
    
    Returns list of dicts with keys: question, evidence, candidate_answer, label, question_id, metadata
    """
    samples = []
    
    question = item["question"]
    gold_answer = item["answer"]
    question_id = item["_id"]
    context = item["context"]
    supporting_facts = item["supporting_facts"]
    question_type = item.get("type", "unknown")
    
    # Get supporting and distractor paragraphs
    supporting_paras = extract_supporting_paragraphs(context, supporting_facts)
    distractor_paras = extract_distractor_paragraphs(context, supporting_facts)
    
    if len(supporting_paras) == 0:
        return samples  # skip if no supporting paragraphs found
    
    # --- Positive sample: ALL supporting paragraphs → label=1 ---
    full_evidence = "\n\n".join(
        f"Title: {title}\n{text}" for title, text in supporting_paras
    )
    samples.append({
        "question": question,
        "evidence": full_evidence,
        "candidate_answer": gold_answer,
        "label": 1,
        "question_id": question_id,
        "metadata": {
            "type": question_type,
            "evidence_type": "full_supporting",
            "num_supporting_paras": len(supporting_paras),
            "hop": len(supporting_paras),
        }
    })
    
    # --- Negative samples: partial supporting paragraphs → label=0 ---
    if len(supporting_paras) >= 2:
        # For each proper subset of supporting paragraphs
        for i in range(1, len(supporting_paras)):
            partial_paras = supporting_paras[:i]
            partial_evidence = "\n\n".join(
                f"Title: {title}\n{text}" for title, text in partial_paras
            )
            samples.append({
                "question": question,
                "evidence": partial_evidence,
                "candidate_answer": gold_answer,
                "label": 0,
                "question_id": question_id,
                "metadata": {
                    "type": question_type,
                    "evidence_type": "partial_supporting",
                    "num_supporting_paras": i,
                    "hop": i,
                }
            })
    
    # --- Negative sample: distractor-only evidence → label=0 ---
    if distractor_paras:
        # Pick 1-2 random distractors
        num_distractors = min(2, len(distractor_paras))
        selected_distractors = random.sample(distractor_paras, num_distractors)
        distractor_evidence = "\n\n".join(
            f"Title: {title}\n{text}" for title, text in selected_distractors
        )
        samples.append({
            "question": question,
            "evidence": distractor_evidence,
            "candidate_answer": gold_answer,
            "label": 0,
            "question_id": question_id,
            "metadata": {
                "type": question_type,
                "evidence_type": "distractor_only",
                "num_supporting_paras": 0,
                "hop": 0,
            }
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Construct training data for DeBERTa sufficiency classifier"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to HotpotQA raw JSON file (e.g., downloads/hotpotqa/train.json or dev.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="training_data.jsonl",
        help="Output JSONL file for training data"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.85,
        help="Train/val split ratio (default: 0.85 train, 0.15 val)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Max number of questions to process (for debugging)"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load HotpotQA data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    if args.max_questions:
        data = data[:args.max_questions]
    
    print(f"Loaded {len(data)} questions")
    
    # Construct training samples
    all_samples = []
    label_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    for item in data:
        samples = construct_samples_for_question(item)
        for s in samples:
            label_counts[s["label"]] += 1
            type_counts[s["metadata"]["evidence_type"]] += 1
        all_samples.extend(samples)
    
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  Label 1 (sufficient): {label_counts[1]}")
    print(f"  Label 0 (insufficient): {label_counts[0]}")
    print(f"  By evidence type:")
    for etype, count in sorted(type_counts.items()):
        print(f"    {etype}: {count}")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split into train and val
    split_idx = int(len(all_samples) * args.split_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Write output
    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(args.output_file)[0]
    train_file = f"{base_name}_train.jsonl"
    val_file = f"{base_name}_val.jsonl"
    
    with open(train_file, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(val_file, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\nTrain samples: {len(train_samples)} → {train_file}")
    print(f"Val samples: {len(val_samples)} → {val_file}")
    
    # Also write a combined file (some training scripts prefer this)
    with open(args.output_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Combined: {len(all_samples)} → {args.output_file}")


if __name__ == "__main__":
    main()
