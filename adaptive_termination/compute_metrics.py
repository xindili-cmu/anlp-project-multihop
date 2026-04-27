#!/usr/bin/env python3
"""
Step C, Script 1: compute_metrics.py

Compute classifier performance metrics on real-distribution triples.
Pure numpy implementation — no sklearn required.

Input:
  - triples_with_scores.jsonl (from Step B)

Output:
  - step_c_metrics.json: all numbers organized for paper reference
  - stdout: human-readable summary
"""

import json
import argparse
import numpy as np


DEFAULT_INPUT = "triples_with_scores.jsonl"
DEFAULT_OUTPUT = "step_c_metrics.json"


# ============================================================
# Pure-numpy AUC (ROC, trapezoidal rule)
# ============================================================
def compute_auc(scores, labels):
    """
    Compute ROC AUC via the Mann-Whitney U formula.
    AUC = P(score_positive > score_negative) (+ ties * 0.5)
    Returns None if only one class present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    # Rank-based AUC: rank all scores, then
    # AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    # We use average ranks for ties (same as scipy.stats.rankdata(method='average'))
    all_scores = np.concatenate([pos_scores, neg_scores])
    # argsort then compute ranks with ties averaged
    order = all_scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks: handle ties
    sorted_scores = all_scores[order]
    ranks_raw = np.arange(1, len(all_scores) + 1, dtype=np.float64)
    # For ties, average the raw ranks
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks_raw[i:j + 1].mean()
            ranks_raw[i:j + 1] = avg_rank
        i = j + 1
    ranks[order] = ranks_raw

    rank_sum_pos = ranks[:n_pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# ============================================================
# Classification metrics at a fixed threshold
# ============================================================
def compute_classification_metrics(scores, labels, threshold=0.5):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    preds = (scores > threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    n = len(labels)
    acc = (tp + tn) / n if n > 0 else 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "n": n,
        "positive_rate": float(labels.mean()) if n > 0 else 0.0,
        "threshold": threshold,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_ece(scores, labels, n_bins=10):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        avg_score = scores[mask].mean()
        actual_pos_rate = labels[mask].mean()
        ece += (bin_size / n) * abs(avg_score - actual_pos_rate)
    return float(ece)


def full_metric_block(scores, labels, label_desc=""):
    m = compute_classification_metrics(scores, labels, threshold=0.5)
    m["auc"] = compute_auc(scores, labels)
    m["ece"] = compute_ece(scores, labels)
    m["label_desc"] = label_desc
    return m


def print_block(name, m):
    print(f"\n--- {name} ---")
    print(f"  n={m['n']} | positive_rate={m['positive_rate']:.3f}")
    print(f"  Acc={m['accuracy']:.4f} | P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f}")
    auc_str = f"{m['auc']:.4f}" if m['auc'] is not None else "N/A"
    print(f"  AUC={auc_str} | ECE={m['ece']:.4f}")
    print(f"  Confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    triples = []
    with open(args.input) as f:
        for line in f:
            triples.append(json.loads(line))
    print(f"  {len(triples)} triples loaded")

    scores = np.array([t["sufficiency_score"] for t in triples])
    primary_labels = np.array([t["primary_label"] for t in triples])
    secondary_labels = np.array([t["secondary_label"] for t in triples])
    miss_rates = np.array([t["miss_rate"] for t in triples])
    hops = np.array([t["hop_idx"] for t in triples])

    results = {}

    # 1. Overall
    print("\n" + "=" * 70)
    print("1. OVERALL METRICS")
    print("=" * 70)
    results["overall_primary"] = full_metric_block(scores, primary_labels, "primary=SF")
    print_block("Overall (PRIMARY label = SF coverage)", results["overall_primary"])
    results["overall_secondary"] = full_metric_block(scores, secondary_labels, "secondary=EM")
    print_block("Overall (SECONDARY label = EM)", results["overall_secondary"])

    # 2. Clean vs Drift
    print("\n" + "=" * 70)
    print("2. CLEAN vs DRIFT SUBSETS (primary label)")
    print("=" * 70)
    clean_mask = miss_rates == 0.0
    drift_mask = ~clean_mask
    results["clean_primary"] = full_metric_block(scores[clean_mask], primary_labels[clean_mask], "primary=SF, clean")
    results["drift_primary"] = full_metric_block(scores[drift_mask], primary_labels[drift_mask], "primary=SF, drift")
    print_block("Clean subset (miss_rate=0)", results["clean_primary"])
    print_block("Drift subset (miss_rate>0)", results["drift_primary"])
    results["clean_secondary"] = full_metric_block(scores[clean_mask], secondary_labels[clean_mask], "secondary=EM, clean")
    results["drift_secondary"] = full_metric_block(scores[drift_mask], secondary_labels[drift_mask], "secondary=EM, drift")

    # 3. By-hop
    print("\n" + "=" * 70)
    print("3. BY-HOP BREAKDOWN (primary label)")
    print("=" * 70)
    by_hop = {}
    for h in sorted(set(hops.tolist())):
        mask = hops == h
        if mask.sum() == 0:
            continue
        m = full_metric_block(scores[mask], primary_labels[mask], f"primary=SF, hop={h}")
        by_hop[str(h)] = m
        auc_str = f"{m['auc']:.3f}" if m['auc'] is not None else "N/A"
        print(f"  Hop {h:>2}: n={m['n']:>4} | pos_rate={m['positive_rate']:.3f} | "
              f"F1={m['f1']:.4f} | AUC={auc_str} | ECE={m['ece']:.4f}")
    results["by_hop_primary"] = by_hop

    # 4. By miss-rate buckets
    print("\n" + "=" * 70)
    print("4. BY-MISS-RATE BUCKET (primary label)")
    print("=" * 70)
    buckets = [
        ("miss=0.0", miss_rates == 0.0),
        ("0<miss<=0.2", (miss_rates > 0) & (miss_rates <= 0.2)),
        ("0.2<miss<=0.5", (miss_rates > 0.2) & (miss_rates <= 0.5)),
        ("0.5<miss<1.0", (miss_rates > 0.5) & (miss_rates < 1.0)),
        ("miss=1.0", miss_rates == 1.0),
    ]
    by_miss = {}
    for bucket_name, mask in buckets:
        if mask.sum() == 0:
            continue
        m = full_metric_block(scores[mask], primary_labels[mask], f"primary=SF, {bucket_name}")
        by_miss[bucket_name] = m
        auc_str = f"{m['auc']:.3f}" if m['auc'] is not None else "N/A"
        print(f"  {bucket_name:>16}: n={m['n']:>4} | pos_rate={m['positive_rate']:.3f} | "
              f"F1={m['f1']:.4f} | AUC={auc_str}")
    results["by_miss_rate_primary"] = by_miss

    # Save
    print(f"\nSaving metrics to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("★ SUMMARY — PAPER'S KEY NUMBERS ★")
    print("=" * 70)
    print(f"\n{'Subset':<25} {'n':>6} {'F1':>8} {'AUC':>8} {'ECE':>8}")
    print("-" * 60)

    def row(name, m):
        auc_str = f"{m['auc']:.4f}" if m['auc'] is not None else "  N/A "
        print(f"{name:<25} {m['n']:>6} {m['f1']:>8.4f} {auc_str:>8} {m['ece']:>8.4f}")

    row("Overall (primary)", results["overall_primary"])
    row("Overall (secondary)", results["overall_secondary"])
    row("Clean subset", results["clean_primary"])
    row("Drift subset", results["drift_primary"])

    print("\nClean validation F1 (from training_info.json): 0.9978")
    print(f"Distribution shift: 0.9978 → {results['overall_primary']['f1']:.4f} "
          f"(Δ = {results['overall_primary']['f1'] - 0.9978:+.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()