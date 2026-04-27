#!/usr/bin/env python3
"""
Step C, Script 4: failure_mode_breakdown.py

Classify each question by Milestone's three failure modes
(Mode 1: Answer Format Collapse, Mode 2: Retrieval Cascade,
 Mode 3: Reasoning Chain Stagnation), then compute classifier
metrics within each mode.

Note: modes are NOT mutually exclusive.
"""

import json
import argparse
import numpy as np
from collections import defaultdict


DEFAULT_INPUT = "triples_with_scores.jsonl"
DEFAULT_OUTPUT = "step_c_failure_modes.json"


# --- Mode detection heuristics (aligned with Milestone Section 2) ---

def is_mode1_answer_format_collapse(final_answer):
    """Mode 1: Answer Format Collapse.
    Milestone: 'a prediction exceeding 100 characters almost always represents a failure'.
    Also check for hedged-phrase markers.
    """
    if final_answer is None:
        return False
    if len(final_answer) > 100:
        return True
    hedged_markers = [
        "however", "i could not find", "following the format",
        "a series of steps", "in the interest of efficiency",
        "this is then a", "it has been decided",
    ]
    fa_lower = final_answer.lower()
    # Only trigger for multi-marker presence to avoid false positives
    marker_count = sum(1 for m in hedged_markers if m in fa_lower)
    return marker_count >= 2


def is_mode2_retrieval_cascade(num_hops, final_em):
    """Mode 2: Retrieval Cascade.
    Milestone: 'exhausted the maximum allowed retrieval steps without the
    exit controller ever triggering a stop. In all 75 cases, the final
    score was 0.'
    """
    return num_hops >= 10 and final_em == 0


def is_mode3_reasoning_stagnation(cots):
    """Mode 3: Reasoning Chain Stagnation.
    Milestone: 'reasoning chain contains repeated phrases such as
    "This is not a fact verified in any of the mentioned Wikipedia Titles"
    or "There is no information in the given Wikipedia articles."'

    Take the union of CoTs from all hops (since CoT is cumulative),
    check for uncertainty markers.
    """
    if not cots:
        return False
    # Use last CoT (most cumulative)
    last_cot = cots[-1].lower()
    uncertainty_markers = [
        "not a fact verified",
        "no information in the given",
        "cannot be concluded",
        "there is no information",
        "not verified in any of",
    ]
    marker_hits = sum(1 for m in uncertainty_markers if m in last_cot)
    # Also check if any of these markers appear repeatedly
    return marker_hits >= 1


# --- Metrics computation (reuse from compute_metrics.py logic) ---

def compute_f1_at_05(scores, labels):
    preds = (scores > 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n = len(labels)
    acc = (tp + tn) / n if n > 0 else 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"n": n, "acc": acc, "p": p, "r": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "positive_rate": float(labels.mean()) if n > 0 else 0.0}


def compute_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    all_scores = np.concatenate([pos_scores, neg_scores])
    order = all_scores.argsort()
    ranks_raw = np.arange(1, len(all_scores) + 1, dtype=np.float64)
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks_raw[i:j + 1] = ranks_raw[i:j + 1].mean()
        i = j + 1
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = ranks_raw
    rank_sum_pos = ranks[:n_pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_ece(scores, labels, n_bins=10):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    if n == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(scores[mask].mean() - labels[mask].mean())
    return float(ece)


def full_metrics(scores, labels):
    m = compute_f1_at_05(scores, labels)
    m["auc"] = compute_auc(scores, labels)
    m["ece"] = compute_ece(scores, labels)
    return m


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
    print(f"  {len(triples)} triples")

    # Group triples by question_id
    by_qid = defaultdict(list)
    for t in triples:
        by_qid[t["question_id"]].append(t)
    print(f"  {len(by_qid)} unique questions")

    # Classify each question by failure mode
    print("\nClassifying questions into failure modes...")
    question_modes = {}  # qid -> {"mode1": bool, "mode2": bool, "mode3": bool, "success": bool}
    for qid, qtriples in by_qid.items():
        qtriples = sorted(qtriples, key=lambda t: t["hop_idx"])
        final_answer = qtriples[0]["pipeline_final_answer"]
        num_hops = qtriples[0]["total_hops"]
        final_em = qtriples[0]["secondary_label"]
        cots = [t["candidate_answer"] for t in qtriples]

        m1 = is_mode1_answer_format_collapse(final_answer)
        m2 = is_mode2_retrieval_cascade(num_hops, final_em)
        m3 = is_mode3_reasoning_stagnation(cots)
        success = (final_em == 1)

        question_modes[qid] = {
            "mode1": m1, "mode2": m2, "mode3": m3,
            "success": success, "any_mode": m1 or m2 or m3,
        }

    # Print question-level mode distribution
    n_qids = len(by_qid)
    n_m1 = sum(1 for v in question_modes.values() if v["mode1"])
    n_m2 = sum(1 for v in question_modes.values() if v["mode2"])
    n_m3 = sum(1 for v in question_modes.values() if v["mode3"])
    n_any = sum(1 for v in question_modes.values() if v["any_mode"])
    n_success = sum(1 for v in question_modes.values() if v["success"])

    print("\n" + "=" * 70)
    print("QUESTION-LEVEL MODE DISTRIBUTION")
    print("=" * 70)
    print(f"  Total questions:                    {n_qids}")
    print(f"  Mode 1 (Answer Format Collapse):    {n_m1} ({100*n_m1/n_qids:.1f}%)")
    print(f"  Mode 2 (Retrieval Cascade):         {n_m2} ({100*n_m2/n_qids:.1f}%)")
    print(f"  Mode 3 (Reasoning Stagnation):      {n_m3} ({100*n_m3/n_qids:.1f}%)")
    print(f"  Any mode:                           {n_any} ({100*n_any/n_qids:.1f}%)")
    print(f"  Successful (EM=1):                  {n_success} ({100*n_success/n_qids:.1f}%)")

    # Overlap matrix
    print("\n  Mode overlap (question count):")
    for m_combo in ["M1 only", "M2 only", "M3 only", "M1+M2", "M1+M3", "M2+M3", "M1+M2+M3"]:
        count = 0
        for v in question_modes.values():
            sig = ("M1" if v["mode1"] else "") + ("M2" if v["mode2"] else "") + ("M3" if v["mode3"] else "")
            if m_combo == "M1 only" and sig == "M1": count += 1
            elif m_combo == "M2 only" and sig == "M2": count += 1
            elif m_combo == "M3 only" and sig == "M3": count += 1
            elif m_combo == "M1+M2" and sig == "M1M2": count += 1
            elif m_combo == "M1+M3" and sig == "M1M3": count += 1
            elif m_combo == "M2+M3" and sig == "M2M3": count += 1
            elif m_combo == "M1+M2+M3" and sig == "M1M2M3": count += 1
        print(f"    {m_combo:<12}: {count}")

    # Assign each triple a mode membership (from its question)
    # Then compute metrics per mode group
    scores = np.array([t["sufficiency_score"] for t in triples])
    primary_labels = np.array([t["primary_label"] for t in triples])
    secondary_labels = np.array([t["secondary_label"] for t in triples])

    # Build masks
    qids = [t["question_id"] for t in triples]
    m1_mask = np.array([question_modes[q]["mode1"] for q in qids])
    m2_mask = np.array([question_modes[q]["mode2"] for q in qids])
    m3_mask = np.array([question_modes[q]["mode3"] for q in qids])
    success_mask = np.array([question_modes[q]["success"] for q in qids])
    any_mode_mask = np.array([question_modes[q]["any_mode"] for q in qids])
    no_mode_mask = ~any_mode_mask & ~success_mask  # failed but no specific mode tagged

    # Compute classifier metrics per group (primary label)
    groups = [
        ("Mode 1 (Answer Format Collapse)", m1_mask),
        ("Mode 2 (Retrieval Cascade)", m2_mask),
        ("Mode 3 (Reasoning Stagnation)", m3_mask),
        ("Successful (EM=1)", success_mask),
        ("Failed, no mode matched", no_mode_mask),
        ("Any mode", any_mode_mask),
    ]

    print("\n" + "=" * 70)
    print("CLASSIFIER PERFORMANCE BY MODE (primary=SF label)")
    print("=" * 70)
    print(f"{'Group':<38} {'n':>6} {'pos_rate':>9} {'F1':>7} {'AUC':>7} {'ECE':>7}")
    print("-" * 78)
    group_results = {}
    for name, mask in groups:
        if mask.sum() == 0:
            print(f"{name:<38}  (empty)")
            continue
        m = full_metrics(scores[mask], primary_labels[mask])
        m_sec = full_metrics(scores[mask], secondary_labels[mask])
        auc_str = f"{m['auc']:.3f}" if m['auc'] is not None else "  N/A"
        print(f"{name:<38} {m['n']:>6} {m['positive_rate']:>9.3f} "
              f"{m['f1']:>7.3f} {auc_str:>7} {m['ece']:>7.3f}")
        group_results[name] = {"primary": m, "secondary": m_sec}

    # Additional view: SF vs EM disagreement within each mode
    print("\n" + "=" * 70)
    print("SF vs EM AGREEMENT BY MODE")
    print("=" * 70)
    print(f"{'Group':<38} {'n':>6} {'SF=1,EM=0':>10} {'SF=0,EM=1':>10} {'agree%':>8}")
    print("-" * 76)
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        p = primary_labels[mask]
        s = secondary_labels[mask]
        sf1_em0 = int(((p == 1) & (s == 0)).sum())
        sf0_em1 = int(((p == 0) & (s == 1)).sum())
        agree = int((p == s).sum())
        total = int(mask.sum())
        print(f"{name:<38} {total:>6} {sf1_em0:>10} {sf0_em1:>10} {100*agree/total:>7.1f}%")

    # Save
    out = {
        "question_mode_distribution": {
            "total": n_qids,
            "mode1": n_m1,
            "mode2": n_m2,
            "mode3": n_m3,
            "any_mode": n_any,
            "success": n_success,
        },
        "per_group_metrics": {
            name: {
                "n": int(mask.sum()),
                "primary": group_results.get(name, {}).get("primary"),
                "secondary": group_results.get(name, {}).get("secondary"),
            }
            for name, mask in groups
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")
    print("\nDone.")


if __name__ == "__main__":
    main()