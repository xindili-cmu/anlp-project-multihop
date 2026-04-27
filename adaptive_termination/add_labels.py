#!/usr/bin/env python3
"""
Step A, Script 4: add_labels.py

Add two proxy labels to each triple (Strategy 4):
  - primary_label (SF-based): whether E contains all gold supporting facts
  - secondary_label (EM-based): whether question's final_answer EM-matches gold

Input:
  - triples_raw.jsonl (from Script 3)
  - title_index.json (from Script 2, has gold_answer + supporting_facts)
  - predictions json (for final_answer)

Output:
  - triples_with_labels.jsonl
  - step_a_stats.json: final stats for Step A
"""

import json
import re
import string
import argparse
from collections import Counter


DEFAULT_TRIPLES = "triples_raw.jsonl"
DEFAULT_INDEX = "title_index.json"
DEFAULT_PREDS = "/Users/bozhang/Downloads/11711 - project/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json"
DEFAULT_OUTPUT = "triples_with_labels.jsonl"
DEFAULT_STATS = "step_a_stats.json"


# ============================================================
# HotpotQA-official EM normalization (from hotpot_evaluate_v1.py)
# ============================================================

def normalize_answer(s):
    """Lowercase, remove punctuation, articles, extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_match(pred, gold):
    """Exact match after HotpotQA normalization."""
    if pred is None or gold is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)


# ============================================================
# SF coverage label
# ============================================================

def normalize_title(title):
    return " ".join(title.lower().strip().split())


def compute_sf_coverage_label(
    retrieved_titles, missed_titles, supporting_facts
):
    """
    Check if all SF titles are present in retrieved_titles AND were matched.

    Returns: (label, coverage_ratio, missing_sf_titles)
      - label: 1 if all SF titles covered, else 0
      - coverage_ratio: fraction of unique SF titles covered (0.0-1.0)
      - missing_sf_titles: list of SF titles not covered
    """
    # Unique SF titles (a title can appear multiple times in SF with different sent_ids)
    sf_titles_unique = list({sf[0] for sf in supporting_facts})

    # Normalized set of retrieved titles that were successfully matched
    retrieved_set_norm = {
        normalize_title(t)
        for t in retrieved_titles
        if t not in missed_titles  # only count matched titles
    }

    covered = 0
    missing_sf_titles = []
    for sf_title in sf_titles_unique:
        if normalize_title(sf_title) in retrieved_set_norm:
            covered += 1
        else:
            missing_sf_titles.append(sf_title)

    total = len(sf_titles_unique)
    coverage_ratio = covered / total if total > 0 else 0.0
    label = 1 if covered == total else 0

    return label, coverage_ratio, missing_sf_titles


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default=DEFAULT_TRIPLES)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--preds", default=DEFAULT_PREDS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    args = parser.parse_args()

    # Load index (has SF + gold_answer)
    print(f"Reading index: {args.index}")
    with open(args.index, "r") as f:
        index = json.load(f)

    # Load predictions (has final_answer)
    print(f"Reading predictions: {args.preds}")
    with open(args.preds, "r") as f:
        preds = json.load(f)

    # Compute per-question EM once (shared across all hops of the same question)
    print("Computing per-question EM labels...")
    question_em = {}
    question_gold = {}
    question_pred = {}
    for qid, pred_answer in preds.items():
        if qid not in index:
            question_em[qid] = 0
            continue
        gold = index[qid]["gold_answer"]
        em = 1 if em_match(pred_answer, gold) else 0
        question_em[qid] = em
        question_gold[qid] = gold
        question_pred[qid] = pred_answer

    em1_count = sum(question_em.values())
    print(f"  Questions with EM=1: {em1_count} / {len(question_em)} "
          f"({100*em1_count/len(question_em):.1f}%)")

    # Process triples
    print(f"\nReading triples: {args.triples}")
    output_triples = []
    primary_label_dist = Counter()
    secondary_label_dist = Counter()
    agreement_counter = Counter()
    primary_by_hop = {}  # hop_idx -> {0: count, 1: count}
    coverage_ratio_sum = 0.0
    n_processed = 0

    with open(args.triples, "r") as f:
        for line in f:
            t = json.loads(line)
            qid = t["question_id"]

            # Primary label (SF coverage)
            if qid in index:
                sf = index[qid]["supporting_facts"]
                primary_label, coverage_ratio, missing_sf = compute_sf_coverage_label(
                    t["retrieved_titles"],
                    t["missed_titles"],
                    sf,
                )
            else:
                primary_label = 0
                coverage_ratio = 0.0
                missing_sf = []
                sf = []

            # Secondary label (EM broadcast)
            secondary_label = question_em.get(qid, 0)

            # Enrich triple
            t["primary_label"] = primary_label
            t["primary_coverage_ratio"] = coverage_ratio
            t["missing_sf_titles"] = missing_sf
            t["num_sf_total"] = len({s[0] for s in sf})  # unique SF titles
            t["secondary_label"] = secondary_label
            t["question_em"] = secondary_label
            t["gold_answer"] = question_gold.get(qid, None)
            t["pipeline_final_answer"] = question_pred.get(qid, None)

            output_triples.append(t)

            # Stats
            primary_label_dist[primary_label] += 1
            secondary_label_dist[secondary_label] += 1
            agreement_counter[(primary_label, secondary_label)] += 1
            coverage_ratio_sum += coverage_ratio

            h = t["hop_idx"]
            if h not in primary_by_hop:
                primary_by_hop[h] = {0: 0, 1: 0}
            primary_by_hop[h][primary_label] += 1

            n_processed += 1

    # Write enriched triples
    print(f"\nWriting {len(output_triples)} labeled triples to {args.output}")
    with open(args.output, "w") as f:
        for t in output_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ===============================================
    # Print stats
    # ===============================================
    print()
    print("=" * 70)
    print("=== Step A final stats ===")
    print("=" * 70)

    # Label distributions
    total = n_processed
    print(f"\nTotal triples: {total}")
    print(f"\nPrimary label (SF coverage):")
    for lbl in [1, 0]:
        c = primary_label_dist[lbl]
        print(f"  label={lbl}: {c:>5} ({100*c/total:.1f}%)")
    print(f"  mean SF coverage ratio: {coverage_ratio_sum/total:.3f}")

    print(f"\nSecondary label (EM broadcast):")
    for lbl in [1, 0]:
        c = secondary_label_dist[lbl]
        print(f"  label={lbl}: {c:>5} ({100*c/total:.1f}%)")

    # Agreement
    print(f"\nAgreement between primary & secondary:")
    agree = agreement_counter[(0, 0)] + agreement_counter[(1, 1)]
    print(f"  Agreement rate: {agree}/{total} = {100*agree/total:.1f}%")
    print(f"  Both label=1 (SF=1 & EM=1): {agreement_counter[(1, 1)]:>5}")
    print(f"  Both label=0 (SF=0 & EM=0): {agreement_counter[(0, 0)]:>5}")
    print(f"  SF=1 but EM=0:              {agreement_counter[(1, 0)]:>5}  "
          f"(SF coverage achieved, but LLM answered wrong)")
    print(f"  SF=0 but EM=1:              {agreement_counter[(0, 1)]:>5}  "
          f"(LLM answered correctly despite incomplete SF coverage — parametric knowledge?)")

    # Clean vs Drift breakdown (for Step C primary/secondary split)
    print(f"\nClean vs Drift subset (by miss_rate):")
    n_clean = sum(1 for t in output_triples if t["miss_rate"] == 0.0)
    n_drift = total - n_clean
    print(f"  Clean (miss_rate=0): {n_clean} ({100*n_clean/total:.1f}%)")
    print(f"  Drift (miss_rate>0): {n_drift} ({100*n_drift/total:.1f}%)")

    # Primary label stats within clean subset
    primary_in_clean = Counter()
    primary_in_drift = Counter()
    for t in output_triples:
        lbl = t["primary_label"]
        if t["miss_rate"] == 0.0:
            primary_in_clean[lbl] += 1
        else:
            primary_in_drift[lbl] += 1
    print(f"  Primary label in clean subset:")
    for lbl in [1, 0]:
        c = primary_in_clean[lbl]
        pct = 100 * c / n_clean if n_clean > 0 else 0
        print(f"    label={lbl}: {c} ({pct:.1f}%)")
    print(f"  Primary label in drift subset:")
    for lbl in [1, 0]:
        c = primary_in_drift[lbl]
        pct = 100 * c / n_drift if n_drift > 0 else 0
        print(f"    label={lbl}: {c} ({pct:.1f}%)")

    # By hop position
    print(f"\nPrimary label by hop:")
    for h in sorted(primary_by_hop.keys()):
        counts = primary_by_hop[h]
        total_h = counts[0] + counts[1]
        pos_pct = 100 * counts[1] / total_h if total_h > 0 else 0
        print(f"  Hop {h:>2}: {total_h:>4} triples | "
              f"label=1: {counts[1]:>3} ({pos_pct:.1f}%) | "
              f"label=0: {counts[0]:>3}")

    # Save stats to JSON
    stats_out = {
        "total_triples": total,
        "primary_label_distribution": dict(primary_label_dist),
        "secondary_label_distribution": dict(secondary_label_dist),
        "mean_sf_coverage": coverage_ratio_sum / total,
        "agreement_rate": agree / total,
        "agreement_breakdown": {
            "both_1": agreement_counter[(1, 1)],
            "both_0": agreement_counter[(0, 0)],
            "primary_1_secondary_0": agreement_counter[(1, 0)],
            "primary_0_secondary_1": agreement_counter[(0, 1)],
        },
        "clean_subset_size": n_clean,
        "drift_subset_size": n_drift,
        "primary_label_in_clean": dict(primary_in_clean),
        "primary_label_in_drift": dict(primary_in_drift),
        "primary_label_by_hop": primary_by_hop,
    }

    with open(args.stats, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nStats saved to: {args.stats}")

    print("\nStep A complete.")


if __name__ == "__main__":
    main()