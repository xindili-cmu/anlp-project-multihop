#!/usr/bin/env python3
"""
Step A, Script 3: extract_triples.py

For each hop in each question's trajectory, generate one (Q, E, A) triple by:
  1. Looking up each retrieved title's paragraph text (exact match first,
     then normalized fallback)
  2. Tracking title match/miss rate (Scheme gamma: keep all triples, mark misses)
  3. Concatenating all looked-up paragraphs as E
  4. Using the cumulated CoT as A

Input:
  - parsed_trajectories.jsonl
  - title_index.json

Output:
  - triples_raw.jsonl: one triple per hop per question
"""

import json
import argparse
from collections import Counter


DEFAULT_TRAJ = "parsed_trajectories.jsonl"
DEFAULT_INDEX = "title_index.json"
DEFAULT_OUTPUT = "triples_raw.jsonl"

# Placeholder for missed titles (keep concise so it doesn't dominate E's token budget)
MISSING_PLACEHOLDER = (
    "[Paragraph text not available in HotpotQA distractor set; "
    "this title was retrieved from external Wikipedia by IRCoT.]"
)


def normalize_title(title):
    """Same normalization used in build_title_index.py."""
    return " ".join(title.lower().strip().split())


def lookup_paragraph(title, index_entry):
    """
    Look up a title's paragraph in the per-question index.

    Returns: (paragraph_text, matched_flag)
      - paragraph_text: actual text if matched, else MISSING_PLACEHOLDER
      - matched_flag: True if title was found (exact or normalized)
    """
    paragraphs = index_entry["paragraphs"]
    normalized_titles = index_entry["normalized_titles"]

    # 1. Exact match
    if title in paragraphs:
        return paragraphs[title], True

    # 2. Normalized fallback
    norm = normalize_title(title)
    if norm in normalized_titles:
        original = normalized_titles[norm]
        return paragraphs[original], True

    # 3. Miss
    return MISSING_PLACEHOLDER, False


def build_evidence(retrieved_titles, index_entry):
    """
    Construct the E string from a list of retrieved titles.

    Returns: (evidence_string, num_matched, num_missed, missed_titles_list)
    """
    parts = []
    num_matched = 0
    num_missed = 0
    missed_titles = []

    for title in retrieved_titles:
        para_text, matched = lookup_paragraph(title, index_entry)
        parts.append(f"Wikipedia Title: {title}\n{para_text}")
        if matched:
            num_matched += 1
        else:
            num_missed += 1
            missed_titles.append(title)

    evidence = "\n\n".join(parts)
    return evidence, num_matched, num_missed, missed_titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default=DEFAULT_TRAJ)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Load trajectories
    print(f"Reading trajectories from: {args.trajectories}")
    trajectories = []
    with open(args.trajectories, "r") as f:
        for line in f:
            trajectories.append(json.loads(line))
    print(f"  {len(trajectories)} questions")

    # Load index
    print(f"Reading title index from: {args.index}")
    with open(args.index, "r") as f:
        index = json.load(f)
    print(f"  {len(index)} questions indexed")

    # Process
    print("Extracting triples...")
    all_triples = []
    questions_missing_index = 0
    total_titles_seen = 0
    total_titles_missed = 0
    miss_rate_buckets = Counter()  # e.g. 0.0, 0.1-0.2, 0.2-0.4, etc.
    hops_with_any_miss = 0

    for traj in trajectories:
        qid = traj["question_id"]
        question = traj["question"]
        total_hops = traj["num_hops"]

        if qid not in index:
            questions_missing_index += 1
            continue

        idx_entry = index[qid]

        for hop in traj["hops"]:
            hop_idx = hop["hop_idx"]
            titles = hop["retrieved_titles"]
            cot = hop["cot"]

            evidence, num_matched, num_missed, missed_titles = build_evidence(titles, idx_entry)
            num_titles = len(titles)
            miss_rate = num_missed / num_titles if num_titles > 0 else 0.0

            total_titles_seen += num_titles
            total_titles_missed += num_missed
            if num_missed > 0:
                hops_with_any_miss += 1

            # Bucket miss_rate for stats
            if miss_rate == 0.0:
                miss_rate_buckets["0.0"] += 1
            elif miss_rate < 0.2:
                miss_rate_buckets["0.0-0.2"] += 1
            elif miss_rate < 0.5:
                miss_rate_buckets["0.2-0.5"] += 1
            elif miss_rate < 1.0:
                miss_rate_buckets["0.5-1.0"] += 1
            else:
                miss_rate_buckets["1.0"] += 1

            triple = {
                "question_id": qid,
                "question": question,
                "hop_idx": hop_idx,
                "total_hops": total_hops,
                "is_final_hop": (hop_idx == total_hops),
                "retrieved_titles": titles,
                "evidence": evidence,
                "candidate_answer": cot,
                "num_titles": num_titles,
                "num_titles_matched": num_matched,
                "num_titles_missed": num_missed,
                "miss_rate": miss_rate,
                "missed_titles": missed_titles,
            }
            all_triples.append(triple)

    # Write output
    print(f"\nWriting {len(all_triples)} triples to: {args.output}")
    with open(args.output, "w") as f:
        for t in all_triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Stats
    print()
    print("=" * 60)
    print("=== Extract triples stats ===")
    print("=" * 60)
    print(f"  Total triples:                  {len(all_triples)}")
    print(f"  Questions missing from index:   {questions_missing_index}")
    print(f"  Total retrieved titles seen:    {total_titles_seen}")
    print(f"  Total titles missed (lookup):   {total_titles_missed}")
    print(f"  Overall title miss rate:        {total_titles_missed / total_titles_seen:.3f}"
          if total_titles_seen > 0 else "  No titles seen!")
    print(f"  Hops with any miss:             {hops_with_any_miss} / {len(all_triples)} "
          f"({100*hops_with_any_miss/len(all_triples):.1f}%)")
    print()
    print("  Miss rate distribution across triples:")
    for bucket in ["0.0", "0.0-0.2", "0.2-0.5", "0.5-1.0", "1.0"]:
        count = miss_rate_buckets.get(bucket, 0)
        pct = 100 * count / len(all_triples) if all_triples else 0
        print(f"    {bucket:>10}: {count:>5} ({pct:.1f}%)")

    # Hop-level stats
    print()
    print("  Miss rate by hop position (may reveal drift):")
    hop_miss = {}  # hop_idx -> (total, missed)
    for t in all_triples:
        h = t["hop_idx"]
        if h not in hop_miss:
            hop_miss[h] = {"triples": 0, "miss_sum": 0.0, "titles_seen": 0, "titles_missed": 0}
        hop_miss[h]["triples"] += 1
        hop_miss[h]["miss_sum"] += t["miss_rate"]
        hop_miss[h]["titles_seen"] += t["num_titles"]
        hop_miss[h]["titles_missed"] += t["num_titles_missed"]

    for h in sorted(hop_miss.keys()):
        info = hop_miss[h]
        avg_miss = info["miss_sum"] / info["triples"]
        overall_miss = info["titles_missed"] / info["titles_seen"] if info["titles_seen"] > 0 else 0
        print(f"    Hop {h:>2}: {info['triples']:>4} triples | "
              f"avg miss rate {avg_miss:.3f} | "
              f"title-level miss rate {overall_miss:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()