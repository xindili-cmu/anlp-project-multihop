#!/usr/bin/env python3
"""
Step A, Script 2: build_title_index.py

Build a per-question title->paragraph lookup index from HotpotQA dev set.

Input:
  - HotpotQA dev.json (from /Users/bozhang/Downloads/ircot/downloads/hotpotqa/dev.json)
  - parsed_trajectories.jsonl (from Script 1) — to know which 500 qids we need

Output:
  - title_index.json: {qid: {title: paragraph_text, ...}, ...}
    Also includes supporting_facts per qid for later use by add_labels.py.

Structure of output:
{
  "<qid>": {
    "question": "...",
    "gold_answer": "...",
    "supporting_facts": [["title", sent_id], ...],
    "paragraphs": {
        "<title_original>": "<paragraph text>",
        ...  (10 paragraphs per question)
    },
    "normalized_titles": {
        "<normalized_title>": "<original_title>",
        ...
    }
  },
  ...
}
"""

import json
import argparse
from pathlib import Path


DEFAULT_DEV = "/Users/bozhang/Downloads/ircot/downloads/hotpotqa/dev.json"
DEFAULT_TRAJ = "parsed_trajectories.jsonl"
DEFAULT_OUTPUT = "title_index.json"


def normalize_title(title):
    """Normalize a title for fuzzy matching: lowercase, strip, collapse spaces."""
    return " ".join(title.lower().strip().split())


def build_paragraph_text(sentences):
    """Concatenate a list of sentences into one paragraph text."""
    # HotpotQA sentences often have leading spaces/punctuation already;
    # direct concatenation preserves their formatting.
    return "".join(sentences).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default=DEFAULT_DEV)
    parser.add_argument("--trajectories", default=DEFAULT_TRAJ)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Load the qids we care about (from parse_chains.py output)
    print(f"Reading parsed trajectories from: {args.trajectories}")
    needed_qids = set()
    with open(args.trajectories, "r") as f:
        for line in f:
            rec = json.loads(line)
            needed_qids.add(rec["question_id"])
    print(f"  {len(needed_qids)} qids to index")

    # Load HotpotQA dev
    print(f"Reading HotpotQA dev from: {args.dev}")
    with open(args.dev, "r") as f:
        dev_data = json.load(f)
    print(f"  {len(dev_data)} total dev questions")

    # Build index for needed qids
    print("Building per-question paragraph index...")
    index = {}
    not_found = []

    for q in dev_data:
        qid = q["id"]
        if qid not in needed_qids:
            continue

        # Parse context (column-oriented format)
        ctx_titles = q["context"]["title"]       # list of 10 titles
        ctx_sentences = q["context"]["sentences"]  # list of 10 lists of sentences
        assert len(ctx_titles) == len(ctx_sentences), \
            f"Context arity mismatch for {qid}: {len(ctx_titles)} titles vs {len(ctx_sentences)} sent lists"

        paragraphs = {}
        normalized_titles = {}
        for title, sents in zip(ctx_titles, ctx_sentences):
            paragraph_text = build_paragraph_text(sents)
            paragraphs[title] = paragraph_text
            normalized_titles[normalize_title(title)] = title

        # Parse supporting_facts (also column-oriented)
        sf_titles = q["supporting_facts"]["title"]
        sf_sent_ids = q["supporting_facts"]["sent_id"]
        assert len(sf_titles) == len(sf_sent_ids), \
            f"SF arity mismatch for {qid}"
        supporting_facts = [[t, sid] for t, sid in zip(sf_titles, sf_sent_ids)]

        index[qid] = {
            "question": q["question"],
            "gold_answer": q["answer"],
            "supporting_facts": supporting_facts,
            "paragraphs": paragraphs,
            "normalized_titles": normalized_titles,
            # Also keep raw sentences for later SF coverage checking
            "sentences_by_title": {
                title: sents
                for title, sents in zip(ctx_titles, ctx_sentences)
            },
        }

    # Check for qids we needed but didn't find
    found = set(index.keys())
    missing = needed_qids - found
    if missing:
        print(f"  WARNING: {len(missing)} qids in trajectories NOT found in dev.json:")
        for m in list(missing)[:10]:
            print(f"    - {m}")
    else:
        print(f"  All {len(needed_qids)} qids successfully indexed.")

    # Stats
    print()
    print("=== Index stats ===")
    print(f"  Questions indexed:         {len(index)}")
    if index:
        paras_per_q = [len(v["paragraphs"]) for v in index.values()]
        print(f"  Paragraphs per question:   min={min(paras_per_q)}, mean={sum(paras_per_q)/len(paras_per_q):.2f}, max={max(paras_per_q)}")
        sf_per_q = [len(v["supporting_facts"]) for v in index.values()]
        print(f"  SF per question:           min={min(sf_per_q)}, mean={sum(sf_per_q)/len(sf_per_q):.2f}, max={max(sf_per_q)}")

    # Save (pretty-printed for debuggability; file will be large-ish ~30-50 MB)
    print(f"\nWriting index to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(index, f, ensure_ascii=False)

    # File size
    from os.path import getsize
    size_mb = getsize(args.output) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()