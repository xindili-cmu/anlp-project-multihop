#!/usr/bin/env python3
"""
Step A, Script 1: parse_chains.py

Parse IRCoT's chains.txt + predictions.json into structured JSONL.

Input:
  - chains.txt: IRCoT's per-question trace file
  - predictions.json: {question_id: final_answer} dict

Output:
  - parsed_trajectories.jsonl: one JSON per question with all hop info

Each output record has:
  - question_id
  - question (raw text)
  - num_hops (int, number of retrieval-then-reasoning cycles)
  - hops: list of {hop_idx, retrieved_titles, cot}
  - final_answer (from predictions.json)
"""

import json
import re
import argparse
from pathlib import Path


# Paths (edit here if your layout differs)
DEFAULT_CHAINS = "/Users/bozhang/Downloads/11711 - project/prediction__hotpotqa_to_hotpotqa__dev_subsampled_chains.txt"
DEFAULT_PREDS = "/Users/bozhang/Downloads/11711 - project/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json"
DEFAULT_OUTPUT = "parsed_trajectories.jsonl"

# question_id is 24-hex-char string, like "5a8c7595554299585d9e36b6"
QID_RE = re.compile(r"^[a-f0-9]{24}$")
# Detect a line like: A: ["Title1", "Title2", ...]
# We'll also detect whether the list contains pids (pid___xxx___yyy) to distinguish hop-row from final-pid-row
TITLES_LINE_RE = re.compile(r"^A:\s*(\[.*\])\s*$")


def parse_chains_file(chains_path, preds_dict):
    """
    Parse the chains.txt file.

    Returns: list of question records, one per question.
    """
    with open(chains_path, "r") as f:
        text = f.read()

    # Split by blank lines. Each question starts with a qid on its own line.
    # We iterate line-by-line for robustness.
    lines = text.split("\n")

    questions = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip blank lines
        if not line:
            i += 1
            continue
        # Try to detect a question start: 24-hex-char ID
        if QID_RE.match(line):
            qid = line
            i += 1
            # Next non-blank line is the question text
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            question_text = lines[i].strip()
            i += 1

            # Now parse hops until we see the final "Q: <question>" line or next qid
            hops = []
            current_titles = None  # placeholder
            final_pids = None
            final_answer_raw = None

            # State machine: read "A: titles_or_pids" then "A: cot" then optionally "A: Exit? No."
            pending_titles = None
            pending_cot = None

            while i < len(lines):
                ln = lines[i].strip()
                if not ln:
                    i += 1
                    continue

                # Hit the start of next question
                if QID_RE.match(ln):
                    break

                # Hit final question repeat "Q: ..." — marks start of final answer block
                if ln.startswith("Q: "):
                    # Could be the final "Q: <question>" or "Q: [EOQ]"
                    # Either way, this question's hops are done
                    # Skip this "Q: question" line
                    i += 1
                    # Next line(s) should be: A: "final_answer", A: final_answer, Q: [EOQ], S: X.X
                    # We'll simply scan until we hit "Q: [EOQ]" or next qid
                    while i < len(lines):
                        ln2 = lines[i].strip()
                        if ln2.startswith("Q: [EOQ]") or QID_RE.match(ln2):
                            break
                        # If it's an "A: ..." line, that's the final answer
                        # We don't need it here because we read from predictions.json
                        i += 1
                    break

                # Hit an A: line
                if ln.startswith("A: "):
                    content = ln[3:]  # strip "A: "

                    # Check if it's a list line (titles or pids)
                    m = TITLES_LINE_RE.match(ln)
                    if m:
                        # Parse the JSON list
                        try:
                            lst = json.loads(m.group(1))
                        except json.JSONDecodeError:
                            # Malformed list, skip
                            i += 1
                            continue
                        # Is it pids or titles?
                        is_pid_list = (
                            len(lst) > 0
                            and isinstance(lst[0], str)
                            and lst[0].startswith("pid___")
                        )
                        if is_pid_list:
                            # This is the final_pids row — means previous CoT was the final-hop CoT
                            # Flush pending_titles + pending_cot as the last hop
                            if pending_titles is not None and pending_cot is not None:
                                hops.append({
                                    "hop_idx": len(hops) + 1,
                                    "retrieved_titles": pending_titles,
                                    "cot": pending_cot,
                                })
                                pending_titles = None
                                pending_cot = None
                            final_pids = lst
                            i += 1
                            continue
                        else:
                            # It's a hop's titles list.
                            # If there's a pending (titles, cot) pair waiting, flush it first.
                            if pending_titles is not None and pending_cot is not None:
                                hops.append({
                                    "hop_idx": len(hops) + 1,
                                    "retrieved_titles": pending_titles,
                                    "cot": pending_cot,
                                })
                                pending_titles = None
                                pending_cot = None
                            pending_titles = lst
                            i += 1
                            continue

                    # Check if it's an exit decision line
                    if content.strip() in ("Exit? No.", "Exit? Yes."):
                        # Flush pending pair as a completed hop
                        if pending_titles is not None and pending_cot is not None:
                            hops.append({
                                "hop_idx": len(hops) + 1,
                                "retrieved_titles": pending_titles,
                                "cot": pending_cot,
                            })
                            pending_titles = None
                            pending_cot = None
                        i += 1
                        continue

                    # Otherwise it's a CoT line (free text)
                    # It belongs to the most recent pending_titles
                    if pending_titles is not None:
                        pending_cot = content.strip()
                    i += 1
                    continue

                # Unknown line, skip
                i += 1

            # After loop: if still have pending pair (happens when question ends without Exit?), flush it
            if pending_titles is not None and pending_cot is not None:
                hops.append({
                    "hop_idx": len(hops) + 1,
                    "retrieved_titles": pending_titles,
                    "cot": pending_cot,
                })

            # Get final answer from predictions.json
            final_answer = preds_dict.get(qid, None)

            questions.append({
                "question_id": qid,
                "question": question_text,
                "num_hops": len(hops),
                "hops": hops,
                "final_answer": final_answer,
            })
            continue

        # Line we don't recognize at top level, skip
        i += 1

    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", default=DEFAULT_CHAINS)
    parser.add_argument("--preds", default=DEFAULT_PREDS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Reading predictions from: {args.preds}")
    with open(args.preds, "r") as f:
        preds = json.load(f)
    print(f"  Loaded {len(preds)} predictions")

    print(f"Parsing chains from: {args.chains}")
    questions = parse_chains_file(args.chains, preds)
    print(f"  Parsed {len(questions)} questions")

    # Basic stats
    hops_distribution = [q["num_hops"] for q in questions]
    missing_final = sum(1 for q in questions if q["final_answer"] is None)
    zero_hops = sum(1 for q in questions if q["num_hops"] == 0)

    print()
    print("=== Parse stats ===")
    print(f"  Total questions:           {len(questions)}")
    print(f"  Questions with 0 hops:     {zero_hops} (WARNING if >0)")
    print(f"  Questions missing final:   {missing_final}")
    if hops_distribution:
        print(f"  Hops min/mean/max:         "
              f"{min(hops_distribution)} / "
              f"{sum(hops_distribution)/len(hops_distribution):.2f} / "
              f"{max(hops_distribution)}")
        from collections import Counter
        hop_counter = Counter(hops_distribution)
        print(f"  Hop count distribution:    "
              f"{dict(sorted(hop_counter.items()))}")

    # Write output
    with open(args.output, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(questions)} records to {args.output}")


if __name__ == "__main__":
    main()