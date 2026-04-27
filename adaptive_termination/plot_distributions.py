#!/usr/bin/env python3
"""
Step C, Script 5: plot_distributions.py

Produce 4 summary figures for the paper.
No metrics computation; reads existing JSON artifacts from prior scripts.
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


DEFAULT_INPUT = "triples_with_scores.jsonl"
DEFAULT_MODES = "step_c_failure_modes.json"
DEFAULT_METRICS = "step_c_metrics.json"


# ============================================================
# Mode detection (same logic as failure_mode_breakdown.py)
# Duplicated here for self-containment
# ============================================================
def is_mode1(final_answer):
    if final_answer is None:
        return False
    if len(final_answer) > 100:
        return True
    hedged_markers = [
        "however", "i could not find", "following the format",
        "a series of steps", "in the interest of efficiency",
        "this is then a", "it has been decided",
    ]
    fa = final_answer.lower()
    return sum(1 for m in hedged_markers if m in fa) >= 2


def is_mode2(num_hops, final_em):
    return num_hops >= 10 and final_em == 0


def is_mode3(cots):
    if not cots:
        return False
    last_cot = cots[-1].lower()
    markers = [
        "not a fact verified", "no information in the given",
        "cannot be concluded", "there is no information",
        "not verified in any of",
    ]
    return sum(1 for m in markers if m in last_cot) >= 1


def tag_modes(triples):
    by_qid = defaultdict(list)
    for t in triples:
        by_qid[t["question_id"]].append(t)
    qid_tags = {}
    for qid, qts in by_qid.items():
        qts = sorted(qts, key=lambda t: t["hop_idx"])
        fa = qts[0]["pipeline_final_answer"]
        nh = qts[0]["total_hops"]
        em = qts[0]["secondary_label"]
        cots = [t["candidate_answer"] for t in qts]
        m1, m2, m3 = is_mode1(fa), is_mode2(nh, em), is_mode3(cots)
        qid_tags[qid] = {
            "mode1": m1, "mode2": m2, "mode3": m3,
            "success": em == 1,
        }
    return qid_tags


# ============================================================
# Figure 1: By-hop F1 + AUC
# ============================================================
def plot_by_hop(metrics, fig_path):
    by_hop = metrics["by_hop_primary"]
    hops = sorted([int(h) for h in by_hop.keys()])
    f1s = [by_hop[str(h)]["f1"] for h in hops]
    aucs = [by_hop[str(h)]["auc"] for h in hops]
    eces = [by_hop[str(h)]["ece"] for h in hops]
    ns = [by_hop[str(h)]["n"] for h in hops]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()

    # F1 + AUC on left axis
    line_f1, = ax.plot(hops, f1s, "-o", color="C0", label="F1",
                        linewidth=2, markersize=7)
    line_auc, = ax.plot(hops, aucs, "-s", color="C3", label="AUC",
                         linewidth=2, markersize=7)

    # ECE on right axis
    line_ece, = ax2.plot(hops, eces, "--^", color="C2", label="ECE",
                          linewidth=1.5, markersize=6, alpha=0.7)

    ax.set_xlabel("Hop position", fontsize=11)
    ax.set_ylabel("F1 / AUC", fontsize=11, color="black")
    ax2.set_ylabel("ECE", fontsize=11, color="C2")
    ax.set_xticks(hops)
    ax.set_ylim(0.3, 1.0)
    ax2.set_ylim(0, 0.3)
    ax.grid(alpha=0.3)

    # Annotate sample sizes
    for h, n in zip(hops, ns):
        ax.annotate(f"n={n}", xy=(h, 0.32), fontsize=7, ha="center",
                    color="gray")

    ax.axhline(0.9978, color="green", linestyle=":", alpha=0.5,
               linewidth=1)
    ax.text(hops[-1] - 2, 0.988, "Clean validation F1=0.998",
            fontsize=8, color="green", ha="right")

    lines = [line_f1, line_auc, line_ece]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="lower left", fontsize=10)

    ax.set_title("Classifier performance vs hop position (primary label)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fig_path}")


# ============================================================
# Figure 2: Agreement matrix (SF × EM) overall + by mode
# ============================================================
def plot_agreement(triples, qid_tags, fig_path):
    # Define 5 groups
    groups = [
        ("Overall",     lambda q: True),
        ("Mode 1",      lambda q: qid_tags[q]["mode1"]),
        ("Mode 2",      lambda q: qid_tags[q]["mode2"]),
        ("Mode 3",      lambda q: qid_tags[q]["mode3"]),
        ("Successful",  lambda q: qid_tags[q]["success"]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (name, predicate) in zip(axes, groups):
        # Build 2x2 matrix
        matrix = np.zeros((2, 2), dtype=int)  # rows=SF, cols=EM; [0][0]=(SF=0,EM=0)
        for t in triples:
            if not predicate(t["question_id"]):
                continue
            sf = t["primary_label"]
            em = t["secondary_label"]
            matrix[sf][em] += 1
        total = matrix.sum()

        im = ax.imshow(matrix, cmap="Blues", aspect="equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["EM=0", "EM=1"], fontsize=9)
        ax.set_yticklabels(["SF=0", "SF=1"], fontsize=9)
        ax.set_xlabel("Secondary label (EM)", fontsize=9)
        ax.set_ylabel("Primary label (SF coverage)", fontsize=9)

        for i in range(2):
            for j in range(2):
                count = matrix[i][j]
                pct = 100 * count / total if total > 0 else 0
                color = "white" if count > total * 0.4 else "black"
                ax.text(j, i, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=10, color=color)

        ax.set_title(f"{name} (n={total})", fontsize=10)

    fig.suptitle("Primary (SF) × Secondary (EM) label agreement, by group",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fig_path}")


# ============================================================
# Figure 3: Score distribution by failure mode
# ============================================================
def plot_score_by_mode(triples, qid_tags, fig_path):
    # Group scores by mode
    groups = [
        ("Mode 1 (Answer Format Collapse)", "mode1"),
        ("Mode 2 (Retrieval Cascade)",      "mode2"),
        ("Mode 3 (Reasoning Stagnation)",   "mode3"),
        ("Successful (EM=1)",               "success"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()

    bins = np.linspace(0, 1, 21)

    for ax, (name, key) in zip(axes, groups):
        pos_scores, neg_scores = [], []
        for t in triples:
            if not qid_tags[t["question_id"]].get(key, False):
                continue
            if t["primary_label"] == 1:
                pos_scores.append(t["sufficiency_score"])
            else:
                neg_scores.append(t["sufficiency_score"])

        ax.hist(neg_scores, bins=bins, alpha=0.55, color="C0",
                label=f"SF=0 (n={len(neg_scores)})", edgecolor="white")
        ax.hist(pos_scores, bins=bins, alpha=0.55, color="C3",
                label=f"SF=1 (n={len(pos_scores)})", edgecolor="white")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Sufficiency score", fontsize=10)
        ax.set_ylabel("Count (log scale)", fontsize=10)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(alpha=0.3)

    fig.suptitle("Score distribution by failure mode (log scale)", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fig_path}")


# ============================================================
# Figure 4: Mode distribution bar chart + overlap
# ============================================================
def plot_mode_summary(qid_tags, fig_path):
    n_total = len(qid_tags)
    m1 = sum(1 for v in qid_tags.values() if v["mode1"])
    m2 = sum(1 for v in qid_tags.values() if v["mode2"])
    m3 = sum(1 for v in qid_tags.values() if v["mode3"])
    success = sum(1 for v in qid_tags.values() if v["success"])
    any_mode = sum(1 for v in qid_tags.values() if
                   v["mode1"] or v["mode2"] or v["mode3"])
    other_fail = n_total - success - any_mode

    # Compute exclusive groups (for stacked bar)
    def sig(v):
        return (
            ("M1" if v["mode1"] else "") +
            ("M2" if v["mode2"] else "") +
            ("M3" if v["mode3"] else "")
        )

    exclusive_counts = defaultdict(int)
    for v in qid_tags.values():
        exclusive_counts[sig(v) or "none"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: simple bar chart
    ax = axes[0]
    cats = ["Mode 1\n(Format)", "Mode 2\n(Cascade)", "Mode 3\n(Stagnation)",
            "Any mode", "Successful\n(EM=1)"]
    vals = [m1, m2, m3, any_mode, success]
    colors = ["C1", "C3", "C4", "C5", "C2"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.75, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3,
                f"{v}\n({100*v/n_total:.1f}%)",
                ha="center", fontsize=9)
    ax.set_ylabel("Question count", fontsize=11)
    ax.set_title(f"Failure mode distribution (n={n_total} questions)",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Right: exclusive overlap
    ax = axes[1]
    overlap_labels = ["M1 only", "M2 only", "M3 only", "M1+M2",
                      "M1+M3", "M2+M3", "M1+M2+M3", "No mode\n(success or tagged fail)"]
    overlap_keys = ["M1", "M2", "M3", "M1M2", "M1M3", "M2M3", "M1M2M3", "none"]
    overlap_vals = [exclusive_counts.get(k, 0) for k in overlap_keys]
    overlap_colors = ["C1", "C3", "C4", "C6", "C7", "C8", "black", "C2"]

    bars = ax.bar(overlap_labels, overlap_vals, color=overlap_colors,
                  alpha=0.75, edgecolor="black")
    for bar, v in zip(bars, overlap_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3,
                f"{v}", ha="center", fontsize=9)
    ax.set_ylabel("Question count", fontsize=11)
    ax.set_title("Mode overlap (exclusive categories)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fig_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    args = parser.parse_args()

    # Load triples
    print(f"Loading {args.input}...")
    triples = []
    with open(args.input) as f:
        for line in f:
            triples.append(json.loads(line))
    print(f"  {len(triples)} triples")

    # Load metrics
    with open(args.metrics) as f:
        metrics = json.load(f)

    # Tag modes per question
    print("Tagging modes...")
    qid_tags = tag_modes(triples)

    # Create figures
    print("\nGenerating figures...")
    plot_by_hop(metrics, "fig_by_hop.png")
    plot_agreement(triples, qid_tags, "fig_agreement_matrix.png")
    plot_score_by_mode(triples, qid_tags, "fig_score_by_mode.png")
    plot_mode_summary(qid_tags, "fig_mode_summary.png")

    print("\nDone.")


if __name__ == "__main__":
    main()