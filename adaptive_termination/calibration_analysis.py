#!/usr/bin/env python3
"""
Step C, Script 3: calibration_analysis.py

Reliability diagram + per-bin ECE decomposition.
Visualizes classifier's confidence calibration on real distribution.

Output:
  - step_c_calibration.json: per-bin stats
  - calibration_reliability.png: reliability diagram (3 panels: overall/clean/drift)
  - score_histogram.png: distribution of scores by label
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT = "triples_with_scores.jsonl"
DEFAULT_OUTPUT = "step_c_calibration.json"


def compute_bin_stats(scores, labels, n_bins=10):
    """Return per-bin: count, avg_score, actual_positive_rate, ece_contribution."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        bin_n = int(mask.sum())
        if bin_n == 0:
            bins.append({
                "bin_idx": i,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "n": 0,
                "avg_score": None,
                "actual_pos_rate": None,
                "ece_contribution": 0.0,
            })
            continue
        avg = float(scores[mask].mean())
        pos_rate = float(labels[mask].mean())
        ece_contrib = (bin_n / n) * abs(avg - pos_rate)
        bins.append({
            "bin_idx": i,
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n": bin_n,
            "avg_score": avg,
            "actual_pos_rate": pos_rate,
            "ece_contribution": ece_contrib,
        })
    total_ece = sum(b["ece_contribution"] for b in bins)
    return bins, total_ece


def plot_reliability(ax, bins, title):
    """Plot a reliability diagram panel."""
    # Diagonal = perfect calibration
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    # Bar heights = actual positive rate in each bin
    # Bar positions = bin midpoints
    xs = []
    ys = []
    widths = []
    counts = []
    for b in bins:
        if b["n"] == 0:
            continue
        mid = (b["bin_low"] + b["bin_high"]) / 2
        xs.append(mid)
        ys.append(b["actual_pos_rate"])
        widths.append(b["bin_high"] - b["bin_low"])
        counts.append(b["n"])

    # Bar plot
    ax.bar(xs, ys, width=[w * 0.9 for w in widths],
           alpha=0.6, edgecolor="C0", color="C0", label="Actual positive rate")

    # Overlay: dots sized by bin count
    max_count = max(counts) if counts else 1
    for x, y, c in zip(xs, ys, counts):
        size = 30 + 200 * (c / max_count)
        ax.scatter(x, y, s=size, color="red", zorder=5, edgecolor="black",
                   alpha=0.7)

    # Annotate count on each bar
    for x, y, c in zip(xs, ys, counts):
        ax.annotate(f"n={c}", xy=(x, y), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted score (bin)")
    ax.set_ylabel("Actual positive rate")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def plot_score_histogram(scores, labels, title, ax, n_bins=30):
    """Overlaid histogram: score distribution by label."""
    pos_scores = np.asarray(scores)[np.asarray(labels) == 1]
    neg_scores = np.asarray(scores)[np.asarray(labels) == 0]
    bins = np.linspace(0, 1, n_bins + 1)

    ax.hist(neg_scores, bins=bins, alpha=0.5, color="C0",
            label=f"label=0 (n={len(neg_scores)})", density=False)
    ax.hist(pos_scores, bins=bins, alpha=0.5, color="C3",
            label=f"label=1 (n={len(pos_scores)})", density=False)
    ax.set_xlabel("Sufficiency score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")  # log-scale because of extreme bimodality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fig_reliability", default="calibration_reliability.png")
    parser.add_argument("--fig_hist", default="score_histogram.png")
    args = parser.parse_args()

    # Load
    print(f"Loading {args.input}...")
    triples = []
    with open(args.input) as f:
        for line in f:
            triples.append(json.loads(line))
    print(f"  {len(triples)} triples")

    scores = np.array([t["sufficiency_score"] for t in triples])
    labels = np.array([t["primary_label"] for t in triples])
    miss_rates = np.array([t["miss_rate"] for t in triples])

    clean_mask = miss_rates == 0.0
    drift_mask = ~clean_mask

    # Compute bins
    overall_bins, overall_ece = compute_bin_stats(scores, labels)
    clean_bins, clean_ece = compute_bin_stats(scores[clean_mask], labels[clean_mask])
    drift_bins, drift_ece = compute_bin_stats(scores[drift_mask], labels[drift_mask])

    # Save
    out = {
        "overall": {"bins": overall_bins, "ece": overall_ece},
        "clean": {"bins": clean_bins, "ece": clean_ece},
        "drift": {"bins": drift_bins, "ece": drift_ece},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")

    # Print per-bin breakdown
    print("\n" + "=" * 70)
    print("PER-BIN BREAKDOWN (overall, primary label)")
    print("=" * 70)
    print(f"{'bin':<12} {'n':>5} {'avg_score':>11} {'pos_rate':>10} {'ece_contrib':>12}")
    for b in overall_bins:
        bin_range = f"[{b['bin_low']:.1f}, {b['bin_high']:.1f})"
        avg_str = f"{b['avg_score']:.4f}" if b['avg_score'] is not None else "   —"
        pos_str = f"{b['actual_pos_rate']:.4f}" if b['actual_pos_rate'] is not None else "   —"
        print(f"{bin_range:<12} {b['n']:>5} {avg_str:>11} {pos_str:>10} {b['ece_contribution']:>12.4f}")
    print(f"\n  Total ECE = {overall_ece:.4f}")
    print(f"  Clean ECE = {clean_ece:.4f}")
    print(f"  Drift ECE = {drift_ece:.4f}")

    # Reliability diagram
    print(f"\nPlotting reliability diagram to {args.fig_reliability}...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_reliability(axes[0], overall_bins, f"Overall (n={len(scores)}, ECE={overall_ece:.3f})")
    plot_reliability(axes[1], clean_bins, f"Clean (n={clean_mask.sum()}, ECE={clean_ece:.3f})")
    plot_reliability(axes[2], drift_bins, f"Drift (n={drift_mask.sum()}, ECE={drift_ece:.3f})")
    fig.suptitle("Reliability Diagram (primary label = SF coverage)", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.fig_reliability, dpi=150, bbox_inches="tight")
    print(f"✓ Saved")

    # Score histogram
    print(f"\nPlotting score histogram to {args.fig_hist}...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_score_histogram(scores, labels, "Overall", axes[0])
    plot_score_histogram(scores[clean_mask], labels[clean_mask], "Clean", axes[1])
    plot_score_histogram(scores[drift_mask], labels[drift_mask], "Drift", axes[2])
    fig.suptitle("Score distribution by label (log-scale y-axis)", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.fig_hist, dpi=150, bbox_inches="tight")
    print(f"✓ Saved")

    print("\nDone.")


if __name__ == "__main__":
    main()