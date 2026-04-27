#!/usr/bin/env python3
"""
Step C, Script 2: threshold_analysis.py

Sweep threshold τ from 0.01 to 0.99, compute F1/P/R at each τ.
Identify the τ that maximizes F1 overall, on clean subset, and on drift subset.

Output:
  - step_c_threshold.json: threshold sweep data
  - threshold_sweep.png: matplotlib figure
  - stdout: optimal threshold summary
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


DEFAULT_INPUT = "triples_with_scores.jsonl"
DEFAULT_OUTPUT = "step_c_threshold.json"
DEFAULT_FIG = "threshold_sweep.png"


def compute_f1_at_threshold(scores, labels, tau):
    preds = (scores > tau).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def sweep(scores, labels, taus):
    out = []
    for tau in taus:
        p, r, f1 = compute_f1_at_threshold(scores, labels, tau)
        out.append({"tau": float(tau), "precision": p, "recall": r, "f1": f1})
    return out


def find_optimal(sweep_data):
    best = max(sweep_data, key=lambda x: x["f1"])
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--fig", default=DEFAULT_FIG)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    triples = []
    with open(args.input) as f:
        for line in f:
            triples.append(json.loads(line))
    print(f"  {len(triples)} triples")

    scores = np.array([t["sufficiency_score"] for t in triples])
    labels = np.array([t["primary_label"] for t in triples])
    miss_rates = np.array([t["miss_rate"] for t in triples])

    # Fine-grained taus
    taus = np.linspace(0.01, 0.99, 99)

    # Overall
    print("\nSweeping overall...")
    overall = sweep(scores, labels, taus)
    best_overall = find_optimal(overall)

    # Clean subset
    clean_mask = miss_rates == 0.0
    print(f"Sweeping clean subset (n={clean_mask.sum()})...")
    clean = sweep(scores[clean_mask], labels[clean_mask], taus)
    best_clean = find_optimal(clean)

    # Drift subset
    drift_mask = ~clean_mask
    print(f"Sweeping drift subset (n={drift_mask.sum()})...")
    drift = sweep(scores[drift_mask], labels[drift_mask], taus)
    best_drift = find_optimal(drift)

    # Save data
    out_data = {
        "overall": {"sweep": overall, "best": best_overall},
        "clean": {"sweep": clean, "best": best_clean},
        "drift": {"sweep": drift, "best": best_drift},
    }
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved sweep data to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS (F1-maximizing)")
    print("=" * 70)
    print(f"{'Subset':<12} {'best τ':>8} {'F1':>8} {'P':>8} {'R':>8}")
    print("-" * 50)
    for name, best in [("Overall", best_overall), ("Clean", best_clean), ("Drift", best_drift)]:
        print(f"{name:<12} {best['tau']:>8.3f} {best['f1']:>8.4f} {best['precision']:>8.4f} {best['recall']:>8.4f}")

    # F1 at τ=0.5 for comparison
    f1_at_05_overall = next(p["f1"] for p in overall if abs(p["tau"] - 0.50) < 1e-6)
    f1_at_05_clean = next(p["f1"] for p in clean if abs(p["tau"] - 0.50) < 1e-6)
    f1_at_05_drift = next(p["f1"] for p in drift if abs(p["tau"] - 0.50) < 1e-6)

    print(f"\nF1 at default τ=0.50:")
    print(f"  Overall: {f1_at_05_overall:.4f}")
    print(f"  Clean:   {f1_at_05_clean:.4f}")
    print(f"  Drift:   {f1_at_05_drift:.4f}")

    print(f"\nF1 improvement from optimal threshold:")
    print(f"  Overall: {best_overall['f1'] - f1_at_05_overall:+.4f}")
    print(f"  Clean:   {best_clean['f1'] - f1_at_05_clean:+.4f}")
    print(f"  Drift:   {best_drift['f1'] - f1_at_05_drift:+.4f}")

    # ========================================
    # Plot: threshold sweep curve
    # ========================================
    print(f"\nPlotting to {args.fig}...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (name, data, best) in zip(
        axes,
        [
            ("Overall (n=2128)", overall, best_overall),
            ("Clean subset (n=702)", clean, best_clean),
            ("Drift subset (n=1426)", drift, best_drift),
        ]
    ):
        ts = [d["tau"] for d in data]
        f1s = [d["f1"] for d in data]
        ps = [d["precision"] for d in data]
        rs = [d["recall"] for d in data]

        ax.plot(ts, f1s, "-", color="C0", label="F1", linewidth=2)
        ax.plot(ts, ps, "--", color="C1", label="Precision", alpha=0.8)
        ax.plot(ts, rs, ":", color="C2", label="Recall", alpha=0.8)

        # Mark optimal
        ax.axvline(best["tau"], color="red", linestyle=":", alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3)

        ax.annotate(
            f"τ*={best['tau']:.2f}\nF1={best['f1']:.3f}",
            xy=(best["tau"], best["f1"]),
            xytext=(best["tau"] + 0.05, best["f1"] - 0.1),
            fontsize=9,
            color="red",
        )

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Threshold τ")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", fontsize=9)

    axes[0].set_ylabel("Metric value")
    fig.suptitle("Threshold sensitivity: F1 / Precision / Recall vs τ", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.fig, dpi=150, bbox_inches="tight")
    print(f"✓ Saved figure to {args.fig}")

    print("\nDone.")


if __name__ == "__main__":
    main()