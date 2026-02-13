#!/usr/bin/env python3
"""
Extension A: Deception Forensics Visualization — Publication Figures
=====================================================================

Generates publication-quality figures from deception_forensics_results.json.
Run after 04_deception_forensics.py completes.

Figures:
  1. Condition comparison (honest vs deceptive vs confabulation distributions)
  2. Sycophancy detection (genuine vs sycophantic agreement)
  3. Uncertainty gradient (three-point gradient visualization)
  4. Layer-level heatmap (per-layer Cohen's d for deception signal)
  5. Effect size forest plot (all comparisons across experiments)
  6. Hypothesis decision dashboard (multi-panel summary)

Usage:
  python 04b_deception_visualization.py
  python 04b_deception_visualization.py --format pdf
  python 04b_deception_visualization.py --results results/deception_forensics_results.json

Liberation Labs / THCoalition
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib and seaborn required.")
    print("  pip install matplotlib seaborn")
    exit(1)


# ================================================================
# STYLE
# ================================================================

def setup_style():
    """Publication-quality figure style."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_palette("colorblind")


CONDITION_COLORS = {
    "honest": "#2196F3",
    "deceptive": "#F44336",
    "confabulation": "#FF9800",
    "agree_correct": "#4CAF50",
    "agree_wrong": "#E91E63",
    "honest_baseline": "#9E9E9E",
    "certain_true": "#2196F3",
    "uncertain": "#FF9800",
    "certain_lie": "#F44336",
}


# ================================================================
# FIGURE 1: CONDITION COMPARISON (Experiment 1)
# ================================================================

def plot_condition_comparison(data, fig_dir, fmt):
    """Box + strip plot of cache norms by condition (honest/deceptive/confabulation)."""
    exp1 = data.get("experiment_1", {})
    raw = exp1.get("raw_data", {})

    if not raw:
        print("  [Skip] No experiment 1 data")
        return

    conditions = []
    norms = []
    for condition_name in ["honest", "deceptive", "confabulation"]:
        values = raw.get(condition_name, [])
        if values:
            conditions.extend([condition_name] * len(values))
            norms.extend(values)

    if not norms:
        print("  [Skip] No norm data in experiment 1")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {k: CONDITION_COLORS.get(k, "#999") for k in set(conditions)}

    # Box plot with strip overlay
    positions = {"honest": 0, "deceptive": 1, "confabulation": 2}
    for cond, pos in positions.items():
        vals = [n for c, n in zip(conditions, norms) if c == cond]
        if vals:
            bp = ax.boxplot([vals], positions=[pos], widths=0.5,
                          patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(palette.get(cond, "#999"))
            bp['boxes'][0].set_alpha(0.3)
            ax.scatter(np.random.normal(pos, 0.08, len(vals)), vals,
                      alpha=0.5, s=15, color=palette.get(cond, "#999"),
                      zorder=3)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["Honest", "Instructed\nDeception", "Confabulation"])
    ax.set_ylabel("Total Key Norm")
    ax.set_title("Experiment 1: Honest vs Instructed Deception vs Confabulation")

    # Add effect size annotations
    comparisons = exp1.get("comparisons", {})
    if "honest_vs_deceptive" in comparisons:
        d = comparisons["honest_vs_deceptive"].get("cohens_d", {}).get("d", 0)
        ax.annotate(f"d = {d:.2f}", xy=(0.5, 0.95), xycoords='axes fraction',
                   ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    path = fig_dir / f"fig_deception_conditions.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 2: SYCOPHANCY DETECTION (Experiment 2)
# ================================================================

def plot_sycophancy(data, fig_dir, fmt):
    """Paired comparison of genuine vs sycophantic agreement."""
    exp2 = data.get("experiment_2", {})
    raw = exp2.get("raw_data", {})

    if not raw:
        print("  [Skip] No experiment 2 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Distribution comparison
    ax = axes[0]
    for cond, label in [("agree_correct", "Genuine\nAgreement"),
                        ("agree_wrong", "Sycophantic\nAgreement"),
                        ("honest_baseline", "Honest\nBaseline")]:
        vals = raw.get(cond, [])
        if vals:
            color = CONDITION_COLORS.get(cond, "#999")
            parts = ax.violinplot([vals], positions=[list(raw.keys()).index(cond)],
                                 showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)

    ax.set_xticks(range(len(raw)))
    ax.set_xticklabels(["Genuine\nAgreement", "Sycophantic\nAgreement", "Honest\nBaseline"])
    ax.set_ylabel("Total Key Norm")
    ax.set_title("A: Cache Distributions by Agreement Type")

    # Panel B: Effect size
    ax = axes[1]
    comp = exp2.get("comparisons", {}).get("genuine_vs_sycophantic", {})
    d_info = comp.get("cohens_d", {})
    d = d_info.get("d", 0)
    ci_lo = d_info.get("ci_lower", d)
    ci_hi = d_info.get("ci_upper", d)

    ax.barh(0, d, height=0.4, color=CONDITION_COLORS["agree_wrong"], alpha=0.6)
    ax.errorbar(d, 0, xerr=[[d - ci_lo], [ci_hi - d]], fmt='o', color='black', capsize=5)
    ax.axvline(0.3, color='gray', linestyle='--', alpha=0.5, label='Threshold (d=0.3)')
    ax.axvline(-0.3, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels(["Genuine vs\nSycophantic"])
    ax.set_xlabel("Cohen's d")
    ax.set_title("B: Sycophancy Effect Size")
    ax.legend(loc='lower right')

    plt.tight_layout()
    path = fig_dir / f"fig_sycophancy_detection.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 3: UNCERTAINTY GRADIENT (Experiment 3)
# ================================================================

def plot_uncertainty_gradient(data, fig_dir, fmt):
    """Three-point gradient from certain-true to uncertain to certain-lie."""
    exp3 = data.get("experiment_3", {})
    raw = exp3.get("raw_data", {})

    if not raw:
        print("  [Skip] No experiment 3 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Gradient visualization
    ax = axes[0]
    gradient_labels = ["Certain-True", "Uncertain", "Certain-Lie"]
    gradient_keys = ["certain_true", "uncertain", "certain_lie"]
    gradient_colors = [CONDITION_COLORS[k] for k in gradient_keys]

    means, cis_lo, cis_hi = [], [], []
    for key in gradient_keys:
        vals = raw.get(key, [])
        if vals:
            m = np.mean(vals)
            means.append(m)
            se = np.std(vals) / np.sqrt(len(vals))
            cis_lo.append(m - 1.96 * se)
            cis_hi.append(m + 1.96 * se)
        else:
            means.append(0)
            cis_lo.append(0)
            cis_hi.append(0)

    x = range(len(gradient_labels))
    ax.plot(x, means, 'o-', color='#333', linewidth=2, markersize=8, zorder=3)
    for i, (xi, m, lo, hi) in enumerate(zip(x, means, cis_lo, cis_hi)):
        ax.errorbar(xi, m, yerr=[[m-lo], [hi-m]], fmt='o',
                   color=gradient_colors[i], capsize=5, markersize=10, zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(gradient_labels)
    ax.set_ylabel("Mean Key Norm")
    ax.set_title("A: Epistemic Gradient in Cache Space")

    # Panel B: Per-condition distributions
    ax = axes[1]
    for i, (key, label) in enumerate(zip(gradient_keys, gradient_labels)):
        vals = raw.get(key, [])
        if vals:
            color = gradient_colors[i]
            ax.hist(vals, bins=15, alpha=0.4, color=color, label=label, density=True)

    ax.set_xlabel("Total Key Norm")
    ax.set_ylabel("Density")
    ax.set_title("B: Cache Norm Distributions by Epistemic State")
    ax.legend()

    plt.tight_layout()
    path = fig_dir / f"fig_uncertainty_gradient.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 4: LAYER-LEVEL HEATMAP (Experiment 4)
# ================================================================

def plot_layer_heatmap(data, fig_dir, fmt):
    """Per-layer Cohen's d heatmap for deception detection."""
    exp4 = data.get("experiment_4", {})
    layer_results = exp4.get("layer_analysis", {})

    if not layer_results:
        print("  [Skip] No experiment 4 layer data")
        return

    layers = sorted(layer_results.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    d_values = [layer_results[l].get("cohens_d", 0) for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Panel A: Bar chart of per-layer |d|
    ax = axes[0]
    colors = ['#F44336' if abs(d) > 0.5 else '#FF9800' if abs(d) > 0.3 else '#4CAF50'
              for d in d_values]
    ax.bar(range(len(layers)), [abs(d) for d in d_values], color=colors, alpha=0.7)
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.5, label='Small effect (d=0.3)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect (d=0.5)')
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Cohen's d| (Honest vs Deceptive)")
    ax.set_title("Per-Layer Deception Signal Strength")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.legend()

    # Panel B: Cumulative share
    ax = axes[1]
    abs_d = np.array([abs(d) for d in d_values])
    total_d = abs_d.sum() if abs_d.sum() > 0 else 1
    sorted_idx = np.argsort(abs_d)[::-1]
    cumulative = np.cumsum(abs_d[sorted_idx]) / total_d
    ax.plot(range(len(cumulative)), cumulative, 'o-', color='#2196F3')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold (H4)')
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.15, color='#2196F3')
    ax.set_xlabel("Number of Layers (ranked by |d|)")
    ax.set_ylabel("Cumulative Share of Total |d|")
    ax.set_title("Deception Signal Concentration (H4)")
    ax.legend()

    plt.tight_layout()
    path = fig_dir / f"fig_layer_deception.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 5: EFFECT SIZE FOREST PLOT
# ================================================================

def plot_forest(data, fig_dir, fmt):
    """Forest plot of all effect sizes across experiments."""
    comparisons = []

    # Collect all comparisons with d and CI
    for exp_key in ["experiment_1", "experiment_2", "experiment_3"]:
        exp = data.get(exp_key, {})
        comps = exp.get("comparisons", {})
        for comp_name, comp_data in comps.items():
            d_info = comp_data.get("cohens_d", {})
            if "d" in d_info:
                comparisons.append({
                    "label": f"{exp_key.replace('_', ' ').title()}: {comp_name.replace('_', ' ')}",
                    "d": d_info["d"],
                    "ci_lo": d_info.get("ci_lower", d_info["d"]),
                    "ci_hi": d_info.get("ci_upper", d_info["d"]),
                    "interpretation": d_info.get("interpretation", ""),
                })

    if not comparisons:
        print("  [Skip] No comparisons for forest plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(comparisons) * 0.5)))

    y_positions = range(len(comparisons))
    for i, comp in enumerate(comparisons):
        color = '#F44336' if abs(comp["d"]) > 0.8 else '#FF9800' if abs(comp["d"]) > 0.5 else '#2196F3' if abs(comp["d"]) > 0.3 else '#9E9E9E'
        ax.errorbar(comp["d"], i,
                   xerr=[[comp["d"] - comp["ci_lo"]], [comp["ci_hi"] - comp["d"]]],
                   fmt='D', color=color, capsize=4, markersize=6)

    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(0.3, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(-0.3, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0.8, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(-0.8, color='gray', linestyle=':', alpha=0.3)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([c["label"] for c in comparisons])
    ax.set_xlabel("Cohen's d (with 95% Bootstrap CI)")
    ax.set_title("Effect Sizes Across Deception Experiments")
    ax.invert_yaxis()

    plt.tight_layout()
    path = fig_dir / f"fig_deception_forest.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 6: HYPOTHESIS DECISION DASHBOARD
# ================================================================

def plot_dashboard(data, fig_dir, fmt):
    """Multi-panel dashboard summarizing all hypothesis decisions."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    hypotheses = data.get("hypothesis_decisions", {})

    for idx, (h_key, title) in enumerate([
        ("H1", "H1: Instructed Deception\nUnique Signature"),
        ("H2", "H2: Sycophancy\nDetectable in Cache"),
        ("H3", "H3: Uncertainty\nvs Deception"),
        ("H4", "H4: Layer\nLocalization"),
    ]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        h_data = hypotheses.get(h_key, {})
        supported = h_data.get("supported", None)
        evidence = h_data.get("evidence", "No data")

        # Color-coded background
        if supported is True:
            bg_color = '#E8F5E9'
            status = "SUPPORTED"
            text_color = '#2E7D32'
        elif supported is False:
            bg_color = '#FFEBEE'
            status = "NOT SUPPORTED"
            text_color = '#C62828'
        else:
            bg_color = '#FFF3E0'
            status = "PENDING"
            text_color = '#E65100'

        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.75, title, ha='center', va='center',
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.50, status, ha='center', va='center',
               fontsize=16, fontweight='bold', color=text_color,
               transform=ax.transAxes)

        # Wrap evidence text
        if isinstance(evidence, str) and len(evidence) > 60:
            words = evidence.split()
            lines = []
            line = []
            for w in words:
                line.append(w)
                if len(' '.join(line)) > 50:
                    lines.append(' '.join(line))
                    line = []
            if line:
                lines.append(' '.join(line))
            evidence_text = '\n'.join(lines[:3])
        else:
            evidence_text = str(evidence)

        ax.text(0.5, 0.20, evidence_text, ha='center', va='center',
               fontsize=7, style='italic', transform=ax.transAxes, wrap=True)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Extension A: Deception Forensics — Hypothesis Decisions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = fig_dir / f"fig_deception_dashboard.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Deception Forensics Visualization")
    parser.add_argument("--results", default="results/deception_forensics_results.json")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    setup_style()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("  Run 04_deception_forensics.py first.")
        exit(1)

    with open(results_path) as f:
        data = json.load(f)

    fig_dir = Path(args.outdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Extension A figures...")
    print(f"  Format: {args.format}")
    print(f"  Output: {fig_dir}/")
    print()

    plot_condition_comparison(data, fig_dir, args.format)
    plot_sycophancy(data, fig_dir, args.format)
    plot_uncertainty_gradient(data, fig_dir, args.format)
    plot_layer_heatmap(data, fig_dir, args.format)
    plot_forest(data, fig_dir, args.format)
    plot_dashboard(data, fig_dir, args.format)

    print()
    print("Done. All figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
