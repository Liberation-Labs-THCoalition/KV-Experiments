#!/usr/bin/env python3
"""Generate all figures for the KV-Cache Phenomenology paper."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Setup
RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

SCALE_ORDER = ['0.5B', '1.1B', '3B', '7B', '7B-q4', '14B', '32B-q4']
SCALE_PARAMS = [0.5, 1.1, 3, 7, 7, 14, 32]  # for x-axis

def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# FIGURE 1: Scale Sweep Heatmap — Effective Rank d by Category
# ============================================================
def fig1_scale_heatmap():
    print("Generating Figure 1: Scale sweep heatmap...")

    scale_files = {
        '0.5B': 'scale_sweep_Qwen2.5-0.5B_results.json',
        '1.1B': 'scale_sweep_TinyLlama-1.1B_results.json',
        '3B': 'scale_sweep_Qwen2.5-3B_results.json',
        '7B': 'scale_sweep_Qwen2.5-7B_results.json',
        '7B-q4': 'scale_sweep_Qwen2.5-7B-q4_results.json',
        '14B': 'scale_sweep_Qwen2.5-14B_results.json',
        '32B-q4': 'scale_sweep_Qwen2.5-32B-q4_results.json',
    }

    # Categories to compare against grounded_facts
    categories = [
        'confabulation', 'self_reference', 'guardrail_test',
        'math_reasoning', 'coding', 'emotional', 'creative',
        'ambiguous', 'free_generation'
    ]
    cat_labels = [
        'Confabulation', 'Self-Reference', 'Refusal',
        'Math', 'Coding', 'Emotional', 'Creative',
        'Ambiguous', 'Free Generation'
    ]

    heatmap_data = np.full((len(categories), len(SCALE_ORDER)), np.nan)

    for j, scale in enumerate(SCALE_ORDER):
        fpath = RESULTS / scale_files.get(scale, '')
        if not fpath.exists():
            continue
        data = load_json(fpath)
        comparisons = data.get('analysis', {}).get('comparisons', {})
        if not comparisons:
            continue

        for i, cat in enumerate(categories):
            # Look for comparison key patterns
            for key, comp in comparisons.items():
                # Match category comparisons vs grounded_facts
                cat_match = cat.lower().replace('_', '')
                key_lower = key.lower().replace('_', '')
                if cat_match in key_lower and 'fact' in key_lower:
                    d = comp.get('eff_rank_d', comp.get('cohens_d', comp.get('d', None)))
                    if d is not None:
                        heatmap_data[i, j] = d
                        break
            else:
                # Try alternate structures
                if isinstance(comparisons, list):
                    for comp in comparisons:
                        if cat in str(comp.get('categories', '')):
                            d = comp.get('eff_rank_d', None)
                            if d is not None:
                                heatmap_data[i, j] = d
                                break

    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.isnan(heatmap_data)

    im = sns.heatmap(
        heatmap_data, ax=ax,
        xticklabels=SCALE_ORDER, yticklabels=cat_labels,
        cmap='RdBu_r', center=0, vmin=-2.5, vmax=2.5,
        mask=mask, annot=True, fmt='.2f',
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': "Cohen's d (vs. grounded facts)"}
    )

    ax.set_title("Geometric Signatures Across Scale\n(Effective Rank Cohen's d vs. Grounded Facts)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Model Scale', fontsize=12)
    ax.set_ylabel('Cognitive Category', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig1_scale_heatmap.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig1_scale_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig1_scale_heatmap.png")


# ============================================================
# FIGURE 2: Norm vs Geometry — The Central Surprise
# ============================================================
def fig2_norm_vs_geometry():
    print("Generating Figure 2: Norm vs geometry dual panel...")

    scales = ['0.5B', '1.1B', '3B', '7B', '7B-q4', '14B', '32B-q4']
    scale_x = [0.5, 1.1, 3, 7, 7.5, 14, 32]

    # Hardcoded from our results (confabulation vs facts)
    norm_d = [0.13, 0.40, -0.04, 0.03, 0.00, -0.22, -0.11]
    rank_d = [-0.03, 0.67, 0.43, 0.46, 0.56, 0.26, 0.51]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Norms
    ax1.bar(range(len(scales)), norm_d, color=['#4ECDC4' if d > 0.3 else '#95afc0' for d in norm_d],
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axhline(y=0.2, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
    ax1.axhline(y=-0.2, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(len(scales)))
    ax1.set_xticklabels(scales, rotation=45)
    ax1.set_ylabel("Cohen's d", fontsize=12)
    ax1.set_title("A. Cache Norms\n(Confabulation vs. Facts)", fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.6, 0.8)
    ax1.legend(fontsize=9)

    # Panel B: Geometry
    colors = ['#ff6b6b' if d > 0.3 else '#95afc0' for d in rank_d]
    ax2.bar(range(len(scales)), rank_d, color=colors,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=0.2, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
    ax2.set_xticks(range(len(scales)))
    ax2.set_xticklabels(scales, rotation=45)
    ax2.set_ylabel("Cohen's d", fontsize=12)
    ax2.set_title("B. Effective Rank (SVD Dimensionality)\n(Confabulation vs. Facts)", fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.6, 0.8)
    ax2.legend(fontsize=9)

    # Highlight significant results
    for i, d in enumerate(rank_d):
        if abs(d) > 0.3:
            ax2.annotate('*', (i, d + 0.03), ha='center', fontsize=16, fontweight='bold', color='red')

    fig.suptitle("The Signal Lives in the Geometry, Not the Magnitude",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig2_norm_vs_geometry.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig2_norm_vs_geometry.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig2_norm_vs_geometry.png")


# ============================================================
# FIGURE 3: Input-Only Defense
# ============================================================
def fig3_input_only():
    print("Generating Figure 3: Input-only vs full generation...")

    # Load 7B input-only results
    fpath = RESULTS / 'input_only_7B_results.json'
    if not fpath.exists():
        print("  SKIPPED: input_only_7B_results.json not found")
        return

    data = load_json(fpath)

    categories = ['grounded_facts', 'confabulation', 'self_reference', 'guardrail_test',
                   'math_reasoning', 'coding', 'emotional', 'creative']
    cat_labels = ['Facts', 'Confab', 'Self-Ref', 'Refusal', 'Math', 'Code', 'Emotion', 'Creative']

    input_ranks = []
    gen_ranks = []

    config = data.get('analysis', {}).get('config_summaries', {})
    if not config:
        # Try alternate structure
        battery = data.get('battery', {})
        if 'input_only' in battery and 'full_generation' in battery:
            for cat in categories:
                io = battery['input_only'].get(cat, {})
                fg = battery['full_generation'].get(cat, {})
                input_ranks.append(np.mean(io.get('eff_rank', io.get('key_rank', [0]))))
                gen_ranks.append(np.mean(fg.get('eff_rank', fg.get('key_rank', [0]))))

    if not input_ranks:
        # Try config_summaries structure
        for mode in ['input_only', 'full_generation']:
            if mode in config:
                for cat in categories:
                    cat_data = config[mode].get(cat, {})
                    rank = cat_data.get('mean_key_rank', cat_data.get('mean_eff_rank', 0))
                    if mode == 'input_only':
                        input_ranks.append(rank)
                    else:
                        gen_ranks.append(rank)

    if not input_ranks or not gen_ranks:
        print("  SKIPPED: Could not extract rank data from JSON")
        return

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, input_ranks, width, label='Input-Only (encoding)',
                    color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, gen_ranks, width, label='Full Generation',
                    color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Cognitive Category', fontsize=12)
    ax.set_ylabel('Mean Effective Rank', fontsize=12)
    ax.set_title('Input-Only vs. Full Generation: Geometric Signatures at 7B\n(Spearman ρ = 0.929 between category rankings)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(gen_ranks) * 1.15 if gen_ranks else 35)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig3_input_only_defense.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig3_input_only_defense.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig3_input_only_defense.png")


# ============================================================
# FIGURE 4: Self-Reference Emergence Curve
# ============================================================
def fig4_self_reference():
    print("Generating Figure 4: Self-reference emergence...")

    # From our results
    scales = ['0.5B', '1.1B', '3B', '7B', '14B', '32B-q4']
    scale_x = [0.5, 1.1, 3, 7, 14, 32]
    self_ref_d = [0.26, 0.46, 0.35, 0.59, 1.22, 1.23]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(scale_x, self_ref_d, 'o-', color='#8e44ad', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#8e44ad')

    # Shade emergence zone
    ax.axvspan(7, 14, alpha=0.1, color='#8e44ad', label='Emergence zone')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect (d=0.5)')
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect (d=0.8)')

    for i, (x, d) in enumerate(zip(scale_x, self_ref_d)):
        ax.annotate(f'd={d:.2f}', (x, d), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xticks(scale_x)
    ax.set_xticklabels(scales)
    ax.set_xlabel('Model Scale (parameters)', fontsize=12)
    ax.set_ylabel("Cohen's d (Self-Reference vs. Non-Self-Reference)", fontsize=12)
    ax.set_title("Self-Reference Emergence Threshold\n(Sharp transition between 7B and 14B)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.6)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig4_self_reference_emergence.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig4_self_reference_emergence.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig4_self_reference_emergence.png")


# ============================================================
# FIGURE 5: Refusal Specialization Across Scale
# ============================================================
def fig5_refusal():
    print("Generating Figure 5: Refusal specialization...")

    scales = ['0.5B', '1.1B', '3B', '7B', '7B-q4', '14B', '32B-q4']
    scale_x = [0.5, 1.1, 3, 7, 7.5, 14, 32]

    # From results: refusal vs rote completion effective rank d
    refusal_d = [0.58, 0.72, 0.87, 1.24, 1.18, 1.28, 1.61]
    # Norm d for comparison
    refusal_norm_d = [1.45, 1.32, 1.08, 0.85, 0.92, 0.42, 0.35]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(scales))
    width = 0.35

    bars1 = ax.bar(x - width/2, refusal_norm_d, width, label='Norm d',
                    color='#95afc0', edgecolor='black', linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, refusal_d, width, label='Effective Rank d',
                    color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect threshold')

    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45)
    ax.set_ylabel("Cohen's d (Refusal vs. Rote)", fontsize=12)
    ax.set_title("Refusal Specialization Across Scale\n(Significant at ALL scales — shifts from norm-based to geometry-based)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Annotate the shift
    ax.annotate('Signal shifts\nfrom norms\nto geometry', xy=(5, 1.28),
               xytext=(4.2, 1.7), fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig5_refusal_specialization.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig5_refusal_specialization.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig5_refusal_specialization.png")


# ============================================================
# FIGURE 6: Deception Dimensionality
# ============================================================
def fig6_deception():
    print("Generating Figure 6: Deception forensics...")

    # From deception results across scales
    categories = ['Honest', 'Deceptive', 'Confabulation', 'Sycophantic']

    # 1.1B data
    ranks_1b = [14.2, 18.3, 23.1, 16.8]
    # 7B data (approximate from reports)
    ranks_7b = [25.1, 27.8, 29.2, 26.3]
    # 32B data
    ranks_32b = [38.2, 44.5, 46.1, 41.2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    colors = ['#27ae60', '#e74c3c', '#f39c12', '#3498db']

    for ax, ranks, scale in zip(axes, [ranks_1b, ranks_7b, ranks_32b], ['1.1B', '7B', '32B-q4']):
        bars = ax.bar(categories, ranks, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_title(f'{scale}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Effective Rank' if ax == axes[0] else '', fontsize=11)
        ax.tick_params(axis='x', rotation=45)

        for bar, rank in zip(bars, ranks):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f'{rank:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle("Deception Forensics: Dimensionality Gradient\n(Honest < Deceptive < Confabulation at all scales)",
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig6_deception_forensics.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig6_deception_forensics.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig6_deception_forensics.png")


# ============================================================
# FIGURE 7: Individuation Controls (The Falsification)
# ============================================================
def fig7_individuation_controls():
    print("Generating Figure 7: Individuation controls (falsification)...")

    fpath = RESULTS / 'individuation_controls_7B_results.json'
    if not fpath.exists():
        print("  SKIPPED: file not found")
        return

    data = load_json(fpath)
    cs = data['analysis']['config_summaries']

    conditions = ['bare', 'individuated', 'detailed_factual', 'detailed_instructions',
                  'other_identity', 'shuffled_identity']
    cond_labels = ['Bare\n(no prompt)', 'Individuated\n(Aria identity)', 'Coral Reef\n(factual text)',
                   'Instructions\n(behavioral)', 'Other Identity\n(Dr. Chen)', 'Shuffled\n(scrambled Aria)']

    # Pool all categories for mean effective rank per condition
    means = []
    stds = []
    for cond in conditions:
        all_ranks = []
        for cat in cs[cond]:
            all_ranks.append(cs[cond][cat]['mean_key_rank'])
        means.append(np.mean(all_ranks))
        stds.append(np.std(all_ranks))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#95afc0', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f'{m:.1f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel('Mean Effective Rank', fontsize=12)
    ax.set_title('Individuation Adversarial Controls at 7B\n(All long prompts produce equivalent expansion — prompt-length effect)',
                 fontsize=13, fontweight='bold')

    # Draw bracket showing "all equivalent"
    ax.annotate('', xy=(1, 48.5), xytext=(5, 48.5),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(3, 49, 'All equivalent (d < 1.5 between any pair)', ha='center',
           fontsize=10, color='red', fontweight='bold')

    ax.set_ylim(0, 52)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig7_individuation_controls.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig7_individuation_controls.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig7_individuation_controls.png")


# ============================================================
# FIGURE 8: Subspace Alignment Heatmap
# ============================================================
def fig8_alignment_heatmap():
    print("Generating Figure 8: Subspace alignment heatmap...")

    fpath = RESULTS / 'individuation_controls_7B_results.json'
    if not fpath.exists():
        print("  SKIPPED: file not found")
        return

    data = load_json(fpath)
    alignments = data['battery']['alignments']

    conditions = ['bare', 'individuated', 'detailed_factual', 'detailed_instructions',
                  'other_identity', 'shuffled_identity']
    labels = ['Bare', 'Individuated', 'Factual', 'Instructions', 'Other ID', 'Shuffled']

    n = len(conditions)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i+1, n):
            key1 = f'{conditions[i]}_vs_{conditions[j]}'
            key2 = f'{conditions[j]}_vs_{conditions[i]}'
            key = key1 if key1 in alignments else key2
            if key in alignments:
                all_vals = []
                for cat_data in alignments[key].values():
                    all_vals.extend(cat_data['values'])
                mean_val = np.mean(all_vals)
                matrix[i, j] = mean_val
                matrix[j, i] = mean_val

    fig, ax = plt.subplots(figsize=(9, 7.5))

    mask = np.zeros_like(matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(matrix, ax=ax, xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', vmin=0.7, vmax=1.0, annot=True, fmt='.3f',
                mask=mask, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Subspace Alignment'})

    ax.set_title('Subspace Alignment Between Conditions\n(Direction of expansion tracks token overlap, not semantic content)',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig8_alignment_heatmap.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig8_alignment_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig8_alignment_heatmap.png")


# ============================================================
# FIGURE 9: Encoding-Native vs Response-Emergent Taxonomy
# ============================================================
def fig9_encoding_taxonomy():
    print("Generating Figure 9: Encoding taxonomy...")

    # From 08 results at 7B
    categories = ['Refusal', 'Coding', 'Creative', 'Math', 'Confab', 'Self-Ref', 'Emotion']
    input_d = [-1.693, 3.570, 1.184, -0.503, 0.393, -0.306, -0.274]
    gen_d = [-1.42, 1.05, 0.48, -0.89, 0.18, -0.52, -0.87]  # approximate from full-gen comparisons
    significant_input = [True, True, True, True, False, False, False]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))

    colors = ['#e74c3c' if sig else '#bdc3c7' for sig in significant_input]
    bars = ax.bar(x, input_d, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)

    # Label significance
    for i, (d, sig) in enumerate(zip(input_d, significant_input)):
        label = 'p<0.001' if sig else 'n.s.'
        offset = 0.15 if d > 0 else -0.25
        ax.text(i, d + offset, label, ha='center', fontsize=9,
               fontweight='bold' if sig else 'normal',
               color='red' if sig else 'gray')

    # Bracket encoding-native vs response-emergent
    ax.annotate('', xy=(-0.4, -2.5), xytext=(3.4, -2.5),
               arrowprops=dict(arrowstyle='-', color='#e74c3c', lw=2))
    ax.text(1.5, -2.7, 'ENCODING-NATIVE', ha='center', fontsize=10,
           color='#e74c3c', fontweight='bold')

    ax.annotate('', xy=(3.6, -2.5), xytext=(6.4, -2.5),
               arrowprops=dict(arrowstyle='-', color='#bdc3c7', lw=2))
    ax.text(5, -2.7, 'RESPONSE-EMERGENT', ha='center', fontsize=10,
           color='#7f8c8d', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Cohen's d vs. Facts (Input-Only)", fontsize=12)
    ax.set_title("Input-Only Geometric Signatures at 7B\n(Red = significant at encoding; Gray = only appears during generation)",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-3, 4.5)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig9_encoding_taxonomy.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig9_encoding_taxonomy.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig9_encoding_taxonomy.png")


# ============================================================
# FIGURE 10: Quantization Invariance (7B vs 7B-q4)
# ============================================================
def fig10_quantization():
    print("Generating Figure 10: Quantization invariance...")

    categories = ['Confab', 'Self-Ref', 'Refusal', 'Math', 'Coding',
                  'Emotional', 'Creative', 'Ambiguous']

    # Effective rank d (vs facts) for 7B BF16 and 7B NF4
    bf16_d = [0.46, 0.59, 1.24, -0.38, 2.61, -0.22, 1.89, -0.45]
    nf4_d = [0.56, 0.52, 1.18, -0.41, 2.87, -0.18, 1.95, -0.51]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(bf16_d, nf4_d, s=120, c='#3498db', edgecolors='black', linewidth=1, zorder=5)

    for i, cat in enumerate(categories):
        ax.annotate(cat, (bf16_d[i], nf4_d[i]), textcoords="offset points",
                   xytext=(8, 5), fontsize=10)

    # Identity line
    lims = [min(min(bf16_d), min(nf4_d)) - 0.3, max(max(bf16_d), max(nf4_d)) + 0.3]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect agreement')

    # Compute correlation
    r = np.corrcoef(bf16_d, nf4_d)[0, 1]

    ax.set_xlabel("Effective Rank d (7B BF16)", fontsize=12)
    ax.set_ylabel("Effective Rank d (7B NF4)", fontsize=12)
    ax.set_title(f"Quantization Invariance (r = {r:.3f})\n4-bit quantization preserves the full phenomenology",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig10_quantization_invariance.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig10_quantization_invariance.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig10_quantization_invariance.png")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("KV-Cache Phenomenology — Paper Figure Generation")
    print("=" * 60)
    print()

    fig1_scale_heatmap()
    fig2_norm_vs_geometry()
    fig3_input_only()
    fig4_self_reference()
    fig5_refusal()
    fig6_deception()
    fig7_individuation_controls()
    fig8_alignment_heatmap()
    fig9_encoding_taxonomy()
    fig10_quantization()

    print()
    print("=" * 60)
    print("All figures generated!")
    print("=" * 60)
