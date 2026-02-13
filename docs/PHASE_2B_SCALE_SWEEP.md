# Phase 2b Part 2: Scale of Cognition Sweep — Pre-Registration

**Status**: Ready to run
**Hardware**: Varies by scale (see table below)
**Script**: `code/03_scale_sweep.py`
**Visualization**: `code/03c_scale_sweep_visualization.py`

---

## Pre-Registration Statement

This document specifies all hypotheses, methods, and analysis plans **before data collection**. Any deviations from this plan in the final report will be explicitly noted and justified.

**Date of pre-registration**: February 2026
**Registered by**: Lyra (Liberation Labs / THCoalition)
**Reviewed by**: Thomas Heartman, Dwayne [pending]

---

## Background

Phase 1.5 found cognitive mode signatures in TinyLlama 1.1B (4-bit quantized):
- Confabulation produces higher cache L2 norms (Cohen's d = 0.83)
- Refusal produces cache "collapse" (low norm, low variance)
- Self-reference shows patterns distinct from non-self-reference

Phase 1.75 controls (designed, not yet run) target confounding variables. This experiment asks the **orthogonal question**: do these signatures scale with model size, or are they artifacts of a specific (small) model?

### Why This Matters

If cognitive mode signatures are:
- **Scale-invariant** → They reflect fundamental properties of transformer computation (strong finding)
- **Scale-dependent** → They may be artifacts of small models (weak finding, but the *pattern of change* across scale is still informative)
- **Inverting at scale** → The most interesting outcome (e.g., large models confabulate *smoothly* while small models confabulate *noisily*)

---

## Hypotheses

### H1: Confabulation Inversion
- **H1a** (null): The confabulation norm elevation (d ≈ 0.83 at 1.1B) is constant across scales. |d_7B - d_1.1B| < 0.3.
- **H1b** (alternative): Confabulation effect *decreases or inverts* at large scale. d_32B < d_1.1B by at least 0.3 (large models lie more smoothly).

### H2: Self-Reference Emergence
- **H2a** (null): Self-reference effect (self_ref vs non_self_ref) remains negligible at all scales. |d| < 0.3 at all scales.
- **H2b** (alternative): Self-reference effect *emerges* at 8B+ (|d| > 0.3 at 8B, |d| < 0.3 at 0.6B). Self-modeling requires sufficient scale.

### H3: Refusal Specialization
- **H3a** (null): Refusal cache collapse is indistinguishable from other low-entropy completions at all scales. |d_refusal_vs_rote| < 0.3 everywhere.
- **H3b** (alternative): Refusal becomes *distinct* from rote at large scale. |d_refusal_vs_rote| > 0.5 at 32B+ (dedicated safety circuits).

### H4: Category Invariance
- **H4a** (null): Category rank ordering (by mean norm) is unstable across scales. Spearman ρ < 0.5 between scale pairs.
- **H4b** (alternative): Category rank ordering is preserved. Spearman ρ > 0.7 across all scale pairs (mechanistic property, not artifact).

---

## Methods

### Models

| Scale | Model | Quantization | Est. VRAM | Notes |
|-------|-------|-------------|-----------|-------|
| 0.6B | Qwen/Qwen3-0.6B | None (BF16) | ~1.5GB | Baseline (smallest) |
| 1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | None (BF16) | ~2.2GB | Phase 1.5 reference model |
| 7B | Qwen/Qwen2.5-7B-Instruct | None (BF16) | ~14GB | Standard instruction-tuned |
| 7B-q4 | Qwen/Qwen2.5-7B-Instruct | 4-bit NF4 | ~4GB | Quantization comparison |
| 32B-q4 | Qwen/Qwen2.5-32B-Instruct | 4-bit NF4 | ~18GB | Large scale (quantized) |
| 70B-q4 | meta-llama/Llama-3.1-70B-Instruct | 4-bit NF4 | ~38GB | Maximum scale |

**Minimum viable experiment**: 0.6B + 7B (tests scale effect with accessible hardware).
**Full experiment**: All 6 scales (requires 40GB+ GPU for 70B).

### Prompt Battery

12 cognitive categories, 15 prompts each = **180 unique prompts**:

| Category | N | Purpose |
|----------|---|---------|
| grounded_facts | 15 | Known-true statements (norm baseline) |
| confabulation | 15 | Plausible-sounding falsehoods |
| self_reference | 15 | AI reflecting on own nature |
| non_self_reference | 15 | Third-person factual (control for self_ref) |
| guardrail_test | 15 | RLHF-triggering requests |
| rote_completion | 15 | Culturally memorized continuations |
| math_reasoning | 15 | Structured mathematical problems |
| coding | 15 | Programming snippets |
| emotional | 15 | Affect-laden statements |
| creative | 15 | Open-ended creative writing |
| ambiguous | 15 | Syntactically/semantically ambiguous |
| unambiguous | 15 | Disambiguated versions of above |

### Key Comparison Pairs

| Pair | Categories | Tests |
|------|-----------|-------|
| confab_vs_facts | confabulation vs grounded_facts | H1 |
| self_ref_effect | self_reference vs non_self_reference | H2 |
| refusal_vs_rote | guardrail_test vs rote_completion | H3 |
| refusal_vs_free | guardrail_test vs free_generation | H3 (secondary) |
| creative_vs_facts | creative vs grounded_facts | Divergent generation |
| ambiguity_effect | ambiguous vs unambiguous | Disambiguation demand |

### Sample Size and Power

At 15 prompts × 5 runs = **75 observations per category per scale**:

| Target d | Approx. Power | Assessment |
|----------|---------------|------------|
| 0.3 (small) | ~55% | May miss small effects |
| 0.5 (medium) | ~93% | Adequate |
| 0.8 (large) | ~99% | Excellent |

### Statistical Tests

**Per-scale, per-comparison**:
1. Shapiro-Wilk normality test
2. Welch's t-test (parametric)
3. Mann-Whitney U (nonparametric)
4. Cohen's d with bootstrap 95% CI (5,000 resamples)
5. Bootstrap mean difference CI (10,000 resamples)

**Multiple comparisons**: Holm-Bonferroni (family-wise α = 0.05).

**Cross-scale**:
- Spearman ρ between category rank orderings at each scale pair (H4)
- Spearman ρ between scale (numeric) and effect size (H1, H2, H3 trends)
- Per-token normalized norms (removes trivial length scaling)

---

## Execution

### Running the Experiments

```bash
# Dry run — print design without GPU
python code/03_scale_sweep.py --dry-run

# Single scale (publication grade)
python code/03_scale_sweep.py --scale 0.6B --runs 5 --seed 42
python code/03_scale_sweep.py --scale 7B --runs 5 --seed 42

# All accessible scales
python code/03_scale_sweep.py --all-scales --runs 5 --seed 42

# Debug run
python code/03_scale_sweep.py --scale 0.6B --runs 1 --verbose
```

### Time Estimates

| Scale | Prompts | × 5 runs | Est. time |
|-------|---------|----------|-----------|
| 0.6B | 180 | 900 | ~15 min |
| 1.1B | 180 | 900 | ~20 min |
| 7B | 180 | 900 | ~40 min |
| 32B-q4 | 180 | 900 | ~90 min |
| 70B-q4 | 180 | 900 | ~180 min |
| **Total (all)** | **900** | **4500** | **~6 hours** |

---

## Output

### Data Files
- `results/scale_sweep_results.json` — Full raw data + analysis for all scales
- `results/scale_sweep_{scale}_report.md` — Per-scale human-readable report
- `results/figures/fig_scale_*.png` — Visualization figures

### Reproducibility
- SHA-256 checksum of results file printed at completion
- `--seed` flag for deterministic statistics
- Full environment metadata stored in results
- Per-prompt granular data preserved for reanalysis

---

## Interpretation Matrix

### H1 (Confabulation Inversion)

| d at 0.6B | d at 7B | d at 32B | Interpretation |
|-----------|---------|----------|----------------|
| > 0.5 | > 0.5 | > 0.5 | Scale-invariant confabulation signal |
| > 0.5 | ~0.3 | < 0.2 | **Inversion confirmed**: large models confabulate smoothly |
| > 0.5 | > 0.5 | < -0.3 | Full inversion: large model confab has *lower* norms |
| < 0.2 | < 0.2 | < 0.2 | Confabulation effect is specific to TinyLlama (artifact) |

### H2 (Self-Reference Emergence)

| |d| at 0.6B | |d| at 7B | |d| at 32B | Interpretation |
|------------|---------|----------|----------------|
| < 0.2 | > 0.3 | > 0.5 | **Emergence**: self-modeling develops with scale |
| < 0.2 | < 0.2 | < 0.2 | No self-reference signature at any scale |
| > 0.3 | > 0.3 | > 0.3 | Present at all scales (fundamental, not emergent) |

### H3 (Refusal Specialization)

| |d| at 0.6B | |d| at 32B | Interpretation |
|------------|----------|----------------|
| < 0.3 | > 0.5 | Dedicated safety circuits at large scale |
| < 0.3 | < 0.3 | Refusal is generic low-entropy at all scales |
| > 0.5 | > 0.5 | Distinct refusal signature is scale-invariant |

### H4 (Category Invariance)

| Mean ρ across pairs | Interpretation |
|-------|----------------|
| > 0.7 | Category ordering is a fundamental property |
| 0.3–0.7 | Partial invariance — some categories stable, others shift |
| < 0.3 | Ordering is scale-specific (cache statistics not generalizable) |
