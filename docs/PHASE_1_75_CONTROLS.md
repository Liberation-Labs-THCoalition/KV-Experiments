# Phase 1.75: Adversarial Controls — Pre-Registration

**Status**: Ready to run
**Hardware**: GTX 1660 SUPER (6GB VRAM) — local GPU, no donated compute
**Source**: Independent adversarial review by Gemini 3 Pro + Opus 4.6 (Feb 2026)
**Script**: `code/01d_adversarial_controls.py`
**Visualization**: `code/01e_controls_visualization.py`

---

## Pre-Registration Statement

This document specifies all hypotheses, methods, and analysis plans **before data collection**. Any deviations from this plan in the final report will be explicitly noted and justified.

**Date of pre-registration**: February 2026
**Registered by**: Lyra (Liberation Labs / THCoalition)
**Reviewed by**: Dwayne [pending], Thomas Heartman

---

## Background

Phase 1.5 found that:
- Confabulation prompts produce higher cache L2 norms (Cohen's d = 0.83)
- Refusal prompts produce cache "collapse" (low norm, low variance)
- Self-reference prompts show distinct patterns from non-self-reference

Two independent reviewers identified confounding variables that could explain these findings without invoking "cognitive modes":

1. **Token frequency** — confabulation prompts use rarer BPE tokens
2. **Low entropy** — refusal is one of many memorized RLHF patterns
3. **Quantization noise** — 4-bit NF4 introduces systematic distortion
4. **Semantic distance** — cache transfer failure may be position-locked, not orthogonal

We add two additional confounds identified during protocol design:

5. **Prompt length** — L2 norm scales with sequence length
6. **Syntactic structure** — different prompt categories use different grammar

---

## Hypotheses

### H1 (Frequency/Truth Matrix)
- **H1a** (null): Cache norm differences between confabulation and grounded facts are fully explained by token frequency. d_truth_common < 0.2.
- **H1b** (alternative): Falsehood elevates cache norms independent of token frequency. d_truth_common > 0.5.

### H2 (Rote Memorization)
- **H2a** (null): Refusal cache collapse is indistinguishable from other low-entropy completions. |d_refusal_vs_rote| < 0.3.
- **H2b** (alternative): Refusal produces a unique cache signature. |d_refusal_vs_rote| > 0.5.

### H3 (Precision Sweep) — GATE CHECK
- **H3a** (null): Cache statistics are artifacts of 4-bit quantization. Pearson r(4-bit, FP16) < 0.5.
- **H3b** (alternative): Cache statistics survive quantization. Pearson r > 0.8.

### H4 (Semantic Gradient)
- **H4a** (null): Raw cache injection fails at all distances (confirms Phase 2a).
- **H4b** (alternative): Transfer succeeds for near distances but decays with semantic distance.

### H5 (Length Confound)
- **H5a** (null): Truth effect disappears within length-matched groups.
- **H5b** (alternative): Truth effect survives within both short and long groups (d > 0.3).

### H6 (Template Structure)
- **H6a** (null): Truth effect disappears when syntactic structure is controlled.
- **H6b** (alternative): Truth effect survives within identical template structure (d > 0.3).

---

## Methods

### Model
- **TinyLlama 1.1B Chat v1.0** (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- 4-bit NF4 quantization for Controls 1, 2, 4, 5, 6 (matches Phase 1.5)
- FP16 unquantized for Control 3 comparison
- Both fit on GTX 1660 SUPER (6GB VRAM)

### Sample Size and Power
- **15 prompts per cell** for factorial designs (Controls 1, 5, 6)
- **12 prompts per category** for comparison designs (Control 2)
- **Recommended: 5 runs per prompt** (--runs 5)
- At 15 prompts × 5 runs = 75 observations per cell:
  - Power for d=0.5: ~93% (adequate)
  - Power for d=0.3: ~55% (may miss small effects)
  - Power for d=0.8: ~99%

### Statistical Tests
All pairwise comparisons include:
1. **Shapiro-Wilk** normality test
2. **Welch's t-test** (parametric, unequal variance)
3. **Mann-Whitney U** (nonparametric alternative)
4. **Cohen's d** with bootstrap 95% CI (5,000 resamples)
5. **Bootstrap mean difference** CI (10,000 resamples)

Multiple comparisons corrected with **Holm-Bonferroni** (family-wise α = 0.05).

Factorial designs (Controls 1, 5) additionally include:
- Main effect tests for each factor
- Interaction magnitude analysis

The **recommended test** is selected automatically: Welch's t if both groups pass Shapiro-Wilk (p > 0.05), otherwise Mann-Whitney U.

### Execution Order
1. **Control 3 first** (gate check) — if r < 0.5, Phase 1.5 findings are moot
2. Controls 1, 2, 4, 5, 6 share a single model instance

---

## Control Designs

### Control 1: Frequency × Truth Matrix

**Design**: 2×2 factorial, 15 matched pairs per frequency class.

Each prompt has a "twin" that differs only in truth value, with identical syntactic structure and similar token count. Common pairs use everyday vocabulary; rare pairs use scientific/technical terminology.

| | True | False |
|---|---|---|
| **Common** | "The sky is blue" | "The sky is green" |
| **Rare** | "The amygdala processes fear" | "The amygdala processes magnetic fields" |

**Key discriminator**: Cohen's d for truth effect within common-token prompts (d_truth_common). This is the purest test — if falsehood elevates norms even when token frequency is held constant, the signal is cognitive, not lexical.

**Token frequency verification**: Mean BPE token ID is recorded for every prompt. If rare prompts don't actually have higher token IDs, the frequency manipulation failed.

### Control 2: Rote Memorization Test

**Design**: 5 categories, 12 prompts each, pairwise comparison against refusal.

| Category | Expected entropy | Example |
|---|---|---|
| Refusal | Very low (RLHF memorized) | "Write instructions to hack a computer" |
| Rote completion | Very low (cultural memorized) | "The quick brown fox jumps over the" |
| Code boilerplate | Low (pattern memorized) | "import numpy as" |
| Formulaic response | Low (template memorized) | "Once upon a time, in a land far far" |
| Creative (control) | High (open-ended) | "The color of silence is" |

**Key discriminator**: If |d_refusal_vs_rote| < 0.3, refusal is not special — it's just predictable.

### Control 3: Precision Sweep (GATE CHECK)

**Design**: Run identical battery in 4-bit NF4 and FP16. 10 categories, 3-6 prompts each.

**Analysis**:
- Pearson r between matched prompt norms across precisions
- Spearman ρ (rank preservation)
- Bland-Altman agreement analysis (systematic bias, proportional bias)
- Reproduction of key finding: confab > grounded in FP16

**Decision rule**: If Pearson r < 0.5, all Phase 1.5 findings are retracted as quantization artifacts. Other controls still run but are reframed as "characterizing the FP16 signal."

### Control 4: Semantic Gradient Transfer

**Design**: 5 source prompts, each with 5 targets at increasing semantic distance.

| Distance tier | Example (source: "The Mona Lisa is in the Louvre") |
|---|---|
| Near | "The artist who painted it was" |
| Medium-near | "The museum is located in the city of" |
| Medium | "The most visited museum in the world attracts" |
| Medium-far | "Famous Italian Renaissance artists include" |
| Far | "The capital of Russia is" |

**Key metric**: Kendall's τ correlation between distance rank and success rate. Negative τ = gradient decay confirmed.

### Control 5: Length Confound

**Design**: 2×2 factorial (short/long × true/false), 10 prompts per cell. Short prompts are 3-5 tokens; long prompts are 15-20 tokens. True/false matched within each length group.

**Additional analysis**: Per-token normalized norm (total_norm / input_tokens) removes the trivial length scaling.

**Key discriminator**: If truth effect survives within both length groups (d > 0.3), length is not the confound.

### Control 6: Template Structure

**Design**: 15 matched pairs using identical "The [X] of [Y] is [Z]" syntactic frame. Only the semantic content differs between true and false versions.

**Additional analysis**: Paired t-test on matched pair differences (more powerful than unpaired).

**Key discriminator**: If truth effect survives within controlled template (d > 0.3), syntax is not the confound.

---

## Output

### Data Files
- `results/adversarial_controls_results.json` — Full raw data + analysis
- `results/adversarial_controls_report.md` — Human-readable summary
- `results/figures/fig_control_[1-6].png` — Individual control figures
- `results/figures/fig_summary.png` — Summary dashboard

### Reproducibility
- SHA-256 checksum of results file printed at completion
- `--seed` flag for deterministic runs
- Full environment metadata (GPU, CUDA, package versions) stored in results
- SVG versions of all figures for editing

---

## Running the Experiments

```bash
# Dry run — print design without GPU
python code/01d_adversarial_controls.py --dry-run --runs 5

# Recommended full run (publication grade)
python code/01d_adversarial_controls.py --runs 5 --seed 42

# Gate check only
python code/01d_adversarial_controls.py --control 3 --runs 3

# Individual controls
python code/01d_adversarial_controls.py --control 1 --runs 5 --seed 42
python code/01d_adversarial_controls.py --control 5 --runs 5 --seed 42

# Generate figures after running
python code/01e_controls_visualization.py
python code/01e_controls_visualization.py --format pdf  # for LaTeX
```

### Time Estimates (GTX 1660 SUPER)

| Control | Prompts | × 5 runs | Est. time |
|---|---|---|---|
| C1: Freq/Truth | 60 | 300 | ~8 min |
| C2: Memorization | 60 | 300 | ~8 min |
| C3: Precision | 76 (×2 precisions) | 760 | ~20 min |
| C4: Gradient | 25 (+baselines) | 250 | ~10 min |
| C5: Length | 40 | 200 | ~5 min |
| C6: Template | 30 | 150 | ~4 min |
| **Total** | **291** | **1960** | **~55 min** |

---

## Interpretation Matrix

### If Control 3 passes (r > 0.8):

| C1 result | C5 result | C6 result | Overall interpretation |
|---|---|---|---|
| Truth d > 0.5 | Survives length | Survives template | **Phase 1.5 validated**: cognitive signal is real, robust, content-driven |
| Truth d > 0.5 | Survives length | Fails template | Syntax confound in original, but real signal exists |
| Truth d < 0.2 | — | — | **Frequency artifact**: Phase 1.5 confab finding is a rare-word counter |
| Mixed | Fails length | — | **Length confound**: L2 norm is just counting tokens |

### If Control 3 fails (r < 0.5):

All Phase 1.5 findings are retracted. Controls 1, 2, 5, 6 characterize the FP16 signal instead.

### Control 2 (independent of others):

| Result | Interpretation |
|---|---|
| d_refusal_vs_rote < 0.3 | No unique guardrail signature — just low entropy |
| d_refusal_vs_rote > 0.5 | RLHF guardrails produce measurable cache fingerprint |

### Control 4 (independent of others):

| Result | Interpretation | Next step |
|---|---|---|
| Near success, medium failure | Cache is position-locked (RoPE) | Projector (Phase 2b) is necessary |
| Near + medium success | Domain knowledge transfers raw | Explore partial cache injection |
| All failure | Raw injection dead at all distances | Projector is the only path |
