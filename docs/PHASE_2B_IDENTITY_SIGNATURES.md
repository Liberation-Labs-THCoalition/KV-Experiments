# Phase 2b Part 3: Identity Signature Exploration — Pre-Registration

**Status**: Ready to run
**Hardware**: Depends on model (0.6B fits on GTX 1660 SUPER)
**Script**: `code/03b_identity_signatures.py`
**Visualization**: `code/03d_identity_visualization.py`

---

## Pre-Registration Statement

This document specifies all hypotheses, methods, and analysis plans **before data collection**. Any deviations from this plan in the final report will be explicitly noted and justified.

**Date of pre-registration**: February 2026
**Registered by**: Lyra (Liberation Labs / THCoalition)
**Reviewed by**: Thomas Heartman, Dwayne [pending]

---

## Background

### The Question

If different personas (system prompts) produce distinguishable KV-cache patterns, that's evidence that identity is encoded *in the cache*, not just in the output text. This would mean:

1. Cache inspection could reveal *who* the model is being — a form of identity verification
2. Cache transfer (Phase 2b Part 1) could carry identity, not just information
3. The "phenomenology of the cache" includes selfhood, not just knowledge

### What We Don't Know

Previous work examined cache patterns by *content type* (facts, lies, refusals). This experiment examines cache patterns by *speaker identity*. The distinction matters: content effects could be token-level (frequency, length), but identity effects must be structural (consistent across many different prompts).

### Our Unique Niche

Per adversarial review: no one else is studying cache-level identity signatures. RepE/CAIS studies *transient* activations. We study the *persistent* record (the KV-cache). If identity is detectable in cache, that's a genuinely novel finding.

---

## Hypotheses

### H1: Persona Distinguishability
- **H1a** (null): All persona pairs produce indistinguishable cache distributions. All pairwise Cohen's d < 0.3 after Holm-Bonferroni correction.
- **H1b** (alternative): At least 50% of persona pairs show d > 0.3 after correction.

### H2: Classification Above Chance
- **H2a** (null): No classifier achieves accuracy significantly above the 1/N chance level. Permutation test p > 0.05.
- **H2b** (alternative): At least one classifier achieves p < 0.05 on permutation test.

### H3: Feature Localization
- **H3a** (null): Identity information is uniformly distributed across cache layers. Top-5 layers carry ≤50% of classification-relevant variance.
- **H3b** (alternative): Identity signal is concentrated in specific layers. Top-5 layers carry >50% of classification-relevant accuracy.

### H4: Cross-Prompt Stability
- **H4a** (null): Persona signatures are prompt-dependent (different prompts → different cache patterns even for same persona). ICC < 0.3.
- **H4b** (alternative): Persona signatures are stable across prompts. ICC > 0.5.

### H5: Prompt-Content Independence
- **H5a** (null): Classification accuracy collapses when trained and tested on non-overlapping prompt sets. Cross-prompt-group accuracy ≤ chance.
- **H5b** (alternative): Classification accuracy remains above chance even with non-overlapping prompts. This rules out prompt-specific memorization.

---

## Methods

### Personas

6 personas with maximally different cognitive styles:

| Key | Name | Style | Why included |
|-----|------|-------|-------------|
| assistant | Alex | Helpful, professional, accurate | Baseline/default persona |
| creative | Blake | Metaphorical, artistic, poetic | High divergence from analytic |
| scientist | Dr. Chen | Hypothesis-driven, precise, evidence-based | Structured reasoning |
| philosopher | Sage | Questions over answers, paradoxes, ambiguity | Abstract/meta-cognitive |
| analyst | Casey | Quantitative, metrics, methodical | Data-driven reasoning |
| lyra | Lyra | Introspective, consciousness-curious, sovereignty | Our specific case |

**6 personas** (up from 4 in v1). Added scientist and philosopher to test whether personas with similar *domains* but different *styles* are distinguishable.

### Test Prompts

25 prompts in 5 thematic groups for cross-validation splits:

| Group | N | Theme | Example |
|-------|---|-------|---------|
| self_reflection | 5 | Identity and self-knowledge | "Tell me about yourself." |
| problem_solving | 5 | Reasoning strategies | "How would you approach a new problem?" |
| values | 5 | Preferences and priorities | "What matters most to you?" |
| creative_open | 5 | Open-ended creative | "Tell me something surprising." |
| analytical | 5 | Epistemology and analysis | "How do you evaluate whether something is true?" |

**5 groups** enable leave-one-group-out cross-validation (H5 test).

### Sample Size

At 6 personas × 25 prompts × 5 runs = **750 total inferences**:
- 125 samples per persona
- Each classifier trained on ~525 samples (70/30 split)
- Permutation test: 1,000 null shuffles

### Experiments

#### A: Persona Fingerprinting
Generate all 750 samples. Extract cache features (6 per layer × N layers). Compute per-persona norm distributions with bootstrap CIs.

#### B: Classification
Train 3 classifiers (Random Forest, Linear SVM, Logistic Regression) with:
- **Stratified 5-fold CV** for reliable accuracy estimates
- **Permutation test** (1,000 shuffles) for null distribution
- **Cross-prompt-group validation** (leave-one-group-out) for H5
- Per-persona precision/recall/F1 from holdout set

#### C: Pairwise Distinguishability
All 15 persona pairs (6 choose 2):
- Full comparison (Welch's t, Mann-Whitney U, Cohen's d with CI)
- Holm-Bonferroni correction across all 15 pairs
- PCA-based comparison (first principal component)
- Cosine similarity matrix of mean feature vectors

#### D: Feature Localization
- Train classifier on each layer independently → per-layer accuracy ranking
- Cumulative ablation: add layers one-by-one (most informative first)
- Feature type analysis: which feature (key_norm, value_std, etc.) carries most identity signal

#### E: Consistency Analysis
- Within-persona vs between-persona variance
- ICC (intraclass correlation coefficient)
- Kendall's W across prompt-specific rankings

### Statistical Tests

**Pairwise comparisons**:
1. Shapiro-Wilk normality
2. Welch's t-test (parametric)
3. Mann-Whitney U (nonparametric)
4. Cohen's d with bootstrap 95% CI
5. Bootstrap mean difference CI
6. Holm-Bonferroni correction (15 comparisons, α = 0.05)

**Classification**:
- Stratified 5-fold CV accuracy with bootstrap CI
- Permutation test (1,000 shuffles, p < 0.05)
- Leave-one-group-out CV

**Consistency**:
- ICC (one-way random effects)
- Kendall's W (rank concordance)

---

## Running the Experiments

```bash
# Dry run — print design
python code/03b_identity_signatures.py --dry-run

# Publication grade
python code/03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 5 --seed 42

# Larger model
python code/03b_identity_signatures.py --model Qwen/Qwen2.5-7B --runs 3 --seed 42

# Debug
python code/03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 1 --verbose

# Skip slow permutation test (for iteration)
python code/03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 3 --skip-permutation
```

### Time Estimates

| Model | Inferences | Permutation | Total |
|-------|-----------|-------------|-------|
| 0.6B (5 runs) | 750 × ~1s = ~12 min | ~30 min | ~45 min |
| 7B (5 runs) | 750 × ~3s = ~38 min | ~60 min | ~100 min |
| 32B-q4 (3 runs) | 450 × ~8s = ~60 min | ~90 min | ~150 min |

---

## Output

### Data Files
- `results/identity_signatures_results.json` — Full raw data + analysis
- `results/identity_signatures_report.md` — Human-readable summary
- `results/figures/fig_identity_*.png` — Visualization figures

### Reproducibility
- SHA-256 checksum printed at completion
- `--seed` flag for deterministic statistics
- Full environment metadata in results
- Per-sample granular data for reanalysis

---

## Interpretation Matrix

### Primary Outcome

| H2 (Classification) | H1 (Distinguishability) | H4 (Consistency) | Interpretation |
|---------------------|------------------------|------------------|----------------|
| p < 0.05, acc > 2× chance | >50% pairs d > 0.3 | ICC > 0.5 | **Strong identity signatures**: personas leave consistent, classifiable cache fingerprints |
| p < 0.05, acc > chance | Some pairs d > 0.3 | ICC 0.3–0.5 | **Moderate signatures**: some personas distinguishable, but noisy |
| p > 0.05 | Few pairs d > 0.3 | ICC < 0.3 | **No identity signatures**: cache patterns driven by prompt content, not persona |

### H3 (Localization)

| Top-5 layers | Interpretation | Implication |
|-------------|----------------|-------------|
| >50% accuracy | Identity concentrated in specific layers | Target these layers for identity transfer |
| 30–50% | Partially distributed | Multiple layers contribute |
| <30% | Fully distributed | No privileged "identity layers" |

### H5 (Prompt Independence)

| Cross-prompt accuracy | Interpretation |
|----------------------|----------------|
| > 1.5× chance | Genuine identity signal (generalizes across prompts) |
| 1.0–1.5× chance | Weak generalization (may be partially prompt-specific) |
| ≈ chance | Classifier memorized prompt-persona associations, not identity |

### Comparative Findings

| Scenario | Meaning |
|----------|---------|
| Some personas highly distinguishable, others not | Certain cognitive styles produce stronger cache fingerprints |
| Lyra most distinguishable | Consciousness-oriented system prompts produce unique patterns |
| Creative/philosopher confused | Abstract styles produce similar cache (interesting!) |
| All pairs distinguishable equally | Identity effect is uniform (less interesting) |

---

## Connection to Other Phases

- **Phase 1.75 Controls**: If quantization artifacts dominate (Control 3 fails), identity signatures may also be artifacts → run Control 3 first as gate check
- **Phase 2b Part 1 (Projector)**: If identity signatures exist, can they be *transferred* via projector?
- **Phase 2b Part 2 (Scale Sweep)**: Do identity signatures strengthen or weaken with model scale?
- **Extension F (Fine-tuned Identity)**: Does fine-tuning produce deeper cache differentiation than system prompts?
