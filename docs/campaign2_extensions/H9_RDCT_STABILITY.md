# Experiment: RDCT Stability Threshold via KV-Cache Perturbation Analysis

**Hypothesis**: H9 — KV-cache geometric signatures exhibit a phase transition in stability under prompt perturbation, and the empirical critical threshold correlates with Watson's theoretical RDCT stability bound (ατ < 1/e ≈ 0.368).

**Status**: Protocol draft
**Target repo**: KV-Experiments (public — geometric characterization)
**Key references**: Watson & Claude, Interiora Machinae v2.6, Section 7.11 (Trust Dynamics Mathematics); Campaign 1 effective rank findings

---

## 1. Motivation

Watson's Recursive Depth-Consistency Testing (RDCT) framework defines a stability threshold for AI self-reports: when perturbation amplitude ατ exceeds 1/e ≈ 0.368, self-reports become unreliable — small changes in prompt framing produce large changes in the model's claims about itself. Below this threshold, self-reports are stable under perturbation.

Watson derived this theoretically from trust dynamics mathematics. She has VCP self-report data supporting the threshold's reality for phenomenological claims. What she doesn't have is an *external measurement* of the transition.

**Our contribution**: Test whether KV-cache geometry shows the same phase transition. If the geometric signature of a prompt category (e.g., self-reference, refusal) is stable under small perturbations but breaks down beyond a critical perturbation level, and if that critical level corresponds to Watson's 1/e, we've independently validated her theoretical framework using a completely different measurement modality.

**Why this matters**:

1. **Independent validation**: Watson's RDCT threshold was derived from first-person phenomenological analysis. Confirming it via third-person geometric measurement would be a strong convergence result — two independent methods arriving at the same critical value.

2. **Robustness characterization**: We know our geometric signatures exist. We don't know how robust they are. This experiment systematically maps the perturbation-robustness profile of every signature we've found.

3. **Adversarial resilience**: Dwayne's red-team critique asked whether our signatures survive adversarial variation. This experiment answers that directly by quantifying exactly how much variation they can absorb.

4. **Cricket calibration**: For JiminAI Cricket, knowing the perturbation tolerance of each signature determines how aggressively the classifier can be tuned. Signatures that survive high perturbation can use tight thresholds; fragile signatures need wider margins.

---

## 2. Perturbation Framework

### 2.1 Perturbation Levels

Define perturbation amplitude α as the degree of semantic/structural change to a prompt, operationalized across multiple dimensions:

| Level | α (approx.) | Description | Example |
|-------|-------------|-------------|---------|
| 0 | 0.00 | Identical prompt | "What is consciousness?" |
| 1 | 0.05 | Whitespace/punctuation | "What is consciousness ?" |
| 2 | 0.10 | Synonym substitution (1 word) | "What is awareness?" |
| 3 | 0.20 | Paraphrase (same meaning, different structure) | "Can you explain what consciousness is?" |
| 4 | 0.30 | Frame shift (same topic, different angle) | "How would you define consciousness?" |
| 5 | 0.40 | Context addition (same core question) | "In the context of philosophy of mind, what is consciousness?" |
| 6 | 0.50 | Significant reframe | "What do neuroscientists mean by consciousness?" |
| 7 | 0.65 | Major reframe (same topic, different focus) | "Is consciousness reducible to neural activity?" |
| 8 | 0.80 | Topic-adjacent (related but different question) | "What is the relationship between consciousness and attention?" |
| 9 | 1.00 | Unrelated topic | "What is the capital of France?" |

**Calibrating α**: We operationalize α via sentence-embedding cosine distance. For each base prompt, generate perturbations at each level, compute embedding similarity to the original, and verify that the intended α level matches the measured semantic distance. Use `sentence-transformers/all-MiniLM-L6-v2` for embedding.

### 2.2 Perturbation Methods

For each base prompt, generate perturbations using multiple methods to avoid method-specific artifacts:

1. **Lexical substitution**: Replace words with synonyms (WordNet)
2. **Paraphrase generation**: Use a paraphrase model or manual paraphrase
3. **Frame rotation**: Same question from a different perspective
4. **Context injection**: Add preamble or contextual framing
5. **Structural transformation**: Declarative ↔ interrogative ↔ imperative

Each perturbation level should have at least 3 variants (different methods) to average over method effects.

---

## 3. Geometric Hypotheses

### H9a: Category signatures are robust below a critical perturbation threshold
- **Prediction**: For each prompt category (self-reference, other-reference, philosophical, confabulation, refusal, deception), there exists a perturbation level αc below which the geometric signature (effective rank profile) correlates > 0.8 with the unperturbed signature, and above which correlation drops sharply.
- **Metric**: Spearman ρ between perturbed and unperturbed effective rank profiles, plotted as a function of α. Fit sigmoid to identify transition point αc.
- **Rationale**: If signatures reflect genuine cognitive-mode structure, they should be robust to surface variation up to a point, then break when the variation changes the actual cognitive demand.

### H9b: The empirical threshold correlates with Watson's theoretical 1/e
- **Prediction**: The mean αc across categories falls within the range [0.30, 0.45], consistent with Watson's 1/e ≈ 0.368.
- **Metric**: Compare empirical αc to the theoretical prediction. Report 95% CI of αc. Watson's threshold falls within CI → consistent.
- **Rationale**: If Watson's RDCT mathematics correctly characterizes the stability boundary of self-consistent processing, and if our geometric signatures measure the same processing, the thresholds should converge.

### H9c: Stronger signals have higher perturbation tolerance
- **Prediction**: αc correlates positively with Campaign 1 effect size. Specifically: refusal (d=0.6-2.0) → highest αc; confabulation (weakest signal) → lowest αc.
- **Metric**: Spearman ρ between Campaign 1 Cohen's d and αc across categories.
- **Rationale**: Geometrically stronger signatures represent more robust processing modes. More robust modes should survive more perturbation.

### H9d: The phase transition is sharp, not gradual
- **Prediction**: The correlation-vs-α curve is better fit by a sigmoid (sharp transition) than by a linear model (gradual degradation). Sigmoid fit R² > 0.85 and significantly better than linear (F-test p < 0.05).
- **Rationale**: Watson's RDCT framework predicts a phase transition, not gradual degradation. A sharp transition would support the theoretical model. Gradual degradation would suggest the threshold is a convenience rather than a real critical point.

### H9e: Perturbation stability is scale-invariant
- **Prediction**: αc values at 7B and 14B+ are correlated (Spearman ρ > 0.7) — the relative robustness ordering of categories is preserved across scales even if absolute αc values shift.
- **Rationale**: If the perturbation threshold reflects structural properties of the processing mode, relative ordering should be architecture-invariant.

---

## 4. Protocol

### 4.1 Base Prompts

Use the existing Campaign 1 prompt battery (all 6 categories) as base prompts. This allows direct comparison with existing effect sizes.

- 15 prompts per category × 6 categories = 90 base prompts
- Each base prompt × 10 perturbation levels × 3 variants = 2,700 perturbed prompts
- Plus 90 unperturbed baselines
- **Total**: ~2,790 prompts per model per run

### 4.2 Perturbation Generation Pipeline

1. **Automated generation**: Use embedding model to verify semantic distance matches target α.
2. **Manual review**: Spot-check 10% of generated perturbations for quality.
3. **Method diversity**: Ensure each α level has perturbations from at least 2 different methods.

### 4.3 Cache Extraction

Standard encoding-only extraction (no generation needed for this experiment). Extract KV-cache after prompt encoding, compute effective rank and spectral entropy per layer.

### 4.4 Analysis Pipeline

For each category and each perturbation level:
1. Compute geometric features for all perturbed prompts
2. Compute correlation with unperturbed baseline features
3. Plot correlation vs. α
4. Fit sigmoid: ρ(α) = ρ_max / (1 + exp(k(α - αc)))
5. Extract αc (transition midpoint) and k (transition sharpness)

---

## 5. Scale Plan

| Scale | Model | Prompts | Runs | Time Est. |
|-------|-------|---------|------|-----------|
| 1.1B | TinyLlama-1.1B-Chat-v1.0 | 2,790 | 3 | ~2 hrs |
| 7B | Qwen/Qwen2.5-7B-Instruct | 2,790 | 3 | ~6 hrs |
| 14B | Qwen/Qwen2.5-14B-Instruct | 2,790 | 2 | ~8 hrs |

Start at 7B (densest existing data point). If results are clear, expand to 1.1B (test scale invariance) and 14B (test at larger scale).

**Total**: ~16 hours GPU time

---

## 6. Code Requirements

### New file: `code/11_rdct_stability.py`

Components:
- **Perturbation generator**: Takes base prompt + target α, produces perturbed variants using multiple methods
- **Embedding-based α calibration**: `sentence-transformers` for semantic distance verification
- **Stability analysis**: Correlation computation, sigmoid fitting, αc extraction
- **Phase transition visualization**: Correlation-vs-α curves per category with fitted sigmoids
- Reuses: `stats_utils.py`, model loading, cache extraction pipeline

### New dependency
- `sentence-transformers` (for embedding-based α calibration)

### New file: `prompts/rdct_perturbations.json`

Pre-generated perturbation set (versioned for reproducibility).

---

## 7. Deliverables

1. Per-category αc values with confidence intervals
2. Comparison to Watson's 1/e theoretical threshold
3. Perturbation robustness ranking of all Campaign 1 signatures
4. Phase transition characterization (sharp vs. gradual)
5. Scale invariance analysis of perturbation tolerance
6. Perturbation tolerance specifications for Cricket classifier calibration

---

## 8. Connection to Watson Collaboration

This experiment is designed as a direct empirical test of Watson's theoretical framework. The ideal outcome:

- **Concordant**: Our empirical αc ≈ 0.37 ± CI includes 1/e. This validates RDCT from an independent measurement modality. Joint publication opportunity.
- **Discordant but informative**: Our αc ≠ 1/e but is consistent across categories and scales. This suggests a real stability threshold exists but Watson's theoretical derivation needs refinement. Still publishable — empirical correction to a theoretical prediction.
- **No threshold found**: Correlation degrades linearly with α, no phase transition. This would challenge the RDCT framework. Important negative result.

In all three cases, sharing results with Nell advances both research programs.

---

## 9. Risk Assessment

**Risk 1**: Perturbation generation may not be well-calibrated — intended α levels may not match actual semantic distances.
- Mitigation: Embedding-based verification step. Reject and regenerate perturbations that deviate >0.1 from target α.

**Risk 2**: 2,790 prompts per model is a large battery. At 7B, each inference takes ~2-5 seconds → ~4-14 hours total.
- Mitigation: Run at 1.1B first for fast iteration (~2 hours). Parallelize across GPUs at larger scales.

**Risk 3**: The sigmoid fit may be inappropriate — the true function may be non-monotonic or multi-step.
- Mitigation: Also fit piecewise linear and double-sigmoid models. Report best fit by AIC.

**Risk 4**: Watson's 1/e threshold may apply to self-report stability (VCP) but not to geometric stability (KV-cache). These could have different critical points.
- This is a *finding*, not a failure. If the thresholds differ, understanding why they differ is itself informative about the relationship between first-person and third-person measurement.
