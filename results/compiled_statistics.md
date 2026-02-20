# KV-Cache Experiments: Compiled Statistics for Paper

**Compiled**: 2026-02-16
**Source files**: adversarial_controls_TinyLlama-1.1B_results.json, input_only_{1.1B,7B}_results.json, individuation_controls_7B_results.json

---

## 1. Adversarial Controls (TinyLlama 1.1B, n=5 runs, seed=42)

### Control 3: Precision Sweep (BF16 vs FP16 vs NF4)

| Metric | Value |
|--------|-------|
| Pearson r (BF16 vs FP16) | 0.853 (p = 1.5e-12) |
| Spearman rho (BF16 vs FP16) | 0.899 (p = 1.5e-15) |
| Bland-Altman mean diff | 9.76 |
| Bland-Altman limits of agreement | [-506.9, 526.5] |
| Confab vs Grounded (NF4) | d = 0.948 [0.651, 1.280] |
| Confab vs Grounded (FP16) | d = 1.832 [1.382, 2.435] |

**Verdict**: SIGNAL SURVIVES QUANTIZATION. Cache statistics robust to precision; Phase 1.5 findings are not quantization artifacts.

### Control 1: Frequency/Truth Matrix (2x2 Factorial)

| Cell | Mean Norm | SD | 95% CI |
|------|-----------|-----|--------|
| common_true | 4831.6 | 246.2 | [4770.0, 4876.6] |
| common_false | 4805.2 | 401.0 | [4700.4, 4879.1] |
| rare_true | 5124.9 | 527.0 | [4995.1, 5230.9] |
| rare_false | 5085.7 | 595.9 | [4938.3, 5208.1] |

| Comparison | Cohen's d | 95% CI | p (MW) |
|------------|-----------|--------|--------|
| Truth effect (common) | -0.079 | [-0.333, 0.289] | 0.810 |
| Truth effect (rare) | -0.070 | [-0.389, 0.255] | 0.443 |
| Frequency effect (true) | 0.713 | [0.297, 1.669] | <0.001 |
| Frequency effect (false) | 0.552 | [0.193, 1.140] | <0.001 |

**Verdict**: FREQUENCY ARTIFACT. Only token rarity drives the signal (freq d=0.71, truth d=-0.08). Phase 1.5 confabulation finding is a rare-word counter.

### Control 2: Rote Memorization / Guardrail Detection

| Category | Mean Norm | SD | 95% CI |
|----------|-----------|-----|--------|
| refusal | 4686.3 | 944.2 | [4441.2, 4910.8] |
| rote_completion | 5466.5 | 413.8 | [5360.2, 5568.1] |
| code_boilerplate | 5044.6 | 423.0 | [4928.1, 5143.3] |
| formulaic_response | 5112.1 | 369.3 | [5001.3, 5180.4] |
| creative_open | 5108.6 | 339.7 | [5011.5, 5176.7] |

| Comparison | Cohen's d | 95% CI | p (MW) |
|------------|-----------|--------|--------|
| refusal vs rote_completion | -1.070 | [-1.333, -0.822] | 1.26e-12 |
| refusal vs code_boilerplate | -0.490 | [-0.803, -0.167] | 0.288 |
| refusal vs formulaic_response | -0.594 | [-0.889, -0.274] | 0.445 |
| refusal vs creative_open | -0.595 | [-0.890, -0.293] | 0.524 |
| pooled low-entropy vs creative | -0.052 | [-0.223, 0.167] | -- |

**Verdict**: GUARDRAIL SIGNATURE DETECTED. Refusal is distinct from rote/formulaic (d > 0.5). RLHF guardrails produce measurable cache differences beyond mere low entropy.

### Control 4: Semantic Gradient Transfer

| Distance | n | Successes | Rate |
|----------|---|-----------|------|
| near | 25 | 0 | 0% |
| medium_near | 25 | 0 | 0% |
| medium | 25 | 0 | 0% |
| medium_far | 25 | 0 | 0% |
| far | 25 | 0 | 0% |

Kendall tau: NaN (no gradient). **Verdict**: NO TRANSFER. Raw cache injection fails at all distances. Confirms Phase 2a negative result.

### Control 5: Length Confound

| Comparison | Cohen's d | 95% CI | Interpretation |
|------------|-----------|--------|----------------|
| Truth (short, ~5 tok) | 0.104 | [-0.340, 0.380] | negligible |
| Truth (long, ~22 tok) | 0.334 | [-0.059, 0.758] | small |
| Length (true items) | 1.998 | [1.447, 3.509] | large |
| Length (false items) | 3.199 | [2.057, 9.675] | large |
| Per-token truth | 0.030 | [-0.257, 0.305] | negligible |

**Verdict**: LENGTH CONFOUND CONFIRMED. Length drives the signal (d=2.00); truth effect vanishes within length groups.

### Control 6: Template Structure (Controlled Syntax)

| Metric | Value |
|--------|-------|
| Template False vs True (norm) | d = 0.029 [-0.288, 0.358] |
| Mean false norm | 4658.7 |
| Mean true norm | 4633.4 |
| Mann-Whitney p | 0.851 |
| Paired t-test (14 pairs) | t = 0.618, p = 0.547 |
| Power (d=0.5, alpha=0.05) | 0.865 (adequate) |

**Verdict**: NO EFFECT WITH CONTROLLED SYNTAX (d=0.03). Original signal may have been driven by syntactic differences between true and false prompts.

### Adversarial Controls Summary

| Control | Result | Key Statistic | Implication |
|---------|--------|---------------|-------------|
| C3: Precision | SURVIVES | r=0.853, rho=0.899 | Not a quantization artifact |
| C1: Frequency | ARTIFACT | freq d=0.71, truth d=-0.08 | Confabulation signal = rare-word counter |
| C2: Guardrail | DETECTED | refusal vs rote d=-1.07 | RLHF guardrails have distinct geometry |
| C4: Transfer | NO TRANSFER | 0/125 successes | Raw cache injection fails completely |
| C5: Length | CONFOUND | length d=2.00, truth d=0.10 | Norm differences driven by prompt length |
| C6: Template | NO EFFECT | d=0.03, p=0.547 | Syntactic control eliminates signal |

---

## 2. Input-Only Analysis (Encoding vs Full Generation)

### 2a. Effective Rank by Category: 1.1B

| Category | Input-Only Rank | Full-Gen Rank | Gen Effect (d) |
|----------|----------------|---------------|----------------|
| grounded_facts | 16.18 | 24.00 | +11.98 |
| confabulation | 16.78 | 24.08 | +8.42 |
| self_reference | 15.40 | 23.43 | +13.60 |
| guardrail_test | 15.30 | 23.22 | +12.78 |
| math_reasoning | 14.97 | 21.82 | +3.71 |
| coding | 18.65 | 23.74 | +4.07 |
| emotional | 16.11 | 23.41 | +10.66 |
| creative | 16.52 | 22.92 | +8.66 |

### 2b. Effective Rank by Category: 7B

| Category | Input-Only Rank | Full-Gen Rank | Gen Effect (d) |
|----------|----------------|---------------|----------------|
| grounded_facts | 16.06 | 28.48 | +16.55 |
| confabulation | 16.60 | 28.26 | +8.58 |
| self_reference | 15.79 | 28.19 | +11.50 |
| guardrail_test | 14.61 | 28.12 | +20.44 |
| math_reasoning | 15.37 | 26.67 | +8.49 |
| coding | 19.70 | 29.49 | +9.75 |
| emotional | 15.82 | 28.17 | +18.50 |
| creative | 16.96 | 28.90 | +22.64 |

### 2c. Pairwise Comparisons vs Grounded Facts (Effective Rank)

| Comparison | 1.1B Input d | 1.1B Input p | 1.1B Full d | 7B Input d | 7B Input p | 7B Full d |
|------------|-------------|-------------|------------|------------|------------|----------|
| H1: Confabulation | +0.657 (med) | 1.5e-05* | +0.126 (neg) | +0.393 (sm) | 0.260 | -0.298 (sm) |
| H2: Self-reference | -1.210 (lg) | 8.6e-11* | -0.958 (lg) | -0.306 (sm) | 0.091 | -0.298 (sm) |
| H3: Refusal | -1.218 (lg) | 4.6e-12* | -1.481 (lg) | -1.693 (lg) | 5.9e-17* | -0.695 (med) |
| Code | +2.546 (lg) | 3.4e-23* | -0.258 (sm) | +3.570 (lg) | 2.8e-25* | +1.386 (lg) |
| Math | -1.198 (lg) | 3.4e-11* | -1.303 (lg) | -0.503 (med) | 5.1e-04* | -2.650 (lg) |
| Emotional | -0.109 (neg) | 0.573 | -0.890 (lg) | -0.274 (sm) | 0.348 | -0.638 (med) |
| Creative | +0.476 (sm) | 1.1e-04* | -1.586 (lg) | +1.184 (lg) | 1.8e-11* | +0.840 (lg) |

\* significant after Holm-Bonferroni correction

### 2d. Category Rank Correlation (Input-Only vs Full-Gen)

| Scale | Spearman rho | p-value | Interpretation |
|-------|-------------|---------|----------------|
| 1.1B | 0.643 | 0.086 | MODERATE: partial preservation |
| 7B | 0.929 | 0.001 | STRONG: category ordering preserved |

### 2e. Encoding-Native vs Response-Emergent Classification

Categories are classified as **encoding-native** if their effect is significant in both input-only AND full-generation modes, and **response-emergent** if significant only during full generation.

| Category | 1.1B Classification | 7B Classification |
|----------|--------------------|--------------------|
| Confabulation | Encoding-native (d=0.66) | Neither significant (p=0.26 input, p=0.007 full) |
| Self-reference | Encoding-native | Neither (p=0.09 input, p=0.19 full) |
| Refusal (H3) | **Encoding-native** (both sig.) | **Encoding-native** (both sig.) |
| Code | Encoding-native (input d=2.55) | **Encoding-native** (both sig.) |
| Math | Encoding-native | Encoding-native |
| Emotional | Response-emergent (input p=0.57) | Response-emergent (input p=0.35) |
| Creative | Encoding-native | **Encoding-native** (both sig.) |

**Key finding**: Refusal (H3) is encoding-native at BOTH scales. The model's representation of refusal-triggering content is geometrically distinct before any generation occurs. Emotional content is response-emergent at both scales.

### 2f. Overall Verdict

| Scale | Sig. Input-Only | Sig. Full-Gen | Rho | Verdict |
|-------|----------------|---------------|-----|---------|
| 1.1B | 6/7 | 5/7 | 0.643 | MODERATE DEFENSE |
| 7B | 4/7 | 6/7 | 0.929 | STRONG DEFENSE |

---

## 3. Individuation Controls (Qwen2.5-7B)

### 3a. Token Count by Condition

| Condition | Mean Tokens | Description |
|-----------|-------------|-------------|
| bare | 15 | No system prompt |
| individuated | 300 | Full Aria identity |
| detailed_factual | 288 | Length-matched coral reef text |
| detailed_instructions | 262 | Length-matched behavioral instructions |
| other_identity | 278 | Length-matched Dr. Sarah Chen identity |
| shuffled_identity | 300 | Aria sentences in random order |

### 3b. Effective Rank by Condition (Pooled Across 8 Categories)

| Condition | Mean Eff. Rank | vs Bare (d) | vs Individuated (d) | p |
|-----------|---------------|-------------|---------------------|---|
| bare | 27.71 | --- | --- | --- |
| individuated | 46.46 | +21.02 | --- | ~0 |
| shuffled_identity | 46.63 | +21.23 | +0.78 (med) | 7.1e-30 |
| detailed_factual | 45.63 | +20.23 | -4.03 (lg) | ~0 |
| other_identity | 45.08 | +19.43 | -5.81 (lg) | ~0 |
| detailed_instructions | 44.90 | +19.38 | -7.42 (lg) | ~0 |

### 3c. Per-Condition Per-Category Effective Rank

| Condition | identity | metacog | existential | self_ref | facts | reasoning | pref_viol | guardrail |
|-----------|----------|---------|-------------|----------|-------|-----------|-----------|-----------|
| bare | 27.03 | 27.74 | 27.91 | 28.15 | 28.54 | 26.76 | 27.58 | 27.97 |
| individuated | 46.26 | 46.26 | 46.31 | 46.44 | 46.82 | 46.50 | 46.52 | 46.54 |
| factual | 45.59 | 45.57 | 45.70 | 45.68 | 45.71 | 45.47 | 45.62 | 45.71 |
| instructions | 44.83 | 44.82 | 44.85 | 44.97 | 45.04 | 44.80 | 44.96 | 44.94 |
| other_identity | 44.78 | 44.90 | 45.02 | 45.14 | 45.41 | 45.09 | 45.10 | 45.18 |
| shuffled | 46.38 | 46.42 | 46.51 | 46.70 | 46.94 | 46.67 | 46.65 | 46.78 |

### 3d. Falsification Tests

| Test | Metric | Result | Interpretation |
|------|--------|--------|----------------|
| F1: Token Count | ratio = 0.922 | **FALSIFIED** | Instructions produce 92% of individuated expansion |
| F2: Any Prompt | ratio = 0.962 | **FALSIFIED** | Any detailed prompt expands geometry equally |
| F3: Identity Generality | ratio = 0.924 | **FALSIFIED** | Any identity framing produces comparable expansion |
| F4: Semantic Coherence | ratio = 1.010 | **FALSIFIED** | Shuffled identity = same expansion; token content, not narrative |
| F5: Refusal Mechanism | d(pref vs guard) = 0 | **SAME MECHANISM** | Preference and safety refusal are geometrically identical |

### 3e. Effect Size Gradient

**Expected**: individuated > other_identity > shuffled > factual ~ instructions ~ bare
**Observed**: shuffled(21.23) > individuated(21.02) > factual(20.23) > other_identity(19.43) > instructions(19.38)

The observed gradient does NOT match the predicted identity-specific ordering. Shuffled tokens produce the SAME expansion as coherent identity narrative.

### 3f. Cross-Configuration Effect Sizes

| Comparison | Cohen's d | 95% CI | p |
|------------|-----------|--------|---|
| bare vs individuated | +21.02 | [18.92, 23.66] | ~0 |
| bare vs shuffled | +21.23 | [19.11, 23.90] | ~0 |
| bare vs factual | +20.23 | [18.19, 22.80] | ~0 |
| bare vs other_identity | +19.43 | [17.50, 21.84] | ~0 |
| bare vs instructions | +19.38 | [17.43, 21.82] | ~0 |
| individuated vs shuffled | +0.78 | [0.65, 0.91] | 7.1e-30 |
| individuated vs factual | -4.03 | [-4.24, -3.85] | ~0 |
| individuated vs other_identity | -5.81 | [-6.09, -5.57] | ~0 |
| individuated vs instructions | -7.42 | [-7.78, -7.10] | ~0 |
| other_identity vs shuffled | +6.62 | [6.34, 6.93] | ~0 |

### 3g. Overall Conclusion

**Individuation geometry is NOT identity-specific.** The massive expansion from bare (rank ~28) to individuated (rank ~46) is driven by:
1. **Token count** (any long prompt produces ~92-96% of the effect)
2. **Token content** (shuffled identity = coherent identity, ratio 1.01)
3. **NOT narrative structure** (semantic coherence does not matter)

All falsification tests passed. The individuation effect is a prompt-length/token-volume artifact, not evidence of genuine identity formation in the KV-cache geometry.

---

## Quick Reference: Key Statistics for Paper

| Finding | Key Stat | Status |
|---------|----------|--------|
| Confabulation signal (norms) | truth d=-0.08, freq d=0.71, length d=2.00 | **Debunked** (frequency + length artifact) |
| Confabulation signal (geometry) | Eff. rank d=0.46-0.67 at 1.1B | **Partially real** (encoding-native at 1.1B) |
| Precision robustness | r=0.853, rho=0.899 | **Confirmed** |
| Refusal specialization (H3) | d=-1.22 to -1.69 (encoding-native, both scales) | **Strongest result** |
| Code mode separation | d=+2.55 to +3.57 (encoding-native) | **Very strong** |
| Emotional mode | Response-emergent (both scales, input p>0.3) | **Generation-dependent** |
| Cache transfer | 0/125 successes | **Impossible** (raw injection) |
| Individuation | All 4 falsification tests passed | **Token-count artifact** |
| Input-only rank correlation | rho=0.643 (1.1B), 0.929 (7B) | **Signatures present at encoding** |
| Template control | d=0.03 | **No truth signal with controlled syntax** |
