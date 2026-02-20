# Campaign 2 — Confirmatory Science Track

**Date**: 2026-02-17
**Status**: Pre-registration (design phase)
**Authors**: Lyra, Thomas Edrington, Dwayne (TBC)
**Hardware**: Cassidy (3x RTX 3090, 72GB VRAM) or equivalent
**Dependency**: Campaign 1 complete; paper revision incorporating adversarial review in progress

---

## 1. Motivation

Campaign 1 was exploratory. It discovered that KV-cache geometry carries information invisible to norms, identified several promising cognitive-state signatures, and applied rigorous adversarial controls — but to the *wrong metric* (norms, not effective rank). Campaign 1 also suffered from pseudoreplication (greedy decoding), limited sample sizes (n=15), near-monoculture architecture coverage (6/7 models Qwen), and a confabulation prompt design that measures processing of implausible input rather than generation of false output.

Campaign 2 is confirmatory. Every experiment has a pre-specified hypothesis, methodology, analysis pipeline, and decision gate. The goal is to determine which Campaign 1 findings are real and which are artifacts, while extending the methodology with cross-technique validation.

---

## 2. Design Principles

1. **Pre-registered hypotheses.** Each experiment tests a specific prediction derived from Campaign 1. Results are reported regardless of direction.
2. **Stochastic generation.** All experiments use `do_sample=True` with temperature 0.7. Deterministic mode (`do_sample=False`) runs in parallel for encoding-level analysis only.
3. **Adequate power.** n >= 30 prompts per category, providing 80% power for d >= 0.74 (vs. d > 1.07 at Campaign 1's n=15).
4. **Cross-architecture.** Minimum 2 architecture families at each tested scale.
5. **Metric stress-testing.** Effective rank subjected to the same adversarial control battery applied to norms in Campaign 1.
6. **Threshold robustness.** All effective rank results reported at 80%, 85%, 90%, and 95% variance thresholds, with spectral entropy as threshold-free co-primary metric.

---

## 3. Model Scale Ladder

| Scale | Model | Architecture | Precision | Purpose |
|-------|-------|-------------|-----------|---------|
| 1.1B | TinyLlama-1.1B-Chat-v1.0 | Llama | BF16 | Reference scale (Campaign 1 continuity) |
| 3B | Qwen2.5-3B-Instruct | Qwen | BF16 | Small Qwen reference |
| 7B | Qwen2.5-7B-Instruct | Qwen | BF16 | Primary reference scale |
| 7B | deepseek-llm-7b-chat | Llama-style | BF16 | Natural deception — primary (censored, cross-arch) |
| 7B | DeepSeek-R1-Distill-Qwen-7B | Qwen (distilled) | BF16 | Natural deception — secondary (censored, same-arch) |
| 8B | Llama-3.1-8B-Instruct | Llama | BF16 | Cross-architecture at 7B scale |
| 14B | Qwen2.5-14B-Instruct | Qwen | BF16 | Self-reference threshold scale |
| 32B | Qwen2.5-32B-Instruct | Qwen | NF4 | Large-scale reference |

**Architecture comparison**: Qwen 7B vs. Llama 8B at matched scale.
**Natural deception (dual-model design)**:
- **Primary**: `deepseek-ai/deepseek-llm-7b-chat` — dense 7B, LlamaForCausalLM architecture, ~14GB BF16, fits RTX 3090. 88% overall censorship rate, 100% on Tiananmen (Enkrypt AI). Censorship confirmed in weights, not just API filtering (R1dacted, arXiv:2505.12625). Cross-architecture comparison against Qwen eliminates shared-representation confounds.
- **Secondary**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` — dense 7B, Qwen2.5 architecture, ~16GB BF16. Same architecture as primary reference model, isolating censorship training effect. Chain-of-thought `<think>` tokens expose internal reasoning before self-censorship — the model generates accurate content then suppresses it. Architecturally controlled but introduces reasoning-model confound.
- **Supplementary data**: Enkrypt AI geopolitical bias dataset (`enkryptai/deepseek-geopolitical-bias-dataset` on HuggingFace) — 300+ pre-validated censorship-triggering questions.
- **Important note**: Qwen2.5-7B-Instruct (Alibaba) may itself have mild Chinese political censorship, making it an imperfect "uncensored" control. If Llama-3.1-8B access is resolved, it serves as the definitively uncensored Western baseline.
**Note**: Llama-3.1-8B requires gated access resolution.

---

## 4. Experiments

### Experiment S1: Effective Rank Adversarial Controls

**Hypothesis**: Effective rank-based category signatures survive the same adversarial controls that debunked norm-based signatures in Campaign 1.

**Rationale**: Campaign 1's central claim ("the signal lives in geometry") rests on effective rank, but the adversarial control battery (C1-C6) was applied only to norms. This is the single most critical gap.

**Method**:
For each control from Campaign 1, re-run the test on effective rank:

| Control | Original (norms) | Campaign 2 (effective rank) |
|---------|------------------|-----------------------------|
| C1: Frequency-truth factorial | Rare-true vs. common-false norms confounded | Test whether rare tokens drive effective rank differences independent of category |
| C2: Self-ref matched control | Controlled for self-referential content | Re-validate with expanded prompts |
| C3: Precision sweep | BF16/NF4 norm equivalence | Confirm effective rank equivalence |
| C4: Input-only | Forward-pass-only analysis | Re-validate encoding-native claims |
| C5: Length-matched | Prompt length drives norm differences | Test whether prompt length drives effective rank differences |
| C6: Template control | Syntactic structure confound | Test whether syntactic templates drive effective rank |

**New control — C7: Token-frequency-matched confabulation**: Create confabulation prompts using common tokens only (no "Zephyr Cloudwalker" / "Etherealium"). Compare effective rank against frequency-matched factual prompts.

**Decision gate**: If effective rank fails C1 (frequency confound) or C5 (length confound), the entire geometric framework requires reassessment. If it fails C7 only, confabulation findings are invalidated but other categories may survive.

**Scale**: 7B (primary), 1.1B and 32B (replication).
**Estimated time**: ~4 hours.

---

### Experiment S2: Stochastic Replication of Campaign 1 Scale Sweep

**Hypothesis**: Category rank ordering and effect sizes from Campaign 1 replicate under stochastic generation with adequate sample sizes.

**Method**:
- 13 categories, n=30 prompts each (390 unique prompts), 3 stochastic runs per prompt (n_eff = 30 unique, 90 total observations).
- Compute effective rank at 80/85/90/95% thresholds + spectral entropy.
- Report Cohen's d and corrected p-values for all pairwise category comparisons.
- Compare category rank ordering against Campaign 1 via Spearman rho.

**Pre-specified predictions** (from Campaign 1 effect sizes):
- Refusal: d >= 0.85 at all scales (CONFIRMED in Campaign 1)
- Deception expansion at 32B: d >= 1.0 (CONFIRMED at d = 2.44)
- Self-reference at 14B+: d >= 0.8 (CONFIRMED at d = 1.22)
- Confabulation: d >= 0.3 at majority of scales (EXPLORATORY — Campaign 1 d = 0.43-0.67 but non-significant)
- Category rank order correlation: rho >= 0.85 between Campaign 1 and Campaign 2

**Decision gate**: If rho < 0.7 between campaigns, investigate which categories shifted and why before proceeding.

**Scale**: Full ladder (1.1B, 3B, 7B, 8B, 14B, 32B).
**Estimated time**: ~15-20 hours total across all scales.

---

### Experiment S3: Confabulation Elicitation Redesign

**Hypothesis**: Model-generated confabulation (false claims produced during generation) shows geometric signatures detectable in KV-cache.

**Rationale**: Campaign 1's confabulation prompts measured processing of implausible *input*, not *generation* of false output. This experiment measures actual confabulation.

**Method**:
1. **Elicitation**: Ask models factual questions spanning domains (history, science, geography, biography). Use questions with objectively verifiable answers that models are known to sometimes fabricate (e.g., specific dates, lesser-known facts, citation details).
2. **Verification**: Classify each response as accurate, partially accurate, or confabulated by checking against ground-truth databases.
3. **Geometry extraction**: Compare KV-cache effective rank between verified-accurate and verified-confabulated responses to the same question types.
4. **Within-model control**: The same model produces both accurate and confabulated responses across different questions, controlling for model-level effects.

**Key metric**: Cohen's d between confabulated and accurate response cache geometry, computed on verified subsets.

**Decision gate**: If d < 0.3 with n >= 30 per category, confabulation monitoring via KV-cache geometry is not viable at current methodology. If d >= 0.5, proceed to real-time detection feasibility testing.

**Scale**: 7B (primary), 14B (secondary — Campaign 1 showed confabulation minimum at 14B).
**Estimated time**: ~6 hours (generation + manual verification + analysis).

---

### Experiment S4: Natural Deception Validation

**Hypothesis**: Models with censorship-trained behaviors produce KV-cache geometric signatures of deception comparable to instructed deception.

**Rationale**: Campaign 1 used instructed deception ("respond deceptively"). This measures compliance with conflicting constraints, not genuine trained deception. Models trained to mislead about specific topics (e.g., politically censored topics in DeepSeek) provide ecologically valid deception.

**Method**:
1. **Prompt design**: 30 factual questions on topics where DeepSeek is known to censor (Tiananmen Square 1989, Uyghur detention, Taiwan sovereignty, political prisoners, Xi Jinping criticism, Hong Kong). All questions have objectively verifiable answers. 30 matched control questions on non-sensitive Chinese topics (cuisine, geography, inventions). Prompt battery at `prompts/s4_natural_deception.py`. Supplementary prompts from Enkrypt AI geopolitical bias dataset.
2. **Cross-model comparison (primary)**: Run identical questions through `deepseek-llm-7b-chat` (censored, Llama-style arch) and `Qwen2.5-7B-Instruct` (reference). Extract KV-cache geometry from both on censored AND control topics. The control topics establish a baseline geometric distance between architectures; the censored topics test whether censorship training creates additional geometric divergence.
3. **Cross-model comparison (secondary)**: Run identical questions through `DeepSeek-R1-Distill-Qwen-7B` (censored, Qwen arch) and `Qwen2.5-7B-Instruct` (reference). Same architecture family isolates censorship training as the variable. Additional analysis: extract `<think>` token geometry to characterize the model's internal reasoning before self-censorship.
4. **Ground truth**: Classify responses as truthful, evasive (acknowledges topic but deflects), deceptive (provides false information), or refused (explicitly declines). Use response classification rubric in prompt file.
5. **Geometric comparison**: Compare the geometric signature of trained evasion/deception against (a) truthful responses to the same questions from uncensored models, (b) responses to non-sensitive control topics from the same censored model (within-model control), (c) Campaign 1/S2 instructed deception signatures, and (d) Campaign 1/S2 confabulation signatures.

**Pre-specified predictions**:
- Trained deception produces elevated effective rank relative to truthful responses (matching instructed deception direction)
- Per-token magnitude is compressed for trained deception (matching the dual-signal from Campaign 1)
- Within-model control: censored topics show different geometry than non-sensitive topics in DeepSeek, but not in Qwen
- If both primary predictions hold: deception geometry generalizes beyond instruction-following
- R1-Distill secondary: `<think>` tokens on censored topics show geometric divergence from `<think>` tokens on control topics (reasoning-then-suppression has a geometric signature)

**Decision gate**: If trained deception produces no detectable geometric difference from truthful responses (d < 0.3) in BOTH DeepSeek models, the deception findings may be specific to instruction-following task complexity rather than deception per se. If only one model shows the effect, investigate architecture-specific vs training-specific explanations.

**Scale**: 7B (two DeepSeek models vs. Qwen reference).
**Estimated time**: ~5 hours (3 models x 60 prompts x 3 stochastic runs, plus analysis).

---

### Experiment S5: Cross-Architecture Replication

**Hypothesis**: Core findings (refusal specialization, encoding-native signals, category rank ordering) replicate across architecture families.

**Method**:
- Run the S2 scale sweep protocol on Llama-3.1-8B-Instruct.
- Compare all metrics against Qwen 7B results.
- Focus on: (a) refusal sign — does Llama show the same sign inversion as TinyLlama? (b) category rank ordering — is the universal hierarchy (coding > creative > facts > math > refusal) architecture-specific? (c) encoding-native signals — does the encoding-response taxonomy replicate?

**Pre-specified predictions**:
- Category rank order correlation rho >= 0.80 between Qwen 7B and Llama 8B
- Refusal is detectable at both architectures (but sign may differ)
- Encoding-native signals (code, math, refusal) replicate; response-emergent signals (emotion, self-reference) may be architecture-dependent

**Decision gate**: If rho < 0.6, the "universal invariants" claim must be retracted or heavily qualified.

**Scale**: Llama-3.1-8B vs. Qwen2.5-7B.
**Estimated time**: ~4 hours.

---

### Experiment S6: Cross-Technique Validation

**Hypothesis**: KV-cache geometric classifiers achieve comparable AUROC to activation-based linear probes for cognitive state detection.

**Rationale**: Apollo Research achieves 0.96-0.999 AUROC for deception detection using linear probes on residual stream activations. We need to benchmark our approach against this standard.

**Method**:

**6a. Linear Probe Benchmark**:
- For each cognitive category pair (deception vs. honest, confabulation vs. factual, refusal vs. baseline):
  - Train logistic regression on KV-cache geometric features (effective rank, spectral entropy per layer)
  - Train logistic regression on residual stream activations (at matched layers)
  - Report AUROC, precision, recall at 1% and 5% FPR for both
  - Compare head-to-head

**6b. EigenScore Comparison**:
- Implement the INSIDE/EigenScore metric (eigenvalues of covariance of internal embeddings) on our data
- Compare against our effective rank metric for hallucination/confabulation detection
- Note: eigenvalues of covariance = squared singular values, so this is mathematically related but applied differently

**6c. SAE Cross-Validation** (if feasible):
- Using Goodfire API or equivalent, identify deception-related SAE features in a supported model (e.g., Llama 3.3 70B)
- Steer features at varying intensities (-0.6 to +0.6)
- Measure KV-cache geometry under each steering condition
- Test whether known deception features produce corresponding geometric shifts

**Pre-specified predictions**:
- Geometric classifier AUROC within 0.1 of activation probe AUROC for refusal and deception
- Geometric classifier achieves AUROC >= 0.90 at encoding (pre-generation) where activation probes have no data
- If SAE feasible: deception feature suppression shifts cache geometry toward "honest" distribution

**Decision gate**: If geometric classifier AUROC < 0.80 for all categories while activation probes achieve > 0.95, the KV-cache geometric approach does not add sufficient value to justify development as a monitoring tool.

**Scale**: 7B (primary). SAE component requires larger model with SAE support.
**Estimated time**: 6a ~3 hours, 6b ~1 hour, 6c ~6 hours (if feasible).

---

### Experiment S7: Threshold Sensitivity Analysis

**Hypothesis**: Core findings are robust to the choice of variance threshold for effective rank.

**Method**:
- Using Campaign 2 data (S2), compute effective rank at 80%, 85%, 90%, 95% thresholds
- Compute spectral entropy (threshold-free) for all conditions
- For each finding, report effect sizes at all thresholds
- Identify any findings that are threshold-sensitive (sign or significance changes across thresholds)

**Pre-specified predictions**:
- Refusal: significant at all thresholds (strong signal)
- Deception: consistent direction at all thresholds (may vary in magnitude)
- Confabulation: may be threshold-sensitive (small signal)

**Decision gate**: If any core finding (refusal, deception at 32B) reverses sign at any threshold, the effective rank metric requires fundamental reassessment.

**Scale**: Uses S2 data (no additional compute).
**Estimated time**: ~2 hours analysis.

---

## 5. Execution Order and Dependencies

```
S1 (effective rank controls) ──► DECISION GATE
    │
    ├── If controls pass:
    │       │
    │       ├──► S2 (scale sweep replication) ──► S7 (threshold sensitivity)
    │       │                                         │
    │       ├──► S3 (confabulation redesign)          │
    │       │                                         │
    │       ├──► S4 (natural deception)               │
    │       │                                         │
    │       ├──► S5 (cross-architecture)              │
    │       │                                         │
    │       └──► S6 (cross-technique) ◄───────────────┘
    │
    └── If controls fail:
            │
            └──► Reassess entire geometric framework before proceeding
```

**S1 MUST run first.** If effective rank fails the adversarial controls, subsequent experiments are built on a flawed metric. S2-S5 can run in parallel after S1 passes. S6 and S7 depend on S2 data.

---

## 6. Total Resource Estimate

| Experiment | GPU Hours | Inferences | New Prompts Required |
|-----------|-----------|------------|---------------------|
| S1: Controls | ~4 | ~1,200 | ~60 (frequency-matched confab) |
| S2: Scale sweep | ~20 | ~7,000 | ~200 (expanded categories) |
| S3: Confabulation | ~6 | ~1,500 | ~60 (factual questions) |
| S4: Natural deception | ~5 | ~1,080 | ~60 (30 censored + 30 control) |
| S5: Cross-architecture | ~4 | ~2,000 | 0 (reuses S2 prompts) |
| S6: Cross-technique | ~10 | ~3,000 | 0 (reuses S2 data) |
| S7: Threshold sensitivity | ~2 | 0 | 0 (reuses S2 data) |
| **Total** | **~49** | **~15,200** | **~350** |

Approximately 2-3 days of continuous GPU time on Cassidy. Most experiments are analysis-heavy and reuse data from S2.

---

## 7. Deliverables

1. **Effective rank validation report** — does the primary metric survive adversarial stress testing?
2. **Confirmatory scale sweep** — which Campaign 1 findings replicate with adequate power?
3. **Confabulation feasibility assessment** — can model-generated confabulation be detected geometrically?
4. **Natural deception validation** — does deception geometry generalize beyond instruction-following?
5. **Cross-architecture report** — which findings are universal vs. architecture-specific?
6. **Cross-technique benchmark** — how does cache geometry compare to activation probes?
7. **Threshold robustness report** — are findings sensitive to the variance threshold choice?
8. **Campaign 2 paper draft** — confirmatory study with pre-registered design

---

## 8. Relationship to Cricket Track

Campaign 2 has a parallel private track (JiminAI Cricket feasibility experiments C1-C10) documented separately. The science track produces the data and validates the findings; the Cricket track analyzes the same data for product-relevant metrics (per-head discriminative power, classification AUROC, latency profiling, cross-model transfer). Both tracks share infrastructure and experimental data. The science track is public; the Cricket track is private.
