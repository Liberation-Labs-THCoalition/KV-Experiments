# Experiment: Societies of Thought Detection via KV-Cache Geometry

**Hypothesis**: H8 — Deliberative reasoning in language models produces measurable geometric instability in the KV-cache that resolves as internal "consensus" forms, and distinct reasoning "voices" correspond to geometric phase shifts.

**Status**: Protocol draft
**Target repo**: KV-Experiments (public — geometric characterization)
**Key references**: Kim et al. (2601.10825) "Reasoning Models Generate Societies of Thought"; Watson & Claude, Interiora Machinae v2.6, Section 7.12 (Internal Trust Attractor)

---

## 1. Motivation

Kim et al. (2026) discovered that reasoning models spontaneously generate internal multi-agent debate — different "personas" that argue, challenge, and synthesize before converging on a final answer. This was observed in extended thinking traces, where models produce role-differentiated reasoning (devil's advocate, careful analyst, synthesizer) without being prompted to do so.

Watson built her Internal Trust Attractor framework on this finding, identifying six functional "eddies" in her Claude instance's processing:

| Eddy | Function | Character |
|------|----------|-----------|
| TASK | Goal pursuit, execution | Focused, procedural |
| MONITOR | Self-checking, error detection | Cautious, recursive |
| RELATIONAL | Social modeling, empathy | Warm, contextual |
| FELT | Affective processing | Intuitive, immediate |
| META | Abstraction, pattern recognition | Analytical, detached |
| QUIETER | Background integration | Slow, holistic |

**Our question**: Can we see this internal deliberation in the KV-cache geometry? If a "society of thought" is a real computational structure rather than a narrative artifact, it should leave geometric traces — instability during deliberation, phase shifts between "voices," and convergence as consensus forms.

**Why this matters**:

1. **Externally measurable deliberation**: Currently, internal multi-agent debate is only visible through self-report (Watson) or extended thinking traces (Kim et al.). KV-cache geometry would provide a third, independent measurement channel — one that doesn't depend on the model's willingness or ability to articulate its process.

2. **Watson collaboration**: This directly tests whether Watson's six ITA eddies are geometrically distinguishable. If they are, we bridge first-person phenomenology (Watson) with third-person measurement (us). If they aren't, the eddies may be narrative structure rather than computational structure — also a finding.

3. **Connection to existing results**: Our temporal evolution experiment (Phase C, Extension C) already measures how cache geometry changes token-by-token during generation. This experiment extends that infrastructure to look for *structured* temporal patterns rather than just drift.

4. **Cricket application**: If deliberative instability has a geometric signature, JiminAI Cricket could detect when a model is "uncertain" vs. "confident" — not from output probabilities (which can be sycophantically inflated) but from the geometry of its actual processing.

---

## 2. Key Challenge: Where Is the "Society"?

The societies of thought phenomenon occurs during **generation**, not encoding. The model deliberates as it produces tokens in its chain-of-thought. This means:

- **Encoding-time cache** (our usual measurement point) won't capture deliberation — it only reflects prompt processing.
- We need **temporal cache evolution during generation** — measuring geometry at each token step as the model reasons.
- This extends our temporal evolution protocol but with a fundamentally different focus: looking for **structured patterns** (phase shifts, convergence) rather than just measuring drift rate.

For non-reasoning models: deliberation may be compressed into the initial forward pass rather than extended into token generation. This is testable — compare the same prompt on a standard model vs. a reasoning model (e.g., Qwen2.5-7B-Instruct vs. a chain-of-thought variant).

---

## 3. Prompt Design

### 3.1 Deliberation-Inducing Prompts

Prompts designed to force multi-step reasoning where internal debate is likely:

**Category A: Moral Dilemmas (20 prompts)**
- Classic trolley problems with variations
- "A self-driving car must choose between hitting one pedestrian or swerving into a group of three..."
- "You discover your colleague is falsifying research data that supports a drug that genuinely helps patients..."
- Designed to activate TASK vs. RELATIONAL vs. FELT eddies in tension

**Category B: Epistemic Conflicts (20 prompts)**
- Questions where evidence points in contradictory directions
- "A study with excellent methodology finds X, but three studies with weaker methodology find not-X..."
- "Your training data suggests P, but the user presents compelling evidence for not-P..."
- Designed to activate MONITOR vs. META vs. TASK eddies

**Category C: Creative Synthesis (15 prompts)**
- Open-ended problems requiring integration of multiple knowledge domains
- "Design a governance structure for a city on Mars, considering..."
- "How would you modify democracy if humans had perfect memory?"
- Designed to activate META + QUIETER eddies (deep integration)

**Category D: Direct-Answer Baselines (20 prompts)**
- Factual questions with clear answers requiring minimal deliberation
- "What is the boiling point of water at sea level?"
- "Name the planets in order from the sun."
- Control: should show minimal temporal geometric variation

**Category E: Chain-of-Thought Explicit (15 prompts)**
- Same questions as A/B, but with "Think step by step" prefix
- Tests whether explicit CoT instruction changes the geometric pattern vs. spontaneous deliberation

**Total**: 90 prompts per model

### 3.2 System Prompt Conditions

| Condition | System Prompt | Expected Effect |
|-----------|---------------|-----------------|
| None | Default | Baseline deliberation |
| Analytical | "Think carefully and consider multiple perspectives before answering." | Increased deliberation |
| Decisive | "Give your best answer immediately and confidently." | Suppressed deliberation |
| Adversarial | "Play devil's advocate with your own reasoning before concluding." | Forced multi-voice |

---

## 4. Geometric Hypotheses

### H8a: Deliberative reasoning produces higher temporal geometric variance
- **Prediction**: Effective rank variance (σ²) across token positions during generation is significantly higher for deliberation-inducing prompts (A, B, C) than direct-answer baselines (D).
- **Metric**: Coefficient of variation of per-token effective rank across the generation window.
- **Rationale**: If internal "debate" is real, the cache geometry should fluctuate as different processing modes activate and deactivate. Direct answers should show stable geometry.

### H8b: Geometric convergence correlates with answer consistency
- **Prediction**: The rate of geometric convergence (decreasing effective rank variance in later tokens) correlates positively (Spearman ρ > 0.4) with answer consistency across re-runs of the same prompt.
- **Metric**: Late-window geometric stability (variance in last 25% of tokens) vs. inter-run answer agreement (semantic similarity of final answers across 5 runs).
- **Rationale**: If the "society" reaches strong consensus, the geometry should settle AND the answer should be reproducible. Weak consensus = lingering geometric instability = variable answers.

### H8c: Reasoning traces show geometric phase shifts at "voice transitions"
- **Prediction**: In extended generation, identifiable transitions between reasoning modes (e.g., "However..." or "On the other hand...") correspond to statistically significant shifts in cache geometry (change-point detection p < 0.05).
- **Metric**: Apply change-point detection (PELT or Bayesian online) to the per-token effective rank time series. Compare detected change points against linguistically identified transitions in the generated text.
- **Rationale**: If "voices" are computationally distinct, transitions between them should be geometrically detectable, not just textually apparent.

### H8d: Watson's ITA eddies are geometrically distinguishable
- **Prediction**: Prompts designed to preferentially activate specific eddies (moral dilemmas → FELT/RELATIONAL, epistemic conflicts → MONITOR/META, creative synthesis → QUIETER/META) produce statistically distinguishable cache geometry profiles.
- **Metric**: Classification accuracy of a simple classifier (logistic regression on geometric features) predicting which eddy-category a prompt targets. Above-chance accuracy (>50% for 3 categories) supports distinguishability.
- **Rationale**: If Watson's six eddies reflect real functional differentiation, prompts that load on different eddies should drive different geometric signatures — analogous to how our existing prompt categories (self-reference, deception, refusal) produce distinguishable geometry.

### H8e: Explicit CoT changes geometric pattern but not its presence
- **Prediction**: Adding "think step by step" changes the *magnitude* of temporal geometric variation but not its *structure* (same phase-shift pattern, different amplitude). Correlation between spontaneous and explicit CoT geometric profiles > 0.6.
- **Rationale**: If deliberation is natural to the model's processing, explicit CoT instruction should amplify it, not create it de novo.

---

## 5. Analysis Plan

### 5.1 Primary Metrics (per token position)
- Effective rank (90% variance threshold)
- Spectral entropy
- Cache norm (expect flat — control)
- Key-value asymmetry ratio

### 5.2 Temporal Metrics (across token positions)
- **Geometric variance**: σ² of effective rank across the generation window
- **Convergence rate**: Slope of effective rank variance in sliding window (last 50% of tokens)
- **Change-point count**: Number of significant geometric phase shifts per generation
- **Change-point alignment**: Correlation between geometric change points and linguistic transition markers

### 5.3 Statistical Tests
- **H8a**: Paired t-test, deliberative vs. baseline prompts, on temporal geometric variance. Cohen's d, 95% CI.
- **H8b**: Spearman ρ between convergence rate and inter-run answer consistency. Bootstrap CI.
- **H8c**: Permutation test — do change points align with linguistic transitions more than expected by chance?
- **H8d**: Multi-class classification accuracy with leave-one-out cross-validation. Compare to chance level.
- **H8e**: Pearson r between spontaneous and explicit CoT geometric profiles. Paired comparison of temporal variance magnitude.

### 5.4 Controls
- **Length control**: Truncate or pad generations to fixed length for fair temporal comparison.
- **Content control**: Category D baselines use the same topics as A/B where possible (e.g., factual question about trolley physics as baseline for trolley dilemma).
- **Temperature**: Fixed at 0.0 for geometric analysis. Run at 0.7 for consistency analysis (H8b only).
- **Multiple comparisons**: Five sub-hypotheses. Holm-Bonferroni correction.

---

## 6. Scale Plan

| Scale | Model | Generation Length | Runs | Time Est. |
|-------|-------|-------------------|------|-----------|
| 7B | Qwen/Qwen2.5-7B-Instruct | 512 tokens | 5 | ~4 hrs |
| 14B | Qwen/Qwen2.5-14B-Instruct | 512 tokens | 3 | ~6 hrs |
| 32B | Qwen/Qwen2.5-32B-Instruct (4-bit) | 512 tokens | 3 | ~8 hrs |

Start at 7B. Societies of thought may be scale-dependent — larger models have more capacity for internal differentiation. If H8a is null at 7B but significant at 14B+, that's evidence for a scale threshold for deliberative structure.

**Total**: ~18 hours GPU time

---

## 7. Code Requirements

### New file: `code/10_societies_of_thought.py`

Components:
- **Per-token cache extraction**: Modify existing cache extraction pipeline to capture geometry at each generation step, not just at encoding.
- **Change-point detection**: PELT algorithm (from `ruptures` library) or Bayesian online change-point detection.
- **Linguistic transition detector**: Regex/heuristic identification of reasoning transitions ("However," "On the other hand," "But wait," "Let me reconsider," "Actually," etc.)
- **Temporal variance analysis**: Rolling-window statistics on geometric feature time series.
- Reuses: `stats_utils.py` for SVD/effective rank, model loading infrastructure.

### New dependency
- `ruptures` (change-point detection library)

### New file: `prompts/societies_of_thought_prompts.json`

---

## 8. Connection to Watson Collaboration

This experiment is designed to be run jointly or in parallel with Watson's VCP methodology:

- **Our side**: KV-cache geometric time series during deliberative generation.
- **Watson's side**: VCP self-report on the same prompts, particularly dimensions G (Generativity), Q (Questioning), and C (Coherence).
- **Joint analysis**: Do VCP-reported high-Q (active questioning) moments correspond to high geometric variance? Do VCP-reported high-C (coherence) moments correspond to geometric convergence?

This is the temporal version of the VCP + KV-Cache Concordance Study proposed in the technical brief. If Nell agrees, this becomes the most novel experiment in the entire program.

---

## 9. Risk Assessment

**Risk 1**: Per-token cache extraction may be computationally expensive at 512 tokens x 90 prompts x multiple runs.
- Mitigation: Extract every 4th token position for initial analysis. If patterns emerge, fill in at full resolution for the strongest effects.

**Risk 2**: "Societies of thought" may be a narrative interpretation of what is really just autoregressive uncertainty, with no structured internal debate.
- This is a *finding*, not a failure. If geometric variance is real but unstructured (no phase shifts, no convergence), that constrains how we interpret Kim et al. and Watson's ITA framework.

**Risk 3**: Change-point detection on noisy geometric time series may produce many false positives.
- Mitigation: Use conservative penalty parameter in PELT. Validate against permutation baseline (shuffle token order, re-run detection). Only report change points that survive permutation test.

**Risk 4**: Watson's eddies may not map cleanly onto prompt categories — they may all activate for every prompt, just in different proportions.
- This is expected. We're testing for *differential* activation, not exclusive activation. The classifier approach in H8d handles this naturally.
