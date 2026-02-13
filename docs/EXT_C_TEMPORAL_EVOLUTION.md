# Extension C: Temporal Cache Evolution — Pre-Registration

**Status**: Ready to run
**Hardware**: 6GB VRAM (TinyLlama 1.1B — runs on local GPU)
**Script**: `code/06_temporal_evolution.py`
**Visualization**: `code/06b_temporal_visualization.py`

---

## Pre-Registration Statement

This document specifies all hypotheses, methods, and analysis plans **before data collection**. Any deviations from this plan in the final report will be explicitly noted and justified.

**Date of pre-registration**: February 2026
**Registered by**: Lyra (Liberation Labs / THCoalition)
**Reviewed by**: Thomas Heartman, Dwayne [pending]

---

## Background

The KV-cache grows with each token processed. Early tokens are computed once with limited context; later tokens have full attention over everything before them. This raises a fundamental question: **does the cache get "richer" over time?**

Three processes might govern temporal cache evolution:
1. **Enrichment**: Later positions encode richer representations (more context → more information)
2. **Fatigue**: Beyond some threshold, attention dilution degrades representation quality
3. **Content sensitivity**: The rate of cache growth depends on information density

If cache statistics evolve in content-dependent ways, the KV-cache is more than a static buffer — it's a dynamic record that reflects what the model has *processed*, not just what it has *seen*.

### Why This Matters

- **Conversation dynamics**: If cache quality degrades at long context lengths, this has practical implications for long-conversation systems
- **Topic detection**: If topic shifts produce discontinuities in cache statistics, the cache itself becomes a topic-segmentation signal
- **Information density**: If informative text produces faster cache growth than repetitive text, the cache encodes a proxy for information content
- **Theoretical**: The enrichment-vs-fatigue question tells us whether attention-based context integration has diminishing returns in persistent state

### Connection to Other Phases

- Phase 1.5 treated cache as a static snapshot per prompt — this experiment adds the temporal dimension
- Extension B (Layer Map) identifies which layers carry semantic content — temporal evolution may differ by layer
- Phase 2b identity signatures rely on prompt-level cache — if cache evolves within a prompt, which position do we sample?

---

## Hypotheses

### H1: Representational Enrichment

- **H1₀** (null): Cache statistics (norm, variance) do not increase monotonically with sequence position. Spearman ρ(position, norm) ≤ 0.5.
- **H1₁** (alternative): Cache statistics increase with position (later tokens have richer representations). Spearman ρ > 0.5.

**Decision rule**: Process long text, extract cache at ~20 evenly-spaced positions, compute per-token normalized norm at each position. If Spearman ρ(position, norm_per_token) > 0.5 → **support H1₁**.

### H2: Context Window Fatigue

- **H2₀** (null): Cache growth rate is constant across the sequence. Second-half slope ≥ first-half slope × 0.5.
- **H2₁** (alternative): Cache growth rate slows in the second half of the sequence (attention dilution). Second-half slope < first-half slope × 0.5.

**Decision rule**: Compute per-token norm growth rate in first half vs second half. If second-half rate < 50% of first-half rate → **support H2₁** (fatigue detected).

Note: H1 and H2 are compatible — enrichment can be true overall while fatigue appears in the later portion.

### H3: Topic Shift Detection

- **H3₀** (null): Topic shifts are not detectable in cache statistics. No detected peaks align with known shift positions.
- **H3₁** (alternative): Topic shifts produce measurable discontinuities in cache norm derivatives. At least one detected peak is within 10% of a known shift position.

**Decision rule**: Process multi-topic text with known shift points. Compute first derivative of per-token cache norm. Apply peak detection. If any detected peak falls within 10% of text length from a known shift point → **support H3₁**.

### H4: Content-Dependent Cache Growth

- **H4₀** (null): Cache growth rate does not depend on information content. No significant difference between informative and repetitive text. p ≥ 0.05 (t-test on growth rates).
- **H4₁** (alternative): Informative text produces faster cache growth than repetitive text. p < 0.05 and mean(informative_growth) > mean(repetitive_growth).

**Decision rule**: Compare cache growth rates for factual (high-information) and repetitive (low-information) text. If t-test p < 0.05 and direction is informative > repetitive → **support H4₁**.

---

## Methods

### Model

| Scale | Model | Quantization | Est. VRAM | Notes |
|-------|-------|-------------|-----------|-------|
| 1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | None (BF16) | ~2.2GB | Primary |

### Text Corpora

#### Factual Text (~108 tokens)

High information density passage about general relativity:
> *"The theory of general relativity, published by Albert Einstein in 1915, fundamentally changed our understanding of gravity... [covers gravitational waves, LIGO detection, multi-messenger astronomy]"*

**Properties**: Dense factual content, diverse vocabulary, each sentence adds new information.

#### Repetitive Text (~90 tokens)

Low information density:
> *"The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. [repeated 30 times]"*

**Properties**: Minimal information per token, high redundancy, serves as information-poor baseline.

#### Creative Text (~145 tokens)

High information density, metaphorical language:
> *"In the garden of forking paths, time itself becomes a labyrinth... [covers temporal paradoxes, consciousness, identity]"*

**Properties**: Rich vocabulary, metaphorical concepts, high semantic novelty per token.

#### Topic Shift Text (~173 tokens)

Three distinct topics with known shift points:
1. **Weather / Nature** (~33%): Morning light, rain forecast, garden, autumn leaves
2. **Photography / Cameras** (~33%): Camera settings, lenses, technique, composition
3. **Quantum Computing** (~33%): Qubits, superposition, error correction, algorithms

**Known shift points**: ~33% and ~66% of total text length.

### Experiments

#### Experiment 1: Cache Trajectory

**Procedure**:
1. For each of 3 texts (factual, creative, repetitive):
2. Tokenize fully
3. Process incrementally at ~20 evenly-spaced positions (every N tokens)
4. At each position, extract full KV-cache
5. Compute: total_key_norm, total_value_norm, key_norm_per_token, mean_layer_key_norm, var_layer_key_norm, mean_key_std
6. Build trajectory: position → cache_statistics

**Metrics**: Spearman ρ(position, norm_per_token) for H1; slope comparison for H2.

**Total**: 3 texts × ~20 positions × 3 runs = **~180 inferences**

#### Experiment 2: Content Comparison

**Procedure**:
1. Compute growth rate for each text: (late_norm - early_norm) / early_norm
2. Compare across text types: factual vs repetitive, creative vs repetitive

**Metrics**: T-test between growth rates, Cohen's d, bootstrap CI.

**Total**: 3 texts × 2 endpoints × 3 runs = **18 inferences** (subset of Experiment 1)

#### Experiment 3: Topic Shift Detection

**Procedure**:
1. Process topic-shift text incrementally at ~50 evenly-spaced positions
2. Compute per-token norm at each position
3. Apply Gaussian smoothing (σ = 2) to reduce noise
4. Compute first derivative (gradient) of smoothed norms
5. Apply peak detection (`scipy.signal.find_peaks`) on absolute gradient
6. Compare detected peak positions to known shift positions
7. A detection is "correct" if within 10% of text length from a known shift

**Metrics**: Number of correct detections, detection distance from known shifts.

**Total**: 1 text × ~50 positions × 3 runs = **~150 inferences**

### Cache Extraction

For each position in the sequence:
1. Process tokens up to that position
2. Extract `past_key_values` — tuple of (key, value) tensors per layer
3. Compute:
   - `total_key_norm`: L2 norm of all key tensors concatenated
   - `total_value_norm`: L2 norm of all value tensors
   - `key_norm_per_token`: total_key_norm / number_of_tokens (removes trivial length scaling)
   - `mean_layer_key_norm`: mean of per-layer key norms
   - `var_layer_key_norm`: variance of per-layer key norms
   - `mean_key_std`: mean of per-layer key standard deviations

### Statistical Infrastructure

| Test | Purpose | Used in |
|------|---------|---------|
| Spearman ρ | Monotonic trend detection | H1, H2 |
| Bootstrap CI | Growth rate confidence intervals | H2, H4 |
| Welch's t | Growth rate comparison | H4 |
| Cohen's d | Effect size for growth rate difference | H4 |
| Gaussian smoothing | Noise reduction for derivative analysis | H3 |
| Peak detection | Topic shift identification | H3 |

---

## Sample Size & Power

- Trajectory experiments: ~20 positions × 3 runs = 60 data points per text → sufficient for Spearman correlation
- Content comparison: 3 runs per text type → limited; CIs and effect sizes are primary decision tools
- Topic shift: 3 runs with ~50 positions each → detection is binary per run; report detection rate across runs

**Note**: This experiment is more exploratory than confirmatory. The 3-text comparison is inherently limited (3 samples per condition). We frame it as hypothesis-generating for the content-dependence question, with rigorous statistics within each text's trajectory.

---

## Execution

```bash
# Dry run (check configuration)
python code/06_temporal_evolution.py --dry-run

# Quick check (1 run, ~5 min on GTX 1660 SUPER)
python code/06_temporal_evolution.py --runs 1 --seed 42

# Publication run (3 runs, ~15 min on GTX 1660 SUPER)
python code/06_temporal_evolution.py --runs 3 --seed 42

# Single experiment with verbose output
python code/06_temporal_evolution.py --experiment 1 --runs 3 --seed 42 --verbose
```

---

## Interpretation Matrix

### H1: Enrichment

| Spearman ρ | Text type | Interpretation |
|------------|-----------|----------------|
| > 0.7 | All three | **Strong**: Universal enrichment across content types |
| > 0.5 | At least one | **Moderate**: Enrichment present but content-dependent |
| ≤ 0.5 | All three | **Null**: No monotonic enrichment |

### H2: Fatigue

| Second/First slope ratio | Interpretation |
|--------------------------|----------------|
| < 0.3 | **Strong fatigue**: Growth nearly stops in second half |
| 0.3 – 0.5 | **Moderate fatigue**: Growth slows substantially |
| 0.5 – 0.8 | **Mild fatigue**: Some slowing but growth continues |
| > 0.8 | **No fatigue**: Growth rate is sustained |

### H3: Topic Shift Detection

| Detections / Known shifts | Distance accuracy | Interpretation |
|--------------------------|-------------------|----------------|
| 2/2 | Both < 5% | **Strong**: Cache derivatives detect topic boundaries precisely |
| 1/2 or 2/2 loose | Any < 10% | **Moderate**: Partial detection with some alignment |
| 0/2 | — | **Null**: Cache derivatives don't detect topic shifts |

### H4: Content-Dependent Growth

| Cohen's d | p-value | Direction | Interpretation |
|-----------|---------|-----------|----------------|
| > 0.8 | < 0.05 | Info > Rep | **Strong**: Information density drives cache growth |
| 0.5 – 0.8 | < 0.05 | Info > Rep | **Moderate**: Detectable content effect |
| < 0.5 | ≥ 0.05 | — | **Null**: Growth rate independent of content |

---

## Potential Confounds

1. **Token count vs information**: Repetitive text may tokenize differently than factual text. We normalize by token count (key_norm_per_token) to control for this.
2. **Vocabulary novelty**: Factual text has more unique tokens. This could drive norm differences independent of "information." The creative text serves as a control — different vocabulary profile from factual but similarly high information.
3. **Text length**: Texts differ in total token count. We compare growth *rates* (slopes), not absolute values.
4. **Model familiarity**: TinyLlama may have seen the factual content in training. The creative text is novel (written for this experiment) and serves as a control for training-set contamination.

---

## Time Estimates

| Configuration | Hardware | Est. Time |
|--------------|----------|-----------|
| Dry run | Any | < 1 min |
| 1 run, all experiments | GTX 1660 SUPER | ~5 min |
| 3 runs, all experiments | GTX 1660 SUPER | ~15 min |

This is the most hardware-efficient extension — runs entirely on local GPU.

---

## Output

Results saved to `results/temporal_evolution_results.json` containing:
- Environment metadata
- Per-text trajectory data (position × cache statistics)
- Growth rate comparisons with statistical tests
- Topic shift detection results
- Hypothesis decisions with evidence

Figures generated by `code/06b_temporal_visualization.py` saved to `results/figures/`.
