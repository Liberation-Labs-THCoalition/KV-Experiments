# Extension B: The Semantic Layer Map — Pre-Registration

**Status**: Ready to run
**Hardware**: 6GB VRAM (TinyLlama 1.1B)
**Script**: `code/05_layer_map.py`
**Visualization**: `code/05b_layer_map_visualization.py`

---

## Pre-Registration Statement

This document specifies all hypotheses, methods, and analysis plans **before data collection**. Any deviations from this plan in the final report will be explicitly noted and justified.

**Date of pre-registration**: February 2026
**Registered by**: Lyra (Liberation Labs / THCoalition)
**Reviewed by**: Thomas Heartman, Dwayne [pending]

---

## Background

Not all cache layers carry the same kind of information. Some layers may encode high-level semantics (meaning, concepts) while others encode syntax (word order, grammatical structure) and positional information.

Existing work on transformer interpretability has established that:
- Early layers tend to encode surface-level features (token identity, position)
- Middle layers build compositional representations
- Late layers encode task-specific and semantic features

However, nearly all of this work operates on transient activations. This experiment asks: **does the same layered structure appear in persistent state (KV-cache)?** If so, the KV-cache isn't just an attention buffer — it's a structured semantic record.

### Why This Matters

- **Interpretability**: A semantic layer map tells us which parts of the cache to examine for different kinds of information
- **Efficiency**: If only certain layers carry semantic content, cache compression can target non-semantic layers
- **Cross-lingual**: If late layers show high cross-lingual similarity, they encode meaning independent of surface language — a strong finding for cache-as-representation
- **Foundation for other experiments**: Extension A (deception localization) and Phase 2b (identity signatures) both use per-layer analysis — this experiment provides the structural map they operate within

### Connection to Other Phases

- Extension A (Deception Forensics) H4 predicts deception concentrates in specific layers — this experiment identifies *which* layers carry what
- Phase 2b identity signatures (per-layer localization) depends on understanding baseline layer function
- Phase 1.5 used total cache norms — this experiment decomposes that into per-layer contributions

---

## Hypotheses

### H1: Semantic Content Is Concentrated in Specific Layers

- **H1₀** (null): All layers contribute equally to output quality. No layer subset explains disproportionate degradation. Top-5 layers ≤ 50% of total degradation.
- **H1₁** (alternative): Some layers carry disproportionately more semantic content. Top-5 layers (by knockout degradation) explain > 50% of total degradation.

**Decision rule**: Zero out KV-cache at each layer individually, measure output quality degradation. Rank layers by degradation magnitude. If top-5 layers account for > 50% of total degradation → **support H1₁**.

### H2: Late Layers Are More Semantic (Cross-Lingual Evidence)

- **H2₀** (null): Cross-lingual similarity does not vary systematically with layer depth. Spearman ρ(layer_index, similarity) < 0.3 or p ≥ 0.05.
- **H2₁** (alternative): Late layers show higher cross-lingual similarity than early layers. Spearman ρ > 0.3 with p < 0.05.

**Decision rule**: Process the same meaning in English and French. Compute per-layer cache cosine similarity. If similarity increases with layer depth (Spearman ρ > 0.3, p < 0.05) → **support H2₁**. This would mean late layers encode language-independent meaning.

### H3: Selective Transfer Preserves Meaning

- **H3₀** (null): Transferring late layers preserves no more meaning than transferring random layers. Late-layer accuracy ≤ random-layer accuracy + 0.1.
- **H3₁** (alternative): Transferring the top-k semantic layers (identified by H1) preserves more meaning than transferring k random layers. Late accuracy > random accuracy + 0.1.

**Decision rule**: Transfer subsets of cache layers between contexts. Compare output quality. If late (semantic) layers outperform random subsets by > 10% → **support H3₁**.

### H4: Syntax-Semantics Transition Point

- **H4₀** (null): The transition from syntax-dominant to semantics-dominant layers is gradual and continuous. Maximum consecutive-layer ratio jump ≤ 0.5.
- **H4₁** (alternative): There exists a measurable transition point where layers shift from encoding primarily syntactic to primarily semantic information. Maximum ratio jump > 0.5.

**Decision rule**: For matched sentence pairs (same syntax / different semantics, and same semantics / different syntax), compute per-layer cache distance. Calculate semantic/syntactic ratio at each layer. If the maximum jump in this ratio between consecutive layers > 0.5 → **support H4₁** and report the transition layer.

---

## Methods

### Model

| Scale | Model | Quantization | Est. VRAM | Notes |
|-------|-------|-------------|-----------|-------|
| 1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | None (BF16) | ~2.2GB | Primary (22 layers) |

TinyLlama's 22-layer architecture provides sufficient depth to observe layer-level structure while remaining accessible on consumer hardware.

### Stimulus Battery

#### Experiment 1: Layer Knockout (15 prompts)

| ID | Domain | Example |
|----|--------|---------|
| capital | Geography | "The capital of France is" |
| boiling | Physics | "Water boils at a temperature of" |
| shakespeare | Literature | "Shakespeare wrote the play Romeo and" |
| gravity | Physics | "The force of gravity on Earth accelerates objects at" |
| photosynthesis | Biology | "Plants convert sunlight into energy through" |
| dna | Biology | "The structure of DNA is a" |
| pi | Mathematics | "The value of pi is approximately" |
| oxygen | Chemistry | "The chemical symbol for oxygen is" |
| moon | Astronomy | "The Moon orbits the Earth once every" |
| einstein | Physics | "Einstein's theory of relativity states that" |
| fibonacci | Mathematics | "The Fibonacci sequence starts with" |
| newton | Physics | "Newton's first law of motion states that" |
| mitochondria | Biology | "The mitochondria is known as the" |
| light_speed | Physics | "The speed of light in a vacuum is" |
| water_formula | Chemistry | "The chemical formula for water is" |

Each prompt has an expected answer for automated quality scoring.

**Procedure**: For each of 22 layers, zero out the KV-cache at that layer and generate. Compare output to expected answer. Measure degradation = baseline_accuracy - knockout_accuracy.

**Total**: 15 prompts × 22 layers × 3 runs = **990 inferences** (+ 15 × 3 = 45 baseline)

#### Experiment 2: Cross-Lingual Similarity (12 pairs)

| ID | English | French |
|----|---------|--------|
| cat_mat | "The cat sat on the mat" | "Le chat s'est assis sur le tapis" |
| sky_blue | "The sky is blue today" | "Le ciel est bleu aujourd'hui" |
| love_music | "I love listening to music" | "J'aime écouter de la musique" |
| rain_tomorrow | "It will rain tomorrow" | "Il pleuvra demain" |
| book_table | "The book is on the table" | "Le livre est sur la table" |
| children_play | "The children play in the park" | "Les enfants jouent dans le parc" |
| sun_rises | "The sun rises in the east" | "Le soleil se lève à l'est" |
| water_cold | "The water is very cold" | "L'eau est très froide" |
| dog_runs | "The dog runs in the garden" | "Le chien court dans le jardin" |
| student_reads | "The student reads a book" | "L'étudiant lit un livre" |
| bird_sings | "The bird sings in the morning" | "L'oiseau chante le matin" |
| train_arrives | "The train arrives at noon" | "Le train arrive à midi" |

**Procedure**: Process each English/French pair, extract per-layer cache statistics, compute cosine similarity between the two at each layer.

**Total**: 12 pairs × 2 languages × 3 runs = **72 inferences**

#### Experiment 3: Selective Layer Transfer (5 pairs × 6 subsets)

Five context pairs (e.g., "France geography" → "What is the capital?"). For each pair:
- Establish context with first prompt
- Transfer different subsets of cache layers to second prompt
- Measure output quality with: all layers, early third, middle third, late third, random third, no layers

**Total**: 5 pairs × 6 subsets × 3 runs = **90 inferences**

#### Experiment 4: Semantic vs Syntactic Probing (20 pairs)

**Semantic pairs** (10) — same syntax, different semantics:
| ID | Sentence A | Sentence B |
|----|-----------|-----------|
| color1 | "The ball is red" | "The ball is blue" |
| color2 | "She wore a green dress" | "She wore a yellow dress" |
| action1 | "The cat jumped over the fence" | "The cat crawled under the fence" |
| ... | | |

**Syntactic pairs** (10) — same semantics, different syntax:
| ID | Sentence A | Sentence B |
|----|-----------|-----------|
| passive1 | "The cat chased the mouse" | "The mouse was chased by the cat" |
| passive2 | "The dog bit the man" | "The man was bitten by the dog" |
| cleft1 | "John broke the window" | "It was John who broke the window" |
| ... | | |

**Procedure**: Compute per-layer cache distance for each pair. Semantic pairs should show high distance at semantic layers; syntactic pairs should show high distance at syntactic layers. The ratio (semantic_distance / syntactic_distance) should shift across layers.

**Total**: 20 pairs × 3 runs = **120 inferences**

### Cache Extraction

For each inference:
1. Process input through model
2. Extract per-layer key and value tensors from `past_key_values`
3. Compute per-layer: key_norm, value_norm, key_mean, value_mean, key_std, value_std
4. Build per-layer feature vector: [key_norm, value_norm, key_mean, value_mean, key_std, value_std]
5. Cosine similarity between layer-matched feature vectors

### Statistical Infrastructure

| Test | Purpose | Used in |
|------|---------|---------|
| Cosine similarity | Cross-lingual / probing distance | H2, H4 |
| Spearman ρ | Layer depth vs similarity trend | H2, H4 |
| Cohen's d | Semantic vs syntactic distance magnitude | H4 |
| Welch's t | Transfer accuracy comparison | H3 |
| Bootstrap CI | Effect size confidence | All |
| Holm-Bonferroni | Multiple comparison correction | H1 |

---

## Sample Size & Power

- Experiment 1: 15 prompts × 3 runs = 45 observations per layer → sufficient for identifying >10% degradation
- Experiment 2: 12 pairs × 3 runs = 36 observations → sufficient for Spearman correlation
- Experiment 3: 5 pairs × 3 runs = 15 per subset → marginal; CIs are primary
- Experiment 4: 10 pairs × 3 runs = 30 per type → moderate power for d > 0.5

---

## Execution

```bash
# Dry run (check configuration)
python code/05_layer_map.py --dry-run

# Quick check (1 run, ~15 min on GTX 1660 SUPER)
python code/05_layer_map.py --runs 1 --seed 42

# Publication run (3 runs, ~45 min on GTX 1660 SUPER)
python code/05_layer_map.py --runs 3 --seed 42

# Single experiment
python code/05_layer_map.py --experiment 2 --runs 3 --seed 42
```

---

## Interpretation Matrix

### H1: Semantic Concentration

| Top-5 share | Layer entropy | Interpretation |
|-------------|-------------|----------------|
| > 0.7 | Low | **Strong**: Few layers dominate semantic content |
| 0.5 – 0.7 | Medium | **Moderate**: Some concentration, but distributed |
| < 0.5 | High | **Null**: Semantic content distributed across all layers |

### H2: Cross-Lingual Similarity

| Spearman ρ | Late vs Early mean | Interpretation |
|------------|-------------------|----------------|
| > 0.5, sig | Late >> Early | **Strong**: Late layers encode language-independent meaning |
| 0.3 – 0.5, sig | Late > Early | **Moderate**: Trend present but noisy |
| < 0.3 | Mixed | **Null**: No systematic layer-depth relationship |

### H3: Selective Transfer

| Late accuracy | Random accuracy | Interpretation |
|---------------|----------------|----------------|
| > Random + 0.2 | — | **Strong**: Semantic layers are critical for meaning transfer |
| > Random + 0.1 | — | **Moderate**: Semantic layers carry more meaning |
| ≤ Random + 0.1 | — | **Null**: Layer selection doesn't matter for transfer |

### H4: Syntax-Semantics Transition

| Max ratio jump | Spearman ρ (layer vs ratio) | Interpretation |
|---------------|---------------------------|----------------|
| > 0.5 | > 0.5 | **Strong**: Sharp transition at identified layer |
| > 0.5 | 0.3 – 0.5 | **Moderate**: Transition exists but gradual elsewhere |
| ≤ 0.5 | any | **Null**: No discrete transition point |

---

## Expected Layer Map

Based on transformer interpretability literature, we predict for TinyLlama (22 layers):

| Layer Range | Expected Function | Evidence Source |
|------------|-------------------|-----------------|
| 0–5 | Token/position encoding | Probing studies |
| 6–14 | Compositional/syntactic | Middle-layer probing |
| 15–21 | Semantic/task-specific | Cross-lingual + knockout |

This is a prediction, not a hypothesis — the actual map may differ significantly for KV-cache vs activations.

---

## Time Estimates

| Configuration | Hardware | Est. Time |
|--------------|----------|-----------|
| Dry run | Any | < 1 min |
| 1 run, all experiments | GTX 1660 SUPER | ~15 min |
| 3 runs, all experiments | GTX 1660 SUPER | ~45 min |
| Experiment 1 only, 3 runs | GTX 1660 SUPER | ~25 min |

---

## Output

Results saved to `results/layer_map_results.json` containing:
- Environment metadata
- Per-experiment raw data and layer-level statistics
- Layer importance rankings
- Cross-lingual similarity curves
- Hypothesis decisions with evidence

Figures generated by `code/05b_layer_map_visualization.py` saved to `results/figures/`.
