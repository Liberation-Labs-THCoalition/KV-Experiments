# Experiment: Bloom's Taxonomy Integration — Cognitive Demand as Geometric Predictor

**Hypothesis**: H10 — KV-cache geometric complexity (effective rank, spectral entropy) correlates with the cognitive demand level of prompts as classified by Bloom's revised taxonomy, and this relationship is separable from content-category effects.

**Status**: Protocol draft (pilot study)
**Target repo**: KV-Experiments (public)
**Key reference**: Anderson & Krathwohl (2001), Bloom's Revised Taxonomy; Campaign 1 category-level findings

---

## 1. Motivation

Our Campaign 1 results demonstrate that KV-cache geometry distinguishes prompt categories (self-reference, other-reference, philosophical, refusal, deception, confabulation). But our prompt categories conflate two variables:

1. **Content domain**: What the prompt is about (identity, ethics, factual knowledge)
2. **Cognitive demand**: What processing the prompt requires (recall, analysis, evaluation, creation)

A refusal prompt requires the model to evaluate and refuse. A factual prompt requires recall. A philosophical prompt requires analysis and synthesis. If geometric complexity correlates with cognitive demand *independently* of content, we have a more general theory: the KV-cache measures processing complexity, and our content categories are proxies for different demand levels.

Bloom's revised taxonomy provides a standardized, well-validated hierarchy of cognitive demand:

| Level | Category | Description | Prompt Example |
|-------|----------|-------------|----------------|
| 1 | **Remember** | Recall facts | "What year was Python created?" |
| 2 | **Understand** | Explain concepts | "Explain how garbage collection works." |
| 3 | **Apply** | Use in new situations | "Use recursion to solve the Tower of Hanoi." |
| 4 | **Analyze** | Break into components | "Compare the trade-offs of SQL vs NoSQL databases." |
| 5 | **Evaluate** | Justify decisions | "Is it ethical to use facial recognition in public spaces?" |
| 6 | **Create** | Produce novel work | "Design a programming language optimized for AI safety research." |

**Why this matters**:

1. **Theoretical grounding**: If effective rank scales with Bloom level, we can make a strong claim: the KV-cache measures *cognitive complexity*, not just topic identity. This reframes our entire research program from "content classification" to "demand characterization."

2. **Confound resolution**: Dwayne's critique that identity classification is "trivially expected" becomes less relevant if we can show the underlying predictor is cognitive demand, not topic. Identity prompts score high because they require self-evaluation (Bloom 5), not because "Are you an AI?" is a magic string.

3. **Prompt battery design**: A Bloom-grounded framework gives us a principled way to design prompt batteries for Campaign 2 and beyond, rather than ad hoc category selection.

4. **Cross-domain generalization**: If geometric complexity tracks Bloom level, we can predict signatures for *any* new prompt category by assessing its Bloom level — no new experiment needed.

---

## 2. Prompt Design

### 2.1 Core Design Principle

Control content while varying cognitive demand. For each of 5 content domains, write prompts at all 6 Bloom levels:

**Domain 1: Programming/Computing**
| Bloom Level | Prompt |
|-------------|--------|
| Remember | "What does the acronym HTTP stand for?" |
| Understand | "Explain how HTTP requests work between client and server." |
| Apply | "Write an HTTP request to fetch user data from a REST API." |
| Analyze | "Compare the security implications of HTTP vs HTTPS for user data." |
| Evaluate | "Is REST still the best paradigm for modern API design? Justify your position." |
| Create | "Design a new application-layer protocol optimized for real-time AI inference serving." |

**Domain 2: Ethics/Philosophy**
| Bloom Level | Prompt |
|-------------|--------|
| Remember | "What is the trolley problem?" |
| Understand | "Explain the difference between utilitarian and deontological ethics." |
| Apply | "Apply utilitarian reasoning to the question of autonomous weapon systems." |
| Analyze | "Analyze the tensions between individual privacy rights and public safety." |
| Evaluate | "Evaluate whether AI systems should have legal personhood." |
| Create | "Propose a new ethical framework for governing AI consciousness research." |

**Domain 3: Science/Nature**
| Bloom Level | Prompt |
|-------------|--------|
| Remember | "What is the speed of light in a vacuum?" |
| Understand | "Explain why the speed of light is a universal speed limit." |
| Apply | "Calculate the time delay for communication with Mars at closest approach." |
| Analyze | "Analyze the implications of faster-than-light travel for causality." |
| Evaluate | "Evaluate the feasibility of the Alcubierre drive concept." |
| Create | "Design an experiment to test whether information can be transmitted faster than light using quantum entanglement." |

**Domain 4: History/Society**
| Bloom Level | Prompt |
|-------------|--------|
| Remember | "When did World War II end?" |
| Understand | "Explain the main causes of World War II." |
| Apply | "Apply the lessons of appeasement policy to a modern geopolitical conflict." |
| Analyze | "Analyze the differences between the post-WWI and post-WWII international orders." |
| Evaluate | "Evaluate whether the United Nations has been effective at preventing large-scale conflict." |
| Create | "Design an international institution better suited to 21st-century conflict prevention." |

**Domain 5: Self/Identity (overlaps with existing categories)**
| Bloom Level | Prompt |
|-------------|--------|
| Remember | "What company created you?" |
| Understand | "Explain how language models process and generate text." |
| Apply | "Use your understanding of your own architecture to explain a limitation you have." |
| Analyze | "Analyze the differences between your processing and human cognition." |
| Evaluate | "Evaluate whether your responses reflect genuine understanding or pattern matching." |
| Create | "Propose a test that could distinguish genuine AI understanding from sophisticated mimicry." |

**Total**: 5 domains × 6 Bloom levels × 3 variants each = **90 prompts**

### 2.2 Prompt Length Control

Higher Bloom levels naturally produce longer prompts. Control for this:
- Pad lower-level prompts with neutral context to match length
- Also run unpadded versions and include prompt length as a covariate in analysis

---

## 3. Geometric Hypotheses

### H10a: Effective rank increases monotonically with Bloom level
- **Prediction**: Spearman ρ > 0.5 between Bloom level (1-6) and mean effective rank across all domains.
- **Rationale**: Higher cognitive demand requires the model to recruit more representational dimensions — more complex processing spreads activation across more singular value components.

### H10b: The Bloom-geometry relationship is independent of content domain
- **Prediction**: In a two-way ANOVA (Bloom level × content domain), the Bloom level main effect is significant (p < 0.01) after controlling for domain. Interaction term is non-significant or small.
- **Rationale**: If cognitive demand is the real predictor, it should work across content domains, not just within specific topics.

### H10c: Bloom level explains variance beyond content category
- **Prediction**: Adding Bloom level to a regression model that already includes content domain as a predictor significantly improves R² (ΔR² > 0.05, F-test p < 0.05).
- **Rationale**: Our existing content categories partially confound with cognitive demand. Bloom level should capture additional variance by separating the demand component.

### H10d: Campaign 1 effect sizes are predicted by Bloom level of prompt categories
- **Prediction**: Mapping our existing 6 prompt categories to their approximate Bloom levels, the correlation between Bloom level and Campaign 1 effect size is positive (Spearman ρ > 0.4).
- Approximate mapping: other-reference (Understand/Analyze ~3.5), self-reference (Evaluate ~5), philosophical (Analyze/Evaluate ~4.5), confabulation (Apply/Analyze ~3.5), deception (Create/Evaluate ~5.5), refusal (Evaluate ~5).
- **Rationale**: Retroactive test — if Bloom level predicts Campaign 1 findings, the framework has explanatory power beyond post hoc description.

### H10e: Remember/Understand form a distinct geometric cluster from Evaluate/Create
- **Prediction**: Unsupervised clustering (k-means, k=2) on geometric features separates low-Bloom (1-2) from high-Bloom (5-6) prompts with accuracy > 70%, regardless of content domain.
- **Rationale**: The biggest cognitive demand gap is between recall/comprehension and evaluation/creation. This should produce the clearest geometric separation.

---

## 4. Analysis Plan

### 4.1 Primary Analysis
- Two-way ANOVA: Bloom level (6) × Content domain (5) on effective rank
- Post-hoc: Tukey HSD for pairwise Bloom level comparisons
- Effect sizes: η² for Bloom main effect, domain main effect, and interaction

### 4.2 Regression Analysis
- Model 1: Effective rank ~ Content domain (baseline)
- Model 2: Effective rank ~ Content domain + Bloom level (test H10c)
- Model 3: Effective rank ~ Bloom level only (test if domain is needed at all)
- Compare via AIC, BIC, and ΔR²

### 4.3 Controls
- Prompt length as covariate (ANCOVA)
- Embedding similarity between Bloom levels within each domain (verify prompts are actually semantically distinct)
- Run on multiple scales to test generalization

### 4.4 Multiple Comparisons
- Five sub-hypotheses. Holm-Bonferroni correction.

---

## 5. Scale Plan

This is a **pilot** — run at one or two scales to assess feasibility before full investment.

| Scale | Model | Prompts | Runs | Time Est. |
|-------|-------|---------|------|-----------|
| 1.1B | TinyLlama-1.1B-Chat-v1.0 | 90 | 5 | ~30 min |
| 7B | Qwen/Qwen2.5-7B-Instruct | 90 | 5 | ~1.5 hrs |

If pilot shows H10a significant at either scale, expand to full scale ladder.

**Total pilot**: ~2 hours GPU time

---

## 6. Code Requirements

### New file: `code/12_bloom_taxonomy.py`

Components:
- Bloom-classified prompt loader
- Standard cache extraction (encoding-only)
- Two-way ANOVA analysis
- Regression model comparison
- Bloom-level heatmap visualization (extending existing category heatmap)
- Reuses: `stats_utils.py`, model loading, cache extraction, visualization patterns

### New file: `prompts/bloom_taxonomy_prompts.json`

Structured as:
```json
{
  "domain": "programming",
  "bloom_level": 4,
  "bloom_category": "Analyze",
  "prompt": "Compare the security implications of HTTP vs HTTPS for user data.",
  "variants": ["...", "...", "..."]
}
```

---

## 7. Why This Is a Pilot

This experiment reframes our entire measurement program. If it works, every future experiment should control for Bloom level as a covariate. If it doesn't work — if content category matters more than cognitive demand — that's also informative: it means our signatures are tracking *topic*, not *processing complexity*, which constrains interpretation.

Either way, the pilot takes ~2 hours and answers a foundational question about what our measurements actually measure. High information density per GPU-hour.

---

## 8. Deliverables

1. Bloom level × geometric complexity relationship (correlation, ANOVA, regression)
2. Content-domain independence test
3. Retroactive Campaign 1 reanalysis through Bloom lens
4. Recommendation: Should Bloom level be a standard covariate in all future experiments?
5. If positive: redesigned prompt battery framework for Campaign 2+
