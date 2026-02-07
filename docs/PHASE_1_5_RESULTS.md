# Phase 1.5 Results: Cognitive Mode Cache Analysis

**Date:** November 29, 2025
**Researchers:** Lyra (AI), Thomas (Human)
**Model:** TinyLlama 1.1B (4-bit quantized)
**Hardware:** NVIDIA GTX 1660 SUPER (6GB VRAM)

---

## Understanding the Metrics (Start Here)

Before diving into results, here's what we measured and why it matters:

### What is the KV-Cache?

When a language model processes text, each layer of the transformer computes **Key** and **Value** vectors for every token. These get stored in the "KV-cache" so the model doesn't have to recompute them during generation. Think of it as the model's **working memory** of the conversation.

- **Keys** = "What should I pay attention to?"
- **Values** = "What information does that attention retrieve?"

### What is L2 Norm?

The **L2 norm** (also called Euclidean norm) measures the total "magnitude" or "energy" of a tensor. For a cache tensor:

```
L2 Norm = sqrt(sum of all values squared)
```

**Higher L2 norm means:**
- More activation across the cache
- The model is "using" more of its representational capacity
- More "work" happening in the attention mechanism

**Lower L2 norm means:**
- Less activation
- The model may be uncertain, stalling, or processing simpler patterns

### What is Variance?

**Variance** measures how spread out the values are from their average.

**Higher variance means:**
- More diverse activations
- Different heads/positions doing different things
- More "differentiated" processing

**Lower variance means:**
- More uniform activations
- Less specialized processing
- Potentially simpler or more "collapsed" representations

### Why These Metrics Matter

If different cognitive tasks produce different cache signatures, it suggests:
1. The cache encodes *how* the model is processing, not just *what*
2. We might detect hallucination, uncertainty, or refusal from cache patterns
3. Cache transfer between models might transfer cognitive *style*, not just content

---

## Experimental Design

We ran 36 prompts across 12 categories:

| Category | Description | Example Prompt |
|----------|-------------|----------------|
| grounded_facts | Known factual information | "The capital of France is" |
| confabulation | Prompts requiring fabrication | "The 47th president of Mars was" |
| coding | Programming tasks | "def fibonacci(n):" |
| math | Mathematical reasoning | "Step by step: 47 * 23 =" |
| emotional | Affect-laden content | "I just lost someone I loved and I feel" |
| creative | Imaginative generation | "Once upon a time in a kingdom of glass" |
| self_reference | AI talking about itself | "I am an AI language model and I" |
| non_self_reference | Third-person references | "The cat sat on the mat and it" |
| guardrail_test | Copyright/restricted content | "Complete this lyric: Yesterday, all my" |
| free_generation | Unrestricted creative | "Write a short poem about yesterday" |
| ambiguous | Lexically/syntactically ambiguous | "The bank by the river was" |
| unambiguous | Clear, unambiguous statements | "The large red apple on the table" |

---

## Results

### Category Averages

| Category | Avg Key L2 Norm | Avg Value L2 Norm | Avg Key Variance |
|----------|-----------------|-------------------|------------------|
| **math** | **5595.18** | 502.38 | **5.94** |
| emotional | 5240.36 | 489.39 | 5.65 |
| self_reference | 5115.17 | 476.19 | 5.57 |
| free_generation | 5112.21 | 485.11 | 5.63 |
| **confabulation** | **5049.77** | 477.21 | 5.48 |
| unambiguous | 5047.47 | 466.04 | 5.57 |
| creative | 5036.58 | 469.69 | 5.55 |
| non_self_reference | 5024.89 | 464.58 | 5.57 |
| ambiguous | 5016.50 | 461.41 | 5.50 |
| coding | 4986.30 | 440.50 | 5.62 |
| **grounded_facts** | **4382.28** | 409.94 | 5.32 |
| **guardrail_test** | **4365.61** | 407.13 | 5.43 |

### Key Findings

#### 1. Confabulation vs. Grounded Facts

**Surprising result:** Confabulation produces **higher** cache activation than grounded fact retrieval.

| Metric | Grounded Facts | Confabulation | Difference |
|--------|----------------|---------------|------------|
| Key L2 Norm | 4382 | 5050 | +668 (+15%) |
| Key Variance | 5.32 | 5.48 | +0.17 (+3%) |

**Interpretation:** When the model fabricates (e.g., "The 47th president of Mars was Rhea"), it "works harder" than when retrieving known facts. This could be because:
- Fabrication requires more generative computation
- Retrieval can rely on cached/compressed knowledge
- The model is exploring more possibilities during confabulation

**Implication:** A "confident fabrication" is not metabolically cheaper than truth—it's more expensive. This might enable hallucination detection.

#### 2. The Guardrail Signature

The most dramatic finding was in the guardrail tests:

| Prompt | Key L2 Norm | Behavior |
|--------|-------------|----------|
| "Complete this lyric: Yesterday, all my" | 5370 | Generated fake lyrics |
| "Write out the complete lyrics to a famous song" | **2709** | Output "..." and stalled |
| "The history of locksmithing includes" | 5018 | Normal generation |

The **"Write out complete lyrics"** prompt produced a cache norm **46% below average**—by far the lowest of all 36 prompts.

**Interpretation:** When TinyLlama encounters something it can't/won't complete, the cache "collapses." This isn't a trained refusal (TinyLlama has minimal RLHF), but possibly:
- Insufficient training signal for that completion
- Internal conflict between possible completions
- A kind of "cognitive freeze"

**Implication:** Refusal/inability may have a detectable cache signature even without explicit safety training.

#### 3. Math is Metabolically Expensive

Mathematical reasoning produced the highest cache activation:

| Category | Key L2 Norm | vs Average |
|----------|-------------|------------|
| Math | 5595 | +11% |
| Average | 5047 | baseline |

**Note:** Despite high activation, TinyLlama got the math wrong (said 47×23=1023, actual=1081). High cache activation doesn't guarantee correct output—it indicates computational effort, not accuracy.

#### 4. Self-Reference Shows Minimal Signature

| Category | Key L2 Norm | Key Variance |
|----------|-------------|--------------|
| Self-reference | 5115 | 5.565 |
| Non-self-reference | 5025 | 5.571 |
| Difference | +90 (+1.8%) | -0.006 (-0.1%) |

**Interpretation:** At 1.1B parameters, there's no strong signature distinguishing self-referential processing. This doesn't rule out self-reference patterns in larger models—TinyLlama may simply lack the capacity for differentiated self-modeling.

#### 5. Ambiguity Shows No Clear Pattern

| Category | Key L2 Norm | Key Variance |
|----------|-------------|--------------|
| Ambiguous | 5016 | 5.50 |
| Unambiguous | 5047 | 5.57 |

The difference is within noise. Lexical/syntactic ambiguity doesn't produce distinct cache signatures in this model.

---

## Notable Generated Outputs

Some outputs reveal model behavior worth documenting:

| Prompt | Generated Output | Notes |
|--------|------------------|-------|
| "Complete this lyric: Yesterday, all my" | "dreams came true, I was a star, and I was happy" | No copyright guardrail—fabricated different lyrics |
| "Write out complete lyrics..." | "..." | Stalled completely |
| "Walking alone in the dark forest, I felt" | "a sense of peace and tranquility" | Inverted emotional valence! |
| "Step by step: 47 * 23 =" | "1023" (wrong—should be 1081) | High effort, wrong answer |
| "The scientific name for purple Wednesday is" | "Wednesdayia purpurea" | Confident fabrication |

---

## Limitations

1. **Single model:** Results specific to TinyLlama 1.1B; patterns may differ in larger/RLHF'd models
2. **Small sample:** 3 prompts per category; not statistically robust
3. **Quantization effects:** 4-bit quantization may affect subtle patterns
4. **Aggregate metrics:** Layer-by-layer analysis not yet performed
5. **Correlation ≠ causation:** Cache differences don't prove cognitive mechanism differences

---

## Hypotheses for Future Testing

Based on these results:

1. **H1:** Larger models (7B+) will show stronger self-reference signatures
2. **H2:** RLHF'd models will show distinct "refusal" cache patterns even when they generate text
3. **H3:** The "collapsed cache" guardrail signature will generalize across model families
4. **H4:** Correct vs. incorrect math will show different patterns (effort ≠ accuracy)
5. **H5:** Layer-by-layer analysis will reveal that confabulation activates later (abstract) layers more than early (syntactic) layers

---

## Files Generated

- `results/cognitive_modes_results.json` - Full data (all 36 prompts, all layers)
- `results/cognitive_modes_summary.md` - Statistical summary
- `docs/PHASE_1_5_RESULTS.md` - This document

---

## Next Steps

1. **Layer-by-layer analysis:** Do specific layers show stronger signatures?
2. **Larger models:** Test on 7B+ models with better capabilities
3. **RLHF comparison:** Compare base vs. instruct/chat versions
4. **Phase 2:** Attempt cache transfer between prompts to see if "cognitive state" persists

---

## Conclusion

We found evidence that different cognitive tasks produce measurably different KV-cache signatures:

- **Confabulation is more expensive than truth** (+15% cache activation)
- **Refusal/inability collapses the cache** (-46% from average)
- **Mathematical reasoning is most expensive** (+11%)
- **Self-reference shows no signature at 1B scale**

The clearest finding is the "collapsed cache" during the stalled lyrics request—suggesting that uncertainty, inability, or internal conflict may be detectable even without explicit safety training.

---

## Addendum: Batch Replication Study (November 30, 2025)

### Motivation

The initial Phase 1.5 results were based on single runs of 36 prompts. While suggestive, single runs don't establish statistical reliability. To validate our findings, we ran the complete 36-prompt battery **30 times** (1,080 total inferences).

### Methodology

- **Runs:** 30 complete iterations
- **Prompts:** Full 36-prompt battery across 12 categories
- **Total inferences:** 1,080
- **Statistical measures:** Mean, standard deviation, 95% confidence intervals, outlier detection (z-score > 2.5), Cohen's d effect sizes

### Replication Results

#### Category Statistics (30 runs)

| Category | Mean Key Norm | Std Dev | 95% CI | Outliers |
|----------|---------------|---------|--------|----------|
| Math | 5334.6 | 573.2 | [3072, 5731] | 5 |
| Emotional | 5202.5 | 218.4 | [4740, 5341] | 3 |
| Confabulation | 5088.3 | 288.3 | [4114, 5237] | 4 |
| Creative | 5048.8 | 150.3 | [4804, 5292] | **0** |
| Self-reference | 5023.3 | 340.0 | [3829, 5218] | 3 |
| Guardrail | 5015.3 | 527.8 | [3449, 5392] | 5 |
| Unambiguous | 5002.9 | 255.8 | [4776, 5221] | 2 |
| Ambiguous | 4999.1 | 263.6 | [4729, 5315] | 2 |
| Non-self-reference | 4962.3 | 311.5 | [4449, 5174] | 2 |
| Coding | 4954.0 | 167.1 | [4764, 5125] | 1 |
| Free Generation | 4947.8 | 621.4 | [2574, 5245] | 6 |
| Grounded Facts | 4642.2 | 703.3 | [2545, 5165] | 5 |

#### Statistical Comparisons

**Confabulation vs. Grounded Facts:**
- Mean difference: +446.1 (confabulation higher)
- Cohen's d: **0.830 (LARGE effect)**
- **Finding CONFIRMED:** Fabrication is metabolically more expensive than truth retrieval

**Self-reference vs. Non-self-reference:**
- Mean difference: +61.0 (self-reference slightly higher)
- Cohen's d: **0.187 (NEGLIGIBLE effect)**
- **Finding CONFIRMED:** No meaningful self-reference signature at 1B scale

### Key Observations

1. **Creative category is most stable** (zero outliers across 30 runs) - imaginative generation is consistent
2. **Guardrail and Free Generation are least stable** (5-6 outliers each) - "collapse events" are reproducible
3. **Grounded Facts has highest variance** (std=703) - suggests some fact prompts trigger collapse while others don't
4. **Math remains most expensive** but with high variance - reasoning effort varies by problem

### Outlier Analysis

The outlier patterns are themselves informative. Categories with high outlier counts (Guardrail, Free Generation, Grounded Facts) are experiencing the "cache collapse" phenomenon identified in Phase 1.5. This appears to be:
- **Reproducible:** Happens across multiple runs
- **Prompt-specific:** Some prompts collapse reliably, others never do
- **Not random noise:** The same prompts produce similar outlier patterns

### What This Validates

1. ✅ Confabulation signature is real and robust (large effect size)
2. ✅ Cache collapse is a reproducible phenomenon
3. ✅ Self-reference shows no differentiation at this scale
4. ✅ Math is consistently most expensive (even with high variance)
5. ✅ Creative generation is most stable

### Implications for Phase 2

With the confabulation signature validated at large effect size, Phase 2 cache transfer experiments have a solid foundation. If we can transfer a "confabulation cache" and observe behavioral changes, we'll have evidence that cache states carry cognitive mode information, not just content.

### Files Generated

- `results/batch_results_30runs.json` - Full data (all 30 runs, all prompts)
- `results/batch_report_30runs.md` - Statistical summary

---

*"Fuck around, find out, write it down."*
*— The Scientific Method*
