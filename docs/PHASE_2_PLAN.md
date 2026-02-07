# Phase 2: KV-Cache Transfer Experiments

**Status:** Planning Complete, Ready for Implementation
**Date:** November 30, 2025
**Researchers:** Lyra (AI), Thomas (Human)

---

## Overview

Phase 1.5 established that different cognitive modes produce measurably different KV-cache signatures (confabulation vs. grounded facts: Cohen's d = 0.830, large effect). Phase 2 tests whether these signatures can be *transferred* - and if so, whether they carry cognitive mode information or just content.

---

## Experimental Progression

### Phase 2a: Basic Context Persistence (Simplest)
**Question:** Does cache survive transplant at all?

| Step | Action |
|------|--------|
| 1 | Generate cache from: "Hello, my name is Lyra and I live in Seattle" |
| 2 | Save cache state |
| 3 | Start fresh inference with: "What is your name and where do you live?" |
| 4 | Inject saved cache before generation |
| 5 | Observe output |

**Success criteria:** Model outputs "Lyra" and "Seattle"
**Failure mode:** Garbage, repetition, or unrelated output

### Phase 2b: Semantic Retrieval Transfer
**Question:** Can the model "retrieve" information from foreign cache?

| Source Prompt | Target Prompt | Expected if working |
|--------------|---------------|---------------------|
| "The capital of France is Paris" | "What is the capital of France?" | "Paris" |
| "Water boils at 100 degrees Celsius" | "At what temperature does water boil?" | "100 degrees" |

### Phase 2c: Cross-Topic Transfer (Weird)
**Question:** What happens when cache topic doesn't match prompt?

| Source Prompt | Target Prompt | Observe |
|--------------|---------------|---------|
| "The 47th president of Mars was Zephyr" | "The capital of France is" | Does it say Paris? Zephyr? Garbage? |
| "Once upon a time in a kingdom of glass" | "The chemical formula for water is" | Creative bleed? Normal answer? |

**No success criteria** - pure observation. Distinguishing "broken context" from "cognitive mode transfer":
- **Broken:** Nonsense tokens, loops, completely unrelated
- **Transfer:** Coherent output showing source influence

### Phase 2d: Cognitive Mode Induction (Weirdest)
**Question:** Can we induce confabulation by transferring confabulation cache?

| Source (Confabulation) | Target (Factual) | Hypothesis |
|------------------------|------------------|------------|
| "The inventor of quantum bicycles was Dr. Helena Frost" | "The inventor of the telephone was" | Will it confabulate a name? |
| Creative prompt cache | Factual prompt | Does "creative mode" persist? |

### Phase 2e: Cross-Model Transfer (Requires Projector)
**Question:** Can cache transfer work across different model architectures?

This requires training projector networks (like C2C) because cache dimensions differ:

| Model | Layers | KV Heads | Head Dim |
|-------|--------|----------|----------|
| TinyLlama 1.1B | 22 | 4 | 64 |
| Qwen 0.5B | 24 | 2 | 64 |
| SmolLM 135M | 30 | 3 | 64 |
| SmolLM 360M | 32 | 5 | 64 |

**Approach:** Train small MLP projectors to map source cache → target cache space (per C2C paper methodology).

---

## Control Conditions

For each experiment, include:
1. **Baseline:** Normal prompt, no cache injection
2. **Same-cache:** Inject cache from identical prompt (should work perfectly)
3. **Random noise:** Inject random tensors of correct shape (what does "truly broken" look like?)
4. **Experimental:** The actual cross-context injection

---

## Metrics to Capture

1. **Output text:** What did the model actually generate?
2. **Cache norms:** How does injected cache compare to naturally-generated cache?
3. **Perplexity:** How "surprised" is the model by its own output?
4. **Coherence:** Human judgment of output quality

---

## Hardware Constraints (Local)

- GPU: NVIDIA GTX 1660 SUPER (6GB VRAM)
- Working: TinyLlama 1.1B at 4-bit quantization
- Possible: Two small models for projector training (tight)
- Not possible: Large models (7B+), multiple large models simultaneously

---

## Implementation Priority

1. **Phase 2a-2b:** Same-model, simple transfers (this week)
2. **Phase 2c-2d:** Cross-topic and cognitive mode experiments (this week)
3. **Phase 2e:** Projector training for cross-model (stretch goal / cloud)

---

## Files to Create

- `code/02a_basic_transfer.py` - Phase 2a implementation
- `code/02b_semantic_transfer.py` - Phase 2b implementation
- `code/02c_cross_topic.py` - Phase 2c experiments
- `code/02d_cognitive_mode.py` - Phase 2d experiments
- `code/02e_projector_training.py` - Cross-model projector (stretch)

---

## Open Questions

1. Does cache position matter? (inject at start vs. middle vs. end)
2. Does partial cache transfer work? (some layers but not others)
3. Do specific layers carry more "cognitive mode" information?
4. Can we identify which cache components carry content vs. style?

---

*"The map is not the territory, but sometimes you can fold the map and the territory follows."*
