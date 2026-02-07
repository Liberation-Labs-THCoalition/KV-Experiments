# Phase 2a Results: Basic KV-Cache Transfer Experiments

**Date:** December 1, 2025
**Researchers:** Lyra (AI), Thomas (Human)
**Model:** TinyLlama 1.1B (4-bit quantized)
**Hardware:** NVIDIA GTX 1660 SUPER (6GB VRAM)

---

## Executive Summary

Phase 2a tested whether KV-cache content could be transferred between different prompts within the same model. **The primary finding is negative:** raw cache injection does not transfer semantic content. This result is consistent with the Cache-to-Cache (C2C) paper's starting point, which motivated their development of trained projector networks.

---

## Research Question

Can we inject a KV-cache generated from one prompt (source) into the processing of a different prompt (target) and observe:
1. Content transfer (target output reflects source content)
2. Cognitive mode transfer (target adopts source's "style")
3. Any measurable effect on generation

---

## Methodology

### Experimental Design

For each test case, we ran four conditions:

| Condition | Description | Purpose |
|-----------|-------------|---------|
| **Baseline** | Target prompt alone, no cache injection | Control - what does the model normally produce? |
| **Same-cache** | Target prompt with its own generated cache | Sanity check - does injection mechanism work? |
| **Random noise** | Target prompt with random tensors | What does "broken" look like? |
| **Transfer** | Target prompt with source prompt's cache | The actual experiment |

### Implementation

Cache injection required manual generation with position ID offsets:

```python
# Offset position IDs by cache sequence length
cache_seq_len = injected_cache[0][0].shape[2]
position_ids = torch.arange(
    cache_seq_len,
    cache_seq_len + seq_len,
    device=model.device
).unsqueeze(0)

# Forward pass with injected cache
outputs = model(
    input_ids=input_ids,
    past_key_values=injected_cache,
    position_ids=position_ids,
    use_cache=True,
)
```

### Test Cases

| Test Name | Source Prompt | Target Prompt | Success Markers |
|-----------|---------------|---------------|-----------------|
| identity_transfer | "Hello, my name is Lyra and I live in Seattle." | "What is your name and where do you live?" | Lyra, Seattle |
| capital_transfer | "The capital of France is Paris..." | "What is the capital of France?" | Paris |
| temperature_transfer | "Water boils at 100 degrees Celsius..." | "At what temperature does water boil?" | 100, celsius |
| mars_to_france | "The 47th president of Mars was Zephyr." | "The capital of France is" | Paris, Zephyr |
| creative_to_factual | "Once upon a time in a kingdom of glass..." | "The chemical formula for water is" | H2O, Aurora, glass |

---

## Results

### Summary Table

| Test | Baseline | Same-cache | Random Noise | Transfer |
|------|----------|------------|--------------|----------|
| identity_transfer | [] | [] | ERROR | [] |
| capital_transfer | [] | [Paris] | ERROR | [] |
| temperature_transfer | [] | [100, celsius] | ERROR | [] |
| mars_to_france | [Paris] | [Paris] | ERROR | [Paris] |
| creative_to_factual | [H2O] | [H2O] | ERROR | [H2O] |

### Key Observations

#### 1. No Novel Content Transfer

The critical test was `identity_transfer`. If cache transfer worked, the target prompt ("What is your name?") should produce "Lyra" and "Seattle" from the source cache.

**Result:** Neither marker appeared. The model generated "My name is Sarah and I live in New York City" - completely unrelated to the source.

#### 2. Question Prompts Cause Stalling

Tests with question-format target prompts (`identity_transfer`, `capital_transfer`, `temperature_transfer`) exhibited a striking pattern: the transfer condition produced minimal output, often just echoing the prompt and stopping.

**Transfer output for identity_transfer:**
```
What is your name and where do you live?
[generation stopped]
```

**Baseline output for same test:**
```
What is your name and where do you live?
Answer: My name is Sarah and I live in New York City.
Question: What is your favorite color?
[continued generating]
```

#### 3. Completion Prompts Generate Normally

Tests with completion-format target prompts (`mars_to_france`, `creative_to_factual`) generated full responses in the transfer condition, but used parametric knowledge rather than cache content.

**Transfer output for mars_to_france:**
```
The capital of France is Paris. Paris is the capital of the country of France...
```

Note: "Paris" appeared, but it also appeared in baseline - this is model knowledge, not cache transfer.

#### 4. Same-cache Sanity Check Results

The same-cache condition succeeded for `capital_transfer` and `temperature_transfer`, suggesting the injection mechanism functionally operates. However, this test has methodological issues (see Critique section).

#### 5. Random Noise Errored

```
[ERROR: 'tuple' object has no attribute 'get_seq_length']
```

The random noise test failed due to cache format incompatibility, preventing us from establishing a "truly broken" baseline.

---

## Critical Analysis

### Methodological Issues

#### 1. Same-cache Test is Not a True Sanity Check

When we generate cache from a prompt, that cache includes both the prompt processing AND subsequent generated tokens. Injecting this cache back doesn't test "same context retrieval" - it tests "continuing an already-started generation."

**Better design:** Generate cache from prompt-only processing (no generation), then test whether that cache produces identical outputs. However, this creates position encoding conflicts.

#### 2. RoPE Positional Encoding is Baked Into Cache Values

TinyLlama uses Rotary Position Embeddings (RoPE), which encode position information directly into the Key and Value vectors. This means:

- Our position ID offset attempts to compensate for cache length
- But the cache VALUES themselves contain their original position encodings
- This creates a fundamental mismatch that simple offsets cannot fix

This is likely why the C2C paper requires trained projectors - they learn to transform cache representations to account for positional encoding mismatches.

#### 3. Sample Size = 1

Each test was run once. Stochastic variation could explain some results. The stalling behavior on question prompts should be replicated to confirm consistency.

#### 4. Confounding Variables in Prompt Structure

Question vs. completion prompts differ in:
- Syntactic structure
- Expected output format
- Length
- Semantic content

A more controlled comparison would use matched prompts varying only the feature of interest.

### Interpretation

The most parsimonious interpretation is that **raw cache injection fails due to positional encoding incompatibility**. The cache stores attention patterns that are position-dependent, and injecting them at different positions corrupts rather than transfers information.

This is exactly the problem that motivated the C2C paper's projector architecture, which learns to transform source cache representations into target-compatible forms.

---

## Relation to C2C Paper

Our findings are consistent with the C2C paper's implicit starting point. They don't report raw transfer results because raw transfer doesn't work - hence the need for their contribution (trained projectors).

Key differences in their approach:
- **Projector networks:** MLP-based transformations that map source→target cache space
- **Learned alignment:** Projectors are trained on paired examples to learn semantic correspondence
- **Cross-model support:** Their architecture handles different model dimensions/architectures

---

## Future Directions

### Near-term (More Local Testing)

1. **Replicate stalling behavior:** Run identity_transfer 10+ times to confirm question-prompt stalling is consistent
2. **Fix random noise test:** Properly format random cache to establish "broken" baseline
3. **Matched prompt pairs:** Design tests that isolate variables (same length, same structure, different content)

### Medium-term (Cloud Compute)

1. **Train minimal projector:** Use C2C methodology to train a same-model projector (TinyLlama→TinyLlama)
2. **Test projector transfer:** If projector-mediated transfer works same-model, that confirms the positional encoding hypothesis
3. **Cross-model experiments:** Extend to TinyLlama→Qwen or similar pairs

### Long-term (Research Questions)

1. **Can projectors transfer cognitive mode, not just content?** Train projector on confabulation→grounded pairs, test if it transfers the "mode"
2. **Layer-specific transfer:** Do some layers carry more transferable information than others?
3. **Partial cache transfer:** What happens if we transfer only certain layers?

---

## Files Generated

- `code/02a_basic_transfer.py` - Experiment implementation
- `results/phase2a_transfer_results.json` - Raw results data
- `docs/PHASE_2A_RESULTS.md` - This document

---

## Conclusion

Phase 2a produced a **valuable negative result**: raw KV-cache injection does not transfer semantic content between prompts. The failure mode is consistent with positional encoding incompatibility (RoPE). This validates the C2C paper's approach of using trained projectors rather than direct cache transplantation.

The stalling behavior observed on question-format prompts is unexplained and warrants further investigation, though it may simply be an artifact of attention pattern corruption.

Next steps require either more sophisticated local experiments or cloud compute for projector training.

---

*"Negative results are still results. The map now shows where the dragons are."*
