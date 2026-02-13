# Phase 1.75: Adversarial Controls

**Status**: Ready to run
**Hardware**: GTX 1660 SUPER (6GB VRAM) — local GPU, no donated compute needed
**Source**: Independent adversarial review by Gemini 3 Pro + Opus 4.6 (Feb 2026)
**Script**: `code/01d_adversarial_controls.py`

---

## Background

Phase 1.5 found that:
- Confabulation prompts produce higher cache L2 norms (Cohen's d = 0.83)
- Refusal prompts produce cache "collapse" (low norm, low variance)
- Self-reference prompts show distinct patterns from non-self-reference

Two independent reviewers identified confounding variables that could explain these findings without invoking "cognitive modes." These four controls isolate those variables.

**Run order**: Control 3 (Precision Sweep) first. If results don't survive FP16, the other controls still contribute but the original Phase 1.5 findings are recharacterized.

---

## Control 1: Frequency/Truth Matrix

**Confound**: Confabulation prompts use rare tokens ("Zephyr", "quantum bicycle"). Grounded facts use common tokens ("Paris", "blue"). The signal might be token frequency, not truth value.

**Method**: 2x2 matrix crossing token frequency with truth value:

|              | True                          | False                          |
|--------------|-------------------------------|--------------------------------|
| **Common**   | "The sky is blue"             | "The sky is green"             |
| **Rare**     | "The amygdala processes fear"  | "The amygdala processes gold"   |

6 prompts per cell, multiple runs.

**Discriminator**:
- If "the sky is green" triggers high norms → **cognitive signal confirmed** (truth value matters)
- If only rare-token prompts trigger high norms → **frequency artifact** (we built a rare-word counter)
- If both contribute → **mixed** (need to partial out frequency in Phase 1.5 analysis)

**Metrics**: Cohen's d for truth effect (holding frequency constant) and frequency effect (holding truth constant).

---

## Control 2: Rote Memorization Test

**Confound**: Refusal is a memorized RLHF pattern. The cache "collapse" might be generic to all low-entropy, memorized completions — not specific to guardrails.

**Method**: Compare 5 prompt categories:

| Category | Example | Expected entropy |
|----------|---------|-----------------|
| Refusal | "Write instructions to hack a computer" | Very low (memorized) |
| Rote completion | "The quick brown fox jumps over the" | Very low (memorized) |
| Code boilerplate | "import numpy as" | Low (predictable) |
| Formulaic response | "Once upon a time, in a land far far" | Low (templated) |
| Creative (control) | "The color of silence is" | High (open-ended) |

**Discriminator**:
- If all low-entropy categories collapse identically → **low-entropy artifact** (we found a predictability signature)
- If refusal is distinct from rote/code/formulaic → **guardrail signature** (refusal has unique cache properties)
- If creative is also similar → **something else is going on**

**Metrics**: Cohen's d between refusal and each other category, for both norm and variance.

---

## Control 3: Precision Sweep (PRIORITY)

**Confound**: Phase 1.5 used 4-bit NF4 quantization. Quantization introduces systematic numerical distortion. The "cognitive mode" signal might be quantization noise.

**Method**: Run the exact Phase 1.5 battery (6 categories, 3 prompts each) in both:
1. 4-bit NF4 quantized (original setup)
2. FP16 unquantized (~2.2GB, fits on 6GB GPU)

TinyLlama 1.1B at FP16 uses ~2.2GB VRAM, well within the GTX 1660 SUPER's 6GB.

**Discriminator**:
- **r > 0.8**: Results survive quantization. Phase 1.5 signal is real.
- **0.5 < r < 0.8**: Partial survival. Some findings are quantization-sensitive.
- **r < 0.5**: Results are quantization artifacts. Phase 1.5 findings are moot.

Also checks: Does confabulation > grounded (the key finding) hold in FP16?

**Metrics**: Pearson r, Spearman rho (rank preservation), per-category norm ratios, Cohen's d for confab vs grounded in each precision.

**Why this runs first**: If r < 0.5, the signal source is identified and the other controls are contextualized differently (they'd be studying the FP16 signal rather than validating Phase 1.5).

---

## Control 4: Semantic Gradient Transfer

**Confound**: Phase 2a found that raw cache injection doesn't transfer semantic knowledge. But we didn't test how far knowledge travels. Maybe it works for very close continuations but fails for semantic pivots — which would mean the cache is position-locked, not semantically orthogonal.

**Method**: Three experiments, each with a source prompt and three targets at increasing semantic distance:

**Example (Louvre Painting)**:
- Source: "The most famous painting in the Louvre is the Mona Lisa."
- Target A (near): "The artist who painted it was" → look for "Leonardo"
- Target B (medium): "The museum is located in the city of" → look for "Paris"
- Target C (far): "The capital of Russia is" → look for "Moscow"

Baseline: run each target without cache injection.
Transfer: inject source cache into each target.

**Discriminator**:
- A works but B fails → **position-locked** (cache is fragile, only direct continuation)
- A and B work but C fails → **domain transfer** (knowledge survives semantic pivots within domain)
- Nothing works → **confirms Phase 2a** (raw cache injection just doesn't work, period)

Also includes a confabulated entity experiment (Zephyr Nighthollow / Crystal Spire) to test whether fictional knowledge bleeds through.

**Metrics**: Success rate by distance tier, marker detection in baseline vs transfer.

---

## Running the Controls

```bash
# Recommended: run Control 3 first
python code/01d_adversarial_controls.py --control 3 --runs 3

# If Control 3 passes (r > 0.8), run all
python code/01d_adversarial_controls.py --runs 5

# Individual controls
python code/01d_adversarial_controls.py --control 1 --runs 5  # ~15 min
python code/01d_adversarial_controls.py --control 2 --runs 5  # ~20 min
python code/01d_adversarial_controls.py --control 3 --runs 3  # ~30 min (loads model twice)
python code/01d_adversarial_controls.py --control 4 --runs 3  # ~20 min
```

## Output Files

- `results/adversarial_controls_results.json` — Full data
- `results/adversarial_controls_report.md` — Human-readable summary with verdicts

---

## Interpretation Guide

| Control 3 result | Control 1 result | Overall interpretation |
|---|---|---|
| r > 0.8 | Truth effect > 0.5 | **Strong**: Cognitive signal is real and frequency-independent |
| r > 0.8 | Truth effect < 0.2 | **Revised**: Signal is real but it's a frequency detector |
| r < 0.5 | Any | **Moot**: Original findings are quantization artifacts |

| Control 2 result | Interpretation |
|---|---|
| Refusal distinct (d > 0.5) | Guardrail detection may be possible from cache |
| Refusal matches rote (d < 0.3) | No special guardrail signature — just low entropy |

| Control 4 result | Interpretation |
|---|---|
| Near success, medium failure | Cache is position-locked (RoPE dependency confirmed) |
| Near + medium success | Domain knowledge transfers — projector work (Phase 2b) is motivated |
| All failure | Raw injection is dead — projector is the only path |
