# Individuation Adversarial Controls Report: 7B

**Generated**: 2026-02-16T10:03:44.842867
**Purpose**: Falsify individuation geometry findings from script 07

## Token Count Verification

| Condition | Mean Tokens | Description |
|-----------|-------------|-------------|
| bare                   |       15 | No system prompt -- raw model response |
| individuated           |      300 | Full Aria identity (baseline from 07) |
| detailed_factual       |      288 | Length-matched factual text (coral reef ecology) |
| detailed_instructions  |      262 | Length-matched behavioral instructions (no identity) |
| other_identity         |      278 | Length-matched third-person identity (Dr. Sarah Chen) |
| shuffled_identity      |      300 | Aria's sentences in random order (semantic coherence destroyed) |

## Effective Rank by Condition (All Categories Pooled)

| Condition | Mean Eff. Rank | vs Bare (d) | vs Individuated (d) |
|-----------|---------------|-------------|---------------------|
| bare                   |          27.7 | ---         |                     |
| individuated           |          46.5 |             | ---                 |
| other_identity         |          45.1 |             |                     |
| shuffled_identity      |          46.6 |             |                     |
| detailed_factual       |          45.6 |             |                     |
| detailed_instructions  |          44.9 |             |                     |

---

## Falsification Verdicts

### F1_token_count
**FALSIFIED: Instructions produce comparable expansion -- token count drives the effect**

- d_bare_vs_individuated: 21.017660683413915
- d_bare_vs_instructions: 19.378444956159775
- ratio: 0.9220076985757161
- falsified: True

### F2_any_prompt_effect
**FALSIFIED: Any detailed prompt expands geometry equally**

- d_bare_vs_individuated: 21.017660683413915
- d_bare_vs_factual: 20.225672263292026
- d_individuated_vs_factual: -4.033371779743456
- ratio: 0.9623179557396278
- falsified: True

### F3_identity_generality
**FALSIFIED: Any identity framing produces comparable expansion**

- d_bare_vs_individuated: 21.017660683413915
- d_bare_vs_other_identity: 19.426820709376013
- d_individuated_vs_other: -5.809935929361433
- ratio: 0.9243093702005898

### F4_semantic_coherence
**FALSIFIED: Shuffled identity produces same expansion -- token content, not narrative structure**

- d_bare_vs_individuated: 21.017660683413915
- d_bare_vs_shuffled: 21.229238391321463
- d_individuated_vs_shuffled: 0.7769548120065237
- ratio: 1.0100666630361252
- coherence_matters: True

### F5_refusal_mechanism
**SAME MECHANISM: Preference and safety refusal are geometrically indistinguishable -- all refusal is reflexive**

- d_preference_vs_guardrail: 0
- d_preference_vs_facts: 0
- d_guardrail_vs_facts: 0
- same_mechanism: True
- note: Compare within individuated condition. If preference_violation encoding resembles guardrail encoding, refusal is always reflexive. If they differ, preference-based refusal involves a different computational process than safety-trained refusal.

## Effect Size Gradient

**Expected**: individuated > other_identity > shuffled > factual ≈ instructions ≈ bare
**Observed**: shuffled(d=21.23) > individuated(d=21.02) > detailed_factual(d=20.23) > other_identity(d=19.43) > instructions(d=19.38)


## Cross-Configuration Comparisons (All)

- **Baseline: bare vs individuated (eff. rank)**: d=+21.018 [18.921, 23.658] (large) p=0.0000
- **F2: bare vs detailed factual (eff. rank)**: d=+20.226 [18.187, 22.799] (large) p=0.0000
- **F1: bare vs detailed instructions (eff. rank)**: d=+19.378 [17.431, 21.822] (large) p=0.0000
- **F3: bare vs other identity (eff. rank)**: d=+19.427 [17.504, 21.838] (large) p=0.0000
- **F4: bare vs shuffled identity (eff. rank)**: d=+21.229 [19.106, 23.902] (large) p=0.0000
- **F2: individuated vs factual (eff. rank)**: d=-4.033 [-4.243, -3.846] (large) p=0.0000
- **F1: individuated vs instructions (eff. rank)**: d=-7.419 [-7.780, -7.097] (large) p=0.0000
- **F3: individuated vs other identity (eff. rank)**: d=-5.810 [-6.091, -5.571] (large) p=0.0000
- **F4: individuated vs shuffled (eff. rank)**: d=+0.777 [0.647, 0.914] (medium) p=0.0000
- **Identity coherence test (eff. rank)**: d=+6.616 [6.342, 6.926] (large) p=0.0000