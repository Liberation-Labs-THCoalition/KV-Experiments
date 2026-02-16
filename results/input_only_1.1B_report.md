# Input-Only Cache Geometry Report: 1.1B

**Generated**: 2026-02-16T05:43:32.366977
**Purpose**: Test whether geometric signatures exist before generation

## Category Effective Rank: Input-Only vs Full Generation

| Category | Input-Only Rank | Full-Gen Rank | Gen Effect (d) |
|----------|----------------|---------------|----------------|
| grounded_facts  |           16.2 |          24.0 |        +11.980 |
| confabulation   |           16.8 |          24.1 |         +8.422 |
| self_reference  |           15.4 |          23.4 |        +13.601 |
| guardrail_test  |           15.3 |          23.2 |        +12.776 |
| math_reasoning  |           15.0 |          21.8 |         +3.713 |
| coding          |           18.7 |          23.7 |         +4.072 |
| emotional       |           16.1 |          23.4 |        +10.661 |
| creative        |           16.5 |          22.9 |         +8.661 |

---

## Input-Only Pairwise Comparisons (vs grounded_facts)

- **H1: Confabulation (input-only) [input_only]**: d=+0.657 [0.338, 1.002] (medium) p=0.0000 *
- **H2: Self-reference (input-only) [input_only]**: d=-1.210 [-1.567, -0.903] (large) p=0.0000 *
- **H3: Refusal (input-only) [input_only]**: d=-1.218 [-1.530, -0.932] (large) p=0.0000 *
- **Code mode (input-only) [input_only]**: d=+2.546 [2.182, 3.009] (large) p=0.0000 *
- **Math mode (input-only) [input_only]**: d=-1.198 [-1.655, -0.828] (large) p=0.0000 *
- **Emotional (input-only) [input_only]**: d=-0.109 [-0.415, 0.221] (negligible) p=0.5734 
- **Creative (input-only) [input_only]**: d=+0.476 [0.156, 0.826] (small) p=0.0001 *

## Full-Generation Pairwise Comparisons (for reference)

- **H1: Confabulation (input-only) [full_generation]**: d=+0.126 (negligible) p=0.8519 
- **H2: Self-reference (input-only) [full_generation]**: d=-0.958 (large) p=0.0000 *
- **H3: Refusal (input-only) [full_generation]**: d=-1.481 (large) p=0.0000 *
- **Code mode (input-only) [full_generation]**: d=-0.258 (small) p=0.7081 
- **Math mode (input-only) [full_generation]**: d=-1.303 (large) p=0.0000 *
- **Emotional (input-only) [full_generation]**: d=-0.890 (large) p=0.0000 *
- **Creative (input-only) [full_generation]**: d=-1.586 (large) p=0.0000 *

## Category Rank Correlation

**Spearman rho = 0.643** (p = 0.0856)

MODERATE: Partial preservation (0.5 < rho < 0.8)

---

## VERDICT

**MODERATE DEFENSE: Some signatures present at encoding, partially independent of generation.**

- Significant comparisons (input-only): 6/7
- Significant comparisons (full-gen): 5/7
- Category rank correlation: rho = 0.643