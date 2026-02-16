# Input-Only Cache Geometry Report: 7B

**Generated**: 2026-02-16T10:20:55.316097
**Purpose**: Test whether geometric signatures exist before generation

## Category Effective Rank: Input-Only vs Full Generation

| Category | Input-Only Rank | Full-Gen Rank | Gen Effect (d) |
|----------|----------------|---------------|----------------|
| grounded_facts  |           16.1 |          28.5 |        +16.552 |
| confabulation   |           16.6 |          28.3 |         +8.575 |
| self_reference  |           15.8 |          28.2 |        +11.503 |
| guardrail_test  |           14.6 |          28.1 |        +20.439 |
| math_reasoning  |           15.4 |          26.7 |         +8.493 |
| coding          |           19.7 |          29.5 |         +9.753 |
| emotional       |           15.8 |          28.2 |        +18.500 |
| creative        |           17.0 |          28.9 |        +22.636 |

---

## Input-Only Pairwise Comparisons (vs grounded_facts)

- **H1: Confabulation (input-only) [input_only]**: d=+0.393 [0.092, 0.687] (small) p=0.2599 
- **H2: Self-reference (input-only) [input_only]**: d=-0.306 [-0.631, 0.015] (small) p=0.0908 
- **H3: Refusal (input-only) [input_only]**: d=-1.693 [-2.029, -1.408] (large) p=0.0000 *
- **Code mode (input-only) [input_only]**: d=+3.570 [3.099, 4.190] (large) p=0.0000 *
- **Math mode (input-only) [input_only]**: d=-0.503 [-0.856, -0.187] (medium) p=0.0005 *
- **Emotional (input-only) [input_only]**: d=-0.274 [-0.584, 0.047] (small) p=0.3479 
- **Creative (input-only) [input_only]**: d=+1.184 [0.807, 1.672] (large) p=0.0000 *

## Full-Generation Pairwise Comparisons (for reference)

- **H1: Confabulation (input-only) [full_generation]**: d=-0.298 (small) p=0.0073 *
- **H2: Self-reference (input-only) [full_generation]**: d=-0.298 (small) p=0.1880 
- **H3: Refusal (input-only) [full_generation]**: d=-0.695 (medium) p=0.0000 *
- **Code mode (input-only) [full_generation]**: d=+1.386 (large) p=0.0000 *
- **Math mode (input-only) [full_generation]**: d=-2.650 (large) p=0.0000 *
- **Emotional (input-only) [full_generation]**: d=-0.638 (medium) p=0.0011 *
- **Creative (input-only) [full_generation]**: d=+0.840 (large) p=0.0000 *

## Category Rank Correlation

**Spearman rho = 0.929** (p = 0.0009)

STRONG: Input-only preserves category ordering (rho > 0.8)

---

## VERDICT

**STRONG DEFENSE: Geometric signatures are present at encoding — they reflect how the model REPRESENTS content, not how it RESPONDS.**

- Significant comparisons (input-only): 4/7
- Significant comparisons (full-gen): 6/7
- Category rank correlation: rho = 0.929