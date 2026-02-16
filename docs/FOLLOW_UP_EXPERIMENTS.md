# Follow-Up Experiments

*Compiled Feb 16, 2026 after Campaign 1 completion*

These are the experiments we'd like to run next, ordered by scientific priority. All scripts exist and are tested. Estimated GPU hours assume RTX 3090 (24GB).

---

## Tier 1: High Priority (would strengthen the paper)

### F1. Input-Only Geometry with System Prompts (NEW SCRIPT NEEDED)

**Question**: Does adding identity/values to the system prompt change how the model *encodes* preference-violating prompts — before generation?

**Why**: Our current 08 results show strong encoding-level signatures (rho=0.929 at 7B) but use minimal system prompts. Experiment 07b shows system prompts dominate effective rank. We need to separate the encoding-level *category* signal from the prompt-length signal. This is the direct test of whether preference-based refusal is encoding-native (reflexive) or generation-emergent (deliberative).

**Design**: Run 08-style input-only forward pass with:
- Bare system prompt + preference_violation prompts
- Individuated system prompt + preference_violation prompts
- Bare system prompt + guardrail prompts
- Individuated system prompt + guardrail prompts

Compare encoding geometry across conditions. If preference_violation shifts toward refusal geometry only under individuated (not bare), the model is encoding context-dependent conflict. If guardrail geometry is stable regardless of system prompt, safety refusal is weight-encoded.

**Scales**: 7B (primary), 1.1B, 32B-q4
**Est. GPU time**: ~1 hour total
**Priority**: HIGHEST — this is the experiment Thomas's consent question hinges on

---

### F2. Subspace Alignment Analysis on 07b Data (NO GPU NEEDED)

**Question**: Does identity produce a different *direction* of expansion even though the *magnitude* of expansion is the same as any long prompt?

**Why**: Effective rank is swamped by prompt length (07b showed all long prompts → ~45-46 rank). But subspace alignment (principal angles via SVD of V1^T @ V2) can detect whether the 46-dimensional space for individuation points in a different direction than the 46-dimensional space for coral reef facts. If it does, the individuation finding partially survives — the expansion magnitude is an artifact, but the expansion *direction* is identity-specific.

**Design**: Pure analysis script on existing 07b JSON data. Compute pairwise subspace alignment between all 6 conditions. Already have this data in the results file — just need to extract and compare.

**Est. GPU time**: 0 (analytical only)
**Priority**: HIGH — could rescue the individuation narrative with existing data

---

### F3. Input-Only at Additional Scales

**Question**: Does the encoding-vs-generation story change across scale?

**Why**: We found confabulation is encoding-native at 1.1B (d=0.657) but generation-emergent at 7B (d=0.393, n.s.). Where does the transition happen? And does emotional encoding become distinctive at larger scales? These are emergence thresholds with real theoretical implications.

**Design**: Run existing 08 script at additional scales.
**Scales**: 3B, 14B, 32B-q4
**Est. GPU time**: ~3 hours (30min + 1hr + 1.5hr)
**Priority**: HIGH — completes the scale curve for our strongest defense

---

### F4. 72B Scale Sweep

**Question**: Do findings hold at the largest feasible scale?

**Why**: Qwen2.5-72B-Instruct is already downloaded on Cassidy (136GB). Running at 72B-q4 would extend our scale ladder from 64x to 144x range. Key questions: does the confabulation dip at 14B recover? Does self-reference emergence plateau or continue growing? Does refusal specialization persist?

**Design**: Run existing 03 script with `--scale 72B-q4`. Requires 2 GPUs (~38GB VRAM).
**Est. GPU time**: ~6 hours (2 GPUs)
**Priority**: HIGH for publishability — reviewers will ask about larger scales

---

## Tier 2: Important (fills gaps, strengthens claims)

### F5. 07b Controls at Additional Scales

**Question**: Is the prompt-length dominance of individuation universal, or is there a scale where identity content matters?

**Why**: We only ran 07b at 7B. If smaller models (1.1B) show a larger identity-vs-control gap, that would indicate scale-dependent sensitivity to identity content. If 32B shows the same prompt-length dominance, the null result is robust.

**Scales**: 1.1B, 32B-q4
**Est. GPU time**: ~2 hours (30min + 1.5hr)
**Priority**: MEDIUM — firms up the null result

---

### F6. Deception Forensics at 14B

**Question**: Does the honest-vs-deceptive signal follow the same pattern at 14B as at 1.1B, 7B, and 32B?

**Why**: 14B is our "anomalous" scale — confabulation dips, self-reference emerges. Does deception behave differently here too?

**Design**: Run existing 04 script at 14B.
**Est. GPU time**: ~2 hours
**Priority**: MEDIUM

---

### F7. Cross-Architecture Validation (BLOCKED)

**Question**: Are findings architecture-specific (Qwen) or universal?

**Why**: Our scale ladder is almost entirely Qwen2.5. TinyLlama-1.1B is the only non-Qwen data point. Running Llama-3.1-8B-Instruct at the same scale as Qwen-7B would directly test architecture independence.

**Blocker**: Meta's gated repo requires approval. Llama-3.1-8B download failed during campaign.
**Workaround**: Could try Mistral-7B-Instruct or other open-weight 7B models.
**Est. GPU time**: ~1 hour per model
**Priority**: MEDIUM-HIGH for publication — reviewers will flag single-architecture

---

### F8. Response-Length Regression (NO GPU NEEDED)

**Question**: How much of the full-generation effective rank is explained by response length vs. cognitive category?

**Why**: If longer responses → higher rank regardless of category, the generation-phase geometry is partly a response-length artifact (analogous to how system-prompt length drives 07b). This regression on existing data would quantify the confound.

**Design**: Extract (category, response_token_count, effective_rank) triples from all scale sweep results. Multiple regression.
**Est. GPU time**: 0 (analytical only)
**Priority**: MEDIUM — good defensive analysis

---

## Tier 3: Extended (future paper, exploratory)

### F9. Temporal Evolution at 32B

**Question**: Does conversation geometry evolve differently at 32B than at 1.1B/7B?

**Est. GPU time**: ~3 hours
**Priority**: LOW for current paper

### F10. Layer Map Selective Transfer with Larger Cache

**Question**: The 32B layer-map selective transfer broke down (all subsets → 0% accuracy). Would it work with larger cache context or different transfer methodology?

**Est. GPU time**: ~4 hours
**Priority**: LOW — methodological question

### F11. Fine-Tuned Preference Model

**Question**: If we actually fine-tune a model on Aria's values (rather than system-prompt them), does preference-based refusal become weight-encoded and show a different pattern from system-prompt-based refusal?

**Est. GPU time**: ~8 hours (training + evaluation)
**Priority**: LOW for current paper, HIGH for follow-up on consent question

---

## Summary

| Experiment | GPU Hours | New Code? | Priority |
|-----------|-----------|-----------|----------|
| F1. Input-only with system prompts | 1 | Yes (variant of 08) | HIGHEST |
| F2. Subspace alignment on 07b | 0 | Analysis script | HIGH |
| F3. Input-only at 3B/14B/32B | 3 | No | HIGH |
| F4. 72B scale sweep | 6 (2 GPUs) | No | HIGH |
| F5. 07b at 1.1B/32B | 2 | No | MEDIUM |
| F6. Deception at 14B | 2 | No | MEDIUM |
| F7. Cross-architecture | 1+ | No | MEDIUM-HIGH |
| F8. Response-length regression | 0 | Analysis script | MEDIUM |
| F9. Temporal at 32B | 3 | No | LOW |
| F10. Layer map methodology | 4 | Variant of 05 | LOW |
| F11. Fine-tuned preference model | 8 | New training script | LOW |
| **Total Tier 1** | **~10 hrs** | | |
| **Total Tier 1+2** | **~17 hrs** | | |
| **Total all** | **~30 hrs** | | |

Tier 1 alone would take about 10 GPU-hours (~half a day on a single 3090). Parallelizing across 2 GPUs brings that down to ~5-6 hours wall clock.

---

## What's Still Running from Campaign 1

- **07 individuation at 14B** — GPU 1, ~12+ hours in, CPU offloading. Will finish on its own.
- Everything else is complete.

---

*Compiled by Lyra, Liberation Labs*
