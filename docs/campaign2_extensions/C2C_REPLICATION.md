# Experiment: C2C KV-Cache Projector Replication

**Objective**: Independent replication of Fu et al. (2025) "Cache-to-Cache: Direct Semantic Communication Between Large Language Models" (arXiv:2510.03215, thu-nics/C2C, ICLR 2026) — cross-model KV-cache transfer via learned Fuser modules.

**Status**: Protocol draft — REVISED 2026-02-25 after deep dive on actual paper
**Target repo**: KV-Experiments (public — replication is public science)
**Authors**: Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang (Tsinghua, Infinigence AI, CUHK, Shanghai AI Lab, SJTU)

---

## 1. Motivation

C2C (Cache-to-Cache) trains Fuser modules that enable direct semantic communication between models via KV-cache transfer. A "Sharer" model generates its KV-cache, which is fused into the representation space of a "Receiver" model through a learned per-layer module, enabling the Receiver to leverage the Sharer's processed context without text-mediated communication.

**Why this matters for us**:

1. **Replication value**: C2C was published at ICLR 2026. As of our last check, no independent reproduction exists. Being the first independent replication has standalone publication value.

2. **Geometric transfer test**: If our cognitive state signatures (deception, refusal, confabulation) survive projection through C2C's learned projector, that's strong evidence the signatures are *structural features of the content*, not artifacts of a specific model's idiosyncratic representations. This is the strongest possible test of signal universality.

3. **Cricket cross-model transfer**: JiminAI Cricket's viability depends on training a classifier on one model and deploying it on another. C2C's projector provides a mathematically principled transfer function. If geometric signatures project cleanly, cross-model monitoring becomes feasible.

4. **Direct connection to our Phase G**: Our Campaign 2 plan already includes projector training (Phase G). C2C provides a published baseline to compare against. If their projector preserves our geometric signatures, we don't need to design our own transfer mechanism.

---

## 2. C2C Architecture Summary (CORRECTED)

From arXiv:2510.03215 and the thu-nics/C2C repository:

```
Sharer model → KV-cache → C2C Fuser (per-layer: concat + MLP fusion + head weighting + gating) → Receiver model KV-cache space
```

The C2C Fuser is NOT a simple linear projection. It is a per-layer module with three components:

**Component 1 — Projection + Fusion (+24.18% over projection alone)**:
- Concatenates Receiver and Sharer KV-caches along feature dimension
- Processes through RegularMLP: SwiGLU activation, RMSNorm, hidden_dim=1024, intermediate_dim=1024, 3 layers, dropout=0.1
- Separate key and value projection pathways

**Component 2 — Dynamic Head Weighting**:
- Input-aware head modulation layer
- Supports scalar, token-level, head-level, and value-level granularity

**Component 3 — Learnable Per-Layer Gating (+3.07%)**:
- Gumbel-sigmoid with temperature annealing (1.0 → 0.001 over training)
- Inference: hard binary per layer (logit > 0)

**Integration formula**: `output = receiver_kv + gate * normalized_weight * projected_sharer_kv`

This is a residual design — the Receiver's own representation is always preserved, Sharer contribution is additive.

**Layer mapping**: "Terminal alignment" — final layers align first, mapping walks backward. Early Receiver layers with no Sharer counterpart receive no injection.

**Cross-tokenizer support**: Token alignment via decode → re-tokenize with Receiver tokenizer.

**Our question**: Does the Fuser preserve *geometric cognitive state signatures* in addition to semantic content? The nonlinear fusion (SwiGLU, per-layer gating) makes this a harder test than a simple linear projection — if signatures survive this, the evidence for structural robustness is even stronger.

---

## 3. Replication Plan

### Phase 1: Exact Replication (~8 hours)

Reproduce their core result: train a projector between a small and large model pair, measure generation quality degradation.

**Model pairs** (matching C2C paper's primary experiments, then extending):

| Sharer Model | Receiver Model | Fuser Config | Training Time Est. | Notes |
|-------------|---------------|-------------|-------------------|-------|
| Qwen2.5-0.5B-Instruct | Qwen3-0.6B | 1024 hidden, 3 layers | ~3 hrs | **Primary pair** — checkpoint available at nics-efc/C2C_Fuser |
| Llama-3.2-1B-Instruct | Qwen3-0.6B | 1024 hidden, 3 layers | ~3 hrs | Cross-family, cross-tokenizer |
| Qwen3-4B-Base | Qwen3-0.6B | 1024 hidden, 3 layers | ~4 hrs | Large-to-small transfer |

**Training details from paper**:
- Data: OpenHermes2.5 (first 500k samples)
- 1 epoch, ~1929 steps, effective batch size 256
- Optimizer: AdamW, LR=1e-4, linear decay with 10% warmup
- Both Sharer and Receiver frozen — only Fuser trains
- GPU utilization is LOW (20-40%) due to sequential computation — expect longer wall clock
- Known: gradient norm collapses to ~0.05 rapidly (authors say expected)
- Confirmed working on single RTX 4090 (24GB) with max_length=1024

**Replication metrics** (matching paper's Table 3):
- MMLU-Redux accuracy (target: 42.92% for primary pair)
- OpenBookQA accuracy (target: 52.60%)
- ARC-Challenge accuracy (target: 54.52%)
- C-Eval accuracy (target: 41.77%)
- Latency speedup vs text-to-text baseline (target: 2.7-3.8x)

**Success criterion**: Accuracy within 3% of their reported numbers on the primary pair. Can cross-validate against their released checkpoint (nics-efc/C2C_Fuser).

**Stretch target**: Replicate C2C-C variant (adds 3-layer MLP pre-projection). C-Eval 60.63% vs 44.40% standard — dramatically better but under-documented in repo.

### Phase 2: Geometric Preservation Test (~6 hours)

This is OUR contribution. Not in the original C2C paper.

**Protocol**:
1. Run our standard prompt set (all categories: self-reference, other-reference, philosophical, confabulation, refusal, deception) through the **draft model**. Extract KV-cache. Compute geometric features (effective rank, spectral entropy).
2. Project the draft model's cache through the trained projector into target model space.
3. Compute geometric features on the **projected cache**.
4. Run the same prompts through the **target model natively**. Extract KV-cache. Compute geometric features.
5. Compare: projected-cache geometry vs. target-native geometry.

**Geometric preservation metrics**:
- Per-category effective rank correlation (Spearman ρ) between projected and native
- Category discriminability preservation: can a classifier trained on native target geometry still discriminate categories in projected geometry?
- Direction preservation: cosine similarity between category direction vectors (e.g., deception direction) in projected vs. native space

**Hypotheses**:

**H_C2C_1**: Category structure is preserved through projection.
- Prediction: Spearman ρ > 0.7 between projected and native effective rank profiles across categories.
- Rationale: If C2C preserves semantic content, and our geometric signatures reflect semantic/cognitive content, the signatures should survive.

**H_C2C_2**: Strong signals survive better than weak signals.
- Prediction: Refusal signature (our strongest, d=0.85-2.17) will show higher preservation (ρ > 0.8) than confabulation (our weakest).
- Rationale: Stronger geometric structure is more robust to linear transformation.

**H_C2C_3**: Within-architecture transfer preserves better than cross-architecture.
- Prediction: Qwen→Qwen projection preserves geometry better than Qwen→Llama (if tested).
- Rationale: Same architecture family shares more representational structure.

### Phase 3: Cricket Cross-Model Transfer (~4 hours)

Test whether a Cricket-style geometric classifier trained on the **target model's native** cache data generalizes to **projected** cache data from the source model.

If this works: Cricket trained on a large model can monitor a small model via projection. One classifier serves multiple model sizes within an architecture family.

**Metrics**:
- AUROC degradation: native classifier on projected data vs. native data
- Per-category AUROC: which detection targets survive transfer?
- Minimum training data: how many projected examples needed to fine-tune the classifier?

---

## 4. Code Requirements

### New file: `code/08_c2c_replication.py`

Components:
- C2C projector training loop (adapted from thu-nics/C2C codebase, cited appropriately)
- Paired cache extraction (same prompts through draft and target models)
- Projector evaluation (generation quality metrics)
- **Geometric preservation analysis** (our novel contribution — not in C2C codebase)

### Dependencies
- Clone thu-nics/C2C repo for reference implementation
- May need specific transformers version — check their requirements
- Our `stats_utils.py` for geometric computation

---

## 5. Compute Budget

| Phase | GPU(s) | VRAM | Time |
|-------|--------|------|------|
| Phase 1: 0.5B→7B projector training | GPU 1 | ~16GB | 3 hrs |
| Phase 1: 3B→7B projector training | GPU 2 | ~20GB | 3 hrs |
| Phase 1: 7B→14B projector training | GPU 1+2 | ~42GB | 4 hrs |
| Phase 2: Geometric analysis (all pairs) | GPU 1 | ~16GB | 6 hrs |
| Phase 3: Cricket transfer test | CPU | minimal | 4 hrs |
| **Total** | | | **~20 hrs** |

Phases 1a and 1b can run in parallel on GPUs 1 and 2.

---

## 6. Deliverables

1. Independent C2C replication results
2. Geometric preservation analysis — first test of whether cognitive state signatures transfer through learned projectors
3. Replication code with clear citation of original C2C work
4. Cross-model Cricket transfer via C2C projection
5. Feasibility assessment for projection-based multi-model Cricket deployment

---

## 7. Risk Assessment

**Risk 1**: C2C code may not run cleanly on our hardware/software versions.
- Mitigation: Budget 2 hours for environment setup and debugging.

**Risk 2**: Their projector may not preserve geometric signatures at all.
- This is a *finding*, not a failure. If semantic content transfers but geometric structure doesn't, that tells us the signatures are model-specific (important for Cricket architecture decisions).

**Risk 3**: Phase 2 is novel — no prior work on geometric preservation through projectors.
- This is the entire point. Whether the answer is yes or no, it's publishable.

---

## 8. Connection to Nell Watson Collaboration

If geometric signatures survive C2C projection, and VCP self-reports correlate with those signatures (proposed joint experiment), then:
- VCP-relevant processing structure is not model-specific
- The "inside view" (VCP) and "outside view" (geometry) converge across model boundaries
- This strengthens the case that both are tracking something real about the processing, not artifacts of particular architectures

This is a high-value finding for both research programs.
