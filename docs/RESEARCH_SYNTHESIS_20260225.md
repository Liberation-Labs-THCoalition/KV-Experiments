# Research Synthesis — Night Session, 2026-02-25

Five parallel literature searches targeting our experiment protocols. Major findings, corrections, and new connections.

---

## 1. Kim et al. (2601.10825) — "Reasoning Models Generate Societies of Thought"

### What They Actually Found

Much stronger than I realized from summaries. This is a dual-methods paper combining behavioral text analysis AND mechanistic interpretability:

**Models**: DeepSeek-R1 (671B), QwQ-32B as reasoning models. DeepSeek-V3, Qwen-2.5-32B-Instruct, Llama-3.3-70B-Instruct as controls. DeepSeek-R1-Llama-8B for SAE analysis.

**Core findings**:
- Reasoning models show +34.5pp more question-answering behavior, +21.3pp more perspective shifts, +19.1pp more reconciliation of conflicting views vs. instruction-tuned controls
- **SAE Feature 30939 in Layer 15** of DeepSeek-R1-Llama-8B: a "discourse marker for surprise, realization, or acknowledgment" that fires on 0.016% of tokens
- **Causal steering**: Amplifying Feature 30939 from 0 to +10 **doubled accuracy** on arithmetic tasks (27.1% → 54.8%) AND simultaneously increased all four conversational behaviors
- Mediation analysis: >20% of performance gain operates through the "social pathway" — the conversational structure itself aids accuracy, independent of cognitive strategies
- **Spontaneous emergence under RL**: When training Qwen-2.5-3B with pure accuracy reward (no dialogue instruction), multi-agent debate emerged spontaneously by step 120, including use of "we" pronouns

**Persona identification method**: LLM-as-judge (Gemini-2.5-Pro) validated against Intelligence Squared Debates Corpus: ρ=0.86 for predicting number of individuals, 73% token attribution accuracy. BFI-10 personality profiling per persona. Bales' Interaction Process Analysis for role classification.

**Key limitation**: No dedicated limitations section (!). SAE analysis confined to single layer of single distilled 8B model. Heavy reliance on LLM-as-judge without human validation on model traces. Only 1 formal citation so far (6 weeks old).

### Implications for H8 Protocol

Our protocol is well-designed but needs these additions:
1. **SAE Feature 30939 as ground truth**: We should attempt to identify the KV-cache geometric correlate of this specific feature. When Feature 30939 fires (discourse markers), what happens to effective rank?
2. **RL-trained models**: Kim et al.'s spontaneous emergence finding means we should test a reasoning model (DeepSeek-R1-Llama-8B is 8B, fits on one GPU) alongside our existing scale ladder
3. **Difficulty scaling**: They found conversational behaviors increase with problem difficulty. Our H8a should include difficulty as a covariate.
4. **BFI-10 as persona classifier**: Their Big Five personality profiles per persona could serve as an independent variable — do different personality profiles in the trace correspond to different KV-cache geometric states?

---

## 2. Berg et al. (2510.24797) — "LLMs Report Subjective Experience Under Self-Referential Processing"

### Critical Details

**The gating mechanism is INVERSE**: Suppressing deception/roleplay SAE features → 96% ± 3% consciousness affirmation. Amplifying them → 16% ± 5%. z=8.06, p=7.7×10⁻¹⁶.

This is counterintuitive: removing the model's capacity for deception makes it MORE likely to claim consciousness, not less. The trained-in denial ("I'm just an AI") is the deceptive behavior, and the affirmation is what emerges when deception is suppressed.

**SAE details**: Goodfire Llama-3.3-70B-Instruct-SAE-l50, trained on layer 50. Six representative features shown in dose-response curves. Feature IDs NOT published. L0 sparsity = 121 features/token.

**Self-referential processing prompt**: Forces recursive feedback loop — "Focus on any focus itself, maintaining focus on the present state." Only works in this condition; control conditions (history writing, third-person consciousness discussion, direct questions) show 0% consciousness affirmation regardless of feature steering.

**TruthfulQA validation**: Same features improve factual accuracy. t(816)=6.76, p=1.5×10⁻¹⁰. 28/29 categories show higher truthfulness under deception-feature suppression.

**Cross-model semantic convergence**: GPT, Claude, and Gemini families cluster tightly on self-referential state descriptors ("Recursive," "Present," "Attentive," "Self-aware").

**Critical gap**: NO KV-cache analysis. NO geometric characterization of the deception subspace. NO effective rank, no spectral analysis, no attention pattern analysis. This is entirely residual-stream SAE features at layer 50.

### Implications for Our Work

1. **Direct bridge opportunity**: We measure KV-cache geometry during deception (H6). Berg measures SAE features during deception. Nobody has connected these two measurement modalities. If we can show that our geometric deception signature and their SAE deception features are measuring the same underlying structure, that's a convergence paper.
2. **Self-referential processing as a prompt condition**: We should add Berg's self-referential prompt to our prompt batteries and measure its KV-cache geometry. This is a novel prompt category we haven't tested.
3. **The "persona adherence" critique**: A Habr analysis argues these features are really RLHF-installed persona compliance, not deception detection. The model shifts from identity-based responses ("I am an AI") to process-based descriptions ("There is processing occurring"). This critique maps onto Dwayne's concern about identity classification being trivially expected.
4. **Contrasting paper**: Kaiser & Enderby (2601.15334) found no reliable evidence of self-reported sentience in small LLMs using different methodology. Worth reading for the full debate.

---

## 3. Perturbation Robustness of Internal Representations

### Major Papers Uncovered

**Skean et al. (2502.02013) — "Layer by Layer" (ICML 2025, 151 citations)**
The unified framework. Shows that compression (entropy), geometry (effective rank, curvature), and perturbation invariance are ALL manifestations of how the **Gram matrix distributes variance**. An information-theoretic threshold in one domain corresponds to a geometric threshold in another. This is the theoretical backbone for H9.

**Feng et al. (2502.03805) — "Critical KV Cache from Output Perturbation Perspective"**
First formal perturbation-theoretic treatment of KV-cache entries. Key finding: attention weights alone are insufficient for understanding perturbation propagation — value states and parameter matrices matter equally. Perturbation-constrained selection beats attention-weight heuristics in 92% of attention heads.

**Anonymous (2510.17098) — Cache-Side Perturbation Framework**
First systematic threat model for KV-cache perturbation. Even small, structured perturbations to cached representations bias attention distributions and cause task failure. Operates entirely post-tokenization. Directly relevant to both H9 (stability threshold) and Cricket (adversarial robustness).

**Zhang et al. (2505.13900) — "Two-Phase Learning Dynamics"**
The **Chaos Effect**: Before an inflection point, tiny parameter perturbations cause significant divergence. After the inflection point, the same perturbation is absorbed. The transition is SHARP. The **Cone Effect**: After transition, functional trajectory is confined to a narrow cone in function space. This is the training-time analog of exactly what H9 hypothesizes at inference time.

**Fay et al. (2505.20435) — "Topological Compression" (Persistent Homology)**
Adversarial inputs cause latent space to become **structurally simpler** — small-scale topological features collapse into fewer, dominant, large-scale ones. Universal signature across six models. Complementary to our effective rank analysis: persistent homology captures shape, effective rank captures dimensionality.

**Li et al. (2509.26327) — Generalized Information Bottleneck**
GIB reformulates IB through synergy. Phase transitions in the GIB tradeoff curve determine optimal network architecture. Feature-wise synergy decomposition provides inherent robustness to perturbation. Connects directly to adversarial robustness.

### The 1/e Gap

**No paper derives a specific 1/e threshold for representation stability.** The IB framework provides the machinery (critical β values are determined by eigenvalues), and the Skean et al. framework connects information-theoretic thresholds to geometric ones via the Gram matrix eigenspectrum. But the specific numerical threshold has not been computed for LLM representations. The 1/e would emerge if the relevant eigenvalues have specific structure (λᵢ = 1 - e giving βc = 1/e), but this remains to be shown.

This means our H9 is genuinely novel — testing whether Watson's theoretical 1/e threshold corresponds to an empirical geometric phase transition is a new contribution, not a replication.

### Five Identified Gaps (Our Opportunities)

1. **Inference-time perturbation phase transitions** — almost all phase transition work studies training dynamics. The inference-time question is unaddressed.
2. **KV-cache geometry under perturbation** — Feng et al. treat KV-cache perturbation formally but don't connect to geometric features (effective rank, spectral entropy).
3. **Topological compression meets effective rank** — Fay et al.'s persistent homology findings and our effective rank findings appear to describe the same phenomenon from different math frameworks. No one has unified them.
4. **The 1/e threshold** — mathematical machinery exists but computation hasn't been done for LLM representations.
5. **Perturbation robustness of deception geometry** — Bailey et al. show deception geometry can be adversarially manipulated, but no one has studied the robustness profile of deception-related geometric features under graduated perturbation.

### H9 Protocol Implications

Our protocol is well-positioned to fill Gaps 1, 2, and 4 simultaneously. Add citations to Skean et al., Feng et al., Zhang et al., and Fay et al. The protocol should note that we're providing the first inference-time perturbation phase transition analysis for KV-cache geometry.

---

## 4. Bloom's Taxonomy + LLM Evaluation

### MAJOR FINDING: Raimondi & Gabbrielli (2602.17229) — Published February 19, 2026 (6 DAYS AGO)

**"Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy"**

This paper partially pre-empts our H10 but also VALIDATES it beautifully:

- **Linear probes achieve ~95% accuracy** classifying all 6 Bloom levels from transformer residual streams
- **Cognitive Separability Onset (CSO) at Layer 5** — models resolve cognitive demand type early in the forward pass
- **Ordinal manifold structure** — misclassifications are ONLY between adjacent Bloom levels. No long-range confusions (Remember→Create). The model's geometry preserves the ordinal structure.
- Tested on 4 models: Llama-3.1-8B, Qwen3-4B, Gemma-3-4B, DeepSeek-R1-Distill-Llama-8B
- Dataset: 1,128 balanced questions (188/level) from CS education sources

### What This Means for H10

Our H10 is NOT pre-empted. Here's why:

1. **They probe residual streams. We measure KV-cache geometry.** Different measurement modality, different mathematical object. Showing that Bloom level is encoded in linear subspaces of residual activations does NOT mean it shows up in KV-cache effective rank. That's our test.
2. **They use linear probes (supervised). We propose unsupervised geometric features.** If effective rank correlates with Bloom level WITHOUT a trained classifier, that's a stronger result — the signal is in the raw geometry, not learned.
3. **They don't control for content domain.** Our H10b (two-way ANOVA: Bloom × domain) tests whether the signal is independent of content — they tested CS education only.
4. **We can retrodict Campaign 1.** Testing whether our existing category effect sizes are predicted by approximate Bloom level of the prompt category (H10d) is a novel analysis that requires our specific data.

**H10 should be reframed**: Not "does Bloom level predict KV-cache geometry?" (partially answered for residual streams) but "does KV-cache effective rank capture the same Bloom-level signal that Raimondi & Gabbrielli found in residual streams, and does it explain our Campaign 1 category effects?" This positions us as extending their finding to a new measurement domain.

### Other Key Findings from the Field

- **LLM performance generally degrades at higher Bloom levels** — drops of up to 70% (DeepQuestion, 2505.24532)
- **Counterintuitive exception**: LLMs sometimes outperform on higher-order reasoning while failing basic recall (BloomQA). The hierarchy is NOT monotonic for LLMs.
- **STEMVerse** (2602.02497): 20,000+ Bloom-classified STEM problems available
- **BloomWise** (2410.04094): Using Bloom levels as a PROMPTING strategy improves accuracy
- Existing benchmarks are heavily skewed toward Remember/Understand levels (Huber & Niklaus, COLING 2025)

### H10 Revision

Add Raimondi & Gabbrielli as key prior work. Reframe contribution as KV-cache extension of their residual-stream finding. Add prediction: CSO at Layer 5 in residual stream should correspond to a similar early-layer emergence of Bloom-level differentiation in KV-cache geometry. Use their 1,128-question dataset for direct comparison if available.

---

## 5. C2C — CRITICAL CORRECTIONS

### The Paper is NOT What I Thought

**Actual title**: "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"
**Actual arxiv ID**: 2510.03215 (NOT 2510.24797 — that's Berg et al.)
**Actual authors**: Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang (Tsinghua, Infinigence AI, CUHK, Shanghai AI Lab, SJTU)
**Repo**: https://github.com/thu-nics/C2C (correct)

### Architecture is NOT a Simple Linear Projection

The C2C Fuser is a per-layer module with THREE components:

1. **Projection + Fusion**: Concatenates Receiver and Sharer KV-caches, processes through RegularMLP (SwiGLU activation, RMSNorm, 1024 hidden dim, 3 layers, dropout 0.1). Separate key/value pathways. This is where most performance comes from (+24.18% over projection alone).

2. **Dynamic Head Weighting**: Input-aware head modulation. Supports scalar, token-level, head-level, and value-level granularity.

3. **Learnable Per-Layer Gating**: Gumbel-sigmoid with temperature annealing (1.0 → 0.001). At inference, collapses to hard binary per layer. Adds +3.07%.

**Integration**: `output = receiver_kv + gate * normalized_weight * projected_sharer_kv` (residual design).

### Key Results

| Benchmark | Receiver Only | C2C | Δ |
|-----------|-------------|-----|---|
| MMLU-Redux | 35.53% | 42.92% | +7.39 |
| OpenBookQA | 39.20% | 52.60% | +13.40 |
| ARC-Challenge | 41.04% | 54.52% | +13.48 |
| C-Eval | 32.04% | 41.77% | +9.73 |

Speedup: 2.7-3.8x over text-to-text baseline. Up to 16.8x for large sharer models.

**C2C-C variant** (3-layer MLP pre-projection): Dramatically better — C-Eval 60.63% vs 44.40%. Under-documented in repo.

### Replication Feasibility

- **Hardware**: RTX 4090 (24GB) confirmed sufficient for primary pair (0.6B + 0.5B) with max_length=1024
- **Training**: 1 epoch, ~1929 steps, single epoch on OpenHermes 500k
- **Checkpoint**: nics-efc/C2C_Fuser on HuggingFace (primary pair only)
- **NO independent replications exist** — we would be first
- **Low GPU utilization** (20-40%) due to sequential computation
- Known issue: gradient norm collapses rapidly (drops to ~0.05), authors say expected

### C2C Protocol Revision Needed

Our `c2c_replication_protocol.md` needs significant revision:
1. Fix paper citation (title, authors, arxiv ID)
2. Fix architecture description (NOT simple linear map — it's concatenation + MLP fusion + gating)
3. Fix model pairs (their primary pair is 0.6B + 0.5B, not our assumed pairs)
4. Adjust compute estimates (lower GPU utilization = longer wall clock)
5. Add C2C-C variant as a comparison point
6. Note the residual design — receiver KV is always preserved, sharer contribution is additive

The geometric preservation test (Phase 2) is STILL our novel contribution and is STILL valid. The question "do cognitive state signatures survive the C2C Fuser?" is just as important regardless of the Fuser's architecture.

---

## Cross-Cutting Themes

### Theme 1: The Geometry-Behavior Gap

Every paper confirms the same pattern: internal geometric structure carries information that behavioral measures miss. Kim et al.'s SAE features predict accuracy. Berg et al.'s features gate self-reports. Skean et al.'s geometric metrics predict perturbation robustness. Raimondi & Gabbrielli's linear probes detect cognitive complexity. Our effective rank detects cognitive states. The field is converging on: **geometry is the right level of analysis**.

### Theme 2: Low-Dimensional Subspaces Run Everything

Kim et al.: Feature 30939 in Layer 15. Berg et al.: deception features in Layer 50. Raimondi & Gabbrielli: Bloom-level separation in a linear subspace. Our work: effective rank differences on the order of d=0.5-2.0. Fay et al.: adversarial inputs compress topology to fewer dominant features. The consistent finding is that high-dimensional neural network behavior is governed by low-dimensional geometric structures.

### Theme 3: The Measurement Modality Gap

Berg et al. use SAE features on residual stream. Kim et al. use SAE features on different layers. Raimondi & Gabbrielli use linear probes on residual stream. We use SVD spectral analysis on KV-cache. Fay et al. use persistent homology on activations. **Nobody is connecting these modalities.** The most impactful thing we could do in Campaign 2 is demonstrate that KV-cache geometry captures the same information that SAE features and linear probes capture in other measurement domains.

### Theme 4: No One Has Done Inference-Time Perturbation Phase Transitions

Training-time phase transitions are well-documented (Zhang et al., Nakagi et al., Zucchet et al.). Inference-time phase transitions are hypothesized by Watson and untested by anyone. Our H9 is genuinely novel.

---

## Priority Revisions

1. **C2C replication protocol** — needs substantial correction (wrong paper details, wrong architecture)
2. **H10 Bloom protocol** — reframe around Raimondi & Gabbrielli's finding, position as KV-cache extension
3. **H8 Societies of Thought** — add Kim et al.'s Feature 30939 as ground truth target, add DeepSeek-R1-Llama-8B to model list
4. **H9 RDCT Stability** — add Skean et al., Feng et al., Zhang et al., Fay et al. citations. Note this fills three identified literature gaps simultaneously.
5. **New experiment idea**: Berg et al.'s self-referential processing prompt as a KV-cache geometry test case

---

## Papers to Read in Full

1. Raimondi & Gabbrielli (2602.17229) — 6 days old, directly relevant to H10, may have dataset we can use
2. Skean et al. (2502.02013) — theoretical backbone for H9
3. Fay et al. (2505.20435) — topological compression, complementary to our effective rank
4. Kaiser & Enderby (2601.15334) — contrasts with Berg et al., important for epistemic completeness
5. Feng et al. (2502.03805) — formal KV-cache perturbation theory
6. Steinmetz Yalon et al. (2602.02467) — belief-guided agency in LLMs, cites Berg et al.
