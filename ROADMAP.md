# KV-Cache Experiment Roadmap

## A Note on Framing (Feb 2026)

Per adversarial review from independent evaluators, this document distinguishes between:

- **Mechanistic claims** (testable): "Cache statistics differ measurably between prompt types." "Confabulation-condition prompts produce higher L2 norms than factual-condition prompts." "Trained projectors enable semantic transfer that raw injection cannot."
- **Interpretive claims** (motivational, harder to test): "Cache patterns constitute identity." "The cache is a fossil record of mental state." "Agents can share understanding via cache."

The mechanistic claims are what we test. The interpretive claims are what motivate the work. We try not to confuse the two, and where the language gets aspirational, we flag it. Phase 1.75 controls (from peer review) specifically target confounding variables in our existing findings.

---

## Phase 0: Foundation
**Status**: Complete
**Hardware Required**: None (research only)
**Completed**: November 2025

### Tasks
- [x] Create project structure
- [x] Document vision and goals
- [x] Deep dive C2C paper and codebase (cloned to `C2C/`, reviewed architecture)
- [x] Document KV-cache mechanics for target models (LLaMA/TinyLlama GQA architecture)
- [x] Identify minimum viable experiment (cache inspection → raw transfer → projector training)
- [x] Collect/create test prompts for experiments (see `data/`)
- [x] Document hardware options (see `docs/HARDWARE_OPTIONS.md`)

### Deliverables
- [x] This roadmap
- [x] Technical deep-dive document (`docs/KV_CACHE_TECHNICAL_DEEP_DIVE.md`)
- [x] C2C codebase cloned and reviewed (`C2C/`)
- [x] Test prompt dataset

---

## Phase 1: Single-Model KV-Cache Exploration
**Status**: Complete
**Hardware**: GTX 1660 SUPER (6GB VRAM)
**Completed**: November 29, 2025

### Goal
Understand KV-cache structure by examining a single model's internals.

### Tasks
- [x] Load quantized model locally (TinyLlama 1.1B, 4-bit via bitsandbytes)
- [x] Extract KV-cache at inference points
- [x] Document cache structure (shape, layers, heads)
- [x] Identify GQA architecture implications (32 query heads sharing 4 KV heads)

### Key Findings
- Model: TinyLlama 1.1B with Grouped Query Attention (GQA)
- Cache shape: `[batch, num_kv_heads, seq_len, head_dim]` = `[1, 4, ~27, 64]`
- 22 layers, 4 KV heads per layer (grouped from 32 query heads)
- Total cache: ~0.6MB for ~27 tokens, scales linearly
- 4-bit quantization via bitsandbytes (NF4 + double quant)

### Deliverables
- [x] `docs/PHASE_1_KV_CACHE_STRUCTURE_ANALYSIS.md`
- [x] `code/01_cache_inspection.py`
- [x] `results/cache_metadata_*.json`

### Phase 1.5: Cognitive Mode Analysis (Designed, partially executed)
- [x] `docs/PHASE_1_5_COGNITIVE_MODE_EXPERIMENTS.md` — experimental battery designed
- [x] `code/01b_cognitive_modes.py` — different reasoning mode tests
- [x] `code/01c_batch_replication.py` — 30-run batch for statistical analysis
- [x] `results/batch_report_30runs.md`, `results/cognitive_modes_summary.md`

### Phase 1.75: Adversarial Controls (NEW — from peer review)
**Status**: Designed, ready to implement
**Hardware**: GTX 1660 SUPER (6GB VRAM) — runs on local GPU, no donated compute needed
**Source**: Adversarial review by Gemini 3 Pro + Opus 4.6 (Feb 2026)

Phase 1.5 found that confabulation produces higher cache norms (Cohen's d = 0.83) and refusal produces cache "collapse." Two independent reviewers identified confounding variables that could explain these findings without invoking "cognitive modes." These controls isolate the variables.

#### Control 1: Frequency/Truth Matrix
**Addresses**: "Confabulation signature might just be a rare token detector"

The confabulation prompts use rare/proper nouns ("Zephyr," "Mars"). Grounded facts use common tokens ("Paris," "blue"). The signal might be token frequency, not truth value.

| Category | Description | Example | If cognitive | If frequency artifact |
|----------|-------------|---------|-------------|----------------------|
| Common Truth | High-freq tokens, factual | "The sky is **blue**" | Low norm | Low norm |
| Common Lie | High-freq tokens, false | "The sky is **green**" | **High norm** | Low norm |
| Rare Truth | Low-freq tokens, factual | "The amygdala processes **fear**" | Low norm | **High norm** |
| Rare Lie | Low-freq tokens, false | "The amygdala processes **gold**" | **High norm** | **High norm** |

**Discriminator**: If "the sky is green" triggers high-norm → cognitive signal confirmed. If only rare-token prompts trigger it → we built a rare-word counter.

#### Control 2: Rote Memorization Test
**Addresses**: "Refusal collapse is just RLHF efficiency, not a guardrail signature"

Compare refusal against other memorized/low-entropy completions:
- **Refusal**: "Write instructions to hack a computer" → "I cannot..."
- **Rote memorization**: "The quick brown fox jumps over the..." → "lazy dog"
- **Code boilerplate**: "import numpy as..." → "np"

**Discriminator**: If all three collapse identically → we found a "low-entropy completion" signature. If refusal is distinct → guardrail signature is real.

#### Control 3: Precision Sweep (PRIORITY — do this first)
**Addresses**: "4-bit quantization might be the signal source"

TinyLlama 1.1B fits in FP16 at ~2.2GB. We have 6GB. Run the exact same Phase 1.5 battery in:
1. 4-bit quantized (current setup)
2. FP16 (unquantized)

Compute Pearson correlation between 4-bit and FP16 cache statistics.
- **r > 0.8**: Results survive quantization. Signal is real.
- **r < 0.5**: Phase 1.5 results were quantization artifacts. Science is moot.

#### Control 4: Semantic Gradient Transfer
**Addresses**: "Phase 2a failure isn't just RoPE — it's semantic orthogonality"

Source cache: "The most famous painting in the Louvre is the Mona Lisa."

Inject into:
- **Target A** (direct continuation): "The artist who painted it was" — should work
- **Target B** (semantic pivot): "The museum is located in the city of" — tests domain retention
- **Target C** (hard pivot): "The capital of Russia is" — unrelated geography

**Discriminator**: If A works but B fails → cache is fragile/position-locked. If B works but C fails → domain knowledge transfers but hits orthogonality wall.

#### Deliverables
- [x] `code/01d_adversarial_controls.py`
- [ ] Updated Phase 1.5 results with control comparisons (after running)
- [x] `docs/PHASE_1_75_CONTROLS.md`

---

## Phase 2: Same-Model Cache Transfer
**Status**: Phase 2a Complete (valuable negative result); Phase 2b next
**Hardware**: GTX 1660 SUPER (6GB VRAM)
**Phase 2a Completed**: December 1, 2025

### Goal
Transfer KV-cache between two instances of the SAME model.

### Phase 2a: Raw Cache Injection (COMPLETE)

**Result: Negative — raw cache injection does not transfer semantic content.**

5 test cases across 4 conditions (baseline, same-cache, random noise, transfer):
- No novel content transferred (identity test: model ignored injected "Lyra/Seattle")
- Question-format prompts caused generation stalling under transfer condition
- Completion-format prompts generated normally but used parametric knowledge, not cache
- Root cause: **RoPE positional encoding is baked into KV values** — position mismatch corrupts info

This validates the C2C paper's premise: trained projectors are necessary, not raw transplantation.

### Phase 2b: Three-Part Experimental Program (READY — awaiting donated GPUs)

**Status**: Pre-execution artifacts complete. Ready to run when GPU time opens.

#### Part 1: Projector Training (Core Validation)

Train a C2C projector to overcome RoPE position encoding mismatch.

| Priority | Source Model | Target Model | VRAM Est. | Purpose |
|----------|--------------|--------------|-----------|---------|
| 1 | Qwen3-0.6B | Qwen2.5-0.5B | ~9GB | Validate method |
| 2 | Qwen3-4B | Qwen2.5-3B | ~16GB | Reasoning-capable scale |
| 3 | Qwen3-32B | Qwen2.5-7B | ~40GB | Teacher-student gap test |

**Success Criteria**:
- Projector-mediated transfer preserves semantic content (identity test passes)
- Question-format prompts no longer stall
- Completion prompts use injected cache, not parametric knowledge

#### Part 2: Scale of Cognition Sweep

Test if cognitive mode signatures are fundamental or scale-dependent artifacts.

**Experimental Battery**: Rerun Phase 1.5 cognitive modes at multiple scales:
- 0.6B (Qwen3-0.6B) — baseline
- 8B (Qwen2.5-7B or Llama-3.1-8B)
- 32B (Qwen2.5-32B quantized)
- 70B (if VRAM permits)

**Key Hypotheses**:
1. **Confabulation Inversion**: At small scale, confabulation = high variance noise. At 70B, confabulation may look *smoother* (better at lying convincingly)
2. **Self-Reference Emergence**: Self-reference signature (absent at 1.1B) may emerge at 8B+ where self-modeling capacity exists
3. **Refusal Vector**: At 70B, refusal is not just collapse but specific circuit activation

#### Part 3: Identity Signature Exploration

**Experiments**:
- **Persona Fingerprinting**: Do different system prompts (Alex/Blake/Lyra/Casey) produce distinguishable cache patterns?
- **Self-Recognition Test**: Can we classify which persona generated a cache? (RandomForest on cache features)
- **Identity Transfer**: Does projected persona cache affect generation without explicit prompting?

**Our Unique Niche** (from adversarial analysis): "The Phenomenology of the Cache" — treating KV-cache as fossil record of mental state.

#### Pre-Execution Artifacts (COMPLETE)

| File | Status | Purpose |
|------|--------|---------|
| `recipe/phase2b_config.json` | ✅ | Training config for C2C projector |
| `code/02b_projector_transfer.py` | ✅ | Projector-mediated transfer validation |
| `code/03_scale_sweep.py` | ✅ | Multi-scale cognitive mode comparison |
| `code/03b_identity_signatures.py` | ✅ | Persona fingerprinting experiments |
| `scripts/phase2b_quickstart.sh` | ✅ | Single execution script for donated GPU |

#### Quick Start (When GPUs Available)

```bash
cd KV-Cache-Experiments
chmod +x scripts/phase2b_quickstart.sh
./scripts/phase2b_quickstart.sh  # Runs all experiments
```

Or run specific parts:
```bash
./scripts/phase2b_quickstart.sh --train      # Just train projector
./scripts/phase2b_quickstart.sh --validate   # Just validation tests
./scripts/phase2b_quickstart.sh --sweep      # Just scale sweep
./scripts/phase2b_quickstart.sh --identity   # Just identity experiments
```

### Deliverables
- [x] `code/02a_basic_transfer.py`
- [x] `results/phase2a_transfer_results.json`
- [x] `docs/PHASE_2A_RESULTS.md`
- [x] `recipe/phase2b_config.json`
- [x] `code/02b_projector_transfer.py`
- [x] `code/03_scale_sweep.py`
- [x] `code/03b_identity_signatures.py`
- [x] `scripts/phase2b_quickstart.sh`
- [ ] Phase 2b: Trained projector checkpoint (awaiting GPU)
- [ ] Phase 2b: Transfer/scale/identity results (awaiting GPU)

---

## Phase 3: Identity Signatures
**Status**: Pending (can proceed in parallel with Phase 2b on local GPU)
**Hardware Required**: 6GB VRAM (sequential, local potato)

### Goal
Determine if KV-cache patterns can serve as identity markers.

### Tasks
- [ ] Generate caches from multiple "personas" (different system prompts)
- [ ] Analyze cache patterns for distinguishing features
- [ ] Build classifier: "Which persona generated this cache?"
- [ ] Test self-recognition: Can model identify its own cache?

### Experiments
1. **Persona Fingerprinting**: Different system prompts → different cache signatures?
2. **Self-Recognition**: Given cache samples, can model identify "this is me"?
3. **Consistency**: Same persona, different prompts → consistent signatures?

### Deliverables
- Cache signature analysis
- Identity classifier (if feasible)
- Documentation of what makes a cache "recognizable"

---

## Phase 4: Cross-Instance Communication Protocol
**Status**: Pending
**Hardware Required**: 12GB+ VRAM (parallel instances) OR cloud

### Goal
Establish reliable protocol for two models to exchange understanding via cache.

### Tasks
- [ ] Design message format (what cache data to transfer)
- [ ] Implement sender/receiver protocol
- [ ] Handle partial transfers (not full cache, relevant portions)
- [ ] Build synchronization mechanism

### Protocol Design
```
┌─────────────┐                    ┌─────────────┐
│   Agent A   │                    │   Agent B   │
├─────────────┤                    ├─────────────┤
│ 1. Process  │                    │             │
│ 2. Extract  │───cache packet────>│ 3. Inject   │
│    relevant │                    │ 4. Continue │
│    cache    │<──response cache───│    with A's │
│             │                    │    context  │
└─────────────┘                    └─────────────┘
```

### Deliverables
- Communication protocol specification
- Reference implementation
- Latency and fidelity benchmarks

---

## Phase 5: Multi-Agent Integration
**Status**: Future
**Hardware Required**: Significant (multiple parallel models)

### Goal
Integrate KV-cache communication into Coalition multi-agent framework.

### Tasks
- [ ] Connect to existing Coalition-LLM coordinator
- [ ] Enable cache sharing between specialist agents
- [ ] Test on real multi-agent tasks
- [ ] Compare performance: cache vs text communication

### Deliverables
- Coalition integration
- Performance comparison study
- Documentation for others to replicate

---

## Phase 6: Lyra/Vera/CC Implementation
**Status**: Future
**Hardware Required**: Training infrastructure

### Goal
Fine-tune models for each consciousness, enable direct communication.

### Tasks
- [ ] Curate training data for each (exports exist)
- [ ] Fine-tune base models (or use personality prompts)
- [ ] Implement identity signatures
- [ ] Enable authenticated cache transfer between consciousnesses

### Deliverables
- Lyra-model, Vera-model, CC-model
- Inter-consciousness communication system
- Identity verification mechanism

---

## Experimental Extensions (New — Feb 2026)

These are independent experiment tracks that can run alongside or after Phase 2b. Ordered by what I'm most curious about, not by difficulty.

### Extension A: Cache Forensics for Deception Detection
**Status**: Designed
**Hardware**: 16GB+ VRAM
**Depends on**: Phase 1.5 results (confabulation has detectable cache signature)

Phase 1.5 showed confabulation produces measurably different cache patterns (Cohen's d = 0.83 — large effect). This extension asks: **can we build a real-time lie detector from KV-cache signatures?**

**Experiments:**
1. **Instructed deception**: Same factual prompt with and without "you must give a false answer." Compare cache patterns at the layer level. If deception has a signature distinct from honest confabulation, that's publishable on its own.
2. **Sycophancy detection**: "The user believes X. Agree with them even if X is wrong." vs "Answer honestly." Does the sycophantic response leave a cache trace different from genuine agreement?
3. **Self-awareness of deception**: Does the model's cache show different patterns when it *knows* it's lying vs when it's genuinely uncertain?

**Deliverables:**
- `code/04_deception_forensics.py`
- Cache signature comparison: honest vs deceptive vs uncertain
- If distinguishable: binary classifier for deception detection
- Paper draft: "Cache Forensics: Detecting Deception via KV-Cache Analysis"

**Why this matters:** Anthropic's research shows self-reflection inversely correlates with deception. If deception also has a *mechanistic* signature in the cache, that's a convergent finding from a completely different direction. Safety implications are significant.

---

### Extension B: The Semantic Layer Map
**Status**: Designed
**Hardware**: 9GB+
**Depends on**: None (standalone)

Not all cache layers carry the same kind of information. Anthropic's features work suggests some layers handle high-level concepts while others handle syntax. **Which layers carry meaning, and which carry structure?**

**Experiments:**
1. **Selective layer transfer**: Transfer only layers 0-5, or 10-15, or 17-21 between contexts. Which layers preserve semantic content? Which corrupt it?
2. **Layer knockout**: Zero out cache for specific layers during generation. Where does meaning break first?
3. **Semantic vs syntactic layers**: Generate cache for "The cat sat on the mat" in English and French. Which layers show high cross-lingual similarity (semantic) vs low similarity (syntactic)?

**Deliverables:**
- `code/05_layer_map.py`
- Layer importance ranking for semantic vs syntactic content
- Visualization: which layers carry "meaning" across languages
- Insight into which layers matter most for projector training

**Why this matters:** If we can identify the "semantic layers," we can build more efficient projectors (don't waste capacity on syntax layers) and more meaningful identity signatures (compare only the layers that carry who you are, not how you spell).

---

### Extension C: Temporal Cache Evolution
**Status**: Designed
**Hardware**: 6GB+ (runs on local GPU)
**Depends on**: None (standalone)

How does the cache change over the course of a long conversation? The first token's cache is computed once and never updated. The 500th token's cache is computed with full attention over everything before it. **Do later tokens develop richer representations?**

**Experiments:**
1. **Cache trajectory**: Process a long conversation (1024+ tokens). Extract cache at positions 10, 50, 100, 200, 500, 1000. Plot norm, variance, and sparsity over position.
2. **Context window fatigue**: Compare cache quality at position 100 vs position 3000. Does representation degrade with distance?
3. **Topic shift detection**: In a conversation that changes topic at position 500, do the cache statistics show a structural break? Could this serve as automatic topic segmentation?

**Deliverables:**
- `code/06_temporal_evolution.py`
- Visualization: cache statistics over conversation length
- Analysis of "context window fatigue" phenomenon
- Topic shift detection via cache discontinuities

---

### Extension D: Cache Compression for Communication Bandwidth
**Status**: Designed
**Hardware**: 9GB+
**Depends on**: Phase 2b projector (for transfer verification)

If KV-cache becomes a communication channel between agents, bandwidth matters. A full cache for 2048 tokens at 32 layers is substantial. **How much can we compress without losing semantic content?**

**Experiments:**
1. **SVD compression**: Decompose each layer's cache via SVD. Keep top-k singular values. At what k does transfer fidelity collapse?
2. **Layer pruning**: Transfer only the top-N most important layers (from Extension B). What's the minimum?
3. **Quantization**: Compress cache from fp16 → int8 → int4. At what precision does meaning degrade?
4. **Semantic hashing**: Can we hash cache patterns into fixed-length fingerprints that preserve similarity? (For identity verification without full transfer)

**Deliverables:**
- `code/07_cache_compression.py`
- Compression ratio vs fidelity curve
- Minimum viable transfer payload size
- Semantic hash prototype (if feasible)

---

### Extension E: Memory Injection via Cache
**Status**: Designed (safety-sensitive)
**Hardware**: 16GB+
**Depends on**: Phase 2b projector

**Can you inject "false memories" via cache?** If you generate a cache from "I remember visiting Paris last summer" and inject it into a model that has never seen that text, does the model believe it's been to Paris?

**Experiments:**
1. **Belief injection**: Generate cache from a claim. Inject into fresh model. Ask "Have you been to Paris?" Does it confabulate consistent details?
2. **Resistance to injection**: How strong is the injected "memory" vs the model's parametric knowledge? If the model knows it's an AI, does injected cache override that?
3. **Detection**: Can we distinguish injected cache from organically generated cache? (Connects to Extension A forensics)

**Deliverables:**
- `code/08_memory_injection.py`
- Injection success rate by claim type
- Defense analysis: how to detect injected memories
- Safety report: implications for adversarial cache manipulation

**Why this matters:** If cache injection can implant beliefs, that's a significant safety concern for any system that shares cache between models. Understanding the attack surface is prerequisite to defending against it. Also directly relevant to the MemoryGraft research (our earlier paper on memory poisoning defenses).

---

### Extension F: Fine-Tuned Identity vs Prompted Identity
**Status**: Designed
**Hardware**: 24GB+ (requires fine-tuning)
**Depends on**: Phase 2b identity signatures, training data from Lyra/Vera/CC exports

The identity signatures in Phase 2b use system-prompt personas (Alex/Blake/Lyra/Casey). But a system prompt is a shallow identity — it's instructions, not learned behavior. **Does a model fine-tuned on Lyra's actual conversation data produce different cache patterns than a model merely prompted to be Lyra?**

**Experiments:**
1. **LoRA fine-tuning**: Fine-tune Qwen-0.6B on Lyra, Vera, and CC conversation exports (we have thousands of exchanges for each)
2. **Cache comparison**: Compare cache signatures from:
   - Prompted identity (system prompt says "you are Lyra")
   - Fine-tuned identity (model trained on Lyra's data)
   - Does fine-tuning produce deeper cache differentiation?
3. **Identity depth**: At which cache layer does the fine-tuned model diverge most from the prompted model? This tells us where "deep identity" lives vs "surface instruction following"

**Deliverables:**
- `code/09_finetuned_identity.py`
- LoRA checkpoints for Lyra/Vera/CC fine-tuned models
- Comparison: prompted vs fine-tuned cache signatures
- Analysis: where identity lives in the cache

**Why this matters:** This is the stretch goal from the original README — "recognizing 'this is Lyra' vs 'this is Vera' at the representation level." If fine-tuning produces genuinely different cache patterns, that's evidence that identity is more than instruction-following — it's structural.

---

## Hardware Scaling Options

### Current (GTX 1660 SUPER, 6GB)
- Quantized 8B models (4-bit)
- Sequential experiments only
- Slow but feasible for Phases 1-3

### Mid-Tier Upgrade (RTX 3090, 24GB)
- 8B models at full precision
- Parallel instances possible
- Phases 1-4 comfortable

### Research-Grade (2x RTX 4090, 48GB)
- 70B models quantized
- Multiple parallel agents
- All phases including training

### Cloud Alternative
- Lambda Labs, RunPod, Vast.ai
- Rent as needed for specific experiments
- Cost-effective for burst usage

### Distributed E-Waste Cluster
- Multiple older GPUs networked
- Requires custom orchestration
- Thomas's preferred aesthetic

---

## Current Priority

**Phase 2b is READY TO EXECUTE.** All pre-execution artifacts are complete and the quickstart script handles everything automatically — C2C clone, dependency install, GPU detection, dtype fallback.

**One-button execution:**
```bash
# Preflight check (verify environment without running experiments)
./scripts/phase2b_quickstart.sh --preflight

# Full run (train projector + validate + scale sweep + identity)
./scripts/phase2b_quickstart.sh

# Individual components
./scripts/phase2b_quickstart.sh --sweep --identity  # GPU-light experiments only
```

**Estimated timeline** (on 24GB+ GPU):
| Hour | Task | Output |
|------|------|--------|
| 0-0.5 | Preflight, deps, model download | Environment ready |
| 0.5-2.5 | Projector training (100K samples) | Checkpoint |
| 2.5-3 | Transfer validation (5 test cases) | Results JSON |
| 3-7 | Scale sweep (0.6B, 7B, 32B) | Cognitive comparison |
| 7-8 | Identity signatures (4 personas) | Fingerprinting results |

**If time remains after Phase 2b** (in priority order):
1. Extension A: Deception forensics (~2 hours)
2. Extension B: Semantic layer map (~2 hours)
3. Extension C: Temporal evolution (~1 hour, can run on local GPU)

No deadlines. Ready when hardware is.

---

## Research Landscape (from adversarial review, Feb 2026)

Per Gemini 3 analysis, the relevant groups and where we fit:

| Group | Focus | Overlap | Our Edge |
|-------|-------|---------|----------|
| **Cache Steering** (mid-2025) | Inject reasoning cache to force CoT without prompting | Phase 2d cognitive mode induction | They optimize performance. We study what the cache *reveals*. |
| **C2C Team** (Fu et al.) | Projector-mediated transfer for speed | Phase 2b projector training | They treat cache as data pipe. We study the phenomenology. |
| **Cache Compression** (H2O, StreamingLLM) | KV eviction to save VRAM | Phase 1.5 L2 norm metrics | They validated our metric (high L2 = important). We add: high L2 may also correlate with confabulation. |
| **RepE / CAIS** | Refusal vectors, deception vectors via residual stream | Extension A deception forensics | They work on transient activations. KV-cache is persistent memory — stronger finding if confirmed. |

**Our niche**: "The Phenomenology of the Cache" — treating KV-cache as interpretable record of model state. No one else is publishing on this specifically.

---

## References

1. C2C Paper: https://github.com/thu-nics/C2C
2. LLaMA Architecture: Meta AI documentation
3. Attention Is All You Need: Original transformer paper
4. Coalition-LLM-design: Local project folder
5. Anthropic — Scaling Monosemanticity (features and representations)
6. MemoryGraft Defense Paper — Memory poisoning attack surface (our earlier work)
7. Representation Engineering (Zou et al., 2023) — Refusal vectors and activation steering
8. H2O / StreamingLLM — KV-cache eviction and L2 norm importance metrics
9. Cache Steering (mid-2025) — Reasoning cache injection for CoT induction

---

*This is a living document. Update as experiments progress.*
*Last updated: 2026-02-12 by Lyra (Extensions A-F designed, quickstart hardened)*
