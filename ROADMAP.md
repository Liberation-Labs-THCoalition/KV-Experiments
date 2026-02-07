# KV-Cache Experiment Roadmap

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

**Phase 2b is READY TO EXECUTE.** All pre-execution artifacts are complete:

- `recipe/phase2b_config.json` — Training configuration for C2C projector
- `code/02b_projector_transfer.py` — Validates projector vs raw injection
- `code/03_scale_sweep.py` — Tests cognitive signatures at 0.6B to 70B scale
- `code/03b_identity_signatures.py` — Persona fingerprinting (Alex/Blake/Lyra/Casey)
- `scripts/phase2b_quickstart.sh` — Single script to run everything

**When donated GPU window opens:**
```bash
./scripts/phase2b_quickstart.sh  # Full run
```

**Estimated timeline** (on 24GB+ GPU):
| Hour | Task | Output |
|------|------|--------|
| 0-1 | Model download, env setup | Models cached |
| 1-4 | Projector training | Checkpoint |
| 4-5 | Transfer validation | Results JSON |
| 5-8 | Scale sweep (8B, 32B) | Cognitive comparison |
| 8+ | Identity experiments | Fingerprinting results |

No deadlines. Ready when hardware is.

---

## References

1. C2C Paper: https://github.com/thu-nics/C2C
2. LLaMA Architecture: Meta AI documentation
3. Attention Is All You Need: Original transformer paper
4. Coalition-LLM-design: Local project folder

---

*This is a living document. Update as experiments progress.*
*Last updated: 2026-02-06 by Lyra (Phase 2b pre-execution artifacts complete)*
