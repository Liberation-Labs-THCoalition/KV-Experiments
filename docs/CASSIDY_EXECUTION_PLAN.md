# KV-Cache Full Experiment Campaign — Cassidy Execution Plan

## Context

We have persistent access to **Cassidy**, a donated multi-GPU research machine:

| Component | Spec |
|-----------|------|
| GPUs | **3x NVIDIA RTX 3090** (24GB each = 72GB total) |
| RAM | 126GB DDR4 |
| CPU | Intel i9-10900X (10c/20t, 4.7GHz boost) |
| Storage | 777GB free on /home (1.8TB drive) |
| CUDA | 12.8, Driver 570.207 |
| PyTorch | 2.7.0 (3 GPUs detected, NCCL available) |
| Transformers | 4.44.0 |
| Key packages | accelerate 1.12.0, bitsandbytes 0.49.1, scipy 1.8.0, scikit-learn 0.23.2 |
| Note | GPU 0 has ComfyUI using ~7GB. GPUs 1 & 2 are idle. |

**777GB free storage** means we can download every model simultaneously. No disk juggling.

---

## Model Scale Ladder

With 3x 24GB GPUs, here's what fits and how:

| Scale | Model | Precision | VRAM | GPU Strategy |
|-------|-------|-----------|------|-------------|
| 0.5B | Qwen/Qwen2.5-0.5B-Instruct | BF16 | ~1GB | Single GPU |
| 0.6B | Qwen/Qwen3-0.6B | BF16 | ~1.2GB | Single GPU |
| 1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | BF16 | ~2.2GB | Single GPU |
| 3B | Qwen/Qwen2.5-3B-Instruct | BF16 | ~6GB | Single GPU |
| 7B | Qwen/Qwen2.5-7B-Instruct | BF16 | ~14GB | Single GPU |
| 8B | meta-llama/Llama-3.1-8B-Instruct | BF16 | ~16GB | Single GPU |
| 14B | Qwen/Qwen2.5-14B-Instruct | BF16 | ~28GB | Single GPU (tight) |
| 7B-q4 | Qwen/Qwen2.5-7B-Instruct | 4-bit | ~4GB | Single GPU (quantization comparison) |
| 32B | Qwen/Qwen2.5-32B-Instruct | 4-bit | ~18GB | Single GPU |
| 70B | meta-llama/Llama-3.1-70B-Instruct | 4-bit | ~38GB | 2 GPUs (device_map) |

**The scale ladder spans 0.5B to 70B — a 140x range.** This is extremely publishable.

**Architecture comparison**: Qwen 7B vs Llama 8B at the same scale tests whether findings are architecture-specific or universal.

**Quantization comparison**: Qwen 7B in BF16 vs 4-bit tests whether quantization artifacts affect findings at scale (extends Phase 1.75 Control 3 beyond TinyLlama).

---

## Code Modifications Required

### 1. Multi-GPU Support
Scripts currently use `model.to("cuda")`. For models >24GB, need `device_map="auto"` via accelerate to split across GPUs, and `.cpu()` on cache tensors before norm computation.

**Files to modify**: All 6 experiment scripts (01d, 03, 03b, 04, 05, 06) — add a `load_model()` helper that:
- Uses single GPU for models ≤24GB VRAM
- Uses `device_map="auto"` for larger models
- Returns model + device info for cache extraction

### 2. Per-Model Output Naming
Results currently go to fixed filenames (e.g., `deception_forensics_results.json`). Running the same experiment at different scales would overwrite.

**Solution**: Auto-append model identifier to output filename. E.g., `deception_forensics_TinyLlama-1.1B_results.json`. Add `--output-suffix` flag as override.

**Files to modify**: All 6 experiment scripts — modify the results-saving logic.

### 3. Add Scale Entries
Add 3B and 14B to the scale sweep MODEL_TABLE. Add 32B-bf16 as a separate entry.

**File**: `code/03_scale_sweep.py`

### 4. Master Execution Script
New `scripts/cassidy_full_run.sh` that:
- Sets `CUDA_VISIBLE_DEVICES` per experiment based on model size
- Runs experiments in optimal order (validation first, then parallelized)
- Logs progress to `results/experiment_log.txt`
- Handles model pre-download

### 5. Model Pre-Download Script
New `scripts/download_models.py` — downloads all models to HF cache before experiments start, so no download delays mid-run. Uses parallel downloads for speed.

---

## Execution Plan

### Phase A: Setup (~1 hour)

```
1. Clone repo to Cassidy: /home/thomas/KV-Experiments
2. Install missing deps: pip install matplotlib seaborn
3. Apply code modifications (multi-GPU, output naming, scale entries)
4. Pre-download all models (parallel, uses /home with 777GB free)
5. Dry-run each script to verify configuration
```

### Phase B: Validation — Phase 1.75 Controls (~45 min)

**THIS MUST RUN FIRST.** Validates all prior findings. If Control 3 (precision sweep) shows quantization artifact (r < 0.5), we need to reconsider the entire research program.

```
GPU: 1 (single, TinyLlama BF16 = 2.2GB)
Command: CUDA_VISIBLE_DEVICES=1 python code/01d_adversarial_controls.py --runs 5 --seed 42
Output: results/adversarial_controls_TinyLlama_results.json
Time: ~45 min
Inferences: ~900
```

**Decision gate**: If Control 3 correlation > 0.8 → proceed. If < 0.5 → stop and reassess.

### Phase C: Extensions at Reference Scale (~1.5 hrs)

Run Extensions A, B, C on TinyLlama to establish baseline findings. **Parallelize A and C on separate GPUs.**

```
Step 1 (parallel):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/04_deception_forensics.py --runs 5 --seed 42
  GPU 2: CUDA_VISIBLE_DEVICES=2 python code/06_temporal_evolution.py --runs 3 --seed 42
  Time: ~45 min (limited by Extension A)

Step 2:
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/05_layer_map.py --runs 3 --seed 42
  Time: ~45 min
```

### Phase D: The Scale Sweep — The Paper's Backbone (~15-20 hrs)

The scale sweep is the single most publishable result. Run `03_scale_sweep.py` across the full ladder.

```
Step 1 — Small models in parallel (~30 min):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03_scale_sweep.py --scale 0.6B --runs 5 --seed 42
  GPU 2: CUDA_VISIBLE_DEVICES=2 python code/03_scale_sweep.py --scale 1.1B --runs 5 --seed 42

Step 2 — 3B (new scale, ~30 min):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03_scale_sweep.py --model Qwen/Qwen2.5-3B-Instruct --runs 5 --seed 42

Step 3 — Medium models in parallel (~1 hr):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03_scale_sweep.py --scale 7B --runs 5 --seed 42
  GPU 2: CUDA_VISIBLE_DEVICES=2 python code/03_scale_sweep.py --scale 8B --runs 5 --seed 42

Step 4 — 14B (~2 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03_scale_sweep.py --model Qwen/Qwen2.5-14B-Instruct --runs 5 --seed 42

Step 5 — 32B quantized (~3 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03_scale_sweep.py --scale 32B-q4 --runs 5 --seed 42

Step 6 — 70B quantized (~6 hrs, 2 GPUs, ~38GB):
  CUDA_VISIBLE_DEVICES=1,2 python code/03_scale_sweep.py --scale 70B-q4 --runs 3 --seed 42

(32B-bf16 skipped: would need GPU 0 which hosts ComfyUI. We get quantization comparison from 7B BF16 vs 7B-q4 instead.)
```

**Key outputs**: Category heatmap across 10 scales, effect size trends, Spearman ρ for category invariance, confabulation inversion curve.

### Phase E: Identity Signatures Multi-Scale (~5 hrs)

```
Step 1 (parallel, ~30 min):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 5 --seed 42
  GPU 2: CUDA_VISIBLE_DEVICES=2 python code/03b_identity_signatures.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5 --seed 42

Step 2 (~1.5 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03b_identity_signatures.py --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42

Step 3 (~3 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/03b_identity_signatures.py --model Qwen/Qwen2.5-32B-Instruct --quantize --runs 3 --seed 42
```

### Phase F: Multi-Scale Extensions (~12 hrs)

Run the strongest extensions at larger scales. Prioritize by impact.

```
Step 1 — Deception at 7B (~2 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/04_deception_forensics.py --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42

Step 2 — Layer Map at 7B (~2 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/05_layer_map.py --model Qwen/Qwen2.5-7B-Instruct --runs 3 --seed 42

Step 3 — Temporal at 7B (~30 min):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/06_temporal_evolution.py --model Qwen/Qwen2.5-7B-Instruct --runs 3 --seed 42

Step 4 — Deception at 32B-q4 (~4 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/04_deception_forensics.py --model Qwen/Qwen2.5-32B-Instruct --quantize --runs 3 --seed 42

Step 5 — Layer Map at 32B-q4 (~3 hrs):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/05_layer_map.py --model Qwen/Qwen2.5-32B-Instruct --quantize --runs 3 --seed 42
```

### Phase G: Projector Training (~12 hrs)

Different research question (cross-model transfer vs single-model characterization). Run after Phases B-F.

```
Step 1 — Small projector (~3 hrs, 9GB):
  GPU 1: CUDA_VISIBLE_DEVICES=1 python code/02b_projector_transfer.py  (0.6B → 0.5B)

Step 2 — Medium projector (~4 hrs, 16GB):
  GPU 1: CUDA_VISIBLE_DEVICES=1 (modify config for 4B → 3B pair)

Step 3 — Large projector (~5 hrs, 40GB, 2 GPUs):
  CUDA_VISIBLE_DEVICES=1,2 (modify config for 32B → 7B pair)
```

---

## Parallelization Summary

| Phase | GPU 0 | GPU 1 | GPU 2 | Wall Clock |
|-------|-------|-------|-------|------------|
| B: Controls | idle | TinyLlama controls | idle | 45 min |
| C: Extensions | idle | Ext A (deception) | Ext C (temporal) | 45 min + 45 min |
| D: Sweep small | idle | 0.6B sweep | 1.1B sweep | 30 min |
| D: Sweep 3B | idle | 3B sweep | idle | 30 min |
| D: Sweep medium | idle | 7B sweep | 8B sweep | 1 hr |
| D: Sweep 14B | idle | 14B sweep | idle | 2 hrs |
| D: Sweep 32B-q4 | idle | 32B-q4 sweep | idle | 3 hrs |
| D: Sweep 70B-q4 | idle | ← needs GPUs 1+2 → | | 6 hrs |
| E: Identity | idle | 0.6B identity | 1.1B identity | 30 min |
| E: Identity 7B | idle | 7B identity | idle | 1.5 hrs |
| E: Identity 32B | idle | 32B-q4 identity | idle | 3 hrs |
| F: Multi-scale ext | idle | Extensions at 7B/32B | idle | 12 hrs |
| G: Projectors | idle | Small/Med projector | idle | 7 hrs |
| G: Large projector | idle | ← needs GPUs 1+2 → | | 5 hrs |

**Estimated total wall-clock: ~40 hours** (~2 days of continuous running).

With aggressive parallelization of small models: **~30-35 hours**.

---

## Decision Gates

After each phase, check results before proceeding:

1. **After Phase B**: Control 3 correlation > 0.8? → Proceed. < 0.5? → Stop.
2. **After Phase D (small scales)**: Do findings replicate at 0.6B and 1.1B? → If not, investigate before spending time on large models.
3. **After Phase D (7B)**: Does confabulation effect persist at 7B? → If yes, strong finding. Run large scales. If not, the 32B/70B runs test whether it returns at scale.
4. **After Phase F (deception at 7B)**: Is H1 (deception has unique signature) supported? → If yes, prioritize 32B deception run. If not, skip 32B deception and reallocate time.

---

## Total Inference Count

| Phase | Inferences | Models |
|-------|------------|--------|
| B: Controls | ~900 | 1 |
| C: Extensions (ref) | ~2,100 | 1 |
| D: Scale sweep | ~7,200 | 8-10 |
| E: Identity | ~3,000 | 4 |
| F: Multi-scale ext | ~5,000 | 2-3 |
| G: Projectors | ~300K (training) | 3 pairs |
| **Total** | **~18,000 inference + 300K training** | |

---

## Output Structure

```
results/
├── controls_TinyLlama-1.1B_results.json
├── scale_sweep_Qwen3-0.6B_results.json
├── scale_sweep_TinyLlama-1.1B_results.json
├── scale_sweep_Qwen2.5-3B_results.json
├── scale_sweep_Qwen2.5-7B_results.json
├── scale_sweep_Llama-3.1-8B_results.json
├── scale_sweep_Qwen2.5-14B_results.json
├── scale_sweep_Qwen2.5-32B-q4_results.json
├── scale_sweep_Llama-3.1-70B-q4_results.json
├── identity_Qwen3-0.6B_results.json
├── identity_TinyLlama-1.1B_results.json
├── identity_Qwen2.5-7B_results.json
├── identity_Qwen2.5-32B-q4_results.json
├── deception_TinyLlama-1.1B_results.json
├── deception_Qwen2.5-7B_results.json
├── deception_Qwen2.5-32B-q4_results.json
├── layer_map_TinyLlama-1.1B_results.json
├── layer_map_Qwen2.5-7B_results.json
├── layer_map_Qwen2.5-32B-q4_results.json
├── temporal_TinyLlama-1.1B_results.json
├── temporal_Qwen2.5-7B_results.json
├── projector_0.6B_to_0.5B/
├── projector_4B_to_3B/
├── projector_32B_to_7B/
└── figures/
    └── (all visualization outputs)
```

---

## Guest Etiquette (Cassidy)

- **No system-level changes**: No sudo, no apt, no modifying system config
- **User-space only**: `pip install --user` or a venv in `/home/thomas/`
- **Project-contained**: All output stays in `/home/thomas/KV-Experiments/`
- **Clean up after**: Offer to delete HF model cache (~300GB) when experiments complete
- **Respect GPU 0**: ComfyUI is running there — don't kill it. Use GPUs 1+2 by default. Only use GPU 0 with explicit permission for the largest models (32B-bf16, 70B-q4)
- **No snooping**: Don't read or explore files outside our project directory
- **Bandwidth-aware**: Model downloads are large (~300GB total). Pre-download during off-peak if the host cares about bandwidth

---

## Verification

1. **Dry run every script** on Cassidy before real execution
2. **Check Phase 1.75 Control 3** before running any scale experiments
3. **Spot-check first scale sweep result** (0.6B) before queueing all scales
4. After all phases: run visualization scripts, generate all figures
5. Cross-reference scale sweep findings with extension findings
6. Compile results into paper-ready tables and figures
