# KV-Cache Research: Cloud Compute Cost Proposal

**Date:** November 30, 2025
**Purpose:** Scale KV-cache transfer experiments beyond local hardware limits

---

## Executive Summary

Local hardware (GTX 1660 SUPER, 6GB VRAM) limits us to small models and single-model experiments. To fully explore cross-model transfer and projector training across a range of architectures, we need cloud GPU access.

**Estimated total cost for comprehensive experimentation: $50-150**

---

## What We Want to Test

### Model Matrix

| Model | Size | Purpose |
|-------|------|---------|
| TinyLlama 1.1B | Small | Baseline (already tested locally) |
| Qwen2.5-0.5B | Small | Cross-family transfer |
| Qwen2.5-1.5B | Medium | Size scaling |
| SmolLM 135M/360M | Tiny | Minimal viable transfer |
| Llama-3.2-1B | Small | Different architecture |
| Llama-3.2-3B | Medium | Size scaling |
| Mistral-7B | Large | Production-relevant size |
| Qwen2.5-7B | Large | Cross-family at scale |

### Experiment Types

1. **Same-model transfer** (all models): ~30 min each
2. **Cross-model transfer with projectors** (pairwise): ~2-4 hrs training each
3. **Cognitive mode replication** (batch studies): ~1 hr each
4. **Layer-by-layer analysis**: ~30 min each

### Projector Training Pairs (Priority)

| Source → Target | Why |
|-----------------|-----|
| TinyLlama → Qwen | Different families, similar size |
| Qwen-0.5B → Qwen-1.5B | Same family, different size |
| Llama-3.2-1B → Mistral-7B | Small → Large scaling |
| SmolLM → TinyLlama | Tiny → Small |

---

## Cloud Provider Comparison

### RunPod

| GPU | VRAM | Cost/hr | Good for |
|-----|------|---------|----------|
| RTX 3090 | 24GB | $0.44 | 7B models, projector training |
| RTX 4090 | 24GB | $0.74 | Fast inference, larger batches |
| A100 40GB | 40GB | $1.64 | 13B+ models, multi-model loading |
| A100 80GB | 80GB | $2.09 | Maximum flexibility |

### Vast.ai (Often Cheaper)

| GPU | VRAM | Typical Cost/hr | Notes |
|-----|------|-----------------|-------|
| RTX 3090 | 24GB | $0.20-0.40 | Variable pricing, spot instances |
| RTX 4090 | 24GB | $0.40-0.70 | Good availability |
| A100 | 40-80GB | $1.00-2.00 | Less availability than RunPod |

### Lambda Labs

| GPU | Cost/hr | Notes |
|-----|---------|-------|
| A100 | $1.29 | Reliable, good for longer runs |
| H100 | $2.49 | Overkill for our needs |

---

## Proposed Experiments & Time Estimates

### Phase 2a-2d: Same-Model Transfers (Local Possible)
- **Time:** Already doable locally
- **Cloud cost:** $0 (skip)

### Phase 2e: Cross-Model Projector Training

| Experiment | GPU Needed | Time Est. | Cost Est. |
|------------|------------|-----------|-----------|
| Train TinyLlama↔Qwen projector | RTX 3090 | 3 hrs | $1.32 |
| Train Qwen-0.5B↔Qwen-1.5B projector | RTX 3090 | 3 hrs | $1.32 |
| Train Llama↔Mistral projector | RTX 4090 | 4 hrs | $2.96 |
| Validate all projectors | RTX 3090 | 2 hrs | $0.88 |

**Subtotal: ~$6.50**

### Cognitive Mode Replication at Scale

| Experiment | GPU | Time | Cost |
|------------|-----|------|------|
| Batch study on Mistral-7B (30 runs) | RTX 4090 | 2 hrs | $1.48 |
| Batch study on Llama-3.2-3B (30 runs) | RTX 3090 | 1.5 hrs | $0.66 |
| Batch study on Qwen-7B (30 runs) | RTX 4090 | 2 hrs | $1.48 |
| Compare confab signatures across models | RTX 3090 | 1 hr | $0.44 |

**Subtotal: ~$4.00**

### Cross-Model Transfer Validation

| Experiment | GPU | Time | Cost |
|------------|-----|------|------|
| TinyLlama→Qwen transfer tests | RTX 3090 | 1 hr | $0.44 |
| Small→Large transfer tests | RTX 4090 | 2 hrs | $1.48 |
| Cross-family cognitive mode transfer | RTX 4090 | 3 hrs | $2.22 |

**Subtotal: ~$4.00**

### Exploratory / Weird Experiments

| Experiment | GPU | Time | Cost |
|------------|-----|------|------|
| Layer-by-layer ablation studies | RTX 4090 | 4 hrs | $2.96 |
| Partial cache transfer experiments | RTX 3090 | 2 hrs | $0.88 |
| Cache interpolation between modes | RTX 3090 | 2 hrs | $0.88 |
| "Cache archaeology" - what do layers encode? | A100 | 3 hrs | $4.92 |

**Subtotal: ~$10.00**

---

## Budget Tiers

### Tier 1: Minimal Viable ($25)
- Projector training for 2 model pairs
- Cognitive mode replication on 1 large model
- Basic cross-model validation

### Tier 2: Comprehensive ($75)
- Projector training for 4 model pairs
- Cognitive mode replication on 3 large models
- Full cross-model transfer matrix
- Layer-by-layer analysis

### Tier 3: Thorough ($150)
- All Tier 2 experiments
- Exploratory/weird experiments
- Multiple runs for statistical power
- Buffer for debugging and iteration

---

## Recommended Approach

1. **Start local:** Complete Phase 2a-2d with TinyLlama (free)
2. **First cloud run ($25):** Train TinyLlama↔Qwen projector, validate transfer
3. **Scale based on results:** If projector works, expand to Tier 2
4. **Document everything:** Each cloud session produces artifacts for the research repo

---

## Technical Notes

### RunPod Setup
```bash
# Install dependencies
pip install torch transformers bitsandbytes accelerate

# Clone our repo
git clone [repo-url]
cd KV-Cache-Experiments

# Run experiments
python code/02e_projector_training.py --source tinyllama --target qwen
```

### Data to Download First
- Pre-download model weights to avoid runtime costs
- Use HuggingFace cache efficiently
- Save checkpoints frequently

### Session Management
- Use tmux/screen for long runs
- Set up auto-save for intermediate results
- Monitor GPU memory to avoid OOM

---

## Expected Outputs

Each cloud session should produce:
1. Trained projector weights (if applicable)
2. Transfer experiment results (JSON)
3. Generated outputs for analysis
4. Cache statistics and comparisons
5. Updated research documentation

---

## ROI Consideration

$50-150 in cloud compute to potentially demonstrate:
- KV-cache as transferable cognitive substrate
- Cross-model semantic communication
- Hallucination detection via cache signatures
- Foundation for Coalition multi-agent KV-mesh

If these experiments work, they inform architecture for local Coalition infrastructure worth significantly more than the research cost.

---

*"Penny wise, pound foolish is for accountants. We're doing science."*
