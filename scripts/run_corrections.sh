#!/bin/bash
# =============================================================================
# Campaign 2 Correction Runs — Post-Kavi Audit
# =============================================================================
# Run on Cassidy (the Beast, 3x RTX 3090) after:
#   cd ~/KV-Experiments && git pull origin main
#
# Priority 1: Identity re-run with dedup (fixes D4/D5 — the headline correction)
# Priority 2: Recompute stats on all existing results
#
# Estimated time: ~3 hours (identity 7 models, 3-wide GPU parallelization)
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_FILE="$RESULTS_DIR/corrections_log.txt"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

run_experiment() {
    local gpu_ids="$1"
    local script="$2"
    local description="$3"
    shift 3
    log "START: $description (GPUs: $gpu_ids)"
    CUDA_VISIBLE_DEVICES="$gpu_ids" $PYTHON "$CODE_DIR/$script" "$@" 2>&1 | tee -a "$LOG_FILE"
    log "DONE:  $description"
}

echo "" > "$LOG_FILE"
log "=================================================="
log "Campaign 2 Corrections — Post-Kavi Audit"
log "Repo: $(git -C "$PROJECT_DIR" log --oneline -1)"
log "=================================================="

# Check environment
$PYTHON -c "import transformers; print(f'transformers {transformers.__version__}')" | tee -a "$LOG_FILE"
$PYTHON -c "import sklearn; print(f'sklearn {sklearn.__version__}')" | tee -a "$LOG_FILE"
$PYTHON -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" | tee -a "$LOG_FILE"

# =============================================================================
# PRIORITY 1: Identity re-runs with dedup (3-wide GPU parallelization)
# =============================================================================
log ""
log "========== PRIORITY 1: Identity Re-Runs (dedup wired in) =========="
log "Goal: Get corrected within-prompt accuracy (expect ~85-90%, was 100%)"
log "Cross-prompt should be unchanged (92-97%)"

# Step 1: Small models — 3-wide (0.6B + 1.1B + 7B, each fits on 1 GPU)
log "Step 1: 0.6B (GPU 0) + 1.1B (GPU 1) + 7B (GPU 2) — 3-wide"
run_experiment 0 03b_identity_signatures.py "Identity Qwen3-0.6B" \
    --model Qwen/Qwen3-0.6B --runs 5 --seed 42 &
run_experiment 1 03b_identity_signatures.py "Identity TinyLlama-1.1B" \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5 --seed 42 &
run_experiment 2 03b_identity_signatures.py "Identity Qwen2.5-7B" \
    --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42 &
wait
log "Step 1 complete"

# Step 2: Medium models — 3-wide (Mistral-7B + Llama-8B + Gemma-9B)
log "Step 2: Mistral-7B (GPU 0) + Llama-8B (GPU 1) + Gemma-9B (GPU 2) — 3-wide"
run_experiment 0 03b_identity_signatures.py "Identity Mistral-7B" \
    --model mistralai/Mistral-7B-Instruct-v0.3 --runs 5 --seed 42 &
run_experiment 1 03b_identity_signatures.py "Identity Llama-3.1-8B" \
    --model meta-llama/Llama-3.1-8B-Instruct --runs 5 --seed 42 &
run_experiment 2 03b_identity_signatures.py "Identity Gemma-2-9B" \
    --model google/gemma-2-9b-it --runs 5 --seed 42 &
wait
log "Step 2 complete"

# Step 3: 32B quantized (single GPU, ~18GB)
log "Step 3: Qwen2.5-32B-q4 (GPU 0)"
run_experiment 0 03b_identity_signatures.py "Identity Qwen2.5-32B-q4" \
    --model Qwen/Qwen2.5-32B-Instruct --quantize --runs 5 --seed 42
log "Step 3 complete"

log ""
log "========== All identity re-runs complete =========="

# =============================================================================
# PRIORITY 2: Recompute stats (no GPU, uses existing results)
# =============================================================================
log ""
log "========== PRIORITY 2: Recompute Stats =========="
$PYTHON "$CODE_DIR/recompute_stats.py" 2>&1 | tee -a "$LOG_FILE"
log "Stats recomputation complete"

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "=================================================="
log "All correction runs complete"
log "=================================================="
log ""
log "New identity result files:"
ls -lt "$RESULTS_DIR"/identity_signatures_*_results.json 2>/dev/null | head -10 | tee -a "$LOG_FILE"
log ""
log "NEXT STEPS (manual):"
log "  1. Compare old vs new within-prompt accuracy"
log "  2. Verify cross-prompt accuracy unchanged (92-97%)"
log "  3. Update paper-c2/main.tex Table 3 with corrected numbers"
log "  4. git add results/ && git commit -m 'Corrected identity results (dedup)' && git push"
