#!/bin/bash
# ================================================================
# KV-Cache Full Experiment Campaign — Cassidy Execution Script
# ================================================================
#
# Machine: 3x RTX 3090 (24GB each), 126GB RAM, 777GB free on /home
# GPU 0: ComfyUI (DO NOT TOUCH) | GPUs 1+2: Ours
#
# Usage:
#   bash scripts/cassidy_full_run.sh              # Run all phases
#   bash scripts/cassidy_full_run.sh --phase B    # Run specific phase
#   bash scripts/cassidy_full_run.sh --dry-run    # Dry run all
#   bash scripts/cassidy_full_run.sh --status     # Check progress
#
# Liberation Labs / THCoalition
# ================================================================

set -euo pipefail

# Use python3 explicitly (Cassidy has python3, not python)
PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

# Default args
PHASE=""
DRY_RUN=""
STATUS_ONLY=false
RUNS=5
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)   PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --status)  STATUS_ONLY=true; shift ;;
        --runs)    RUNS="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ================================================================
# Logging
# ================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_gpu() {
    log "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | while read line; do
        log "  $line"
    done
}

# ================================================================
# Status check
# ================================================================

if $STATUS_ONLY; then
    echo "=== Experiment Progress ==="
    echo ""
    echo "Results found:"
    if [ -d "$RESULTS_DIR" ]; then
        find "$RESULTS_DIR" -name "*_results.json" -printf "  %f (%s bytes, %Tc)\n" 2>/dev/null \
            || ls -la "$RESULTS_DIR"/*_results.json 2>/dev/null \
            || echo "  (none yet)"
    fi
    echo ""
    echo "Log tail:"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "  (no log yet)"
    exit 0
fi

# ================================================================
# Helper: run experiment with logging
# ================================================================

run_experiment() {
    local description="$1"
    local gpu_ids="$2"
    local script="$3"
    shift 3
    local extra_args="$@"

    log "━━━ START: $description (GPUs: $gpu_ids) ━━━"
    local start_time=$(date +%s)

    if [ -n "$DRY_RUN" ]; then
        CUDA_VISIBLE_DEVICES="$gpu_ids" $PYTHON "$CODE_DIR/$script" $DRY_RUN $extra_args
    else
        CUDA_VISIBLE_DEVICES="$gpu_ids" $PYTHON "$CODE_DIR/$script" $extra_args
    fi
    local exit_code=$?

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))

    if [ $exit_code -eq 0 ]; then
        log "━━━ DONE: $description (${minutes}m ${seconds}s) ━━━"
    else
        log "━━━ FAILED: $description (exit code $exit_code, ${minutes}m ${seconds}s) ━━━"
    fi

    return $exit_code
}

# ================================================================
# Phase B: Validation — Adversarial Controls (MUST RUN FIRST)
# ================================================================

phase_B() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE B: VALIDATION — ADVERSARIAL CONTROLS             ║"
    log "║  Gate check: Control 3 must show r > 0.8                ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    run_experiment "Adversarial Controls (TinyLlama)" "1" \
        "01d_adversarial_controls.py" --runs "$RUNS" --seed "$SEED"
}

# ================================================================
# Phase C: Extensions at Reference Scale (TinyLlama)
# ================================================================

phase_C() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE C: EXTENSIONS AT REFERENCE SCALE                 ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # Step 1: Deception + Temporal in parallel on separate GPUs
    log "Step C.1: Deception (GPU 1) + Temporal (GPU 2) in parallel"
    run_experiment "Deception Forensics (TinyLlama)" "1" \
        "04_deception_forensics.py" --runs "$RUNS" --seed "$SEED" &
    local pid1=$!

    run_experiment "Temporal Evolution (TinyLlama)" "2" \
        "06_temporal_evolution.py" --runs 3 --seed "$SEED" &
    local pid2=$!

    wait $pid1 $pid2
    log "Step C.1 complete (parallel)"

    # Step 2: Layer Map (sequential, GPU 1)
    run_experiment "Semantic Layer Map (TinyLlama)" "1" \
        "05_layer_map.py" --runs 3 --seed "$SEED"
}

# ================================================================
# Phase D: The Scale Sweep — The Paper's Backbone
# ================================================================

phase_D() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE D: SCALE SWEEP (0.5B → 70B)                     ║"
    log "║  The paper's backbone — 140x parameter range            ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # Step D.1: Smallest models in parallel
    log "Step D.1: 0.5B (GPU 1) + 0.6B (GPU 2) in parallel"
    run_experiment "Scale Sweep 0.5B" "1" \
        "03_scale_sweep.py" --scale 0.5B --runs "$RUNS" --seed "$SEED" &
    local pid1=$!
    run_experiment "Scale Sweep 0.6B" "2" \
        "03_scale_sweep.py" --scale 0.6B --runs "$RUNS" --seed "$SEED" &
    local pid2=$!
    wait $pid1 $pid2

    # Step D.2: 1.1B + 3B in parallel
    log "Step D.2: 1.1B (GPU 1) + 3B (GPU 2) in parallel"
    run_experiment "Scale Sweep 1.1B" "1" \
        "03_scale_sweep.py" --scale 1.1B --runs "$RUNS" --seed "$SEED" &
    pid1=$!
    run_experiment "Scale Sweep 3B" "2" \
        "03_scale_sweep.py" --scale 3B --runs "$RUNS" --seed "$SEED" &
    pid2=$!
    wait $pid1 $pid2

    # Step D.3: Medium models in parallel (7B Qwen vs 8B Llama — architecture comparison)
    log "Step D.3: 7B Qwen (GPU 1) + 8B Llama (GPU 2) in parallel"
    run_experiment "Scale Sweep 7B" "1" \
        "03_scale_sweep.py" --scale 7B --runs "$RUNS" --seed "$SEED" &
    pid1=$!
    run_experiment "Scale Sweep 8B" "2" \
        "03_scale_sweep.py" --scale 8B --runs "$RUNS" --seed "$SEED" &
    pid2=$!
    wait $pid1 $pid2

    # Step D.4: 7B-q4 quantization comparison (GPU 1)
    log "Step D.4: 7B-q4 quantization comparison"
    run_experiment "Scale Sweep 7B-q4" "1" \
        "03_scale_sweep.py" --scale 7B-q4 --runs "$RUNS" --seed "$SEED"

    # Step D.5: 14B (single GPU, ~28GB — tight fit on 24GB, may need quantization)
    log "Step D.5: 14B (single GPU)"
    run_experiment "Scale Sweep 14B" "1" \
        "03_scale_sweep.py" --scale 14B --runs "$RUNS" --seed "$SEED"

    # Step D.6: 32B quantized (single GPU, ~18GB)
    log "Step D.6: 32B-q4 (single GPU)"
    run_experiment "Scale Sweep 32B-q4" "1" \
        "03_scale_sweep.py" --scale 32B-q4 --runs "$RUNS" --seed "$SEED"

    # Step D.7: 70B quantized (2 GPUs, ~38GB)
    log "Step D.7: 70B-q4 (GPUs 1+2, device_map=auto)"
    run_experiment "Scale Sweep 70B-q4" "1,2" \
        "03_scale_sweep.py" --scale 70B-q4 --runs 3 --seed "$SEED"
}

# ================================================================
# Phase E: Identity Signatures Multi-Scale
# ================================================================

phase_E() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE E: IDENTITY SIGNATURES MULTI-SCALE               ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # Step E.1: Small models in parallel
    log "Step E.1: 0.6B (GPU 1) + 1.1B (GPU 2) in parallel"
    run_experiment "Identity 0.6B" "1" \
        "03b_identity_signatures.py" --model "Qwen/Qwen3-0.6B" --runs "$RUNS" --seed "$SEED" &
    local pid1=$!
    run_experiment "Identity 1.1B" "2" \
        "03b_identity_signatures.py" --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --runs "$RUNS" --seed "$SEED" &
    local pid2=$!
    wait $pid1 $pid2

    # Step E.2: 7B
    log "Step E.2: 7B identity"
    run_experiment "Identity 7B" "1" \
        "03b_identity_signatures.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs "$RUNS" --seed "$SEED"

    # Step E.3: 32B quantized
    log "Step E.3: 32B-q4 identity"
    run_experiment "Identity 32B-q4" "1" \
        "03b_identity_signatures.py" --model "Qwen/Qwen2.5-32B-Instruct" --quantize --runs 3 --seed "$SEED"
}

# ================================================================
# Phase F: Multi-Scale Extensions (Strongest findings at larger models)
# ================================================================

phase_F() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE F: MULTI-SCALE EXTENSIONS (7B + 32B)             ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # F.1: Deception at 7B
    run_experiment "Deception 7B" "1" \
        "04_deception_forensics.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs "$RUNS" --seed "$SEED"

    # F.2: Layer Map at 7B
    run_experiment "Layer Map 7B" "1" \
        "05_layer_map.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs 3 --seed "$SEED"

    # F.3: Temporal at 7B
    run_experiment "Temporal 7B" "1" \
        "06_temporal_evolution.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs 3 --seed "$SEED"

    # F.4: Deception at 32B-q4
    run_experiment "Deception 32B-q4" "1" \
        "04_deception_forensics.py" --model "Qwen/Qwen2.5-32B-Instruct" --quantize --runs 3 --seed "$SEED"

    # F.5: Layer Map at 32B-q4
    run_experiment "Layer Map 32B-q4" "1" \
        "05_layer_map.py" --model "Qwen/Qwen2.5-32B-Instruct" --quantize --runs 3 --seed "$SEED"
}

# ================================================================
# Phase G: Projector Training
# ================================================================

phase_G() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE G: PROJECTOR TRAINING                            ║"
    log "║  Cross-model cache transfer learning                    ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # G.1: Small projector (0.6B → 0.5B)
    run_experiment "Projector Small (0.6B→0.5B)" "1" \
        "02b_projector_transfer.py"

    log "NOTE: Medium and large projector configs need manual setup."
    log "See execution plan for 4B→3B and 32B→7B pairs."
}

# ================================================================
# Main execution
# ================================================================

log "╔══════════════════════════════════════════════════════════════╗"
log "║  KV-CACHE FULL EXPERIMENT CAMPAIGN                          ║"
log "║  Machine: Cassidy (3x RTX 3090)                             ║"
log "║  Liberation Labs / THCoalition                              ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""
log "Project: $PROJECT_DIR"
log "Python: $($PYTHON --version 2>&1)"
log "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null)"
log ""

if [ -n "$DRY_RUN" ]; then
    log "*** DRY RUN MODE — no models loaded ***"
fi

# Run specific phase or all
if [ -n "$PHASE" ]; then
    case "$PHASE" in
        B) phase_B ;;
        C) phase_C ;;
        D) phase_D ;;
        E) phase_E ;;
        F) phase_F ;;
        G) phase_G ;;
        *) echo "Unknown phase: $PHASE (valid: B C D E F G)"; exit 1 ;;
    esac
else
    # Run all phases in order
    log "Running ALL phases (B → G)"
    log ""

    phase_B
    log ""; log "Decision gate: Check Control 3 correlation before proceeding."; log ""

    phase_C
    phase_D
    phase_E
    phase_F
    phase_G
fi

log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  CAMPAIGN COMPLETE                                          ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""
log "Results in: $RESULTS_DIR"
log "Log: $LOG_FILE"

# List all result files
log ""
log "Result files:"
find "$RESULTS_DIR" -name "*_results.json" -exec ls -lh {} \; 2>/dev/null | while read line; do
    log "  $line"
done
