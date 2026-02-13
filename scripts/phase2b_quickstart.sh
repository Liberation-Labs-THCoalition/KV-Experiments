#!/bin/bash
# Phase 2b Quick Start Script
# Run this on donated GPU when time window opens
#
# Usage:
#   ./scripts/phase2b_quickstart.sh           # Full run (train + validate + sweep + identity)
#   ./scripts/phase2b_quickstart.sh --train   # Just train projector
#   ./scripts/phase2b_quickstart.sh --validate # Just run validation (needs checkpoint)
#   ./scripts/phase2b_quickstart.sh --sweep   # Just run scale sweep
#   ./scripts/phase2b_quickstart.sh --identity # Just run identity experiments
#   ./scripts/phase2b_quickstart.sh --preflight # Just run preflight checks
#
# Environment:
#   Tested on: Ubuntu 22.04/24.04, CUDA 12.x, Python 3.10+
#   Minimum VRAM: 9GB (base model pair), 16GB recommended, 40GB for 32B models
#
# Time estimates (24GB GPU):
#   Projector training: ~1-2 hours
#   Validation: ~30 min
#   Scale sweep: ~4-8 hours
#   Identity: ~1 hour
#   Total: ~8-12 hours

set -euo pipefail

echo "=============================================="
echo "PHASE 2b: KV-CACHE PROJECTOR EXPERIMENTS"
echo "Liberation Labs / THCoalition"
echo "=============================================="
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/phase2b_projector"
RESULTS_DIR="$PROJECT_ROOT/results"
C2C_DIR="$PROJECT_ROOT/C2C"
LOG_DIR="$PROJECT_ROOT/logs"

# Models
BASE_MODEL="Qwen/Qwen3-0.6B"
TEACHER_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$LOG_DIR"

# Parse arguments
RUN_TRAIN=false
RUN_VALIDATE=false
RUN_SWEEP=false
RUN_IDENTITY=false
RUN_PREFLIGHT=false

if [ $# -eq 0 ]; then
    RUN_TRAIN=true
    RUN_VALIDATE=true
    RUN_SWEEP=true
    RUN_IDENTITY=true
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --train) RUN_TRAIN=true; shift ;;
            --validate) RUN_VALIDATE=true; shift ;;
            --sweep) RUN_SWEEP=true; shift ;;
            --identity) RUN_IDENTITY=true; shift ;;
            --preflight) RUN_PREFLIGHT=true; shift ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
fi

# ============================================
# PREFLIGHT CHECKS
# ============================================

preflight() {
    echo "[PREFLIGHT] Running environment checks..."
    local FAIL=0

    # Python version
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        echo "  ✓ Python $PY_VER"
    else
        echo "  ✗ Python 3.10+ required (found: $PY_VER)"
        FAIL=1
    fi

    # CUDA
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        VRAM_GB=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')")
        echo "  ✓ CUDA available: $GPU_NAME ($VRAM_GB GB)"

        # Check bfloat16 support
        if python3 -c "import torch; assert torch.cuda.is_bf16_supported()" 2>/dev/null; then
            echo "  ✓ bfloat16 supported"
            export DTYPE="bfloat16"
        else
            echo "  ⚠ bfloat16 not supported — falling back to float16"
            export DTYPE="float16"
        fi
    else
        echo "  ✗ CUDA not available"
        FAIL=1
    fi

    # Key packages
    for pkg in torch transformers accelerate bitsandbytes scipy; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo "  ✓ $pkg"
        else
            echo "  ✗ $pkg not installed"
            FAIL=1
        fi
    done

    # scikit-learn (needed for identity experiments)
    if python3 -c "import sklearn" 2>/dev/null; then
        echo "  ✓ scikit-learn"
    else
        echo "  ⚠ scikit-learn not installed (needed for --identity)"
        if [ "$RUN_IDENTITY" = true ]; then
            echo "    Installing scikit-learn..."
            pip install scikit-learn --quiet
        fi
    fi

    # C2C framework
    if [ -d "$C2C_DIR" ] && [ -f "$C2C_DIR/pyproject.toml" ]; then
        echo "  ✓ C2C framework present"
    else
        echo "  ⚠ C2C framework not found"
        if [ "$RUN_TRAIN" = true ]; then
            echo "    Cloning C2C repository..."
            git clone https://github.com/thu-nics/C2C.git "$C2C_DIR"
            cd "$C2C_DIR"
            pip install -e ".[training,evaluation]" --quiet
            cd "$PROJECT_ROOT"
            echo "  ✓ C2C framework installed"
        fi
    fi

    # Disk space (need ~100GB for models + checkpoints)
    DISK_FREE=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "${DISK_FREE:-0}" -ge 50 ]; then
        echo "  ✓ Disk space: ${DISK_FREE}GB free"
    else
        echo "  ⚠ Low disk space: ${DISK_FREE}GB free (50GB+ recommended)"
    fi

    echo ""
    if [ $FAIL -ne 0 ]; then
        echo "[PREFLIGHT] ✗ Critical issues found. Fix above errors before proceeding."
        exit 1
    else
        echo "[PREFLIGHT] ✓ All checks passed."
    fi
}

# Always run preflight
preflight

if [ "$RUN_PREFLIGHT" = true ]; then
    exit 0
fi

# ============================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================

echo ""
echo "[1/6] Installing project dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet 2>/dev/null || true

# Install C2C if present
if [ -d "$C2C_DIR" ] && [ -f "$C2C_DIR/pyproject.toml" ]; then
    cd "$C2C_DIR"
    pip install -e ".[training,evaluation]" --quiet 2>/dev/null || true
    cd "$PROJECT_ROOT"
    echo "  ✓ C2C package installed"
fi

# ============================================
# STEP 2: DOWNLOAD MODELS
# ============================================

echo ""
echo "[2/6] Downloading models..."
python3 -c "
from huggingface_hub import snapshot_download
import sys

models = ['$BASE_MODEL', '$TEACHER_MODEL']
for model in models:
    print(f'  Downloading {model}...')
    try:
        snapshot_download(model)
        print(f'    ✓ {model} ready')
    except Exception as e:
        print(f'    ✗ {model} failed: {e}')
        sys.exit(1)
"

# ============================================
# STEP 3: TRAIN PROJECTOR
# ============================================

if [ "$RUN_TRAIN" = true ]; then
    echo ""
    echo "[3/6] Training C2C projector..."

    if [ -f "$CHECKPOINT_DIR/projector_0.pt" ]; then
        echo "  Checkpoint exists at $CHECKPOINT_DIR. Skipping training."
        echo "  (Delete checkpoint dir to retrain)"
    elif [ -f "$C2C_DIR/script/train/SFT_train.py" ]; then
        echo "  Starting projector training..."
        echo "  Config: $PROJECT_ROOT/recipe/phase2b_config.json"
        echo "  Estimated time: 1-2 hours on 24GB GPU"
        echo ""

        python3 "$C2C_DIR/script/train/SFT_train.py" \
            --config "$PROJECT_ROOT/recipe/phase2b_config.json" \
            2>&1 | tee "$LOG_DIR/projector_training_$(date +%Y%m%d_%H%M%S).log"

        if [ -f "$CHECKPOINT_DIR/projector_0.pt" ]; then
            echo "  ✓ Projector training complete"
        else
            echo "  ✗ Training may have failed — no checkpoint found"
            echo "  Check logs at $LOG_DIR/"
        fi
    else
        echo "  ✗ C2C training script not found at: $C2C_DIR/script/train/SFT_train.py"
        echo "  Run: git clone https://github.com/thu-nics/C2C.git $C2C_DIR"
    fi
else
    echo ""
    echo "[3/6] Skipping projector training (--train not specified)"
fi

# ============================================
# STEP 4: PROJECTOR VALIDATION
# ============================================

if [ "$RUN_VALIDATE" = true ]; then
    echo ""
    echo "[4/6] Running projector transfer validation..."

    if [ -f "$CHECKPOINT_DIR/projector_0.pt" ]; then
        python3 "$PROJECT_ROOT/code/02b_projector_transfer.py" \
            --checkpoint "$CHECKPOINT_DIR" \
            --base-model "$BASE_MODEL" \
            --teacher-model "$TEACHER_MODEL" \
            --verbose \
            2>&1 | tee "$LOG_DIR/validation_$(date +%Y%m%d_%H%M%S).log"

        echo "  ✓ Validation complete. Results in $RESULTS_DIR/"
    else
        echo "  ⚠ No projector checkpoint found. Skipping validation."
        echo "  Run with --train first, or provide checkpoint at $CHECKPOINT_DIR"
    fi
else
    echo ""
    echo "[4/6] Skipping validation (--validate not specified)"
fi

# ============================================
# STEP 5: SCALE SWEEP
# ============================================

if [ "$RUN_SWEEP" = true ]; then
    echo ""
    echo "[5/6] Running cognitive mode scale sweep..."

    # Always run base model
    echo "  Testing 0.6B baseline..."
    python3 "$PROJECT_ROOT/code/03_scale_sweep.py" \
        --scale 0.6B \
        --num-runs 5 \
        2>&1 | tee "$LOG_DIR/sweep_0.6B_$(date +%Y%m%d_%H%M%S).log"

    # Check VRAM for larger models
    VRAM_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))" 2>/dev/null || echo "0")

    if [ "$VRAM_GB" -ge 16 ]; then
        echo ""
        echo "  Testing 7B (VRAM: ${VRAM_GB}GB)..."
        python3 "$PROJECT_ROOT/code/03_scale_sweep.py" \
            --scale 7B \
            --num-runs 5 \
            2>&1 | tee "$LOG_DIR/sweep_7B_$(date +%Y%m%d_%H%M%S).log"
    else
        echo "  Skipping 7B (need 16GB VRAM, have ${VRAM_GB}GB)"
    fi

    if [ "$VRAM_GB" -ge 40 ]; then
        echo ""
        echo "  Testing 32B quantized (VRAM: ${VRAM_GB}GB)..."
        python3 "$PROJECT_ROOT/code/03_scale_sweep.py" \
            --scale 32B \
            --quantize \
            --num-runs 3 \
            2>&1 | tee "$LOG_DIR/sweep_32B_$(date +%Y%m%d_%H%M%S).log"
    else
        echo "  Skipping 32B (need 40GB VRAM, have ${VRAM_GB}GB)"
    fi

    echo "  ✓ Scale sweep complete. Results in $RESULTS_DIR/"
else
    echo ""
    echo "[5/6] Skipping scale sweep (--sweep not specified)"
fi

# ============================================
# STEP 6: IDENTITY SIGNATURES
# ============================================

if [ "$RUN_IDENTITY" = true ]; then
    echo ""
    echo "[6/6] Running identity signature experiments..."

    python3 "$PROJECT_ROOT/code/03b_identity_signatures.py" \
        --model "$BASE_MODEL" \
        --num-samples 10 \
        2>&1 | tee "$LOG_DIR/identity_$(date +%Y%m%d_%H%M%S).log"

    echo "  ✓ Identity experiments complete. Results in $RESULTS_DIR/"
else
    echo ""
    echo "[6/6] Skipping identity experiments (--identity not specified)"
fi

# ============================================
# SUMMARY
# ============================================

echo ""
echo "=============================================="
echo "PHASE 2b COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no new results files)"
echo ""
echo "Logs:"
ls -la "$LOG_DIR"/*.log 2>/dev/null | tail -5 || echo "  (no logs)"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR/"
echo "  2. Compare projector_transfer vs raw_transfer in phase2b_transfer_results.json"
echo "  3. Check scale_sweep_results.json for cross-scale cognitive patterns"
echo "  4. Check identity_signatures_results.json for persona fingerprinting"
echo "  5. Push results: git add results/ logs/ && git commit && git push"
