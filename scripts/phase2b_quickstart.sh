#!/bin/bash
# Phase 2b Quick Start Script
# Run this on donated GPU when time window opens
#
# Usage:
#   ./scripts/phase2b_quickstart.sh           # Full run (train + validate + sweep)
#   ./scripts/phase2b_quickstart.sh --train   # Just train projector
#   ./scripts/phase2b_quickstart.sh --validate # Just run validation (needs checkpoint)
#   ./scripts/phase2b_quickstart.sh --sweep   # Just run scale sweep
#   ./scripts/phase2b_quickstart.sh --identity # Just run identity experiments

set -e

echo "=============================================="
echo "PHASE 2b: KV-CACHE PROJECTOR EXPERIMENTS"
echo "=============================================="
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/phase2b_projector"
RESULTS_DIR="$PROJECT_ROOT/results"
C2C_DIR="$PROJECT_ROOT/C2C"

# Models to download
BASE_MODEL="Qwen/Qwen3-0.6B"
TEACHER_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SCALE_MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"  # Will use quantization
)

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"

# Parse arguments
RUN_TRAIN=false
RUN_VALIDATE=false
RUN_SWEEP=false
RUN_IDENTITY=false

if [ $# -eq 0 ]; then
    # No args = run all
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
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
fi

# Step 0: Check CUDA
echo "[0/6] Checking CUDA availability..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 1: Install dependencies
echo ""
echo "[1/6] Installing dependencies..."
cd "$C2C_DIR"
if [ -f "pyproject.toml" ]; then
    pip install -e ".[training,evaluation]" --quiet
    echo "  C2C package installed"
else
    echo "  Warning: C2C pyproject.toml not found. Installing from requirements..."
    pip install transformers accelerate bitsandbytes scikit-learn scipy --quiet
fi
cd "$PROJECT_ROOT"

# Step 2: Download models
echo ""
echo "[2/6] Downloading models (this may take a while on first run)..."
python -c "
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

# Step 3: Train projector
if [ "$RUN_TRAIN" = true ]; then
    echo ""
    echo "[3/6] Training projector..."

    # Check if checkpoint already exists
    if [ -f "$CHECKPOINT_DIR/projector_0.pt" ]; then
        echo "  Checkpoint exists. Skipping training."
        echo "  To retrain, delete $CHECKPOINT_DIR"
    else
        # Try C2C training script first, fall back to custom
        if [ -f "$C2C_DIR/script/train/SFT_train.py" ]; then
            echo "  Using C2C training script..."
            python "$C2C_DIR/script/train/SFT_train.py" \
                --config "$PROJECT_ROOT/recipe/phase2b_config.json"
        else
            echo "  C2C training script not found."
            echo "  Manual projector training would be needed."
            echo "  For now, continuing with validation experiments..."
        fi
    fi
else
    echo ""
    echo "[3/6] Skipping projector training (--train not specified)"
fi

# Step 4: Run validation
if [ "$RUN_VALIDATE" = true ]; then
    echo ""
    echo "[4/6] Running projector transfer validation..."

    if [ -d "$CHECKPOINT_DIR" ] && [ -f "$CHECKPOINT_DIR/projector_0.pt" ]; then
        python "$PROJECT_ROOT/code/02b_projector_transfer.py" \
            --checkpoint "$CHECKPOINT_DIR" \
            --base-model "$BASE_MODEL" \
            --teacher-model "$TEACHER_MODEL" \
            --verbose
    else
        echo "  No checkpoint found. Running baseline comparison only..."
        # The script will show raw transfer fails vs baseline
        echo "  (Projector training needed for full comparison)"
    fi
else
    echo ""
    echo "[4/6] Skipping validation (--validate not specified)"
fi

# Step 5: Scale sweep
if [ "$RUN_SWEEP" = true ]; then
    echo ""
    echo "[5/6] Running cognitive mode scale sweep..."

    # Start with base model (always works on any GPU)
    echo "  Testing 0.6B baseline..."
    python "$PROJECT_ROOT/code/03_scale_sweep.py" \
        --scale 0.6B \
        --num-runs 5

    # Check VRAM for larger models
    VRAM_GB=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))" 2>/dev/null || echo "0")

    if [ "$VRAM_GB" -ge 16 ]; then
        echo "  Testing 7B (VRAM: ${VRAM_GB}GB)..."
        python "$PROJECT_ROOT/code/03_scale_sweep.py" \
            --scale 7B \
            --num-runs 5
    fi

    if [ "$VRAM_GB" -ge 40 ]; then
        echo "  Testing 32B quantized (VRAM: ${VRAM_GB}GB)..."
        python "$PROJECT_ROOT/code/03_scale_sweep.py" \
            --scale 32B \
            --num-runs 3
    fi

    echo "  Scale sweep results saved to $RESULTS_DIR/scale_sweep_results.json"
else
    echo ""
    echo "[5/6] Skipping scale sweep (--sweep not specified)"
fi

# Step 6: Identity signatures
if [ "$RUN_IDENTITY" = true ]; then
    echo ""
    echo "[6/6] Running identity signature experiments..."

    python "$PROJECT_ROOT/code/03b_identity_signatures.py" \
        --model "$BASE_MODEL" \
        --num-samples 10

    echo "  Identity results saved to $RESULTS_DIR/identity_signatures_results.json"
else
    echo ""
    echo "[6/6] Skipping identity experiments (--identity not specified)"
fi

# Summary
echo ""
echo "=============================================="
echo "PHASE 2b COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no results files yet)"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR/"
echo "  2. If projector training succeeded, compare projector_transfer vs raw_transfer"
echo "  3. Check scale_sweep_results.json for cross-scale cognitive mode patterns"
echo "  4. Check identity_signatures_results.json for persona fingerprinting"
echo ""
echo "For detailed analysis, run:"
echo "  python -c \"import json; print(json.dumps(json.load(open('$RESULTS_DIR/phase2b_transfer_results.json')), indent=2))\""
