#!/bin/bash
# Run SOAR2COT with different models (separate DB and progress.json)
#
# Usage:
#   ./scripts/run_with_model.sh qwen3

set -e

MODEL_NAME="${1:-qwen3}"

echo "=========================================="
echo "Running SOAR2COT with model: $MODEL_NAME"
echo "=========================================="

# Base paths
BASE_DIR="/data/hjkim/soar2cot"
DATA_DIR="$BASE_DIR/data"

# Model-specific directory
MODEL_DATA_DIR="$DATA_DIR/$MODEL_NAME"
mkdir -p "$MODEL_DATA_DIR"

# Set environment variables
export PROGRESS_FILE="$MODEL_DATA_DIR/progress.json"
export MODEL_CONFIG="$MODEL_NAME"  # For config selection in run.py

# Set DB (use model-specific or fallback to default)
case "$MODEL_NAME" in
    "qwen3")
        export NEON_DSN="${NEON_DSN_QWEN3:-$NEON_DSN}"
        ;;
    "gpt-oss")
        export NEON_DSN="${NEON_DSN_GPT_OSS:-$NEON_DSN}"
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        exit 1
        ;;
esac

# Show config
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Progress: $PROGRESS_FILE"
echo "  DB: ${NEON_DSN:0:40}..."
echo ""

# Check existing progress
if [ -f "$PROGRESS_FILE" ]; then
    TOTAL=$(jq '.total_completed // 0' "$PROGRESS_FILE" 2>/dev/null || echo "0")
    echo "Found existing progress: $TOTAL completed"
else
    echo "Starting fresh (no progress file)"
fi

echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run pipeline
echo ""
echo "Starting pipeline..."
cd "$BASE_DIR"

LOG_FILE="$BASE_DIR/logs/run_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Use unbuffered Python output
PYTHONUNBUFFERED=1 /home/ubuntu/miniconda3/envs/soar2cot/bin/python -u -m src.run 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Completed!"
echo "=========================================="
echo "Log: $LOG_FILE"
echo "Progress: $PROGRESS_FILE"
echo ""
