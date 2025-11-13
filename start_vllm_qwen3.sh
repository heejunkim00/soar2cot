#!/bin/bash

# vLLM Server Startup Script for Qwen3-32B
# Runs on GPU 0, 1, 2, 3 with tensor parallelism

set -e

echo "==================================================="
echo "Starting vLLM Server for Qwen3-32B"
echo "==================================================="

# Configuration
MODEL_NAME="Qwen/Qwen3-32B"
GPUS="0,1,2,3"
TENSOR_PARALLEL=4
PORT=8000
LOG_DIR="/data/arclang/logs"
LOG_FILE="$LOG_DIR/vllm_qwen3_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPUs: $GPUS"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "  Port: $PORT"
echo "  Log: $LOG_FILE"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "vLLM not found. Installing..."
    pip install vllm
else
    echo "vLLM is already installed."
fi

echo ""
echo "Starting vLLM server with reasoning mode..."
echo "View logs: tail -f $LOG_FILE"
echo ""

# Start vLLM server with reasoning support
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --disable-custom-all-reduce \
    2>&1 | tee "$LOG_FILE"
