#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace/Lotus}
MODEL_ID=${MODEL_ID:-jingheya/lotus-depth-d-v2-0-disparity}
mkdir -p "$WORKSPACE/models"

hf download "$MODEL_ID" --repo-type model --local-dir "$WORKSPACE/models/lotus-depth-d" --include "*.safetensors" "*.json" "*.pt"
