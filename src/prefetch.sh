#!/usr/bin/env bash
# Optional command INSIDE the image on RunPod, not on the source/VPS host.
set -euo pipefail
if [ -d /runpod-volume ]; then BASE=/runpod-volume; else BASE=/workspace; fi
if [ -z "${WORKSPACE:-}" ] || [ "$WORKSPACE" = /workspace/Lotus ]; then
  WORKSPACE="$BASE/Lotus"
fi
export WORKSPACE
export HF_HOME="${HF_HOME:-$WORKSPACE/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export PYTHONPATH=/app:/opt/lotus
mkdir -p "$HF_HUB_CACHE"
# Complete component files, including tokenizer vocabulary/merges. No filtered
# local-dir snapshot that inference never reads. HF cache handles file locking.
exec /opt/venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
from model_config import MODEL_ID, MODEL_REVISION

snapshot_download(
    MODEL_ID, revision=MODEL_REVISION, cache_dir=os.environ["HF_HUB_CACHE"],
    ignore_patterns=["assets/*", "README.md", ".gitattributes"],
)
print("[prefetch] model snapshot is ready in the inference cache")
PY
