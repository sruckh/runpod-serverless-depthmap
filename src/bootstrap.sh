#!/usr/bin/env bash
# Container entrypoint on RunPod. No installs, clones, or volume-hosted code.
set -euo pipefail
trap 'printf "[bootstrap] startup failed at line %s\n" "$LINENO" >&2' ERR

if [ -d /runpod-volume ]; then
  BASE=/runpod-volume
else
  BASE=/workspace
fi
if [ -z "${WORKSPACE:-}" ] || [ "$WORKSPACE" = /workspace/Lotus ]; then
  WORKSPACE="$BASE/Lotus"
fi
export WORKSPACE
export HF_HOME="${HF_HOME:-$WORKSPACE/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$HF_HOME}}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export RUNPOD_SERVER_PORT="${RUNPOD_SERVER_PORT:-3000}"
# Immutable code comes from the image, never an old copy on the mounted volume.
export PYTHONPATH=/app:/opt/lotus
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$WORKSPACE/output_images"
printf '[bootstrap] runtime=image workspace=%s\n' "$WORKSPACE"
exec /opt/venv/bin/python -u /app/handler.py "$@"
