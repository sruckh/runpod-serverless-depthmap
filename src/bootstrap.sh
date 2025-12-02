#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace/Lotus}
VENV="$WORKSPACE/venv"
SRC="$WORKSPACE/src"
CACHE="$WORKSPACE/cache"
OUTPUT="$WORKSPACE/output_images"
LOGS="$WORKSPACE/logs"
UPSTREAM="$WORKSPACE/upstream"

mkdir -p "$CACHE" "$OUTPUT" "$LOGS" "$SRC"

# Clone upstream Lotus repo if missing (code is used at runtime; not baked in image)
if [ ! -d "$UPSTREAM/.git" ]; then
  git clone https://github.com/EnVision-Research/Lotus.git "$UPSTREAM"
fi

if [ ! -d "$VENV" ]; then
  python -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  pip install \
    accelerate==1.4.0 datasets==3.3.1 diffusers==0.32.2 easydict==1.13 ftfy==6.3.1 \
    geffnet==1.0.2 h5py==3.13.0 huggingface-hub==0.36.0 ImageIO==2.37.0 imageio-ffmpeg==0.6.0 \
    Jinja2==3.1.6 matplotlib==3.10.1 numpy==2.2.3 omegaconf==2.4.0.dev3 opencv-python==4.11.0.86 \
    peft==0.15.1 spaces==0.33.1 tabulate==0.9.0 tensorboard==2.19.0 transformers==4.57.3 \
    requests==2.32.3 pillow==11.0.0 runpod==1.6.1
else
  source "$VENV/bin/activate"
fi

export HF_HOME="$CACHE"
export HUGGINGFACE_HUB_CACHE="$CACHE"
export PYTHONPATH="$SRC:$UPSTREAM:$PYTHONPATH"

# Optional: prefetch model (uncomment if desired)
# MODEL_ID=${MODEL_ID:-jingheya/lotus-depth-d-v2-0-disparity}
# hf download "$MODEL_ID" --repo-type model --local-dir $WORKSPACE/models/lotus-depth-d --include "*.safetensors" "*.json" "*.pt"

# Optional warmup (non-fatal if it fails)
python "$SRC/handler.py" --warmup || true

# Start handler (runpod serverless mode)
exec python "$SRC/handler.py" --rp_serve_api
