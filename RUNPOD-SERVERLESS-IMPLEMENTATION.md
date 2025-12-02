# RunPod Serverless Implementation Plan (Lotus Depth)

This document is a detailed implementation plan to deploy Lotus depth inference as a RunPod serverless endpoint. It includes a ToDo list, code/command examples, and a review checklist. Security: enforce `LOTUS_API_KEY` on all calls.

## ToDo Overview
1) Prepare project layout on the RunPod network volume.
2) Write Dockerfile (lean base; uses mounted volume; runs handler).
3) Add bootstrap script to set up venv, install deps, prefetch model/cache, and warm up.
4) Add handler code (RunPod serverless entry) with API key check, input handling (URL/base64), validation, inference, output save, and response.
5) Add configuration/env wiring (HF cache, model ID, API key).
6) Optional: add prefetch script using `hf download`.
7) Document run/test steps and finalize.
8) Review each item against the checklist.

## 1) Project Layout on Network Volume
- Mount volume at `/workspace`.
- Directory structure:
  - `/workspace/Lotus/venv` — Python 3.12 venv
  - `/workspace/Lotus/src` — handler + bootstrap/prefetch scripts
  - `/workspace/Lotus/upstream` — cloned Lotus repo (runtime; provides `pipeline.py`)
  - `/workspace/Lotus/cache` — HF cache (`HF_HOME`)
  - `/workspace/Lotus/models` — optional snapshot from `hf download`
  - `/workspace/Lotus/output_images` — outputs written here
  - `/workspace/Lotus/logs` — optional logs
- Create dirs (example):
  ```bash
  mkdir -p /workspace/Lotus/{src,cache,models,output_images,logs}
  ```

## 2) Dockerfile
Goal: minimal base, Python 3.12; everything else on the volume. Entrypoint runs bootstrap.

`Dockerfile`:
```Dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/Lotus/cache \
    HUGGINGFACE_HUB_CACHE=/workspace/Lotus/cache \
    WORKSPACE=/workspace/Lotus

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /workspace/Lotus
COPY src/ /workspace/Lotus/src/

# Entrypoint: run bootstrap then handler
CMD ["bash", "/workspace/Lotus/src/bootstrap.sh"]
```

## 3) Bootstrap Script (`src/bootstrap.sh`)
Responsibilities:
- Ensure venv exists; if not, create and install pinned deps.
- Activate venv.
- Ensure cache/output dirs exist.
- Optional prefetch using `hf download`.
- Optional warmup to load model once.
- Start the handler.

`src/bootstrap.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace/Lotus
VENV=$WORKSPACE/venv
SRC=$WORKSPACE/src
CACHE=$WORKSPACE/cache
OUTPUT=$WORKSPACE/output_images
LOGS=$WORKSPACE/logs
UPSTREAM=$WORKSPACE/upstream

mkdir -p "$CACHE" "$OUTPUT" "$LOGS" "$SRC"

# Clone upstream Lotus repo if missing (runtime code source)
if [ ! -d \"$UPSTREAM/.git\" ]; then
  git clone https://github.com/EnVision-Research/Lotus.git \"$UPSTREAM\"
fi

if [ ! -d \"$VENV\" ]; then
  python -m venv \"$VENV\"
  source \"$VENV/bin/activate\"
  pip install --upgrade pip
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  pip install \
    accelerate==1.4.0 datasets==3.3.1 diffusers==0.32.2 easydict==1.13 ftfy==6.3.1 \
    geffnet==1.0.2 h5py==3.13.0 huggingface-hub==1.1.7 ImageIO==2.37.0 imageio-ffmpeg==0.6.0 \
    Jinja2==3.1.6 matplotlib==3.10.1 numpy==2.2.3 omegaconf==2.4.0.dev3 opencv-python==4.11.0.86 \
    peft==0.15.1 spaces==0.33.1 tabulate==0.9.0 tensorboard==2.19.0 transformers==4.49.0 \
    requests==2.32.3 pillow==11.0.0 runpod==1.6.1
else
  source \"$VENV/bin/activate\"
fi

export HF_HOME=\"$CACHE\"
export HUGGINGFACE_HUB_CACHE=\"$CACHE\"
export PYTHONPATH=\"$SRC:$UPSTREAM:$PYTHONPATH\"

# Optional: prefetch model (uncomment if desired)
# MODEL_ID=${MODEL_ID:-jingheya/lotus-depth-d-v2-0-disparity}
# hf download \"$MODEL_ID\" --repo-type model --local-dir $WORKSPACE/models/lotus-depth-d --include \"*.safetensors\" \"*.json\" \"*.pt\"

# Optional warmup (non-fatal if it fails)
python \"$SRC/handler.py\" --warmup || true

# Start handler (runpod serverless mode)
exec python \"$SRC/handler.py\" --rp_serve_api
```

Make executable:
```bash
chmod +x /workspace/Lotus/src/bootstrap.sh
```

## 4) Handler (`src/handler.py`)
Responsibilities:
- Enforce `LOTUS_API_KEY` (env). Reject if missing/mismatch.
- Inputs: `image_url` or `image_base64`; guard size; max resolution 4096x4096.
- Load `LotusDPipeline` once (regression mode) with HF cache at `/workspace/Lotus/cache`; model default `jingheya/lotus-depth-d-v2-0-disparity`.
- Use fp16 on CUDA; enable flash-attn if available.
- Postprocess: grayscale PNG, save to `/workspace/Lotus/output_images/<job_id>.png`, return base64 + min/max.
- Warmup mode: `--warmup` loads model and exits.

`src/handler.py` (skeleton):
```python
import os, base64, io, logging, uuid, argparse, requests
from PIL import Image
import numpy as np
import torch
import runpod
from pipeline import LotusDPipeline  # ensure this file exists in src

API_KEY_ENV = "LOTUS_API_KEY"
MODEL_ID_DEFAULT = "jingheya/lotus-depth-d-v2-0-disparity"
OUTPUT_DIR = "/workspace/Lotus/output_images"
MAX_PIXELS = 4096 * 4096  # resolution guard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pipeline = None

def load_pipeline(model_id=None):
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    model_id = model_id or MODEL_ID_DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = LotusDPipeline.from_pretrained(model_id, torch_dtype=dtype, cache_dir=os.environ.get("HF_HOME"))
    pipe = pipe.to(device)
    try:
        import flash_attn  # noqa: F401
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        logger.info("flash-attn/xformers not enabled; continuing.")
    pipe.set_progress_bar_config(disable=True)
    _pipeline = pipe
    return _pipeline

def check_api_key(job):
    supplied = None
    headers = job.get("http", {}).get("headers") if job.get("http") else None
    if headers:
        supplied = headers.get("lotus-api-key") or headers.get("LOTUS-API-KEY")
    if not supplied:
        supplied = job.get("input", {}).get("api_key")
    expected = os.environ.get(API_KEY_ENV)
    if not expected or supplied != expected:
        raise PermissionError("Unauthorized")

def load_image_from_job(job_input):
    if "image_url" in job_input:
        url = job_input["image_url"]
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.content
    elif "image_base64" in job_input:
        data = base64.b64decode(job_input["image_base64"])
    else:
        raise ValueError("No image provided.")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("Image too large; max 4096x4096 pixels.")
    return img

def run_inference(job):
    check_api_key(job)
    job_input = job["input"]
    img = load_image_from_job(job_input)
    np_img = np.array(img).astype(np.float32)
    tensor = torch.tensor(np_img).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 127.5 - 1.0
    pipe = load_pipeline(job_input.get("model_id"))
    tensor = tensor.to(pipe.device)
    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).to(pipe.device)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    with torch.no_grad():
        pred = pipe(
            rgb_in=tensor,
            prompt="",
            num_inference_steps=1,
            generator=None,
            output_type="np",
            timesteps=[job_input.get("timestep", 999)],
            task_emb=task_emb,
            processing_res=job_input.get("processing_res"),
            match_input_res=not job_input.get("output_processing_res", False),
            resample_method=job_input.get("resample_method", "bilinear"),
        ).images[0]
    out_np = pred.mean(axis=-1)
    out_min, out_max = float(out_np.min()), float(out_np.max())
    norm = (out_np - out_min) / (out_max - out_min + 1e-8)
    png = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    png_path = os.path.join(OUTPUT_DIR, fname)
    png.save(png_path)
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"image_base64": b64, "min": out_min, "max": out_max, "file_path": png_path}

def handler(job):
    try:
        return run_inference(job)
    except PermissionError as e:
        return {"error": str(e), "status": 401}
    except Exception as e:
        return {"error": str(e), "status": 400}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true")
    args, _ = parser.parse_known_args()
    if args.warmup:
        try:
            load_pipeline()
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
        exit(0)
    runpod.serverless.start({"handler": handler})
```

Notes:
- Ensure `pipeline.py` (LotusDPipeline) is in `src/`.
- Enforce API key via `LOTUS_API_KEY` env variable.
- Max resolution guard: 4096x4096.
- Saves PNG to `/workspace/Lotus/output_images`, returns base64 and stats.
- Warmup loads the pipeline to pull weights.

## 5) Env/Config Wiring
Set in RunPod serverless configuration:
- `LOTUS_API_KEY` (required)
- `MODEL_ID` (optional override, default `jingheya/lotus-depth-d-v2-0-disparity`)
- `HF_HOME=/workspace/Lotus/cache`
- `HUGGINGFACE_HUB_CACHE=/workspace/Lotus/cache`

## 6) Optional Prefetch Script (`src/prefetch.sh`)
Run once on the pod with the volume mounted.
```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_ID=${MODEL_ID:-jingheya/lotus-depth-d-v2-0-disparity}
hf download "$MODEL_ID" --repo-type model --local-dir /workspace/Lotus/models/lotus-depth-d --include "*.safetensors" "*.json" "*.pt"
```

## 7) Testing and Warmup Steps
1) Deploy image with volume mounted at `/workspace`.
2) First start runs `bootstrap.sh`: creates venv, installs deps, optional prefetch/warmup.
3) Send a request with correct `LOTUS_API_KEY` header and a small test image (prefer URL for >10MB).
4) Verify output saved in `/workspace/Lotus/output_images` and response contains base64 + min/max.
5) Confirm payload guidance: base64 only for small inputs; use URL for larger (payload limit ~10MB).

## 9) Calling the endpoint (examples)
- The serverless handler expects `LOTUS_API_KEY` via header (`lotus-api-key` / `LOTUS-API-KEY`) or `input.api_key`.
- Preferred input for large images is URL (avoid 10MB `/run` payload limit); base64 is accepted for small images.
- Max resolution: 4096x4096; larger images are rejected.
- Example payload (HTTP POST to RunPod `/run`):
```json
{
  "input": {
    "api_key": "YOUR_KEY",
    "image_url": "https://example.com/sample.jpg",
    "processing_res": 768,
    "disparity": true
  }
}
```
- Response (success):
```json
{
  "image_base64": "...png...",
  "min": 0.0123,
  "max": 1.2345,
  "file_path": "/workspace/Lotus/output_images/abcd1234.png"
}
```
- If unauthorized:
```json
{ "error": "Unauthorized", "status": 401 }
```
- If input invalid (e.g., too large or missing image):
```json
{ "error": "Image too large; max 4096x4096 pixels.", "status": 400 }
```

## 8) Review Checklist
- Dockerfile: uses specified base; minimal; copies `src`; CMD runs `bootstrap.sh`.
- Volume layout: `/workspace/Lotus/{venv,src,cache,models,output_images,logs}` present; permissions OK.
- Bootstrap: creates venv if missing; installs pinned torch/flash-attn + libs; sets HF_HOME; optional `hf download`; warmup non-fatal.
- Handler: API key enforcement against `LOTUS_API_KEY`; rejects missing/mismatch; accepts URL/base64; enforces 4096x4096 cap; runs LotusDPipeline regression; saves PNG; returns base64 + min/max; clear error payloads.
- Security: No bypass of API key; unauthorized attempts rejected.
- Model caching: HF cache at `/workspace/Lotus/cache`; download/prefetch works.
- Flash-attn: imports cleanly or logs and continues.
- Testing: sample request succeeds; output file written; response within payload limits.
- Logs: include startup/warmup info; no secrets logged.
