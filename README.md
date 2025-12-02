# RunPod Serverless Depth Map

This repository packages the Lotus depth regression model (`jingheya/lotus-depth-d-v2-0-disparity`) as a RunPod serverless endpoint that accepts an image and returns a grayscale depth map.

## What this image does
- Clones the upstream Lotus repo at runtime (for `pipeline.py`) into `/workspace/Lotus/upstream`.
- Builds a Python 3.12 venv with pinned inference deps (torch 2.8.0 cu128, flash-attn 2.8.3, diffusers, transformers, etc.).
- Caches models under `/workspace/Lotus/cache` (HF cache).
- Runs a RunPod handler that:
  - Enforces `LOTUS_API_KEY` (header `lotus-api-key`/`LOTUS-API-KEY` or `input.api_key`).
  - Accepts `image_url` (preferred) or `image_base64`.
  - Enforces max resolution 4096x4096; rejects larger.
  - Returns grayscale depth PNG as base64 plus min/max stats; saves PNG to `/workspace/Lotus/output_images`.

## Layout on mounted volume (`/workspace`)
- `/workspace/Lotus/venv` — venv with pinned deps.
- `/workspace/Lotus/src` — handler, bootstrap, prefetch scripts.
- `/workspace/Lotus/upstream` — cloned Lotus repo (runtime).
- `/workspace/Lotus/cache` — HF cache (`HF_HOME`/`HUGGINGFACE_HUB_CACHE`).
- `/workspace/Lotus/models` — optional manual snapshot (`hf download`).
- `/workspace/Lotus/output_images` — saved PNGs.

## Required env vars
- `LOTUS_API_KEY` — required for all calls.
- `HF_HOME` and `HUGGINGFACE_HUB_CACHE` — default to `/workspace/Lotus/cache`.
- `MODEL_ID` — optional override (default `jingheya/lotus-depth-d-v2-0-disparity`).

## Calling the endpoint (RunPod `/run` example)
Use URL for larger images (payload limit ~10MB on `/run`); base64 only for small inputs.
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
Headers (recommended): `lotus-api-key: YOUR_KEY`

Success response:
```json
{
  "image_base64": "...png...",
  "min": 0.0123,
  "max": 1.2345,
  "file_path": "/workspace/Lotus/output_images/<uuid>.png"
}
```
Errors:
- Unauthorized: `{ "error": "Unauthorized", "status": 401 }`
- Invalid/too large image: `{ "error": "Image too large; max 4096x4096 pixels.", "status": 400 }`

## Build / CI
- GitHub Actions workflow: `.github/workflows/docker-publish.yml` builds and pushes `gemneye/runpod-serverless-depthmap` on pushes to `main`, using secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD`.

## Optional prefetch
Run once on the pod to pre-download weights:
```bash
MODEL_ID=jingheya/lotus-depth-d-v2-0-disparity \
HF_HOME=/workspace/Lotus/cache \
hf download "$MODEL_ID" --repo-type model --local-dir /workspace/Lotus/models/lotus-depth-d --include "*.safetensors" "*.json" "*.pt"
```

## Notes
- FP16 on CUDA; attempts flash-attn, falls back gracefully.
- Saves outputs to `/workspace/Lotus/output_images` and returns base64 plus stats.
