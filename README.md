# RunPod Serverless Depth Map

This packages the Lotus depth regression model (`jingheya/lotus-depth-d-v2-0-disparity`) as a RunPod serverless endpoint that accepts an image and returns a grayscale depth map.

## What this image does
- Clones the upstream Lotus repo at runtime (for `pipeline.py`) into the workspace volume.
- Builds a Python 3.12 venv with pinned inference deps (torch 2.8.0 cu128, flash-attn 2.8.3, diffusers 0.32.2, transformers 4.57.3, huggingface-hub 0.36.0, runpod 1.6.1, etc.).
- Caches models under the workspace cache dir (see layout).
- Runs a RunPod handler that:
  - Enforces `LOTUS_API_KEY` (header `lotus-api-key`/`LOTUS-API-KEY` or `input.api_key`).
  - Accepts `image_url` (preferred) or `image_base64`.
  - Enforces max resolution 4096x4096; rejects larger.
  - Returns grayscale depth PNG as base64 plus min/max stats; saves PNG to `<WORKSPACE>/output_images` (auto-prunes files older than 14 days).
  - Defaults: fp32 (higher quality), `processing_res=0` (native resolution), `resample_method="bicubic"`.

## Layout on mounted volume
- The worker prefers `/runpod-volume/Lotus` if `/runpod-volume` exists (RunPod serverless mount). Otherwise it falls back to `/workspace/Lotus`.
- `<WORKSPACE>/venv` — venv with pinned deps.
- `<WORKSPACE>/src` — handler, bootstrap, prefetch scripts (restored from image if missing on volume).
- `<WORKSPACE>/upstream` — cloned Lotus repo (runtime).
- `<WORKSPACE>/cache` — HF cache (`HF_HOME`/`HF_HUB_CACHE`/`HUGGINGFACE_HUB_CACHE`).
- `<WORKSPACE>/models` — optional manual snapshot (`hf download`).
- `<WORKSPACE>/output_images` — saved PNGs (auto-pruned after 14 days).

## Required env vars
- `LOTUS_API_KEY` — required for all calls.
- `HF_HOME` / `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` — default to `<WORKSPACE>/cache`.
- `MODEL_ID` — optional override (default `jingheya/lotus-depth-d-v2-0-disparity`).
- If your endpoint uses Bearer auth, also send `Authorization: Bearer <token>` in requests.

## Calling the endpoint (RunPod `/runsync` recommended)
Use `/runsync` for synchronous handling; payload limits (~10MB) still apply. Prefer `image_url` for larger files and base64 only for small inputs.
```json
{
  "input": {
    "api_key": "YOUR_KEY",
    "image_url": "https://example.com/sample.jpg",
    "processing_res": 0,
    "disparity": true
  }
}
```
Headers (if you can set them): `lotus-api-key: YOUR_KEY` and optionally `Authorization: Bearer YOUR_RUNPOD_TOKEN`.

Success response:
```json
{
  "image_base64": "...png...",
  "min": 0.0123,
  "max": 1.2345,
  "file_path": "/runpod-volume/Lotus/output_images/<uuid>.png"
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
HF_HOME=/runpod-volume/Lotus/cache \
hf download "$MODEL_ID" --repo-type model --local-dir /runpod-volume/Lotus/models/lotus-depth-d --include "*.safetensors" "*.json" "*.pt"
```

## Notes
- Defaults: fp32, `processing_res=0` (native), `resample_method="bicubic"`.
- Attempts flash-attn; falls back gracefully if unavailable.
- Saves outputs to `<WORKSPACE>/output_images` (auto-pruned after 14 days) and returns base64 plus stats.
