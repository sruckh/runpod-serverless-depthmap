import argparse
import base64
import io
import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
import runpod
import torch
from PIL import Image

# Ensure upstream Lotus repo is importable
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace/Lotus"))
UPSTREAM = WORKSPACE / "upstream"
SRC = WORKSPACE / "src"
for p in (str(UPSTREAM), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import LotusDPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_ENV = "LOTUS_API_KEY"
MODEL_ID_DEFAULT = "jingheya/lotus-depth-d-v2-0-disparity"
OUTPUT_DIR = WORKSPACE / "output_images"
MAX_PIXELS = 4096 * 4096  # resolution guard

_pipeline = None

# Silence noisy deprecations from diffusers Lora and torch_dtype notices
warnings.filterwarnings(
    "ignore",
    message=".*LoraLoaderMixin is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch_dtype` is deprecated",
    category=FutureWarning,
)


def load_pipeline(model_id=None):
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    model_id = model_id or MODEL_ID_DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = LotusDPipeline.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=os.environ.get("HF_HOME")
    )
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
    job_input = job.get("input") or {}
    img = load_image_from_job(job_input)
    np_img = np.array(img).astype(np.float32)
    tensor = torch.tensor(np_img).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 127.5 - 1.0
    pipe = load_pipeline(job_input.get("model_id"))
    tensor = tensor.to(pipe.device, dtype=pipe.dtype)
    task_emb = torch.tensor([1, 0], device=pipe.device, dtype=pipe.dtype).unsqueeze(0)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    png_path = OUTPUT_DIR / fname
    png.save(png_path)
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"image_base64": b64, "min": out_min, "max": out_max, "file_path": str(png_path)}


def handler(job):
    try:
        return run_inference(job)
    except PermissionError as e:
        return {"error": str(e), "status": 401}
    except Exception as e:
        logger.exception("Inference failed")
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
        sys.exit(0)
    runpod.serverless.start({"handler": handler})
