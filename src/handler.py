"""RunPod Serverless entrypoint. All runtime work happens inside the image on RunPod."""

import argparse
import base64
import hmac
import io
import logging
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import runpod
import torch
from PIL import Image, features

from image_input import JobError, Options, load_image_from_job, validate_input
from model_config import MODEL_ID, MODEL_REVISION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace/Lotus"))
OUTPUT_DIR = WORKSPACE / "output_images"
_pipeline: Any = None
_task_emb: Any = None
# Diffusers pipelines may mutate scheduler/state. Serialize access rather than
# allowing concurrent calls to race over the same model and exhaust GPU memory.
_inference_lock = threading.Lock()


class StartupError(RuntimeError):
    """A known configuration error whose message is safe to log."""


def job_label(job: Any) -> str:
    value = job.get("id", job.get("requestId")) if isinstance(job, dict) else None
    if isinstance(value, str) and 0 < len(value) <= 128 and value.isascii():
        if all(char.isalnum() or char in "-_" for char in value):
            return value
    return "unknown"


def log_failure(stage: str, error: Exception, job_id: str = "startup") -> None:
    # Traceback locations without exception text: network/model exceptions may
    # contain signed URLs, tokens, or input values. Never log image payloads.
    frames = traceback.extract_tb(error.__traceback__)
    locations = "; ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in frames)
    logger.error("job=%s stage=%s exception=%s locations=%s", job_id, stage, type(error).__name__, locations)


def check_api_key(job: dict[str, Any]) -> None:
    expected = os.environ.get("LOTUS_API_KEY")
    if not expected:
        raise JobError("WORKER_MISCONFIGURED", "Worker authentication is not configured.", 503)
    supplied = None
    http = job.get("http")
    if isinstance(http, dict) and isinstance(http.get("headers"), dict):
        supplied = next(
            (value for key, value in http["headers"].items() if str(key).lower() == "lotus-api-key"),
            None,
        )
    if supplied is None and isinstance(job.get("input"), dict):
        supplied = job["input"].get("api_key")
    if not isinstance(supplied, str) or not hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        raise JobError("UNAUTHORIZED", "Unauthorized", 401)


def load_pipeline() -> Any:
    """Called once by startup in the process that will actually serve jobs."""
    global _pipeline, _task_emb
    if _pipeline is not None:
        return _pipeline
    if not torch.cuda.is_available():
        raise StartupError("A CUDA GPU is required for this RunPod worker.")
    precision = os.environ.get("LOTUS_PRECISION", "fp32")
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise StartupError("LOTUS_PRECISION must be fp32, fp16, or bf16.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise StartupError("The assigned GPU does not support bf16.")
    tf32 = os.environ.get("LOTUS_ALLOW_TF32", "0")
    if tf32 not in {"0", "1"}:
        raise StartupError("LOTUS_ALLOW_TF32 must be 0 or 1.")
    torch.backends.cuda.matmul.allow_tf32 = tf32 == "1"
    from pipeline import LotusDPipeline

    started = time.perf_counter()
    pipe = LotusDPipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtypes[precision],
        cache_dir=os.environ.get("HF_HUB_CACHE"), use_safetensors=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        task = torch.tensor([[1, 0]], device=pipe.device, dtype=pipe.dtype)
        task_emb = torch.cat([torch.sin(task), torch.cos(task)], dim=-1)
    _task_emb = task_emb
    _pipeline = pipe
    # Diffusers uses native SDPA with compatible PyTorch 2 attention processors.
    # flash_attn is not xformers; neither is needed for this default path.
    processors = getattr(pipe.unet, "attn_processors", {})
    backends = sorted({type(processor).__name__ for processor in processors.values()})
    logger.info("stage=model_load seconds=%.3f precision=%s attention=%s", time.perf_counter() - started, precision, backends)
    return pipe


def predict(image: Image.Image, options: Options) -> Any:
    pipe = load_pipeline()
    # from_numpy shares the uint8 buffer; convert/normalize after transfer,
    # avoiding the old NumPy float32 copy plus torch.tensor copy on the CPU.
    array = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        tensor = tensor.to(device=pipe.device, dtype=pipe.dtype).div_(127.5).sub_(1.0)
        return pipe(
            rgb_in=tensor, prompt="", num_inference_steps=1,
            generator=None, output_type="np", timesteps=[options.timestep],
            task_emb=_task_emb, processing_res=options.processing_res,
            match_input_res=not options.output_processing_res,
            resample_method=options.resample_method,
        ).images[0]


def encode_prediction(pred: Any) -> tuple[bytes, float, float]:
    array = np.asarray(pred)
    if (
        array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 1
        or max(array.shape[:2]) > 4096 or not np.isfinite(array).all()
    ):
        raise JobError("INVALID_PREDICTION", "The model returned an invalid depth map.", 500)
    depth = array.mean(axis=-1, dtype=np.float32)
    minimum, maximum = float(depth.min()), float(depth.max())
    if not np.isfinite([minimum, maximum]).all():
        raise JobError("INVALID_PREDICTION", "The model returned an invalid depth map.", 500)
    span = maximum - minimum
    if span <= 1e-8:
        pixels = np.zeros(depth.shape, dtype=np.uint8)
    else:
        pixels = ((depth - minimum) / span * 255).clip(0, 255).astype(np.uint8)
    with io.BytesIO() as buffer:
        Image.fromarray(pixels).save(buffer, format="PNG")
        return buffer.getvalue(), minimum, maximum


def save_output(data: bytes) -> str:
    # Encode once, then atomically publish the same PNG bytes used by base64.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / f"{uuid.uuid4().hex}.png"
    temporary = target.with_suffix(".tmp")
    try:
        with temporary.open("xb") as output:
            output.write(data)
        temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return str(target)


def run_inference(job: Any) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise JobError("INVALID_INPUT", "The job must be an object.")
    check_api_key(job)
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        raise JobError("INVALID_INPUT", "input must be an object.")
    options = validate_input(job_input, MODEL_ID)
    started = time.perf_counter()
    with load_image_from_job(job_input) as image:
        if options.processing_res and int(min(image.size) * options.processing_res / max(image.size)) < 8:
            raise JobError("INVALID_INPUT", "processing_res is too small for this image's aspect ratio.")
        decoded = time.perf_counter()
        with _inference_lock:
            acquired = time.perf_counter()
            prediction = predict(image, options)
        inferred = time.perf_counter()
    png, minimum, maximum = encode_prediction(prediction)
    try:
        path = save_output(png)
    except OSError as error:
        log_failure("output_storage", error, job_label(job))
        raise JobError("OUTPUT_FAILED", "Could not save the depth map.", 503, True) from None
    result = {
        "image_base64": base64.b64encode(png).decode("ascii"),
        "min": minimum, "max": maximum, "file_path": path,
    }
    logger.info(
        "job=%s stage=job_complete download_decode_s=%.3f queue_s=%.3f inference_s=%.3f serialize_s=%.3f total_s=%.3f",
        job_label(job), decoded - started, acquired - decoded, inferred - acquired,
        time.perf_counter() - inferred, time.perf_counter() - started,
    )
    return result


def handler(job: Any) -> dict[str, Any]:
    try:
        return run_inference(job)
    except JobError as error:
        logger.warning("job=%s stage=job_failed code=%s retryable=%s", job_label(job), error.code, error.retryable)
        return error.response()
    except torch.cuda.OutOfMemoryError as error:
        log_failure("gpu_oom", error, job_label(job))
        # The failed call's stack/tensors can retain allocations until this
        # handler returns. Recycle the worker rather than retrying in that state.
        return {
            **JobError("GPU_OUT_OF_MEMORY", "GPU memory exhausted; try a lower processing_res.", 503, True).response(),
            "refresh_worker": True,
        }
    except Exception as error:
        log_failure("inference", error, job_label(job))
        result = JobError("INTERNAL_ERROR", "Depth-map inference failed.", 500).response()
        # CUDA runtime errors can poison the context. Do not reuse that worker.
        if isinstance(error, RuntimeError) and "cuda" in str(error).lower():
            result["refresh_worker"] = True
        return result


def prune_outputs() -> None:
    """Best-effort background retention; never delays accepting the first job."""
    cutoff = time.time() - 14 * 86400
    try:
        for path in OUTPUT_DIR.iterdir():
            try:
                if path.suffix in {".png", ".tmp"} and path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                logger.warning("stage=retention code=DELETE_FAILED")
    except OSError:
        logger.warning("stage=retention code=SCAN_FAILED")


def initialize_worker(warmup: bool = False) -> None:
    if not os.environ.get("LOTUS_API_KEY"):
        raise StartupError("LOTUS_API_KEY must be configured in the RunPod endpoint.")
    if not all(features.check(codec) for codec in ("jpg", "zlib", "webp")):
        raise StartupError("The image must include JPEG, PNG, and WebP codecs.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Check storage before loading the model, without touching previous outputs.
    probe = OUTPUT_DIR / f".probe-{uuid.uuid4().hex}"
    probe.touch(exist_ok=False)
    probe.unlink()
    load_pipeline()
    if warmup:
        with Image.new("RGB", (64, 64), "white") as image:
            predict(image, Options())
        torch.cuda.synchronize()
        logger.info("stage=warmup_complete")
    threading.Thread(target=prune_outputs, name="output-retention", daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true", help="Warm inference in the serving process; do not exit.")
    args, _ = parser.parse_known_args()
    warmup = os.environ.get("LOTUS_WARMUP", "0")
    if warmup not in {"0", "1"}:
        raise SystemExit("LOTUS_WARMUP must be 0 or 1.")
    try:
        initialize_worker(warmup=args.warmup or warmup == "1")
    except StartupError as error:
        logger.error("stage=startup code=CONFIGURATION_ERROR detail=%s", error)
        raise SystemExit(str(error)) from None
    except Exception as error:
        log_failure("startup", error)
        raise SystemExit("Worker initialization failed; check RunPod configuration and startup stage logs.") from None
    # The model remains in THIS process for all jobs. No refresh after success.
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
