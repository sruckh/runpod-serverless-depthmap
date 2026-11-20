# RunPod Serverless Depth Map

A **RunPod Serverless GPU worker** for Lotus-2 depth regression using
`jingheya/lotus-depth-d-v2-0-disparity`. It accepts a JPEG/JPG, PNG, or WebP
reference image and returns an 8-bit grayscale depth PNG.

**This checkout is source only. The VPS/source host does not run the worker and
needs no Python packages, CUDA, virtualenv, or model weights.** Dependencies are
installed during the Docker build in CI. The resulting image runs on RunPod.

## Worker lifecycle

1. GitHub Actions builds the CUDA 12.8 / Python 3.12 image. PyTorch, the inference
   dependencies, application code, and a pinned Lotus source revision are baked
   into immutable `/opt/venv`, `/opt/lotus`, and `/app` paths.
2. RunPod starts `bash /app/bootstrap.sh`. It selects the mounted workspace and
   configures the Hugging Face cache. **There are no runtime installs or clones.**
3. The serving Python process checks authentication configuration, image codecs,
   output storage, and GPU availability, then loads the model **once**. A missing
   GPU or a failed model load fails startup instead of silently running on CPU.
4. Optional inference warmup runs in that **same process**. The process then calls
   `runpod.serverless.start({"handler": handler})` and reuses its pipeline across
   jobs. Successful jobs do not refresh/restart the worker.
5. Outputs older than 14 days are pruned once per worker startup by a background
   thread. Cleanup does not block accepting the first job. Continuously running
   workers do not periodically prune; use separate scheduled storage management
   if strict continuous retention is required.

On a new cache, weights still download on RunPod before the worker is ready.
A shared network volume avoids repeated downloads, but each worker must still
load its own GPU model. Configure the endpoint's initialization timeout to allow
for a first download/model load. Keeping active workers can reduce cold starts,
but incurs RunPod costs; this repository does not change endpoint settings.

## Configure the RunPod endpoint

Deploy the CI-built image as a **GPU Serverless endpoint**. Use the image's default
command; remove any old command override pointing at `/workspace/Lotus/src`.
Configure these variables in the RunPod template/endpoint, not on the VPS:

| Variable | Default / purpose |
| --- | --- |
| `LOTUS_API_KEY` | **Required.** Application key supplied in each job's `input.api_key`. Missing configuration fails startup. |
| `WORKSPACE` | `/runpod-volume/Lotus` when `/runpod-volume` exists; otherwise `/workspace/Lotus`. An explicit non-default path is respected. |
| `HF_HOME` | `<WORKSPACE>/cache`. |
| `HF_HUB_CACHE` | Existing `HUGGINGFACE_HUB_CACHE`, otherwise `HF_HOME`. Used by both prefetch and inference. |
| `MODEL_ID` | `jingheya/lotus-depth-d-v2-0-disparity`. One model per worker. |
| `MODEL_REVISION` | Default model pinned to `10b3e82cbb999274f69982d518d5abfdbe5f7fe8`. An alternate `MODEL_ID` defaults to `main`; set a commit explicitly for reproducibility. |
| `HF_TOKEN` | Optional Hugging Face secret for a private/gated alternate model. Never bake tokens into the image. |
| `HF_HUB_OFFLINE` | Optional `1` after prefetching a complete snapshot; startup then fails rather than downloading missing files. |
| `LOTUS_WARMUP` | `0`; set `1` to run a 64×64 inference before accepting jobs. This warms basic kernels, not every production image shape. |
| `LOTUS_PRECISION` | `fp32`. Optional `fp16` / `bf16` trade precision for speed/memory; bf16 requires a compatible GPU. |
| `LOTUS_ALLOW_TF32` | `0`; set `1` to allow TF32 matrix multiplication on compatible hardware. It changes numerical precision. |

The default remains **fp32, native resolution (`processing_res=0`), bicubic**.
No quality-changing optimization is silently enabled. Native 4096×4096 inference
can exceed available VRAM even though it passes the input size limit; select a
GPU with enough memory or supply a lower `processing_res` such as `768`.

Diffusers uses native PyTorch SDPA on compatible attention processors. The worker
logs the actual processor classes. FlashAttention is not xFormers; neither is
installed as an unrelated startup dependency. `torch.compile` is deliberately not
enabled because compile/recompile costs can worsen short-lived worker latency.

## Storage and migration

Only runtime data belongs on the RunPod volume:

```text
/runpod-volume/Lotus/       # or /workspace/Lotus without a network volume
  cache/                   # shared HF snapshot/blob cache
  output_images/           # UUID-named PNG outputs
```

Old volume directories `venv/`, `src/`, `upstream/`, and `models/` are **not used**
by the new image and are **not deleted automatically**. This prevents stale
volume-hosted code/dependencies from overriding a new deployment. Existing
HF-cache weights and output images retain their previous paths. Old manually
prefetched `models/lotus-depth-d` snapshots are not automatically imported into
that cache.

`file_path` is a path **inside the RunPod worker/volume**, not a public download
URL or a path on the VPS. Without a network volume, outputs/cache are ephemeral.
Concurrent workers share model-cache file locks but have independent pipelines
and UUID output names. Inference is serialized within each worker to avoid
mutable pipeline state races and concurrent GPU-memory spikes.

## Request contract

Use the standard RunPod `/run` or `/runsync` API, with RunPod Bearer authentication
as required by your endpoint. The application key is separate from the RunPod
API key. Send it in **`input.api_key`**: arbitrary HTTP headers are not guaranteed
to be forwarded into a standard queued RunPod job. A case-insensitive
`lotus-api-key` header is accepted only when an integration actually supplies
`job.http.headers`.

Supply **exactly one** of `image_url` or `image_base64`:

```json
{
  "input": {
    "api_key": "YOUR_LOTUS_API_KEY",
    "image_url": "https://example.com/reference.webp",
    "processing_res": 0,
    "resample_method": "bicubic",
    "timestep": 999,
    "output_processing_res": false
  }
}
```

For base64, use a raw base64 string or a matching data URL such as
`data:image/png;base64,...`. Do not include whitespace in the base64 data. RunPod
has its own request/response size limits, which may be lower than the worker's
20 MiB decoded-file limit. Prefer URLs for larger inputs.

| Input | Validation |
| --- | --- |
| Image format | Detected from bytes: JPEG, PNG, WebP only, regardless of extension or server Content-Type. Data-URL media types must match the bytes. |
| Dimensions | Each axis 8–4096 pixels; at most 4096×4096 pixels. Checked before RGB conversion. Very small images are rejected because of VAE downsampling. |
| File bytes | At most 20 MiB, enforced before base64 decoding allocation and while streaming downloads. |
| Orientation / color | EXIF orientation applied; grayscale/CMYK converted to RGB; transparency composited over white. |
| Animation | Animated WebP/APNG rejected instead of silently taking the first frame. |
| `processing_res` | Integer `0` (native) or `8`–`4096`; the resized shorter edge must remain at least 8 pixels. Booleans and strings rejected. |
| `resample_method` | `nearest`, `nearest-exact`, `bilinear`, or `bicubic`. |
| `timestep` | Integer `0`–`999`. |
| `output_processing_res` | Boolean; `false` restores the original input dimensions. |
| `model_id` | Optional compatibility field; must equal this worker's configured `MODEL_ID`. Different per-job models are rejected rather than silently ignored. |

Unknown fields remain ignored for compatibility. The former example's `disparity`
field was not implemented; the default model itself is the disparity variant.

### URL download policy

- HTTP/HTTPS only, using standard ports 80/443. No URL credentials or fragments.
- Every destination, including each redirect, must resolve only to public IPs.
  The connection is pinned to the validated IP; HTTPS retains the original SNI,
  certificate hostname verification, and Host header. Private, loopback,
  link-local/metadata, shared-address-space, and mixed public/private destinations
  are rejected. Proxy environment variables and ambient credentials are not used.
- At most three redirects; byte cap checked even without Content-Length.
- Five-second connect/read timeouts plus a 30-second streaming budget. OS DNS
  resolution has its own system timeout, and a blocking read can overrun the
  budget by up to its read timeout. This is not a strict end-to-end 30-second SLA.
- No automatic retries or compressed HTTP response bodies. Transient failures
  are marked retryable; the caller can decide whether to submit a new job.

This deliberately rejects internal object-store URLs/private buckets exposed
only through a private network. Use a public signed HTTPS URL or base64 instead.

## Responses and errors

Successful handler return (RunPod wraps it in the job's `output`):

```json
{
  "image_base64": "...base64-encoded grayscale PNG...",
  "min": 0.0123,
  "max": 1.2345,
  "file_path": "/runpod-volume/Lotus/output_images/<uuid>.png"
}
```

The PNG is encoded once; the exact same bytes are saved atomically and returned.
Non-finite or malformed model outputs are rejected. Constant depth maps produce
a black image without a divide-by-zero. `min` and `max` are raw predicted-depth
statistics before per-image normalization, not calibrated metric distance.

Example failed handler return:

```json
{
  "error": "The reference image is corrupt or incomplete.",
  "code": "INVALID_IMAGE",
  "status": 400,
  "retryable": false
}
```

**`status` is application metadata, not an HTTP status override.** The pinned
RunPod SDK (1.6.1) promotes a truthy handler `error` to the job's error field and
retains the remaining metadata under `output`. Inspect the RunPod job status and
error as well as its output; do not assume a completed HTTP request means
successful inference.

Codes include `UNAUTHORIZED`, `WORKER_MISCONFIGURED`, `INVALID_INPUT`,
`INVALID_MODEL`, `INVALID_URL`, `INVALID_IMAGE`, `UNSUPPORTED_IMAGE`,
`IMAGE_TOO_LARGE`, `DOWNLOAD_FAILED`, `DOWNLOAD_TIMEOUT`, `INVALID_PREDICTION`,
`GPU_OUT_OF_MEMORY`, `OUTPUT_FAILED`, and `INTERNAL_ERROR`.

CUDA OOM returns a safe error and asks RunPod to refresh the worker using the
SDK's `refresh_worker` return flag. CUDA runtime failures also request refresh;
successful jobs do not. There is no blind inference retry. Retrying an OOM with
the same resolution/GPU can fail again—reduce `processing_res` first.

Logs contain stage timings, error codes, exception types, and traceback locations,
not API keys, base64 payloads, raw exception messages, or signed image URLs.

## Optional prefetch — RunPod only

From an administrative shell **inside this image on RunPod**, with the intended
volume mounted and the same model/cache variables as the serving endpoint:

```bash
bash /app/prefetch.sh
```

It downloads the complete pinned inference snapshot, including tokenizer files,
into the **same HF cache** the handler reads. It does not create a separate local
model directory. This is optional; the worker will otherwise fetch on first
startup. Do not run this command or install its prerequisites on the source VPS.

## CI and deployment verification

`.github/workflows/docker-publish.yml` validates pull requests and builds/publishes
on pushes to `main` or manual dispatch. Docker Hub secrets remain
`DOCKER_USERNAME` and `DOCKER_PASSWORD`.

The Docker `quality` target performs these checks **inside the CI container**:

- actual application/Lotus imports and JPEG/PNG/WebP codec availability;
- Ruff lint, mypy typechecking, ShellCheck, and shell syntax;
- CPU-only pytest tests with mocked model/network calls, covering formats,
  EXIF/transparency, limits, SSRF/redirects, authentication, errors, preprocessing,
  model reuse, persistence, retention, and serving-process initialization.

The published `final` target excludes development tools and test fixtures.
The source VPS needs no installation or local worker/container execution.

Before rolling out to production, validate the built image on a **RunPod GPU**:

1. Test startup with a fresh model cache and with a populated network volume.
2. Send JPEG, PNG, and WebP jobs, then malformed, oversized, and unauthorized jobs.
3. Compare first-job and warm-job latency at representative native resolutions
   and at `processing_res=768`; record peak VRAM and stage timings.
4. Check depth-map quality before opting into reduced precision or TF32.
5. Confirm `FAILED` job behavior and recovery/worker refresh after GPU failures.

CPU tests cannot prove GPU compatibility, real model correctness, first-run
model-download timing, endpoint initialization behavior, or a measured speedup.
No deployment, paid GPU job, publish, push, or commit is performed merely by
editing this repository.
