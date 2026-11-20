"""Bounded, content-based reference-image loading. No model or CUDA imports."""

import base64
import binascii
import io
import ipaddress
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, urljoin, urlsplit

import urllib3
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 4096 * 4096
# Bound Pillow's early header/codec checks too, notably WebP's decoder setup.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
MAX_DIMENSION = 4096
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_URL_LENGTH = 8192
DOWNLOAD_SECONDS = 30.0
FORMATS = ("JPEG", "PNG", "WEBP")
DATA_URL_TYPES = {"image/jpeg": "JPEG", "image/jpg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
RESAMPLE_METHODS = {"nearest", "nearest-exact", "bilinear", "bicubic"}


class JobError(Exception):
    """A safe, public failure; never include secrets or upstream exception text."""

    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable

    def response(self) -> dict[str, Any]:
        # RunPod 1.6.1 promotes `error` to the job's failure field. `status` is
        # application metadata, NOT an HTTP response code for /run or /runsync.
        return {
            "error": str(self),
            "code": self.code,
            "status": self.status,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class Options:
    processing_res: int = 0
    timestep: int = 999
    output_processing_res: bool = False
    resample_method: str = "bicubic"


def validate_input(job_input: Any, model_id: str) -> Options:
    if not isinstance(job_input, dict):
        raise JobError("INVALID_INPUT", "input must be an object.")
    sources = [key for key in ("image_url", "image_base64") if key in job_input]
    if len(sources) != 1:
        raise JobError("INVALID_INPUT", "Provide exactly one of image_url or image_base64.")
    value = job_input[sources[0]]
    if not isinstance(value, str) or not value:
        raise JobError("INVALID_INPUT", "The image source must be a nonempty string.")
    if "model_id" in job_input and job_input["model_id"] != model_id:
        raise JobError("INVALID_MODEL", "model_id must match this worker's configured model.")
    values = {}
    for name, default, minimum, maximum in (
        ("processing_res", 0, 0, MAX_DIMENSION),
        ("timestep", 999, 0, 999),
    ):
        number = job_input.get(name, default)
        if type(number) is not int or not minimum <= number <= maximum:
            raise JobError("INVALID_INPUT", f"{name} must be an integer from {minimum} to {maximum}.")
        values[name] = number
    if 0 < values["processing_res"] < 8:
        raise JobError("INVALID_INPUT", "processing_res must be 0 or at least 8.")
    output_res = job_input.get("output_processing_res", False)
    if type(output_res) is not bool:
        raise JobError("INVALID_INPUT", "output_processing_res must be a boolean.")
    method = job_input.get("resample_method", "bicubic")
    if not isinstance(method, str) or method not in RESAMPLE_METHODS:
        raise JobError("INVALID_INPUT", "Unsupported resample_method.")
    return Options(
        processing_res=values["processing_res"], timestep=values["timestep"],
        output_processing_res=output_res, resample_method=method,
    )


def _public_destination(url: str) -> tuple[SplitResult, str, int, str]:
    """Resolve once; the connection uses this exact IP (including across redirects)."""
    if len(url) > MAX_URL_LENGTH or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise JobError("INVALID_URL", "Invalid image URL.")
    try:
        parts = urlsplit(url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if port != (443 if parts.scheme == "https" else 80):
            raise ValueError
        host = parts.hostname.encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        raise JobError("INVALID_URL", "Use an HTTP(S) image URL on its standard port.") from None
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise JobError("DOWNLOAD_FAILED", "Could not resolve the image host.", 502, True) from None
    ips = {str(entry[4][0]) for entry in addresses}
    if not ips:
        raise JobError("DOWNLOAD_FAILED", "Could not resolve the image host.", 502, True)
    for address in ips:
        ip = ipaddress.ip_address(address)
        # IPv4-mapped IPv6 addresses need an independent IPv4 policy check.
        mapped = getattr(ip, "ipv4_mapped", None)
        if not ip.is_global or (mapped is not None and not mapped.is_global):
            raise JobError("INVALID_URL", "Image URLs must resolve only to public IP addresses.")
    return parts, host, port, sorted(ips)[0]


def download_image(url: str) -> bytes:
    """Stream a bounded response with TLS hostname checking and pinned DNS.

    No proxy/environment credentials, cookies, automatic redirects, or retries.
    DNS resolution itself uses the OS resolver timeout; socket I/O also has a
    bounded timeout and the streaming loop enforces a wall-clock budget.
    """
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    for redirect in range(4):
        parts, host, port, address = _public_destination(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise JobError("DOWNLOAD_TIMEOUT", "Image download timed out.", 504, True)
        timeout = urllib3.Timeout(total=remaining, connect=min(5.0, remaining), read=min(5.0, remaining))
        pool: urllib3.HTTPConnectionPool
        if parts.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                address, port, server_hostname=host, assert_hostname=host,
                cert_reqs="CERT_REQUIRED", timeout=timeout, maxsize=1,
            )
        else:
            pool = urllib3.HTTPConnectionPool(address, port, timeout=timeout, maxsize=1)
        authority = f"[{host}]" if ":" in host else host
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        response = None
        try:
            response = pool.urlopen(
                "GET", path, headers={"Host": authority, "Accept-Encoding": "identity"},
                preload_content=False, redirect=False, retries=False,
            )
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if redirect == 3 or not location:
                    raise JobError("DOWNLOAD_FAILED", "Image URL exceeded the redirect limit.", 502)
                url = urljoin(url, location)
                continue
            if response.status != 200:
                retryable = response.status in {408, 429} or response.status >= 500
                raise JobError("DOWNLOAD_FAILED", "The image server did not return a usable image.", 502, retryable)
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise JobError("INVALID_IMAGE", "Compressed HTTP responses are not supported.")
            size = response.headers.get("Content-Length")
            if size is not None:
                try:
                    declared = int(size)
                except ValueError:
                    raise JobError("DOWNLOAD_FAILED", "Invalid image response length.", 502) from None
                if declared < 0:
                    raise JobError("DOWNLOAD_FAILED", "Invalid image response length.", 502)
                if declared > MAX_IMAGE_BYTES:
                    raise JobError("IMAGE_TOO_LARGE", "Image exceeds the 20 MiB byte limit.", 413)
            data = bytearray()
            # read1 returns available bytes rather than waiting for a full chunk,
            # so a trickling server cannot reset the overall budget indefinitely.
            while True:
                if time.monotonic() >= deadline:
                    raise JobError("DOWNLOAD_TIMEOUT", "Image download timed out.", 504, True)
                chunk = response.read1(64 * 1024, decode_content=False)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_IMAGE_BYTES:
                    raise JobError("IMAGE_TOO_LARGE", "Image exceeds the 20 MiB byte limit.", 413)
            if time.monotonic() >= deadline:
                raise JobError("DOWNLOAD_TIMEOUT", "Image download timed out.", 504, True)
            return bytes(data)
        except urllib3.exceptions.TimeoutError:
            raise JobError("DOWNLOAD_TIMEOUT", "Image download timed out.", 504, True) from None
        except (urllib3.exceptions.HTTPError, OSError, ValueError):
            raise JobError("DOWNLOAD_FAILED", "Could not download the reference image.", 502, True) from None
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise JobError("DOWNLOAD_FAILED", "Image URL exceeded the redirect limit.", 502)


def decode_base64(value: str) -> tuple[bytes, str | None]:
    expected_format = None
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        media_type = header[5:].removesuffix(";base64")
        if not separator or not header.endswith(";base64") or media_type not in DATA_URL_TYPES:
            raise JobError("INVALID_IMAGE", "Use a JPEG, PNG, or WebP base64 data URL.")
        expected_format = DATA_URL_TYPES[media_type]
    if len(value) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise JobError("IMAGE_TOO_LARGE", "Image exceeds the 20 MiB byte limit.", 413)
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise JobError("INVALID_IMAGE", "Invalid base64 image data.") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise JobError("IMAGE_TOO_LARGE", "Image exceeds the 20 MiB byte limit.", 413)
    return data, expected_format


def load_image_from_job(job_input: dict[str, Any]) -> Image.Image:
    expected_format = None
    if "image_url" in job_input:
        data = download_image(job_input["image_url"])
    else:
        data, expected_format = decode_base64(job_input["image_base64"])
    try:
        with Image.open(io.BytesIO(data), formats=FORMATS) as image:
            if expected_format and image.format != expected_format:
                raise JobError("INVALID_IMAGE", "Data URL media type does not match its image.")
            if min(image.size) < 8:
                raise JobError("INVALID_IMAGE", "Image dimensions must be at least 8 pixels on each axis.")
            if max(image.size) > MAX_DIMENSION or image.width * image.height > MAX_PIXELS:
                raise JobError("IMAGE_TOO_LARGE", "Image dimensions must not exceed 4096 on either axis.", 413)
            if getattr(image, "is_animated", False):
                raise JobError("UNSUPPORTED_IMAGE", "Animated reference images are not supported.", 415)
            # Verification catches structural corruption; loading the reopened
            # image also forces decoding (Pillow does not allow truncated data).
            image.verify()
        with Image.open(io.BytesIO(data), formats=FORMATS) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                return Image.alpha_composite(background, rgba).convert("RGB")
            return oriented.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise JobError("INVALID_IMAGE", "Use a valid JPEG, PNG, or WebP reference image.") from None
    except (OSError, SyntaxError, ValueError):
        raise JobError("INVALID_IMAGE", "The reference image is corrupt or incomplete.") from None
