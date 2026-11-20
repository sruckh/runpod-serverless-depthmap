import base64
import io
import socket
from unittest.mock import MagicMock

import pytest
import urllib3
from PIL import Image

import image_input as images
from image_input import JobError, Options, load_image_from_job, validate_input


def encoded_image(fmt="PNG", mode="RGB", color: str | int | tuple[int, ...] = "red", size=(16, 24), **save_options):
    with Image.new(mode, size, color) as image, io.BytesIO() as buf:
        image.save(buf, format=fmt, **save_options)
        return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("source", ["base64", "url", "data"])
def test_three_formats_from_each_source(fmt, source, monkeypatch):
    encoded = encoded_image(fmt)
    if source == "url":
        monkeypatch.setattr(images, "download_image", lambda _: base64.b64decode(encoded))
        payload = {"image_url": "https://example.com/file"}
    elif source == "data":
        mime = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}[fmt]
        payload = {"image_base64": f"data:image/{mime};base64,{encoded}"}
    else:
        payload = {"image_base64": encoded}
    assert validate_input(payload, "model") == Options()
    with load_image_from_job(payload) as result:
        assert result.mode == "RGB"
        assert result.size == (16, 24)


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_transparency_composited_on_white(fmt):
    payload = {"image_base64": encoded_image(fmt, "RGBA", (0, 0, 0, 0))}
    with load_image_from_job(payload) as result:
        assert result.getpixel((0, 0)) == (255, 255, 255)


def test_grayscale():
    with load_image_from_job({"image_base64": encoded_image(mode="L", color=128)}) as result:
        assert result.getpixel((0, 0)) == (128, 128, 128)


def test_palette_transparency():
    with Image.new("P", (16, 16), 0) as image, io.BytesIO() as buf:
        image.save(buf, format="PNG", transparency=0)
        value = base64.b64encode(buf.getvalue()).decode()
    with load_image_from_job({"image_base64": value}) as result:
        assert result.getpixel((0, 0)) == (255, 255, 255)


def test_exif_rotation():
    exif = Image.Exif()
    exif[274] = 6
    value = encoded_image("JPEG", size=(16, 24), exif=exif)
    with load_image_from_job({"image_base64": value}) as result:
        assert result.size == (24, 16)


@pytest.mark.parametrize("fmt", ["WEBP", "PNG"])
def test_animation_rejected(fmt):
    with Image.new("RGB", (16, 16), "red") as first, Image.new("RGB", (16, 16), "blue") as second:
        with io.BytesIO() as buf:
            first.save(buf, format=fmt, save_all=True, append_images=[second], duration=100, loop=0)
            value = base64.b64encode(buf.getvalue()).decode()
    with pytest.raises(JobError, match="Animated"):
        load_image_from_job({"image_base64": value})


@pytest.mark.parametrize("value", ["!!!", "a", "é", "data:image/gif;base64,YQ==", "data:image/png,YQ=="])
def test_invalid_base64(value):
    with pytest.raises(JobError):
        load_image_from_job({"image_base64": value})


def test_mismatched_data_url():
    value = "data:image/webp;base64," + encoded_image("PNG")
    with pytest.raises(JobError, match="does not match"):
        load_image_from_job({"image_base64": value})


@pytest.mark.parametrize("data", [b"not an image", b"", b"\x89PNG\r\n\x1a\n"])
def test_corrupt_images(data):
    with pytest.raises(JobError):
        load_image_from_job({"image_base64": base64.b64encode(data).decode()})


def test_truncated_jpeg():
    value = base64.b64decode(encoded_image("JPEG"))[:-30]
    with pytest.raises(JobError):
        load_image_from_job({"image_base64": base64.b64encode(value).decode()})


def test_other_format_rejected():
    with pytest.raises(JobError):
        load_image_from_job({"image_base64": encoded_image("BMP")})


def test_dimensions_checked_before_rgb_conversion(monkeypatch):
    value = encoded_image(size=(4097, 8))
    convert = MagicMock(side_effect=AssertionError("must not convert oversized image"))
    monkeypatch.setattr(Image.Image, "convert", convert)
    with pytest.raises(JobError, match="dimensions"):
        load_image_from_job({"image_base64": value})
    convert.assert_not_called()


def test_base64_byte_cap(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 2)
    with pytest.raises(JobError) as error:
        images.decode_base64(base64.b64encode(b"abc").decode())
    assert error.value.status == 413


@pytest.mark.parametrize("payload", [None, [], {}, {"image_url": "x", "image_base64": "y"}, {"image_url": 1}, {"image_url": ""}])
def test_invalid_input(payload):
    with pytest.raises(JobError):
        validate_input(payload, "model")


@pytest.mark.parametrize("name,value", [
    ("processing_res", True), ("processing_res", -1), ("processing_res", 4097),
    ("processing_res", 7), ("processing_res", "768"), ("timestep", 1000),
    ("timestep", 1.5), ("timestep", False), ("output_processing_res", "false"),
    ("resample_method", "lanczos"), ("resample_method", []), ("model_id", "other"),
])
def test_invalid_options(name, value):
    with pytest.raises(JobError):
        validate_input({"image_url": "https://example.com/x", name: value}, "model")


def public_dns(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a", "https://user:secret@example.com/a",
    "http://example.com:8080/a", "https://example.com/a\r\nHost:x", "https://example.com/a#fragment",
])
def test_disallowed_urls(url):
    with pytest.raises(JobError):
        images._public_destination(url)


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "100.64.0.1"])
def test_private_destinations(ip, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET6, 1, 6, "", (ip, 443))])
    with pytest.raises(JobError, match="public IP"):
        images._public_destination("https://example.com/x")


def mock_pool(monkeypatch, response):
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    pool = MagicMock()
    pool.urlopen.return_value = response
    constructor = MagicMock(return_value=pool)
    monkeypatch.setattr(urllib3, "HTTPSConnectionPool", constructor)
    return pool, constructor


def response_for(status=200, headers=None, chunks=None):
    response = MagicMock()
    response.status = status
    response.headers = headers or {}
    response.read1.side_effect = chunks or [b"data", b""]
    return response


def test_dns_pinned_and_tls_hostname_preserved(monkeypatch):
    response = response_for()
    pool, constructor = mock_pool(monkeypatch, response)
    assert images.download_image("https://example.com/img?q=1") == b"data"
    assert constructor.call_args.args == ("8.8.8.8", 443)
    assert constructor.call_args.kwargs["server_hostname"] == "example.com"
    assert constructor.call_args.kwargs["assert_hostname"] == "example.com"
    assert pool.urlopen.call_args.args == ("GET", "/img?q=1")
    assert pool.urlopen.call_args.kwargs["headers"]["Host"] == "example.com"
    assert pool.urlopen.call_args.kwargs["retries"] is False
    response.close.assert_called_once()
    pool.close.assert_called_once()


def test_redirect_to_private_host_rejected(monkeypatch):
    response = response_for(302, {"Location": "http://127.0.0.1/private"})
    mock_pool(monkeypatch, response)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kw: [
        (socket.AF_INET, 1, 6, "", ("127.0.0.1" if host == "127.0.0.1" else "8.8.8.8", port)),
    ])
    with pytest.raises(JobError, match="public IP"):
        images.download_image("https://example.com/img")
    response.close.assert_called_once()


@pytest.mark.parametrize("headers,chunks", [
    ({"Content-Length": "999"}, [b"x", b""]),
    ({}, [b"123", b"456", b""]),
])
def test_stream_byte_cap(monkeypatch, headers, chunks):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 5)
    response = response_for(headers=headers, chunks=chunks)
    pool, _ = mock_pool(monkeypatch, response)
    with pytest.raises(JobError) as error:
        images.download_image("https://example.com/x")
    assert error.value.status == 413
    response.close.assert_called_once()
    pool.close.assert_called_once()


@pytest.mark.parametrize("status,retryable", [(404, False), (429, True), (503, True)])
def test_http_error(status, retryable, monkeypatch):
    mock_pool(monkeypatch, response_for(status))
    with pytest.raises(JobError) as error:
        images.download_image("https://example.com/secret?token=private")
    assert error.value.retryable is retryable
    assert "private" not in str(error.value)


def test_timeout_sanitized(monkeypatch):
    pool, _ = mock_pool(monkeypatch, response_for())
    pool.urlopen.side_effect = urllib3.exceptions.ReadTimeoutError(pool, "/?token=private", "timeout")
    with pytest.raises(JobError) as error:
        images.download_image("https://example.com/x")
    assert error.value.code == "DOWNLOAD_TIMEOUT"
    assert "private" not in str(error.value)


def test_deadline_enforced_during_stream(monkeypatch):
    mock_pool(monkeypatch, response_for())
    clock = iter([0, 0, images.DOWNLOAD_SECONDS + 1])
    monkeypatch.setattr(images.time, "monotonic", lambda: next(clock))
    with pytest.raises(JobError) as error:
        images.download_image("https://example.com/x")
    assert error.value.code == "DOWNLOAD_TIMEOUT"


def test_redirect_limit(monkeypatch):
    mock_pool(monkeypatch, response_for(302, {"Location": "/again"}))
    with pytest.raises(JobError, match="redirect limit"):
        images.download_image("https://example.com/x")
