import base64
import io
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

import handler as worker
from image_input import Options
from tests.test_image_input import encoded_image


@pytest.fixture(autouse=True)
def isolated_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("LOTUS_API_KEY", "test-secret")
    monkeypatch.setenv("LOTUS_WARMUP", "0")
    monkeypatch.setattr(worker, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(worker, "_pipeline", None)
    monkeypatch.setattr(worker, "_task_emb", None)


def job(**options):
    return {"input": {"api_key": "test-secret", "image_base64": encoded_image(), **options}}


@pytest.mark.parametrize("payload", [None, [], "invalid", 42])
def test_bad_envelope(payload):
    assert worker.handler(payload)["code"] == "INVALID_INPUT"


@pytest.mark.parametrize("key", [None, [], 42, "wrong"])
def test_bad_api_key(key, monkeypatch):
    predict = MagicMock()
    monkeypatch.setattr(worker, "predict", predict)
    payload = job(api_key=key)
    assert worker.handler(payload)["code"] == "UNAUTHORIZED"
    predict.assert_not_called()


def test_missing_api_key_configuration(monkeypatch):
    monkeypatch.delenv("LOTUS_API_KEY")
    assert worker.handler(job())["code"] == "WORKER_MISCONFIGURED"
    with pytest.raises(RuntimeError, match="LOTUS_API_KEY"):
        worker.initialize_worker()


def test_mixed_case_header():
    worker.check_api_key({"http": {"headers": {"LoTuS-ApI-KeY": "test-secret"}}})


def test_unicode_api_key(monkeypatch):
    monkeypatch.setenv("LOTUS_API_KEY", "clé")
    worker.check_api_key({"input": {"api_key": "clé"}})


def test_success_preserves_contract_and_bytes(monkeypatch):
    prediction = np.arange(16 * 24 * 3, dtype=np.float32).reshape(24, 16, 3)
    predict = MagicMock(return_value=prediction)
    monkeypatch.setattr(worker, "predict", predict)
    result = worker.handler(job())
    assert set(result) == {"image_base64", "min", "max", "file_path"}
    raw = base64.b64decode(result["image_base64"])
    from pathlib import Path

    assert Path(result["file_path"]).read_bytes() == raw
    with Image.open(io.BytesIO(raw)) as image:
        assert image.mode == "L"
        assert image.size == (16, 24)
    assert result["min"] < result["max"]
    assert predict.call_args.args[1] == Options()


def test_too_small_processing_resolution_skips_inference(monkeypatch):
    predict = MagicMock()
    monkeypatch.setattr(worker, "predict", predict)
    assert worker.handler(job(processing_res=8))["code"] == "INVALID_INPUT"
    predict.assert_not_called()


def test_invalid_options_skip_inference(monkeypatch):
    predict = MagicMock()
    monkeypatch.setattr(worker, "predict", predict)
    assert worker.handler(job(processing_res=-1))["code"] == "INVALID_INPUT"
    predict.assert_not_called()


@pytest.mark.parametrize("prediction", [
    np.full((8, 8, 3), np.nan), np.full((8, 8, 3), np.inf),
    np.zeros((8, 8)), np.zeros((8, 8, 4)), np.zeros((0, 8, 3)),
])
def test_invalid_predictions(prediction):
    with pytest.raises(worker.JobError) as error:
        worker.encode_prediction(prediction)
    assert error.value.code == "INVALID_PREDICTION"


def test_constant_predictions():
    png, minimum, maximum = worker.encode_prediction(np.ones((8, 8, 3)))
    assert minimum == maximum == 1.0
    with Image.open(io.BytesIO(png)) as image:
        assert image.getextrema() == (0, 0)


@pytest.mark.parametrize("error,code,status,refresh", [
    (RuntimeError("secret-token"), "INTERNAL_ERROR", 500, False),
    (OSError("secret-token"), "INTERNAL_ERROR", 500, False),
    (torch.cuda.OutOfMemoryError("secret-token"), "GPU_OUT_OF_MEMORY", 503, True),
    (RuntimeError("CUDA error: secret-token"), "INTERNAL_ERROR", 500, True),
])
def test_safe_failures(error, code, status, refresh, monkeypatch, caplog):
    monkeypatch.setattr(worker, "predict", MagicMock(side_effect=error))
    result = worker.handler(job())
    assert result["code"] == code
    assert result["status"] == status
    assert bool(result.get("refresh_worker")) is refresh
    assert "secret-token" not in str(result)
    assert "secret-token" not in caplog.text


def test_storage_failure_classified_separately(monkeypatch):
    monkeypatch.setattr(worker, "predict", lambda *args: np.zeros((8, 8, 3)))
    monkeypatch.setattr(worker, "save_output", MagicMock(side_effect=OSError("secret-token")))
    result = worker.handler(job())
    assert result["code"] == "OUTPUT_FAILED"
    assert result["status"] == 503
    assert result["retryable"] is True
    assert "secret-token" not in str(result)


def test_atomic_output_removes_temporary_file(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("disk error")))
    with pytest.raises(OSError):
        worker.save_output(b"png")
    assert not list(tmp_path.iterdir())


def test_pipeline_loaded_once_and_task_embedding_reused(monkeypatch):
    pipe = MagicMock()
    pipe.to.return_value = pipe
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.float32
    pipe.unet.attn_processors = {}
    pipeline_class = MagicMock()
    pipeline_class.from_pretrained.return_value = pipe
    monkeypatch.setitem(sys.modules, "pipeline", SimpleNamespace(LotusDPipeline=pipeline_class))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("LOTUS_PRECISION", "fp32")
    monkeypatch.setenv("LOTUS_ALLOW_TF32", "0")
    assert worker.load_pipeline() is pipe
    task = worker._task_emb
    assert worker.load_pipeline() is pipe
    assert worker._task_emb is task
    pipeline_class.from_pretrained.assert_called_once()
    assert pipeline_class.from_pretrained.call_args.kwargs["revision"] == worker.MODEL_REVISION
    assert pipeline_class.from_pretrained.call_args.kwargs["torch_dtype"] is torch.float32


def test_no_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        worker.load_pipeline()


def test_predict_normalization_and_options(monkeypatch):
    pipe = MagicMock()
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.float32
    pipe.return_value.images = [np.zeros((8, 8, 3), dtype=np.float32)]
    monkeypatch.setattr(worker, "load_pipeline", lambda: pipe)
    with Image.new("RGB", (8, 8), "white") as image:
        worker.predict(image, Options(processing_res=768, timestep=900, output_processing_res=True))
    arguments = pipe.call_args.kwargs
    assert arguments["rgb_in"].shape == (1, 3, 8, 8)
    assert torch.all(arguments["rgb_in"] == 1.0)
    assert arguments["processing_res"] == 768
    assert arguments["timesteps"] == [900]
    assert arguments["match_input_res"] is False
    assert arguments["resample_method"] == "bicubic"


def test_initialize_warms_in_same_process(monkeypatch):
    load = MagicMock()
    predict = MagicMock()
    monkeypatch.setattr(worker, "load_pipeline", load)
    monkeypatch.setattr(worker, "predict", predict)
    monkeypatch.setattr(torch.cuda, "synchronize", MagicMock())
    monkeypatch.setattr(worker.threading, "Thread", MagicMock())
    worker.initialize_worker(warmup=True)
    load.assert_called_once()
    predict.assert_called_once()


def test_startup_rejects_missing_codec(monkeypatch):
    monkeypatch.setattr(worker.features, "check", lambda _: False)
    with pytest.raises(RuntimeError, match="codecs"):
        worker.initialize_worker()


def test_retention_preserves_recent_and_unrelated_files(tmp_path):
    import os

    old = tmp_path / "old.png"
    old.write_bytes(b"old")
    stamp = time.time() - 15 * 86400
    os.utime(old, (stamp, stamp))
    recent = tmp_path / "recent.png"
    recent.write_bytes(b"recent")
    unrelated = tmp_path / "old.txt"
    unrelated.write_bytes(b"keep")
    os.utime(unrelated, (stamp, stamp))
    worker.prune_outputs()
    assert not old.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_main_initializes_once_before_registering_handler(monkeypatch):
    order = []
    monkeypatch.setattr(sys, "argv", ["handler.py"])
    monkeypatch.setenv("LOTUS_WARMUP", "0")
    monkeypatch.setattr(worker, "initialize_worker", lambda **kw: order.append("initialize"))
    monkeypatch.setattr(worker.runpod.serverless, "start", lambda config: order.append(config["handler"]))
    worker.main()
    assert order == ["initialize", worker.handler]


def test_startup_failure_does_not_register_handler(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["handler.py"])
    monkeypatch.setattr(worker, "initialize_worker", MagicMock(side_effect=RuntimeError("secret-token")))
    start = MagicMock()
    monkeypatch.setattr(worker.runpod.serverless, "start", start)
    with pytest.raises(SystemExit, match="initialization failed") as error:
        worker.main()
    assert "secret-token" not in str(error.value)
    start.assert_not_called()
