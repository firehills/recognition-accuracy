from pathlib import Path

import pytest

from recognition_accuracy import runtime


def test_gpu_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr(runtime.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(runtime.ort, "preload_dlls", lambda: None)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        runtime.session(Path("unused.onnx"), "gpu")


def test_gpu_provider_initialization_fallback_is_rejected(monkeypatch):
    class FakeSession:
        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        runtime.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(runtime.ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(runtime.ort, "InferenceSession", lambda *a, **kw: FakeSession())
    with pytest.raises(RuntimeError, match="failed to activate"):
        runtime.session(Path("unused.onnx"), "gpu")


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="cpu or gpu"):
        runtime.session(Path("unused.onnx"), "other")
