import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from recognition_accuracy import runtime


def install_fake_litert(monkeypatch, interpreter):
    package = ModuleType("ai_edge_litert")
    package.__path__ = []
    module = ModuleType("ai_edge_litert.interpreter")
    module.Interpreter = interpreter
    monkeypatch.setitem(sys.modules, "ai_edge_litert", package)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", module)


def test_tflite_model_runs_with_litert(monkeypatch):
    class FakeInterpreter:
        def __init__(self, model_path):
            assert model_path == "model.tflite"
            self.input = np.zeros((1, 112, 112, 3), dtype=np.float32)
            self.output = np.array([[1, 2, 3]], dtype=np.float32)

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [
                {
                    "name": "images",
                    "shape_signature": np.array([1, 112, 112, 3]),
                    "dtype": np.float32,
                    "index": 5,
                }
            ]

        def get_output_details(self):
            return [{"name": "embedding", "shape": np.array([1, 3]), "index": 7}]

        def set_tensor(self, index, value):
            assert index == 5
            np.testing.assert_array_equal(value, self.input)

        def invoke(self):
            pass

        def get_tensor(self, index):
            assert index == 7
            return self.output

    install_fake_litert(monkeypatch, FakeInterpreter)
    model = runtime.session(Path("model.tflite"), "cpu")
    assert model.get_inputs()[0].shape == [1, 112, 112, 3]
    assert model.get_outputs()[0].name == "embedding"
    np.testing.assert_array_equal(
        model.run(["embedding"], {"images": np.zeros((1, 112, 112, 3), dtype=np.float32)})[0],
        [[1, 2, 3]],
    )


def test_tflite_dynamic_dimensions_are_allowed():
    runtime.require_shape([None, 112, 112, 3], (1, 112, 112, 3))
    runtime.require_shape([-1, 112, 112, 3], (1, 112, 112, 3))


def test_nchw_input_shape_is_accepted_and_transposed():
    base = np.zeros((1, 112, 112, 3), dtype=np.float32)
    transformed = runtime.prepare_input_tensor(base, [1, 3, 112, 112])
    assert transformed.shape == (1, 3, 112, 112)
    np.testing.assert_array_equal(transformed[:, 0], base[:, :, :, 0])
    runtime.require_shape([1, 3, 112, 112], (1, 112, 112, 3))


def test_tflite_gpu_is_rejected_explicitly():
    with pytest.raises(ValueError, match="cpu only"):
        runtime.session(Path("model.tflite"), "gpu")


def test_tflite_missing_litert_is_explicit(monkeypatch):
    package = ModuleType("ai_edge_litert")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "ai_edge_litert", package)
    monkeypatch.delitem(sys.modules, "ai_edge_litert.interpreter", raising=False)
    with pytest.raises(RuntimeError, match="install the tflite extra"):
        runtime.session(Path("model.tflite"), "cpu")


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
