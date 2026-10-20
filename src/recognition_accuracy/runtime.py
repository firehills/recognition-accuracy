import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: list[int | None]
    type: str


class LiteRTSession:
    def __init__(self, path: Path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError(
                "TFLite models require LiteRT; install the tflite extra with "
                "`python -m pip install -e '.[tflite]'`."
            ) from error

        self._interpreter: Any = Interpreter(model_path=str(path))
        self._interpreter.allocate_tensors()
        inputs = self._interpreter.get_input_details()
        if len(inputs) != 1:
            raise ValueError("Expected a model with exactly one input")
        if inputs[0]["dtype"] != np.float32:
            raise ValueError("Expected a float32 model input")
        self._input = inputs[0]
        self._inputs = [
            TensorInfo(
                self._input["name"],
                [
                    None if dimension < 0 else int(dimension)
                    for dimension in self._input["shape_signature"]
                ],
                "tensor(float)",
            )
        ]
        self._outputs = self._interpreter.get_output_details()
        self._output_indices = {output["name"]: output["index"] for output in self._outputs}

    def get_inputs(self) -> list[TensorInfo]:
        return self._inputs

    def get_outputs(self) -> list[TensorInfo]:
        return [
            TensorInfo(
                output["name"],
                [int(dimension) for dimension in output["shape"]],
                "tensor(float)",
            )
            for output in self._outputs
        ]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        if set(input_feed) != {self._input["name"]}:
            raise ValueError(f"Expected input {self._input['name']!r}")
        unknown_outputs = set(output_names) - self._output_indices.keys()
        if unknown_outputs:
            raise ValueError(f"Unknown output(s): {sorted(unknown_outputs)}")
        self._interpreter.set_tensor(self._input["index"], input_feed[self._input["name"]])
        self._interpreter.invoke()
        return [
            self._interpreter.get_tensor(self._output_indices[name]) for name in output_names
        ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session(path: Path, device: str) -> ort.InferenceSession | LiteRTSession:
    if device not in ("cpu", "gpu"):
        raise ValueError("Device must be cpu or gpu")
    if path.suffix.lower() == ".tflite":
        if device == "gpu":
            raise ValueError("LiteRT TFLite inference currently supports --device cpu only")
        return LiteRTSession(path)
    if device == "gpu":
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu instead of "
                "onnxruntime and compatible CUDA/cuDNN libraries, or select --device cpu."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    result = ort.InferenceSession(str(path), providers=providers)
    result.disable_fallback()
    if device == "gpu" and result.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("ONNX Runtime failed to activate CUDAExecutionProvider")
    if len(result.get_inputs()) != 1:
        raise ValueError("Expected a model with exactly one input")
    if result.get_inputs()[0].type != "tensor(float)":
        raise ValueError("Expected a float32 model input")
    return result


def require_shape(actual: list, expected: tuple[int, ...]) -> None:
    if len(actual) != len(expected) or any(
        isinstance(a, int) and a >= 0 and a != e for a, e in zip(actual, expected)
    ):
        raise ValueError(f"Expected model input {expected}, got {actual}")


def unit_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding or template is non-finite or has zero norm")
    return value / norm
