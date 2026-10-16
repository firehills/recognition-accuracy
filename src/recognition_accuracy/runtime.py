import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session(path: Path, device: str) -> ort.InferenceSession:
    if device not in ("cpu", "gpu"):
        raise ValueError("Device must be cpu or gpu")
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
        isinstance(a, int) and a != e for a, e in zip(actual, expected)
    ):
        raise ValueError(f"Expected model input {expected}, got {actual}")


def unit_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding or template is non-finite or has zero norm")
    return value / norm
