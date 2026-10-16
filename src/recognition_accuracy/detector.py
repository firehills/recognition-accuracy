from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .runtime import require_shape, session


@dataclass
class Face:
    box: np.ndarray
    landmarks: np.ndarray
    score: float


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep = []
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2] + 1), axis=1)
    while order.size:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        intersection = np.prod(
            np.maximum(
                0,
                np.minimum(boxes[index, 2:], boxes[remaining, 2:])
                - np.maximum(boxes[index, :2], boxes[remaining, :2])
                + 1,
            ),
            axis=1,
        )
        union = areas[index] + areas[remaining] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[overlap <= threshold]
    return keep


def decode(outputs: list[np.ndarray], height: int, width: int, confidence: float) -> list[Face]:
    """Decode InsightFace RetinaFace net3 probability/bbox/landmark feature maps."""
    if len(outputs) != 9:
        raise ValueError("RetinaFace net3 requires nine NCHW feature-map outputs")
    maps = {}
    for output in outputs:
        if output.ndim != 4 or output.shape[0] != 1:
            raise ValueError(
                "Unsupported detector: expected NCHW RetinaFace feature maps, not SCRFD"
            )
        key = tuple(output.shape[1:])
        if key in maps:
            raise ValueError(f"Duplicate RetinaFace output shape: {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("Detector returned non-finite values")
        maps[key] = output
    faces = []
    for stride, sizes in ((32, (512, 256)), (16, (128, 64)), (8, (32, 16))):
        h, w = height // stride, width // stride
        try:
            cls = maps[(4, h, w)]
            boxes = maps[(8, h, w)]
            landmarks = maps[(20, h, w)]
        except KeyError as error:
            raise ValueError(
                f"Unsupported RetinaFace output shapes for stride {stride}; expected "
                f"(1,4,{h},{w}), (1,8,{h},{w}), (1,20,{h},{w})"
            ) from error
        scores = cls[0, 2:].transpose(1, 2, 0).reshape(-1)
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("RetinaFace class outputs must be probabilities, not logits")
        selected = np.flatnonzero(scores >= confidence)
        if not selected.size:
            continue
        # Anchor order is (row, column, scale); the base anchor has centre (7.5, 7.5).
        row = selected // (w * 2)
        col = (selected // 2) % w
        size = np.asarray(sizes, dtype=np.float64)[selected % 2]
        centre = np.column_stack((col * stride + 7.5, row * stride + 7.5))
        delta = boxes[0].transpose(1, 2, 0).reshape(-1, 4)[selected]
        predicted_centre = centre + delta[:, :2] * size[:, None]
        with np.errstate(over="raise", invalid="raise"):
            predicted_size = np.exp(delta[:, 2:].astype(np.float64)) * size[:, None]
        decoded_boxes = np.column_stack(
            (
                predicted_centre - (predicted_size - 1) / 2,
                predicted_centre + (predicted_size - 1) / 2,
            )
        )
        points = landmarks[0].transpose(1, 2, 0).reshape(-1, 5, 2)[selected]
        points = centre[:, None, :] + points * size[:, None, None]
        for index, box, point in zip(selected, decoded_boxes, points):
            faces.append(Face(box, point, float(scores[index])))
    return faces


PRIOR_SIZES = ((16, 32), (64, 128), (256, 512))
PRIOR_STEPS = (8, 16, 32)
PRIOR_VARIANCE = (0.1, 0.2)
# Pytorch_Retinaface-style exports expect BGR pixels minus this per-channel mean.
PRIOR_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def priors(height: int, width: int) -> np.ndarray:
    """Anchor (cx, cy, w, h) in pixels, ordered (level, row, column, size) as in training."""
    result = []
    for sizes, step in zip(PRIOR_SIZES, PRIOR_STEPS):
        rows, cols = int(np.ceil(height / step)), int(np.ceil(width / step))
        y, x, size = np.meshgrid(np.arange(rows), np.arange(cols), sizes, indexing="ij")
        result.append(
            np.stack(
                ((x + 0.5) * step, (y + 0.5) * step, size, size), axis=-1, dtype=np.float64
            ).reshape(-1, 4)
        )
    return np.concatenate(result)


def decode_priorbox(
    outputs: list[np.ndarray], height: int, width: int, confidence: float
) -> list[Face]:
    """Decode Pytorch_Retinaface-style loc [1,N,4], conf [1,N,2], landmarks [1,N,10]."""
    maps = {}
    for output in outputs:
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] not in (2, 4, 10):
            raise ValueError(f"Unsupported prior-box RetinaFace output shape: {output.shape}")
        if output.shape[2] in maps:
            raise ValueError(f"Duplicate RetinaFace output shape: {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("Detector returned non-finite values")
        maps[output.shape[2]] = output[0].astype(np.float64)
    if set(maps) != {2, 4, 10}:
        raise ValueError("Prior-box RetinaFace requires loc, conf and landmark outputs")
    anchors = priors(height, width)
    if any(len(value) != len(anchors) for value in maps.values()):
        raise ValueError(
            f"RetinaFace returned {len(maps[4])} boxes but {len(anchors)} priors were expected"
        )
    scores = maps[2][:, 1]
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("RetinaFace class outputs must be probabilities, not logits")
    selected = np.flatnonzero(scores >= confidence)
    anchors = anchors[selected]
    delta = maps[4][selected]
    centre = anchors[:, :2] + delta[:, :2] * PRIOR_VARIANCE[0] * anchors[:, 2:]
    with np.errstate(over="raise", invalid="raise"):
        size = anchors[:, 2:] * np.exp(delta[:, 2:] * PRIOR_VARIANCE[1])
    boxes = np.column_stack((centre - size / 2, centre + size / 2))
    points = maps[10][selected].reshape(-1, 5, 2)
    points = anchors[:, None, :2] + points * PRIOR_VARIANCE[0] * anchors[:, None, 2:]
    return [
        Face(box, point, float(scores[index])) for index, box, point in zip(selected, boxes, points)
    ]


class RetinaFace:
    def __init__(
        self,
        model: Path,
        device: str,
        size: int = 640,
        confidence: float = 0.8,
        nms_threshold: float = 0.4,
    ):
        if size < 32 or size % 32:
            raise ValueError("Detector size must be a positive multiple of 32")
        if not 0 < confidence <= 1 or not 0 <= nms_threshold <= 1:
            raise ValueError("Invalid detector confidence or NMS threshold")
        self.session = session(model, device)
        self.input = self.session.get_inputs()[0]
        require_shape(self.input.shape, (1, 3, size, size))
        count = len(self.session.get_outputs())
        if count == 9:
            self.layout = "net3"
        elif count == 3:
            self.layout = "priorbox"
        else:
            raise ValueError(
                f"Unsupported RetinaFace export with {count} outputs; expected InsightFace net3 "
                "(9 feature maps) or Pytorch_Retinaface prior-box (loc, conf, landmarks)"
            )
        self.size = size
        self.confidence = confidence
        self.nms_threshold = nms_threshold

    def detect(self, image: np.ndarray) -> list[Face]:
        height, width = image.shape[:2]
        scale = min(self.size / width, self.size / height)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(image, (resized_w, resized_h))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.float32)
        if self.layout == "net3":
            canvas[:resized_h, :resized_w] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            canvas[:resized_h, :resized_w] = resized.astype(np.float32) - PRIOR_MEAN_BGR
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
        outputs = self.session.run(None, {self.input.name: tensor})
        decoder = decode if self.layout == "net3" else decode_priorbox
        faces = decoder(outputs, self.size, self.size, self.confidence)
        if not faces:
            return []
        indices = nms(
            np.array([face.box for face in faces]),
            np.array([face.score for face in faces]),
            self.nms_threshold,
        )
        factors = np.array([resized_w / width, resized_h / height])
        result = []
        for index in indices:
            face = faces[index]
            face.box = face.box / np.tile(factors, 2)
            face.landmarks = face.landmarks / factors
            centre = (face.box[:2] + face.box[2:]) / 2
            if 0 <= centre[0] < width and 0 <= centre[1] < height:
                result.append(face)
        return result
