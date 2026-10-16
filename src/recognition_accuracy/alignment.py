import cv2
import numpy as np

REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def similarity_transform(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Expected five finite (x, y) landmarks")
    source_mean = points.mean(axis=0)
    target_mean = REFERENCE.mean(axis=0)
    source = points - source_mean
    target = REFERENCE - target_mean
    variance = np.sum(source * source)
    if variance <= 1e-12 or np.linalg.matrix_rank(source) < 2:
        raise ValueError("Degenerate face landmarks")
    u, singular, vt = np.linalg.svd(target.T @ source)
    correction = np.ones(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1
    rotation = u @ np.diag(correction) @ vt
    scale = np.sum(singular * correction) / variance
    if scale <= 1e-12:
        raise ValueError("Degenerate face transform")
    linear = scale * rotation
    translation = target_mean - linear @ source_mean
    return np.column_stack((linear, translation))


def aligned_tensor(bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = similarity_transform(points)
    crop = cv2.warpAffine(
        bgr,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 127.5)[None], dtype="<f4")
