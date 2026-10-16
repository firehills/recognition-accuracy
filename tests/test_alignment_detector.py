import numpy as np
import pytest

from recognition_accuracy.alignment import REFERENCE, aligned_tensor, similarity_transform
from recognition_accuracy.detector import RetinaFace, decode, decode_priorbox, nms, priors
from recognition_accuracy.runtime import require_shape, unit_vector


def test_similarity_recovers_scale_rotation_and_translation():
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = 1.7 * REFERENCE @ rotation.T + [13, -9]
    transform = similarity_transform(source)
    recovered = source @ transform[:, :2].T + transform[:, 2]
    np.testing.assert_allclose(recovered, REFERENCE, atol=1e-10)
    singular = np.linalg.svd(transform[:, :2], compute_uv=False)
    np.testing.assert_allclose(singular[0], singular[1])
    assert np.linalg.det(transform[:, :2]) > 0


def test_raw_tensor_shape_channel_order_normalization():
    image = np.full((112, 112, 3), [0, 127, 255], dtype=np.uint8)
    tensor = aligned_tensor(image, REFERENCE)
    assert tensor.shape == (1, 112, 112, 3)
    assert tensor.dtype == np.dtype("<f4")
    assert tensor.flags.c_contiguous
    assert tensor.nbytes == 150528
    np.testing.assert_allclose(tensor[0, 50, 50], [1, -0.5 / 127.5, -1], atol=1e-7)


@pytest.mark.parametrize("points", [np.zeros((5, 2)), np.ones((4, 2)), REFERENCE * np.nan])
def test_rejects_invalid_landmarks(points):
    with pytest.raises(ValueError):
        similarity_transform(points)


def test_retinaface_onnx_decodes_to_original_image(detector_model):
    detector = RetinaFace(detector_model, "cpu", size=128)
    faces = detector.detect(np.full((112, 112, 3), 127, dtype=np.uint8))
    assert len(faces) == 1
    np.testing.assert_allclose(faces[0].landmarks, REFERENCE, atol=1e-5)
    np.testing.assert_allclose(faces[0].box, np.array([48, 48, 79, 79]) * 112 / 128)
    assert faces[0].score == pytest.approx(0.99)


def test_decode_rejects_scrfd_and_missing_outputs():
    with pytest.raises(ValueError, match="nine"):
        decode([], 640, 640, 0.8)
    with pytest.raises(ValueError, match="SCRFD"):
        decode([np.zeros((1, 128, 1))] * 9, 640, 640, 0.8)


def test_second_anchor_and_nonzero_box_deltas():
    outputs = []
    for stride in (8, 16, 32):
        width = 128 // stride
        maps = [np.zeros((1, channels, width, width), dtype=np.float32) for channels in (4, 8, 20)]
        if stride == 16:
            maps[0][0, 3, 2, 3] = 0.95
            maps[1][0, 4:8, 2, 3] = [0.1, -0.2, np.log(2), np.log(0.5)]
            maps[2][0, 10:20, 2, 3] = np.arange(10) / 10
        outputs.extend(maps)
    faces = decode(outputs, 128, 128, 0.8)
    assert len(faces) == 1
    centre = np.array([3 * 16 + 7.5, 2 * 16 + 7.5])
    predicted_centre = centre + [6.4, -12.8]
    np.testing.assert_allclose(
        faces[0].box,
        np.concatenate((predicted_centre - [63.5, 15.5], predicted_centre + [63.5, 15.5])),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        faces[0].landmarks,
        centre + np.arange(10).reshape(5, 2) * 6.4,
        atol=1e-5,
    )


def test_nms_and_embeddings():
    boxes = np.array([[0, 0, 9, 9], [1, 1, 10, 10], [30, 30, 40, 40]], dtype=float)
    assert nms(boxes, np.array([0.8, 0.9, 0.7]), 0.4) == [1, 2]
    np.testing.assert_allclose(unit_vector(np.array([3, 4])), [0.6, 0.8])
    for vector in (np.zeros(3), np.array([np.nan, 1]), np.array([np.inf, 1])):
        with pytest.raises(ValueError):
            unit_vector(vector)
    require_shape(["batch", 112, 112, 3], (1, 112, 112, 3))
    with pytest.raises(ValueError):
        require_shape([1, 3, 112, 112], (1, 112, 112, 3))


def test_priorbox_decoding():
    anchors = priors(64, 64)
    assert anchors.shape == (8 * 8 * 2 + 4 * 4 * 2 + 2 * 2 * 2, 4)
    index = (3 * 8 + 4) * 2 + 1
    np.testing.assert_allclose(anchors[index], [36, 28, 32, 32])
    loc = np.zeros((1, len(anchors), 4), dtype=np.float32)
    conf = np.tile(np.array([1, 0], dtype=np.float32), (1, len(anchors), 1))
    landmarks = np.zeros((1, len(anchors), 10), dtype=np.float32)
    conf[0, index] = [0.05, 0.95]
    loc[0, index] = [0.5, -0.5, np.log(2) / 0.2, 0]
    landmarks[0, index] = np.arange(10) / 10
    faces = decode_priorbox([landmarks, loc, conf], 64, 64, 0.8)
    assert len(faces) == 1
    np.testing.assert_allclose(faces[0].box, [5.6, 10.4, 69.6, 42.4], atol=1e-4)
    np.testing.assert_allclose(
        faces[0].landmarks, [36, 28] + np.arange(10).reshape(5, 2) / 10 * 3.2, atol=1e-5
    )
    with pytest.raises(ValueError, match="priors"):
        decode_priorbox([loc[:, :5], conf[:, :5], landmarks[:, :5]], 64, 64, 0.8)
