import numpy as np
import pytest

from recognition_accuracy.metrics import equal_error, operating_point, rates, summarize


def brute_eer(genuine, impostor):
    values = np.unique(np.concatenate((genuine, impostor))).astype(np.float64)
    thresholds = np.unique(np.concatenate(([-1], values, np.nextafter(values, np.inf))))
    fmr, fnmr = rates(np.sort(genuine), np.sort(impostor), thresholds)
    difference = fmr - fnmr
    end = np.flatnonzero(difference <= 0)[0]
    if difference[end] == 0:
        return fmr[end]
    start = end - 1
    weight = difference[start] / (difference[start] - difference[end])
    return fmr[start] + weight * (fmr[end] - fmr[start])


@pytest.mark.parametrize(
    "genuine,impostor,eer,auc",
    [
        ([0.8, 0.9], [0.1, 0.2], 0, 1),
        ([0.1, 0.2], [0.8, 0.9], 1, 0),
        ([0.5, 0.5], [0.5, 0.5, 0.5], 0.5, 0.5),
        ([0.2, 0.8], [0.1, 0.9], 0.5, 0.5),
    ],
)
def test_known_metrics(genuine, impostor, eer, auc):
    summary, curve = summarize(
        np.array(genuine, dtype=np.float32),
        np.array(impostor, dtype=np.float32),
        [0, 0.1, 0.5, 1],
        11,
    )
    assert summary["eer"]["eer"] == pytest.approx(eer)
    assert summary["roc_auc"] == pytest.approx(auc)
    assert curve.shape[1] == 4
    np.testing.assert_allclose(curve[:, 3], 1 - curve[:, 2])
    assert (curve[0, 1:3] == [1, 0]).all()
    assert (curve[-1, 1:3] == [0, 1]).all()
    for point in summary["operating_points"]:
        assert point["fmr"] <= point["target_fmr"]


def test_random_metrics_match_exhaustive_reference():
    rng = np.random.default_rng(42)
    for _ in range(100):
        genuine = np.sort(rng.integers(-5, 6, size=17).astype(np.float32) / 5)
        impostor = np.sort(rng.integers(-5, 6, size=39).astype(np.float32) / 5)
        summary, _ = summarize(genuine.copy(), impostor.copy(), [0.01, 0.1, 0.5], 9)
        assert summary["eer"]["eer"] == pytest.approx(brute_eer(genuine, impostor))
        auc = (genuine[:, None] > impostor).mean() + 0.5 * (genuine[:, None] == impostor).mean()
        assert summary["roc_auc"] == pytest.approx(auc)


def test_ties_at_operating_point_are_excluded_conservatively():
    genuine = np.array([0.1, 0.8, 0.9], dtype=np.float32)
    impostor = np.array([0.1, 0.5, 0.5, 0.5], dtype=np.float32)
    point = operating_point(genuine, impostor, 0.5)
    assert point["threshold"] > float(impostor[-1])
    assert point["fmr"] == 0
    assert not np.any(impostor >= point["threshold"])
    assert point["tar"] == pytest.approx(2 / 3)
    assert operating_point(genuine, impostor, 0.01)["below_empirical_resolution"]


def test_float32_neighbor_crossing_and_nonuniform_class_sizes():
    a = np.float32(0.5)
    b = np.nextafter(a, np.float32(1))
    genuine = np.array([a, b], dtype=np.float32)
    impostor = np.array([a] * 21 + [b] * 9, dtype=np.float32)
    assert equal_error(genuine, impostor)["eer"] == pytest.approx(brute_eer(genuine, impostor))


def test_subnormal_scores_do_not_skip_empirical_crossings():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    genuine = np.array([0, tiny, 2 * tiny, 1], dtype=np.float32)
    impostor = np.array([-1, 0, 0, tiny, 2 * tiny, 3 * tiny], dtype=np.float32)
    assert equal_error(genuine, impostor)["eer"] == pytest.approx(brute_eer(genuine, impostor))


def test_invalid_metric_inputs():
    with pytest.raises(ValueError, match="at least one"):
        summarize(np.array([]), np.array([0.2]), [0.1], 5)
    with pytest.raises(ValueError, match="between"):
        operating_point(np.array([0.5]), np.array([0.1]), -0.1)
