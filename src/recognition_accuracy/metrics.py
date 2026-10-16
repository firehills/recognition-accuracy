import numpy as np


def rates(genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray) -> tuple:
    """Scores must be sorted ascending. Accept a comparison iff score >= threshold."""
    fnmr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)
    fmr = (len(impostor) - np.searchsorted(impostor, thresholds, side="left")) / len(impostor)
    return fmr, fnmr


def equal_error(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    # Locate the crossing, then interpolate the two adjacent empirical operating points.
    low = float(min(genuine[0], impostor[0]))
    high = float(np.nextafter(float(max(genuine[-1], impostor[-1])), np.inf))
    while True:
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        fmr, fnmr = rates(genuine, impostor, np.array([middle]))
        if fmr[0] > fnmr[0]:
            low = middle
        else:
            high = middle
    # low/high bracket the same empirical score jump, including tied scores.
    fmr, fnmr = rates(genuine, impostor, np.array([low, high]))
    difference = fmr - fnmr
    weight = (
        0.0 if difference[0] == difference[1] else (difference[0] / (difference[0] - difference[1]))
    )
    eer = fmr[0] + weight * (fmr[1] - fmr[0])
    return {
        "eer": float(eer),
        "threshold_bracket": [low, high],
        "interpolation_weight": float(weight),
        "note": "Linear interpolation between adjacent empirical ROC points; "
        "a tied-score EER may not be achievable by a single threshold.",
    }


def operating_point(genuine: np.ndarray, impostor: np.ndarray, target: float) -> dict:
    if not 0 <= target <= 1:
        raise ValueError("Target FMR must be between 0 and 1")
    allowed = int(np.floor(target * len(impostor)))
    if allowed >= len(impostor):
        threshold = -1.0
    else:
        # Exclude all ties at the boundary to guarantee empirical FMR <= target.
        boundary = float(impostor[len(impostor) - allowed - 1])
        threshold = float(np.nextafter(np.float32(boundary), np.float32(np.inf)))
    fmr, fnmr = rates(genuine, impostor, np.array([threshold]))
    return {
        "target_fmr": target,
        "threshold": threshold,
        "fmr": float(fmr[0]),
        "fnmr": float(fnmr[0]),
        "tar": float(1 - fnmr[0]),
        "expected_false_matches_at_target": target * len(impostor),
        "below_empirical_resolution": target < 1 / len(impostor),
    }


def summarize(
    genuine: np.ndarray,
    impostor: np.ndarray,
    targets: list[float],
    curve_points: int,
) -> tuple[dict, np.ndarray]:
    if not len(genuine) or not len(impostor):
        raise ValueError("Evaluation needs at least one genuine and one impostor comparison")
    if curve_points < 3:
        raise ValueError("At least three curve points are required")
    # In-place memmap sorts bound RAM usage even for tens of millions of impostor scores.
    genuine.sort()
    impostor.sort()
    eer = equal_error(genuine, impostor)
    points = [operating_point(genuine, impostor, target) for target in targets]
    sample = np.linspace(0, 1, curve_points)
    # Sample order statistics, not interpolated quantiles; preserve actual score boundaries.
    sampled = np.concatenate(
        [
            genuine[np.rint(sample * (len(genuine) - 1)).astype(np.int64)],
            impostor[np.rint(sample * (len(impostor) - 1)).astype(np.int64)],
        ]
    ).astype(np.float64)
    # Include all genuine boundaries (small for LFW), operating points, and the EER crossing.
    thresholds = np.unique(
        np.concatenate(
            [
                [-1.0, np.nextafter(1.0, np.inf)],
                sampled,
                np.nextafter(sampled, np.inf),
                genuine.astype(np.float64),
                np.nextafter(genuine.astype(np.float64), np.inf),
                [p["threshold"] for p in points],
                eer["threshold_bracket"],
            ]
        )
    )
    fmr, fnmr = rates(genuine, impostor, thresholds)
    curve = np.column_stack((thresholds, fmr, fnmr, 1 - fnmr))
    # Exact empirical AUC = P(genuine > impostor) + 0.5 * P(tie).
    left = np.searchsorted(impostor, genuine, side="left").astype(np.float64)
    right = np.searchsorted(impostor, genuine, side="right").astype(np.float64)
    auc = np.sum((left + right) / 2) / (len(genuine) * len(impostor))
    summary = {
        "genuine_comparisons": len(genuine),
        "impostor_comparisons": len(impostor),
        "fmr_resolution": 1 / len(impostor),
        "fnmr_resolution": 1 / len(genuine),
        "eer": eer,
        "roc_auc": float(auc),
        "operating_points": points,
        "curve": "Sampled impostor order statistics plus all genuine score boundaries; "
        "EER, AUC and operating points are computed from all scores, not this sample.",
        "accept_rule": "cosine_similarity >= threshold",
    }
    return summary, curve
