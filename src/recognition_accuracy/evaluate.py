import csv
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np

from .dataset import TENSOR_BYTES, TENSOR_SHAPE, identity_from_name
from .metrics import summarize
from .runtime import prepare_input_tensor, require_shape, session, sha256, unit_vector


def find_tensors(dataset: Path, allow_partial: bool) -> tuple[list[Path], list[Path]]:
    metadata_path = dataset / "dataset.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("complete"):
            raise ValueError("Dataset preparation is incomplete")
        if metadata.get("failures") and not allow_partial:
            raise ValueError(
                "Dataset contains failed images; inspect manifest or use --allow-partial"
            )
        contract = {
            "shape": list(TENSOR_SHAPE),
            "dtype": "<f4",
            "layout": "NHWC",
            "color": "RGB",
            "normalization": "(pixel - 127.5) / 127.5",
        }
        if any(metadata.get(key) != value for key, value in contract.items()):
            raise ValueError("Unsupported tensor contract in dataset metadata")
    else:
        metadata = None
    train = sorted((dataset / "train").rglob("*.raw"))
    test = sorted((dataset / "test").rglob("*.raw"))
    if not train or not test:
        raise ValueError("Dataset requires non-empty train/ and test/ directories of .raw tensors")
    stems = set()
    for path in train + test:
        identity_from_name(path)
        if path.stem in stems:
            raise ValueError(f"Duplicate image / train-test leakage: {path.stem}")
        stems.add(path.stem)
        if path.stat().st_size != TENSOR_BYTES:
            raise ValueError(f"Incorrect raw tensor size: {path}; expected {TENSOR_BYTES} bytes")
    if metadata:
        expected = set()
        with (dataset / "manifest.jsonl").open() as manifest:
            for line in manifest:
                record = json.loads(line)
                if record["status"] == "ok" and record["split"] in ("train", "test"):
                    expected.add(record["tensor"])
        actual = {p.relative_to(dataset).as_posix() for p in train + test}
        if actual != expected:
            raise ValueError("Raw tensors do not match the preparation manifest")
    return train, test


def embed(
    paths: list[Path], model_session, output_name: str, input_range: str = "normalized"
) -> np.ndarray:
    if input_range not in ("normalized", "0-255"):
        raise ValueError("Input range must be normalized or 0-255")
    name = model_session.get_inputs()[0].name
    model_shape = model_session.get_inputs()[0].shape
    result = []
    dimension = None
    for index, path in enumerate(paths):
        tensor = np.fromfile(path, dtype="<f4").reshape(TENSOR_SHAPE)
        if not np.isfinite(tensor).all() or tensor.min() < -1 or tensor.max() > 1:
            raise ValueError(f"Tensor contains non-finite values or values outside [-1,1]: {path}")
        if input_range == "0-255":
            tensor = (tensor + np.float32(1)) * np.float32(127.5)
        prepared = prepare_input_tensor(tensor, model_shape)
        value = np.asarray(model_session.run([output_name], {name: prepared})[0])
        if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 2:
            raise ValueError(f"Embedding output must have shape [1,D], D>=2; got {value.shape}")
        if dimension is None:
            dimension = value.shape[1]
        if value.shape[1] != dimension:
            raise ValueError("Model output embedding dimension changed")
        result.append(unit_vector(value[0]))
        if (index + 1) % 100 == 0 or index + 1 == len(paths):
            print(f"Embedded {index + 1}/{len(paths)}", flush=True)
    return np.stack(result)


def plot(curve: np.ndarray, summary: dict, output: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    fmr, fnmr = curve[:, 1], curve[:, 2]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(fmr, 1 - fnmr)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(
        xlabel="FMR",
        ylabel="TAR",
        title=f"ROC (AUC={summary['roc_auc']:.5f})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    epsilon = min(summary["fmr_resolution"], summary["fnmr_resolution"]) / 2
    epsilon = max(epsilon, 1e-10)
    axes[1].semilogx(np.maximum(fmr, epsilon), 1 - fnmr)
    axes[1].set(xlabel="FMR (log scale)", ylabel="TAR", title="Low-FMR ROC", ylim=(0, 1))
    normal = NormalDist()
    transform = np.vectorize(normal.inv_cdf)
    axes[2].plot(
        transform(np.clip(fmr, epsilon, 1 - epsilon)),
        transform(np.clip(fnmr, epsilon, 1 - epsilon)),
    )
    ticks = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99])
    for axis in (axes[2].xaxis, axes[2].yaxis):
        axis.set_major_locator(FixedLocator(transform(ticks)))
        axis.set_major_formatter(FixedFormatter([f"{100 * t:g}%" for t in ticks]))
    axes[2].set(
        xlabel="FMR (normal deviate)",
        ylabel="FNMR (normal deviate)",
        title=f"DET (EER={100 * summary['eer']['eer']:.3f}%)",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Enrolment templates vs test probes (not official LFW pairs protocol)")
    figure.tight_layout()
    figure.savefig(output / "roc-det.png", dpi=160)
    plt.close(figure)


def evaluate(
    dataset: Path,
    output: Path,
    model: Path,
    device: str,
    output_name: str | None,
    targets: list[float],
    chunk_size: int,
    curve_points: int,
    allow_partial: bool,
    input_range: str = "normalized",
) -> dict:
    if input_range not in ("normalized", "0-255"):
        raise ValueError("Input range must be normalized or 0-255")
    if chunk_size < 1 or curve_points < 3:
        raise ValueError("Chunk size must be positive and curve points must be >=3")
    if not targets or any(not 0 <= target <= 1 for target in targets):
        raise ValueError("Supply at least one target FMR, each between 0 and 1")
    train, test = find_tensors(dataset, allow_partial)
    model_session = session(model, device)
    require_shape(model_session.get_inputs()[0].shape, TENSOR_SHAPE)
    outputs = [item.name for item in model_session.get_outputs()]
    if output_name is None:
        if len(outputs) != 1:
            raise ValueError(f"Model has multiple outputs; select --output-name from {outputs}")
        output_name = outputs[0]
    if output_name not in outputs:
        raise ValueError(f"Unknown output {output_name!r}; choices: {outputs}")
    identities = sorted({identity_from_name(path) for path in train})
    if len(identities) < 2:
        raise ValueError("At least two enrolled identities are required for impostor comparisons")
    identity_indices = {name: index for index, name in enumerate(identities)}
    probe_identities = [identity_from_name(path) for path in test]
    missing = sorted(set(probe_identities) - set(identities))
    if missing:
        raise ValueError(f"Test identities have no enrolment tensors: {missing}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    training = embed(train, model_session, output_name, input_range)
    probes = embed(test, model_session, output_name, input_range)
    if training.shape[1] != probes.shape[1]:
        raise ValueError("Train and test embedding dimensions differ")
    sums = np.zeros((len(identities), training.shape[1]), dtype=np.float64)
    for path, embedding in zip(train, training):
        sums[identity_indices[identity_from_name(path)]] += embedding
    templates = np.stack([unit_vector(value) for value in sums])
    np.savez(
        output / "embeddings.npz",
        templates=templates,
        probes=probes,
        template_identities=np.array(identities),
        probe_identities=np.array(probe_identities),
        probe_files=np.array([p.relative_to(dataset).as_posix() for p in test]),
        enrolment_embeddings=training,
        enrolment_files=np.array([p.relative_to(dataset).as_posix() for p in train]),
    )
    genuine = np.lib.format.open_memmap(
        output / "genuine-scores.npy",
        mode="w+",
        dtype="<f4",
        shape=(len(test),),
    )
    impostor = np.lib.format.open_memmap(
        output / "impostor-scores.npy",
        mode="w+",
        dtype="<f4",
        shape=(len(test) * (len(identities) - 1),),
    )
    impostor_offset = 0
    rank_one = 0
    with (output / "probes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tensor", "identity", "predicted_identity", "genuine_score", "best_score"])
        for start in range(0, len(test), chunk_size):
            stop = min(start + chunk_size, len(test))
            scores = np.clip(probes[start:stop] @ templates.T, -1, 1).astype(np.float32)
            truth = np.array([identity_indices[name] for name in probe_identities[start:stop]])
            rows = np.arange(stop - start)
            genuine[start:stop] = scores[rows, truth]
            prediction = np.argmax(scores, axis=1)
            # A top-score tie is not counted as a successful rank-one identification.
            unique_best = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) == 1
            rank_one += int(np.sum((prediction == truth) & unique_best))
            for row, index in enumerate(range(start, stop)):
                writer.writerow(
                    [
                        test[index].relative_to(dataset).as_posix(),
                        probe_identities[index],
                        identities[prediction[row]],
                        float(scores[row, truth[row]]),
                        float(scores[row, prediction[row]]),
                    ]
                )
            mask = np.ones(scores.shape, dtype=bool)
            mask[rows, truth] = False
            other = scores[mask]
            impostor[impostor_offset : impostor_offset + len(other)] = other
            impostor_offset += len(other)
    summary, curve = summarize(genuine, impostor, targets, curve_points)
    genuine.flush()
    impostor.flush()
    summary.update(
        protocol="L2-normalized mean enrolment template vs every test probe; exhaustive "
        "genuine/impostor comparisons. No model training. Not official LFW pairs.",
        model=str(model.resolve()),
        model_sha256=sha256(model),
        device=device,
        output_name=output_name,
        input_range=input_range,
        dataset=str(dataset.resolve()),
        enrolled_identities=len(identities),
        enrolment_images=len(train),
        test_images=len(test),
        embedding_dimension=training.shape[1],
        rank_one_accuracy=rank_one / len(test),
        rank_one_ties="Counted as incorrect; probes.csv breaks displayed ties by identity order.",
        score_dtype="float32",
        score_files="Sorted ascending; use embeddings.npz to reconstruct "
        "identity-labelled pair scores.",
        partial_dataset_accepted=allow_partial,
        threshold_warning="Thresholds are measured on this evaluation set, not independently "
        "calibrated. Use a disjoint calibration set for deployed thresholds. "
        "Pair comparisons are correlated; low FMR resolution is not a "
        "statistical confidence guarantee.",
    )
    metadata_path = dataset / "dataset.json"
    if metadata_path.exists():
        summary["dataset_metadata_sha256"] = sha256(metadata_path)
        summary["preparation_failures"] = json.loads(metadata_path.read_text())["failures"]
    np.savetxt(
        output / "curves.csv",
        curve,
        delimiter=",",
        header="threshold,fmr,fnmr,tar",
        comments="",
        fmt="%.17g",
    )
    plot(curve, summary, output)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"EER={summary['eer']['eer']:.6f}; AUC={summary['roc_auc']:.6f}; results: {output}")
    return summary
