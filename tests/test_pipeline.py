import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from recognition_accuracy.dataset import discover_images, identity_from_name, prepare
from recognition_accuracy.evaluate import evaluate, find_tensors


def write_image(root, name, bgr=(128, 128, 255)):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((112, 112, 3), bgr, dtype=np.uint8))
    return path


def run_prepare(images, output, detector_model, **kwargs):
    options = {
        "source": images,
        "output": output,
        "detector_model": detector_model,
        "device": "cpu",
        "enrolment": 1,
        "max_test": 1,
        "detector_size": 128,
        "confidence": 0.8,
        "allow_partial": False,
        "face_selection": "central",
    }
    options.update(kwargs)
    return prepare(**options)


def run_evaluate(dataset, output, model, **kwargs):
    options = {
        "dataset": dataset,
        "output": output,
        "model": model,
        "device": "cpu",
        "output_name": None,
        "targets": [0.01, 0.001],
        "chunk_size": 1,
        "curve_points": 21,
        "allow_partial": False,
    }
    options.update(kwargs)
    return evaluate(**options)


def test_real_onnx_end_to_end(tmp_path, detector_model, recognition_model):
    images = tmp_path / "images"
    for index in (3, 1, 2):
        write_image(images, f"Alice/Alice_{index:04}.png")
    for index in (2, 1):
        write_image(images, f"Bob/Bob_{index:04}.png", (255, 128, 128))
    # Singleton identities enrol as distractors and do not contribute genuine probes.
    write_image(images, "Charlie/Charlie_0001.png", (128, 255, 128))
    dataset = tmp_path / "dataset"
    prepared = run_prepare(images, dataset, detector_model)
    assert prepared["images"] == 6
    assert prepared["tensors"] == {"train": 3, "test": 2, "unused": 1}
    assert prepared["complete"] and prepared["failures"] == 0
    assert (dataset / "unused/Alice_0003.raw").stat().st_size == 150528
    manifest = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 6
    assert all(record["status"] == "ok" for record in manifest)
    results = tmp_path / "results"
    summary = run_evaluate(dataset, results, recognition_model)
    assert summary["genuine_comparisons"] == 2
    assert summary["impostor_comparisons"] == 4
    assert summary["enrolled_identities"] == 3
    assert summary["eer"]["eer"] == 0
    assert summary["roc_auc"] == 1
    assert summary["rank_one_accuracy"] == 1
    assert summary["embedding_dimension"] == 3
    for point in summary["operating_points"]:
        assert point["tar"] == 1
        assert point["fmr"] == 0
    for name in (
        "metrics.json",
        "curves.csv",
        "roc-det.png",
        "probes.csv",
        "embeddings.npz",
        "genuine-scores.npy",
        "impostor-scores.npy",
    ):
        assert (results / name).stat().st_size > 0
    scores = np.load(results / "genuine-scores.npy")
    assert (scores > 0.99).all()
    exported = np.loadtxt(results / "curves.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(exported[:, 3], 1 - exported[:, 2])
    assert json.loads((results / "metrics.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        run_evaluate(dataset, results, recognition_model)
    (dataset / "test/Alice_0002.raw").rename(dataset / "test/Alice_0004.raw")
    with pytest.raises(ValueError, match="manifest"):
        find_tensors(dataset, False)


def test_numeric_splitting_and_multiple_enrolments(tmp_path, detector_model):
    images = tmp_path / "images"
    for index in (10, 2, 1, 3):
        write_image(images, f"A_Name_{index}.png")
    dataset = tmp_path / "dataset"
    run_prepare(images, dataset, detector_model, enrolment=2)
    assert sorted(p.stem for p in (dataset / "train").glob("*.raw")) == ["A_Name_1", "A_Name_2"]
    assert (dataset / "test/A_Name_3.raw").exists()
    assert (dataset / "unused/A_Name_10.raw").exists()


def test_failures_recorded_without_promoting_later_images(tmp_path, detector_model):
    images = tmp_path / "images"
    images.mkdir()
    (images / "Alice_0001.jpg").write_bytes(b"not an image")
    write_image(images, "Alice_0002.png")
    dataset = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="1/2"):
        run_prepare(images, dataset, detector_model)
    records = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    assert records[0]["status"] == "unreadable_image"
    assert records[0]["split"] == "train"
    assert records[1]["split"] == "test"
    assert not list((dataset / "train").glob("*.raw"))
    assert (dataset / "test/Alice_0002.raw").exists()
    with pytest.raises(ValueError, match="failed images"):
        find_tensors(dataset, False)


def test_no_face_produces_manifest_failure(tmp_path, detector_model, monkeypatch):
    from recognition_accuracy.detector import RetinaFace

    monkeypatch.setattr(RetinaFace, "detect", lambda self, image: [])
    images = tmp_path / "images"
    write_image(images, "Alice_0001.png")
    output = tmp_path / "dataset"
    result = run_prepare(images, output, detector_model, allow_partial=True)
    assert result["failures"] == 1
    assert not list(output.rglob("*.raw"))
    record = json.loads((output / "manifest.jsonl").read_text())
    assert record["status"] == "no_face"


def test_unexpected_detector_error_keeps_dataset_incomplete(tmp_path, detector_model, monkeypatch):
    from recognition_accuracy.detector import RetinaFace

    def broken(self, image):
        raise ValueError("Unsupported output")

    monkeypatch.setattr(RetinaFace, "detect", broken)
    images = tmp_path / "images"
    write_image(images, "Alice_0001.png")
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Unsupported"):
        run_prepare(images, output, detector_model)
    assert not json.loads((output / "dataset.json").read_text())["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        find_tensors(output, True)


def external_dataset(tmp_path):
    dataset = tmp_path / "raw"
    for split, index in (("train", 1), ("test", 2)):
        (dataset / split).mkdir(parents=True)
        for identity in ("Alice", "Bob"):
            np.ones((1, 112, 112, 3), dtype="<f4").tofile(
                dataset / split / f"{identity}_{index:04}.raw",
            )
    return dataset


def test_external_tensors_and_duplicate_leakage(tmp_path):
    dataset = external_dataset(tmp_path)
    train, test = find_tensors(dataset, False)
    assert len(train) == len(test) == 2
    (dataset / "test/Alice_0002.raw").rename(dataset / "test/Alice_0001.raw")
    with pytest.raises(ValueError, match="leakage"):
        find_tensors(dataset, False)


def test_invalid_tensor_size_and_values(tmp_path, recognition_model):
    dataset = external_dataset(tmp_path)
    tensor = dataset / "train/Alice_0001.raw"
    tensor.write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        find_tensors(dataset, False)
    values = np.full((1, 112, 112, 3), np.nan, dtype="<f4")
    values.tofile(tensor)
    with pytest.raises(ValueError, match="non-finite"):
        run_evaluate(dataset, tmp_path / "results", recognition_model)


def test_missing_enrolment_fails_even_allow_partial(tmp_path, recognition_model):
    dataset = external_dataset(tmp_path)
    (dataset / "test/Alice_0002.raw").rename(dataset / "test/Unenrolled_0002.raw")
    with pytest.raises(ValueError, match="no enrolment"):
        run_evaluate(dataset, tmp_path / "results", recognition_model, allow_partial=True)


def test_identification_ties_not_success(tmp_path, recognition_model):
    dataset = external_dataset(tmp_path)
    summary = run_evaluate(dataset, tmp_path / "results", recognition_model)
    assert summary["rank_one_accuracy"] == 0
    assert summary["eer"]["eer"] == 0.5
    assert summary["roc_auc"] == 0.5


def test_enrolment_template_normalizes_each_embedding_before_averaging(tmp_path, recognition_model):
    dataset = tmp_path / "dataset"
    samples = {
        "train/Alice_0001.raw": [1, 0, 0],
        "train/Alice_0002.raw": [0, 0.25, 0],
        "train/Bob_0001.raw": [0, 0, 1],
        "test/Alice_0003.raw": [1, 1, 0],
        "test/Bob_0002.raw": [0, 0, 1],
    }
    for name, values in samples.items():
        destination = dataset / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.broadcast_to(np.array(values, dtype="<f4"), (1, 112, 112, 3)).tofile(destination)
    results = tmp_path / "results"
    summary = run_evaluate(dataset, results, recognition_model)
    assert summary["enrolment_images"] == 3
    assert summary["roc_auc"] == 1
    with np.load(results / "embeddings.npz") as archive:
        np.testing.assert_allclose(archive["templates"][0], [2**-0.5, 2**-0.5, 0])


def test_cli_help_and_error_exit():
    help_result = subprocess.run(
        [sys.executable, "-m", "recognition_accuracy", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "download" in help_result.stdout and "evaluate" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "recognition_accuracy",
            "prepare",
            "--images",
            "/nonexistent",
            "--output",
            "/nonexistent",
            "--detector",
            "/nonexistent",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "ERROR:" in result.stderr


def test_cli_onnx_end_to_end(tmp_path, detector_model, recognition_model):
    images = tmp_path / "images"
    for identity, color in (("Alice", (128, 128, 255)), ("Bob", (255, 128, 128))):
        for index in (1, 2):
            write_image(images, f"{identity}_{index:04}.png", color)
    dataset, results = tmp_path / "dataset", tmp_path / "results"
    commands = [
        [
            "prepare",
            "--images",
            str(images),
            "--output",
            str(dataset),
            "--detector",
            str(detector_model),
            "--detector-size",
            "128",
        ],
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--output",
            str(results),
            "--model",
            str(recognition_model),
            "--target-fmr",
            "0.01",
            "0.001",
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", "recognition_accuracy", *command],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    summary = json.loads((results / "metrics.json").read_text())
    assert summary["eer"]["eer"] == 0
    assert summary["genuine_comparisons"] == 2
    assert summary["impostor_comparisons"] == 2
    assert (results / "roc-det.png").is_file()


def test_image_discovery_errors(tmp_path):
    with pytest.raises(ValueError, match="filename"):
        identity_from_name(Path("bad-name.jpg"))
    with pytest.raises(ValueError, match="No LFW"):
        discover_images(tmp_path)
    write_image(tmp_path, "a/Alice/Alice_0001.png")
    write_image(tmp_path, "b/Alice/Alice_0001.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        discover_images(tmp_path)


def test_rejects_non_lfw_layout(tmp_path):
    write_image(tmp_path, "WIDER_val/images/6--Funeral/6_Funeral_Funeral_6_572.jpg")
    with pytest.raises(ValueError, match="Not an LFW layout"):
        discover_images(tmp_path)
