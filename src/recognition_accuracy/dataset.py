import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

from .alignment import aligned_tensor
from .detector import RetinaFace
from .runtime import sha256

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TENSOR_SHAPE = (1, 112, 112, 3)
TENSOR_BYTES = 1 * 112 * 112 * 3 * 4


def identity_from_name(path: Path) -> str:
    match = re.fullmatch(r"(.+)_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Expected an LFW identity_NNNN filename: {path}")
    return match.group(1)


def check_lfw_layout(path: Path, root: Path) -> str:
    """Return the identity, requiring LFW's <identity>/<identity>_NNNN.jpg (or flat) layout."""
    identity = identity_from_name(path)
    if path.parent != root and path.parent.name != identity:
        raise ValueError(
            f"Not an LFW layout: {path.relative_to(root)} (expected folder {identity!r}). "
            "The source does not look like LFW (e.g. WIDER FACE uses event folders)."
        )
    return identity


def discover_images(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    groups = defaultdict(list)
    stems = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        identity = check_lfw_layout(path, root)
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem: {path.stem}")
        stems.add(path.stem)
        groups[identity].append(path)
    if not groups:
        raise ValueError(f"No LFW images found under {root}")
    return {
        identity: sorted(paths, key=lambda p: (int(p.stem.rsplit("_", 1)[1]), p.name))
        for identity, paths in sorted(groups.items())
    }


def prepare(
    source: Path,
    output: Path,
    detector_model: Path,
    device: str,
    enrolment: int,
    max_test: int,
    detector_size: int,
    confidence: float,
    allow_partial: bool,
    face_selection: str,
) -> dict:
    if enrolment < 1 or max_test < 1:
        raise ValueError("Enrolment count and maximum test count must both be positive")
    groups = discover_images(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    detector = RetinaFace(detector_model, device, detector_size, confidence)
    output.mkdir(parents=True)
    for split in ("train", "test", "unused"):
        (output / split).mkdir()
    counts = {"train": 0, "test": 0, "unused": 0}
    failures = 0
    total = sum(map(len, groups.values()))
    metadata = {
        "schema_version": 1,
        "complete": False,
        "source": str(source.resolve()),
        "shape": list(TENSOR_SHAPE),
        "dtype": "<f4",
        "layout": "NHWC",
        "color": "RGB",
        "normalization": "(pixel - 127.5) / 127.5",
        "alignment": "ArcFace five-point similarity, 112x112",
        "detector": str(detector_model.resolve()),
        "detector_sha256": sha256(detector_model),
        "device": device,
        "detector_size": detector_size,
        "detector_confidence": confidence,
        "face_selection": face_selection,
        "enrolment_per_identity": enrolment,
        "max_test_per_identity": max_test,
        "identities": len(groups),
        "images": total,
    }
    metadata_path = output / "dataset.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "manifest.jsonl").open("w") as manifest:
        processed = 0
        for identity, paths in groups.items():
            for index, path in enumerate(paths):
                split = (
                    "train"
                    if index < enrolment
                    else ("test" if index < enrolment + max_test else "unused")
                )
                record = {
                    "source": str(path.relative_to(source)),
                    "identity": identity,
                    "split": split,
                }
                image = cv2.imread(str(path))
                if image is None:
                    record["status"] = "unreadable_image"
                else:
                    faces = detector.detect(image)
                    record["faces_detected"] = len(faces)
                    if not faces:
                        record["status"] = "no_face"
                    else:
                        if face_selection == "central":
                            height, width = image.shape[:2]
                            face = min(
                                faces,
                                key=lambda f: (
                                    ((f.box[0] + f.box[2]) / 2 - width / 2) ** 2
                                    + ((f.box[1] + f.box[3]) / 2 - height / 2) ** 2,
                                    -f.score,
                                ),
                            )
                        else:
                            face = max(faces, key=lambda f: f.score)
                        try:
                            tensor = aligned_tensor(image, face.landmarks)
                        except ValueError as error:
                            record.update(status="invalid_landmarks", error=str(error))
                        else:
                            destination = output / split / f"{path.stem}.raw"
                            tensor.tofile(destination)
                            record.update(
                                status="ok",
                                tensor=str(destination.relative_to(output)),
                                confidence=face.score,
                                landmarks=face.landmarks.tolist(),
                                box=face.box.tolist(),
                            )
                            counts[split] += 1
                if record["status"] != "ok":
                    failures += 1
                    print(f"FAILED {path}: {record['status']}", flush=True)
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                processed += 1
                if processed % 100 == 0 or processed == total:
                    print(f"Prepared {processed}/{total}; failures={failures}", flush=True)
    metadata.update(complete=True, tensors=counts, failures=failures)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if failures and not allow_partial:
        raise RuntimeError(
            f"{failures}/{total} images failed. See {output / 'manifest.jsonl'}. "
            "No synthetic tensors were created. Inspect failures and rerun to a new output "
            "directory, or explicitly accept this dataset with evaluate --allow-partial."
        )
    return metadata
