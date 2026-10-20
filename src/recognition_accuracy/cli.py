import argparse
import json
import sys
from pathlib import Path

from .dataset import prepare
from .download import download
from .evaluate import evaluate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="LFW preparation and MobileFaceNet template-vs-probe accuracy evaluation",
    )
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download and safely extract the LFW archive")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument(
        "--archive", type=Path, help="Extract an existing archive instead of downloading"
    )
    fetch.add_argument(
        "--url", help="Alternative Google Drive / HTTP URL (default: the project's LFW file id)"
    )
    fetch.add_argument("--sha256", help="Optional expected archive SHA-256")

    dataset = commands.add_parser(
        "prepare", help="Align all LFW images and save normalized tensors"
    )
    dataset.add_argument("--images", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--detector", type=Path, required=True, help="RetinaFace net3 ONNX model")
    dataset.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    dataset.add_argument("--enrolment", type=int, default=1, help="First N images per identity")
    dataset.add_argument(
        "--max-test", type=int, default=10, help="Next at most Y images per identity"
    )
    dataset.add_argument("--detector-size", type=int, default=640)
    dataset.add_argument("--confidence", type=float, default=0.8)
    dataset.add_argument("--face-selection", choices=["central", "confidence"], default="central")
    dataset.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully despite recorded detection/alignment failures",
    )

    score = commands.add_parser(
        "evaluate",
        help="Infer embeddings and compute verification metrics",
        epilog=(
            "Model inputs may use NHWC [1,112,112,3] or NCHW [1,3,112,112]. "
            "The evaluator automatically transposes stored NHWC tensors for NCHW models; "
            "RGB order is unchanged; normalization is unchanged unless --input-range 0-255 "
            "is selected."
        ),
    )
    score.add_argument(
        "--dataset", type=Path, required=True, help="Directory containing train/ and test/"
    )
    score.add_argument("--output", type=Path, required=True)
    score.add_argument(
        "--model", type=Path, required=True, help="MobileFaceNet ONNX or TFLite model"
    )
    score.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    score.add_argument("--output-name", help="Required if the model has multiple outputs")
    score.add_argument(
        "--input-range",
        choices=["normalized", "0-255"],
        default="normalized",
        help="Model input values: normalized [-1,1] (default) or RGB pixels [0,255]",
    )
    score.add_argument("--target-fmr", type=float, nargs="+", default=[0.01, 0.001, 0.0001])
    score.add_argument(
        "--chunk-size", type=int, default=128, help="Probe count per score-matrix chunk"
    )
    score.add_argument(
        "--curve-points", type=int, default=2001, help="Impostor curve sampling budget"
    )
    score.add_argument("--allow-partial", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "download":
            download(args.output, args.archive, args.sha256, args.url)
        elif args.command == "prepare":
            result = prepare(
                args.images,
                args.output,
                args.detector,
                args.device,
                args.enrolment,
                args.max_test,
                args.detector_size,
                args.confidence,
                args.allow_partial,
                args.face_selection,
            )
            print(json.dumps({"tensors": result["tensors"], "failures": result["failures"]}))
        else:
            evaluate(
                args.dataset,
                args.output,
                args.model,
                args.device,
                args.output_name,
                args.target_fmr,
                args.chunk_size,
                args.curve_points,
                args.allow_partial,
                input_range=args.input_range,
            )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
