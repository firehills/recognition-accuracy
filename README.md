# MobileFaceNet accuracy tools

Download LFW, align faces into model-ready raw tensors, and evaluate an ONNX or
TFLite MobileFaceNet model. ONNX models run on CPU or NVIDIA GPU; TFLite models use
the LiteRT runtime on CPU. No models or face data are bundled.

## Environment

Open this folder in VS Code and select **Dev Containers: Reopen in Container**.
The default [devcontainer](.devcontainer/devcontainer.json) installs CPU ONNX Runtime
and test tools into `.venv`. Activate it in the terminal:

```bash
source .venv/bin/activate
```

For GPU, choose the **Face recognition accuracy (NVIDIA GPU)** configuration when
reopening/rebuilding. The [GPU configuration](.devcontainer/gpu/devcontainer.json)
requires an NVIDIA host driver supporting CUDA 12 and NVIDIA Container Toolkit
configured for Docker. It installs ONNX Runtime's CUDA/cuDNN pip extras and preloads
their libraries; no host CUDA toolkit is needed. Both commands accept `--device gpu`.
Unavailable CUDA or provider initialization failure produces an error, not silent
CPU-only execution. ONNX Runtime can still place unsupported individual operators
on CPU; provider selection does not guarantee every operator runs on GPU.

Outside a container, use Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cpu,test]'
# OR in a separate environment: python -m pip install -e '.[gpu,test]'
# For TFLite model evaluation, include the LiteRT runtime:
python -m pip install -e '.[cpu,tflite,test]'
```

Install **one** ONNX runtime extra (`cpu` or `gpu`), not both: they provide the
same `onnxruntime` module. Add `tflite` when evaluating TFLite models.

## 1. Download LFW

```bash
face-accuracy download --output data/lfw
```

Uses `gdown` with the supplied [Google Drive archive](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing).

> **Warning:** as of 2026-10-16 that Drive file (363 MB, SHA-256
> `f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a`) contains the
> **WIDER FACE validation set** (`WIDER_val/images/<event>/...`), not LFW. The tool
> verifies the extracted layout (`<identity>/<identity>_NNNN.jpg`) and refuses
> non-LFW archives rather than producing meaningless "identities". Point it at a
> genuine LFW archive instead, e.g. the original `lfw.tgz` (13,233 images, 5,749
> identities, SHA-256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`):

```bash
face-accuracy download --url 'https://drive.google.com/file/d/<ID>/view' --output data/lfw
face-accuracy download --url https://ndownloader.figshare.com/files/5976018 --output data/lfw
face-accuracy download --archive /path/to/lfw.tgz --output data/lfw
```

Behind a TLS-inspecting proxy, Python may reject the certificate chain. Trust the
proxy CA (do not disable verification), e.g.
`REQUESTS_CA_BUNDLE=/path/to/bundle-with-proxy-ca.pem face-accuracy download ...`.

ZIP and tar archives are supported. Extraction rejects path traversal, links, and
special files. Downloads/extraction use a temporary directory and publish only on
success. The archive SHA-256 is printed; pass `--sha256 HASH` if a trusted hash is
available. Existing outputs are never overwritten.

Check LFW's usage conditions and applicable privacy/data-handling requirements
before downloading or using the face images and embeddings.

## 2. Prepare aligned raw tensors

Supply a compatible RetinaFace ONNX detector:

```bash
face-accuracy prepare \
  --images data/lfw \
  --detector /path/to/retinaface.onnx \
  --output data/tensors \
  --enrolment 1 --max-test 10 \
  --device cpu
```

### Detector contract (important)

Two RetinaFace ONNX export families are supported; the family is chosen from the
number of model outputs.

**Pytorch_Retinaface prior-box exports (3 outputs)** — e.g. the
[biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) family
and the `retinaface_mv1/mv2/r18/r34.onnx` releases of
[yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch/releases):

* Input float32 NCHW `[1,3,S,S]`, **BGR** pixels minus mean `(104,117,123)`.
* Outputs `loc [1,N,4]`, `conf [1,N,2]` (softmax; face is channel 1),
  `landmarks [1,N,10]`, in any order.
* Priors: min sizes `(16,32)`, `(64,128)`, `(256,512)` at steps 8/16/32,
  variances `(0.1,0.2)`.

**InsightFace RetinaFace net3 exports (9 outputs):**

* Input float32 NCHW **RGB** `[1,3,S,S]`, raw 0..255 pixels (no mean subtraction).
* For each stride 8/16/32, classification probabilities `[1,4,H,W]` (two
  background then two face channels), box deltas `[1,8,H,W]`, and five landmark
  deltas `[1,20,H,W]`. Output order does not matter.
* Two square anchors per location, sizes `(32,16)`, `(128,64)`, `(512,256)` at
  strides 8, 16, 32 respectively; base-anchor centre `(7.5,7.5)`
  ([reference](https://github.com/deepinsight/insightface/blob/master/detection/retinaface/retinaface.py)).

Both: `S=640` by default (`--detector-size`, a multiple of 32; dynamic spatial
inputs are accepted). Landmarks are left eye, right eye, nose, left mouth, right
mouth. SCRFD and other layouts fail explicitly. Preprocessing semantics cannot be
inferred from ONNX shapes, so check your export matches one of the above.

Images are aspect-ratio resized and padded at bottom/right for detection; boxes
and landmarks are mapped back to the original image. NMS uses IoU 0.4 and detection
confidence defaults to 0.8 (`--confidence`). If multiple faces are present, the
face with centre closest to the image centre is selected, appropriate for LFW's
subject-centred images. `--face-selection confidence` selects the highest-confidence
face instead. Inspect the manifest when assessing multi-face errors.

### Tensor and split contract

The five landmarks are fitted to the standard ArcFace 112x112 reference using a
least-squares **similarity transform** (uniform scale, rotation, translation; no
reflection/shear). The original image is warped with bilinear interpolation and
black padding. The crop is converted to RGB, then normalized:

```text
input = (pixel - 127.5) / 127.5
shape = [1,112,112,3]  (NHWC)
dtype = little-endian float32
storage = C-contiguous, headerless .raw, exactly 150528 bytes
```

Example output:

```text
data/tensors/
  dataset.json
  manifest.jsonl
  train/Person_Name_0001.raw
  test/Person_Name_0002.raw
  unused/Person_Name_0012.raw
```

Identity is everything before the final `_number` in the original LFW filename.
Images are ordered by their numeric suffix, not traversal order. For each identity,
the first **N** are enrolment (`train/`), the next **up to Y** are test (`test/`).
All remaining images are processed into `unused/`, so every successfully detected
image produces a tensor. `Y` is the maximum number of **test** images, not the
maximum total images. Singleton identities have enrolment only and act as
impostor distractors. "Training" here means enrolment, not fine-tuning the network.

Every source image gets a manifest record, including failures. No-face, unreadable,
or invalid-landmark cases cannot yield a valid aligned tensor; **no fabricated
fallback tensor is generated**. Failed images keep their originally assigned split;
later images do not silently replace enrolment images. Preparation processes all
images, writes failure counts, and exits nonzero if any fail. An unexpected detector
or filesystem error aborts with `complete: false`.

Use `--allow-partial` only after consciously accepting this limitation; evaluation
requires its own `--allow-partial` for datasets with preparation failures. Test
identities without any successful enrolment tensor always cause evaluation to fail.
Use a new output path when rerunning: existing datasets/results are never overwritten.

Read a tensor:

```python
import numpy as np

tensor = np.fromfile("data/tensors/train/Person_Name_0001.raw", dtype="<f4")
tensor = tensor.reshape(1, 112, 112, 3)
```

## 3. Evaluate MobileFaceNet

```bash
face-accuracy evaluate \
  --dataset data/tensors \
  --model /path/to/mobilefacenet.onnx \
  --output results/mobilefacenet \
  --device cpu \
  --target-fmr 0.01 0.001 0.0001
```

The model must have one float32 NHWC `[1,112,112,3]` input with the RGB normalization
above and an embedding output `[1,D]`. Symbolic dimensions are accepted but the
actual inference input is always batch one. If there are multiple outputs, select
the embedding with `--output-name NAME`. Zero/non-finite embeddings fail explicitly.
Pass a `.tflite` file directly as `--model` to evaluate it with LiteRT; no ONNX
model or format conversion is required. TFLite evaluation currently supports CPU
only.

```bash
face-accuracy evaluate \
  --dataset data/tensors \
  --model /path/to/mobilefacenet.tflite \
  --output results/mobilefacenet-tflite \
  --device cpu
```

The evaluator recursively finds `.raw` files in `train/` and `test/`. Filenames
determine identity; `unused/` is ignored. It checks byte size, value range,
duplicate image names across splits, and (when present) the preparation manifest.
Externally generated raw datasets may omit the manifest but must obey the same
directory/naming/tensor contract; their provenance and split correctness are the
caller's responsibility.

### Protocol and metrics

1. L2 normalize each enrolment embedding.
2. Average enrolment embeddings per identity, then L2 normalize the template.
3. L2 normalize each test/probe embedding.
4. Compute cosine similarity against **every** enrolled identity template.
5. A probe/template comparison is genuine for matching identity, impostor otherwise.
   Accept a match iff `score >= threshold`.

Reports include:

* **FMR**: fraction of impostor comparisons accepted.
* **FNMR**: fraction of genuine comparisons rejected.
* **TAR**: `1 - FNMR`.
* **EER**: equal error rate (interpreting the requested "ERR" as EER), interpolated
  between adjacent empirical ROC operating points at the crossing. Ties may mean
  this rate is not achievable with a single deterministic threshold; the threshold
  bracket and interpolation weight are reported.
* **ROC AUC**: exact empirical `P(genuine > impostor) + 0.5 * P(tie)`.
* **Rank-one accuracy**: closed-set identification, in addition to verification
  metrics. Top-score ties count as incorrect.
* TAR/FNMR and achieved FMR at each requested FMR, conservatively excluding boundary
  ties so achieved empirical FMR never exceeds its target.

This is a **custom enrolment-template vs probe protocol**, not LFW's official
6000-pair, ten-fold verification benchmark. Do not compare these numbers directly
to published LFW pair accuracy. Thresholds here are measured on the evaluation
set, not independently calibrated or cross-validated. Use a separate disjoint
calibration set before claiming deployed threshold performance.

Pair comparisons are correlated and heavily imbalanced; no confidence intervals
are implied. Reports include comparison counts and empirical FMR/FNMR resolution.
Targets below `1 / impostor_count` are flagged; zero observed false matches does
not establish a zero population FMR. Model training overlap with LFW, undetected
faces, and wrong face selection can bias results.

### Result files

* `metrics.json`: metrics, counts, protocol, model hash, and threshold caveats.
* `curves.csv`: threshold/FMR/FNMR/TAR for plotting.
* `roc-det.png`: ROC, low-FMR ROC, and normal-deviate DET plots. Zero/one endpoints
  are clipped only for log/probit display, not for reported metrics.
* `probes.csv`: per-probe identity, predicted identity, genuine and best scores.
* `embeddings.npz`: enrolment embeddings, templates, probes, identities, filenames.
* `genuine-scores.npy`, `impostor-scores.npy`: all comparison scores, **sorted
  ascending**, memory-mapped float32, not aligned with probe order.

Scoring uses chunks (`--chunk-size 128`) rather than holding the full score matrix
in memory. Scores are stored on disk and sorted in place for exact metrics. Disk
space is approximately `4 * test_images * enrolled_identities` bytes for scores,
plus embeddings, tensors and images. For 10,000 probes and 5,749 identities this
is about 230 MB of scores. Embeddings and a chunk-sized score matrix remain in RAM.
Metrics are computed on stored float32 cosine scores. Curve export samples impostor
order statistics (`--curve-points 2001`) and includes all genuine boundaries,
requested operating points and the EER crossing; **EER, AUC, and reported operating
points use all comparisons**, not a histogram or plotted-curve approximation.
When reproducing curve thresholds, promote stored scores to float64 before comparison;
some exported thresholds lie immediately above a float32 score to exclude ties.
Requested operating-point thresholds use the next representable float32 instead,
so they can also be applied directly to float32 scores without rounding back onto a tie.

Inference and evaluation are local. Raw face tensors and embeddings are sensitive;
keep datasets/results out of version control and apply appropriate access controls.

## Tests

```bash
python -m pytest
python -m ruff check .
```

Tests use synthetic images/models; no LFW download or real model weights are needed.
They verify alignment, RetinaFace anchor decoding, split/failure behavior, raw tensor
contracts, empirical metrics including ties, archive safety, and end-to-end ONNX
inference/report generation. Real-model CPU/GPU accuracy must be validated separately
with your supplied weights and detector export.

### Reference end-to-end run (2026-10-16)

Verified on real LFW (`lfw.tgz` above, all 13,233 images aligned, 0 failures) with
`retinaface_mv2.onnx` (yakhyo, prior-box) and InsightFace `w600k_mbf.onnx`
MobileFaceNet wrapped with an NHWC→NCHW `Transpose` to present `[1,112,112,3]`,
`--enrolment 1 --max-test 10` (5,749 templates, 4,883 probes):

| Metric | Value |
|---|---|
| EER | 0.348 % |
| TAR @ FMR 1e-2 / 1e-3 / 1e-4 | 99.73 % / 99.57 % / 99.37 % |
| ROC AUC | 0.99951 |
| Rank-1 identification | 98.89 % |

CPU (12 threads): ~18 min preparation, ~2 min evaluation. `--device gpu`
(onnxruntime-gpu 1.31, RTX A2000) gave identical metrics.
