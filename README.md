# Industrial Inspection

A from-scratch visual anomaly detection project for industrial part inspection. The goal is to learn from normal part images, identify defective parts and localize suspicious regions, with a reproducible evaluation and explicit failure analysis.

**Status:** initial implementation and data verification. No final model performance is claimed.

## Approach

Two compact, randomly initialized networks learn reconstruction and defect segmentation from procedurally corrupted normal images. This is a simplified DRAEM-inspired experiment, not a reproduction of the original method. Real defective images are reserved for evaluation. The starting category is `metal_nut`; `screw` and `transistor` follow with separate category checkpoints.

## Setup

Use Python 3.12 or newer in an isolated virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

The installed dependency versions are recorded in `requirements.lock.txt`.

For the same dependency versions, install `requirements.lock.txt` before the editable project. CPU and MPS availability depend on the host; the local environment uses Apple M4 with PyTorch MPS.

## Prepare data

The complete MVTec AD archive is approximately 5.3 GB compressed. Allow sufficient space for extraction, roughly 15 GB including the archive and environment. Raw data is excluded from Git.

```bash
python scripts/prepare_dataset.py
python -m inspection.data --root data/mvtec_ad --categories metal_nut \
  --output data/manifest.json --report data/audit.json \
  --archive-sha256 cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d
```

Training, synthetic validation and normal calibration use disjoint portions of the original normal training set. Test membership is unchanged. See [data provenance and license](docs/DATA.md).

## Train and evaluate

```bash
python -m inspection.train --manifest data/manifest.json \
  --config configs/metal_nut.json --output runs/metal_nut
python -m inspection.evaluate --manifest data/manifest.json \
  --checkpoint runs/metal_nut/checkpoint.pt --output runs/metal_nut/evaluation.json \
  --overlays runs/metal_nut/overlays --pixel-space original
```

The checkpoint includes the configuration, split digest and a decision threshold derived only from normal calibration images. Anomaly scores are not probabilities. A short pilot verifies the training loop; final accuracy requires the planned experiments and frozen evaluation.

## Inspect one photo

```bash
python -m inspection.predict --checkpoint runs/metal_nut/checkpoint.pt \
  --image /path/to/metal-nut-photo.png --output runs/inspection-example
```

The command exports the decision, score, threshold, checkpoint identity, a raw activation map and an overlay at the photo's original dimensions. The checkpoint specifies the part category; the model does not automatically recognize arbitrary part types. Warm inference timing excludes file loading and preprocessing.

See [the implementation plan](docs/PLAN.md) for staged acceptance gates and agent responsibilities, and [the experiment protocol](docs/EXPERIMENTS.md) for comparisons and interpretation.

## References

- [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection](https://arxiv.org/abs/2108.07610)

MVTec AD data and exported derivatives are subject to CC BY-NC-SA 4.0. This project does not establish production inspection performance or a novel research method.
