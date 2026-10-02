# Model card

## Intended use

Research and educational demonstration of visual industrial-part inspection. A user selects the matching part category, supplies an image consistent with the dataset imaging conditions, and reviews an anomaly score, frozen-threshold decision and spatial activation overlay. The model is not a general object recognizer or a production acceptance authority.

## Current demo candidate

The registry `artifacts/models.json` is the authoritative source for the selected demo checkpoint, SHA-256, category, examples and measured performance. Before completion of the controlled study it identifies the exploratory metal-nut foreground pilot; it is explicitly marked as not meeting the target.

The pilot is a joint reconstruction/segmentation network with 977, 876 parameters, trained from random initialization at 256×256 for 100 epochs. It sees only 154 original normal training images with procedural corruptions. Model selection uses separate synthetic validation images; the legacy decision threshold is the 95 th percentile of 33 separate normal calibration scores. The original metal-nut test was examined during development, making its results exploratory.

At the frozen legacy threshold it detects 51/93 defects with 1/22 normal false alarms. Native pixel AP is 0.148435 and image AUROC is 0.769306. These are separate metrics; none is a probability of correctness on an arbitrary upload.

## Learning tracks

| Track | Learned parameters | Training supervision | Interpretation |
| --- | --- | --- | --- |
| MVTec joint / segmentation | Random initialization | Normal photographs + synthetic corruption masks | Anomaly detection under the documented synthetic prior |
| Normal reconstruction | Random initialization | Normal photographs with noise | Reconstruction-error reference |
| KolektorSDD2 supervised | Random initialization | Real defective/normal images and real pixel masks | Separate supervised surface-inspection task |
| PatchCore reference | ImageNet-pretrained features | Normal training feature memory | Pretrained comparison, not our from-scratch model |

## Model selection and calibration

Protocol v2 uses a frozen shared synthetic bank for normal-only candidate comparison. Configuration and seed selection never use real test labels. Three training seeds expose optimization variability. The selected checkpoint's normal calibration scores determine the 90 th-percentile threshold. Small calibration sets do not establish a guaranteed 10% false-alarm rate.

## Inputs, outputs and display

RGB/gray/RGBA inputs are converted to oriented RGB. MVTec uses square bilinear resizing; KolektorSDD2 uses aspect-preserving letterboxing to 640×256. Padding is excluded from scoring, supervised loss and native restoration. Invalid or oversized uploads are rejected.

Scores are uncalibrated activations (or raw reconstruction errors). The main model score averages the strongest 1% of valid image pixels. Overlay colors use a fixed per-checkpoint display range, separate from the decision threshold. Float 32 maps are retained; an 8-bit preview is not used to calculate metrics.

## Evaluation limits

Original-resolution masks are unchanged during pixel evaluation. Maps are restored to native dimensions; interpolation cannot recover image detail lost during resizing. Reported intervals use image-level bootstrap and Wilson intervals, not independent-pixel assumptions. They do not capture camera/domain shift. KSDD2 does not supply physical product or acquisition-group identifiers, so image-level partitioning does not prove product/batch independence.

See the experiment report for error groups, finite-sample uncertainty and failing defect types. Strong behavior on one category does not establish support for another.

## Reproducibility and distribution

Checkpoints carry model kind, preprocessing, scoring, calibration, data-split identity and source/environment provenance. Training checkpoints also retain optimizer/random states for exact CPU resume in the same tested environment. Accelerator kernels may vary. Public demo users cannot submit checkpoint files; release weights are project-owned and SHA-256 verified.

Source and dataset licenses are documented separately in the README and data card. Uploaded images are processed in memory and are not saved by this application.
