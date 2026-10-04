# Model card

## Scope and measured outcome

Research and educational demonstration of industrial visual inspection. A user selects a supported task and reviews a raw score, frozen-threshold decision and spatial activation map. The models do not recognize arbitrary objects, and a passing dataset result does not authorize automated production acceptance.

The completed `study-v2` trained 14 normal-only MVTec models and three real-defect-supervised KolektorSDD2 models. All start from random weights. The selected KolektorSDD2 model meets the project's **dataset point-estimate** target; all selected MVTec models fail it. These are separate tasks, with separate category-specific weights.

| Selected model | Training / selection | Recall | Normal false alarms | Native pixel AP | Outcome |
| --- | --- | --- | --- | --- | --- |
| KolektorSDD2, seed 44, epoch 60 | Real defects and masks; real validation | 105/110, 95.45% | 89/894, 9.955% | 0.815463 | Target met as point estimates |
| Metal nut, seed 44, epoch 95 | Normal images + synthetic defects; synthetic validation | 36/93, 38.7% | 3/22, 13.6% | 0.167204 | Not met; exploratory |
| Screw, seed 42, epoch 80 | Normal images + synthetic defects; synthetic validation | 50/119, 42.0% | 2/41, 4.9% | 0.147046 | Not met; frozen held-out |
| Transistor, seed 42, epoch 90 | Normal images + synthetic defects; synthetic validation | 8/40, 20.0% | 6/60, 10.0% | 0.081671 | Not met; frozen held-out |

See [the complete controlled study](results/CONTROLLED_STUDY.md) and [portable evidence](results/controlled-study.json). Full category summaries preserve exact hashes, provenance, thresholds, metrics and training-seed variation: [surface](model-cards/kolektor_surface.json), [metal nut](model-cards/metal_nut.json), [screw](model-cards/screw.json), [transistor](model-cards/transistor.json).

These are the published v0.1.0 models. The completed acquisition-robustness follow-up below contains experimental weights only; it does not replace this table, the active registry or the historical measurements.

## Selected surface model

The surface model is a single U-Net with **488,705 parameters** and 16 base channels, trained from random initialization using Adam at learning rate 0.0003, batch size 8 and positive-weighted BCE (weight 3) plus Dice. Batches sample four normal and four defective images with replacement. This initial configuration has no geometric or appearance augmentation.

Images preserve aspect ratio within a **640-high × 256-wide** letterbox. Padding is excluded from loss, score, validation pixel AP and restored maps. The image score averages the strongest 1% of valid sigmoid segmentation activations; it is not a calibrated probability. The fixed threshold is **2.276815479262951e-7**, with `score >= threshold` classified as defective. It is the linearly interpolated 90th percentile of 313 separate normal calibration scores.

The official training partition supplies 1,631 training images (1,459 normal / 172 defective), 350 validation images (313 / 37), and 350 calibration images (313 / 37). Positive calibration images remain unused. Duplicate handling, archive identity and the fixed seed-42 split are recorded in the [data documentation](DATA.md). The official 1,004-image test remains outside all three partitions.

Training seeds 42, 43 and 44 share the same split and settings. Early stopping uses 15 epochs without an improvement, within a 100-epoch maximum. Configuration/checkpoint/seed selection uses only the harmonic mean of validation image AUROC and model-resolution pixel AP, with the declared tie-breaks. Seed 44's epoch-60 checkpoint was frozen before official test prediction; a test-winning seed was not selected afterward.

Checkpoint SHA-256: `0c2222771ac7b9719eede442f20f120d81790eba1e20b435b706ebc142f0eed8`.

### Test results and uncertainty

| Measurement | Estimate | 95% interval where available |
| --- | --- | --- |
| Image AUROC | 0.975432 | 0.952732–0.992394 |
| Image average precision | 0.921661 | Not estimated |
| Native pixel average precision | 0.815463 | Not estimated |
| Defect recall | 105/110 = 95.45% | 89.80–98.04% |
| Normal false-alarm rate | 89/894 = 9.955% | 8.16–12.09% |
| Alert precision at test prevalence | 105/194 = 54.12% | 47.10–60.99% |
| Confusion counts | TP 105, FN 5, FP 89, TN 805 | — |

AUROC uncertainty uses 1,000 stratified image-bootstrap samples with seed 42; rate intervals use Wilson scores. The observed FPR is just below 10%, and its upper interval exceeds 10%. The lower recall interval is below 90%. The passing point estimates therefore do not establish either target as a population guarantee.

Across all three training seeds, recall is 94.85% ± 0.52 percentage points and FPR is 9.36% ± 0.85 percentage points (mean ± sample standard deviation). These summaries describe training variation, not confidence intervals, and do not replace the validation-selected model's result.

All five false negatives have native annotated area below 1%: four of 37 images in the 0.1–1% bin, and the sole image below 0.1%. Smaller defects remain a limitation despite the stronger overall localization metric.

### Robustness and timing

| Paired test condition | Defect recall | Normal false alarms |
| --- | ---: | ---: |
| Original | 95.45% | 9.955% |
| Gaussian blur, radius 1 | 94.5% | 8.5% |
| Brightness ×0.8 | 92.7% | 8.6% |
| Brightness ×1.2 | 93.6% | 15.5% |
| JPEG quality 60 | 96.4% | 31.4% |

The same test images undergo these predeclared perturbations without retraining or recalibration. Brighter and compressed images violate the FPR target. Pixel AP is not calculated for stress runs, and these checks do not substitute for new camera or production-batch data.

Local Apple M4 warm forward-and-score latency is 98.94 ms median / 100.71 ms p95 on CPU, and 13.74 ms / 14.39 ms on MPS, over 32 measurements per device. Loading, image decoding, preparation, device transfer, rendering and network/service overhead are excluded. External system idleness cannot be enforced by the experiment scheduler.

## Normal-only MVTec models and reference

The selected joint reconstruction/segmentation model has **977,876 parameters**, base width 16 and 256×256 square input. Category training uses only normal photographs and procedural corruptions with known masks. A fixed synthetic bank ranks five eligible main-model configurations; the reconstruction-only model is a separate baseline. The chosen joint configuration is repeated on three seeds per category.

All selected MVTec categories fail the target. Synthetic validation is a proxy that does not reliably rank performance on real anomalies. The selected metal-nut model catches only 2/23 scratches; the screw model catches 3/24 manipulated-front defects, and transistor recall is low across all four defect types. Their complete failures and nonselected seeds stay in the evidence.

The [PatchCore reference](results/PATCHCORE_REFERENCE.md) uses ImageNet-pretrained features and a normal-image feature memory. It is not one of our randomly initialized networks and does not supply their weights. The earlier metal-nut models retain their original thresholds and [exploratory results](results/EXPERIMENT_REPORT.md), including the alpha demo's 51/93 defect detections and 1/22 false alarms.

## Experimental acquisition-robustness models

The [completed two-run follow-up](results/KSDD2_ROBUSTNESS_RESULTS.md) compares a fresh seed-42 control with acquisition augmentation under the same supervised training budget and frozen splits. Control selected epoch 30 and stopped at 45; augmentation selected epoch 35 and stopped at 50. Augmentation passed the validation-only gate, increasing pooled image-AUROC/pixel-AP harmonic mean from 0.823098 to 0.848836. Its paired validation AUROC difference has a 95% interval of **−0.000708 to +0.016731**, crossing zero. This interval is descriptive, not an independent significance claim or an interval for pixel AP.

| Experimental model | Recall | Normal false alarms | Native pixel AP | Original-test point target |
| --- | --- | --- | --- | --- |
| Fresh control | 104/110 = 94.545% | 98/894 = 10.962% | 0.800701 | Not met |
| Acquisition augmentation | 106/110 = 96.364% | 115/894 = 12.864% | 0.811400 | Not met |

Both use real defect masks, random initialization and seed 42; this comparison does not measure training-seed variation. Each threshold was calibrated once from the same 313 clean normal calibration sources, after the two checkpoints and validation gate were frozen. The thresholds are **4.15221046523584e-7** for control and **1.857622180523322e-7** for augmentation. No test outcome changed them.

Passing validation did not satisfy the joint quality target: both original-test false-alarm rates exceed 10%. Augmentation reduced brightness ×1.2 false alarms from **207/894 to 131/894**, but increased JPEG-quality-60 false alarms from **153/894 to 185/894**. All four augmented-model misses occupy less than 1% of the native image; the sole sub-0.1% defect remains missed. The [full report](results/KSDD2_ROBUSTNESS_RESULTS.md) retains all five conditions, Wilson/bootstrap intervals, image AP, defect-area groups and CPU/MPS timings. [Portable evidence](results/KSDD2_ROBUSTNESS_RESULTS.json) and [24 attributed examples](results/KSDD2_ROBUSTNESS_RESULTS.md#auditable-examples-and-errors) preserve both errors and correct decisions.

All these reused-test results are **development-inspected / exploratory**, because earlier surface-test failures motivated the intervention. They neither establish new independent reliability nor improve the normal-only MVTec result. Neither weight is promoted; v0.1.0 remains deployed. A separate [exactly two-run learned image-decision experiment](KSDD2_DECISION_SCREEN.md) is running under a frozen protocol after 180 local tests, separate review and passing Linux CI. Its [launch record](results/KSDD2_DECISION_LAUNCH.json) verifies execution; no quality result exists yet. Its direct classification objective, a decision head fed detached features, and low-false-alarm partial-AUROC selection are a project adaptation inspired by the official ViCoS mixed segmentation/decision approach, not a reproduction.

## Evaluation discipline and limits

The KolektorSDD2 fallback was triggered by frozen MVTec target outcomes. It is a separate scenario with real-defect supervision, not an improvement measurement on MVTec or a fair direct supervision-matched comparison. Model and seed selection within the fallback stayed validation-only.

Metal-nut tests were inspected during development and remain exploratory. Screw, transistor and KolektorSDD2 were evaluated after the study's methods, weights, selections and thresholds were frozen. Any future method changes informed by these results must treat reused tests as development-inspected/exploratory; fresh independent success claims require an untouched holdout. Dataset providers do not supply all physical product/acquisition-group identities, so pixel-duplicate checks cannot prove production-batch independence.

The subsequent [acquisition-robustness screen](KSDD2_ROBUSTNESS_SCREEN.md) was motivated by these surface-test failures and a validation-only diagnostic. Its completed reused-test evaluation is exploratory. This does not alter the historical frozen v0.1.0 measurements or the current deployed weights.

Original-resolution ground-truth masks are preserved for final pixel AP; predictions are restored to native geometry. Resizing can remove detail that interpolation cannot recover. Image-bootstrap and Wilson intervals assume image-level sampling and do not quantify domain shift or independent-pixel uncertainty. Precision depends on prevalence and will change outside this test mixture.

## Inference, distribution and privacy

The registry `artifacts/models.json` is authoritative for the active demo's checkpoint identity, checksum, examples and results. The model summaries here document the completed study; [deployment verification](DEPLOYMENT.md) records the separately verified hosted state.

RGB, gray and RGBA inputs are converted to oriented RGB. Invalid or oversized uploads are rejected. CLI, evaluation and demo share preprocessing and scoring. The hosted supervised preview uses a fixed 0–1 activation scale; the original checkpoint scale remains recorded and normal-only previews retain it. This display-only change does not alter weights, scores, thresholds or raw float32 maps, which are exported separately from 8-bit previews. Scores this small should be read in scientific notation or directly from JSON, not inferred from rounded gallery headings.

Checkpoints carry model kind, preprocessing, score definition, calibration, split identity and source/environment provenance. Training snapshots also retain optimizer/random states. Exact CPU resume is tested in the same environment; accelerator kernels may differ. Users cannot upload executable checkpoint files, and project release weights are SHA-256 verified.

The application processes uploaded images in memory without saving them. Source, dataset-derived artifacts and third-party code have separate licenses, documented in the README and attribution files. Dataset conditions include noncommercial and share-alike terms.
