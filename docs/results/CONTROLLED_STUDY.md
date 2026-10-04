# Controlled study: frozen selections and measured outcomes

This report was generated only after every declared training, evaluation and conditional fallback step completed and its evidence passed consistency checks. Metal-nut measurements remain **exploratory**. Screw and transistor use frozen held-out tests after all MVTec weights and model/seed selections were fixed. No test-winning seed replaces a validation-selected seed.

## Candidate ablations and validation-only selection

Every candidate uses random initialization. The common canonical synthetic validation bank ranks candidates by the harmonic mean of image AUROC and pixel AP; the normal-only reconstruction baseline is reported but excluded from main-model eligibility. Ties within 1e-12 use predeclared CPU median latency, then checkpoint SHA-256. These are controlled protocol comparisons, with capacity differences disclosed by actual parameter counts; synthetic validation does not establish transfer to real defects.

| Candidate | Method | Input | Actual parameters | Selected epoch | Validation H | Eligible | Exploratory test AUROC / native AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| protocol-v2-joint-256-seed42 | joint_reconstruction_segmentation | 256 | 977,876 | 100 | 0.935432 | yes | 0.7693 / 0.1484 |
| protocol-v2-joint-128-seed42 | joint_reconstruction_segmentation | 128 | 977,876 | 75 | 0.777927 | yes | 0.7546 / 0.2537 |
| protocol-v2-unrestricted-256-seed42 | joint_reconstruction_segmentation | 256 | 977,876 | 75 | 0.921755 | yes | 0.7356 / 0.1773 |
| protocol-v2-no-scratch-256-seed42 | joint_reconstruction_segmentation | 256 | 977,876 | 65 | 0.920943 | yes | 0.7385 / 0.2610 |
| protocol-v2-segmentation-256-seed42 | segmentation_only | 256 | 488,705 | 90 | 0.934315 | yes | 0.8138 / 0.2154 |
| protocol-v2-reconstruction-256-seed42 | normal_only_denoising_reconstruction | 256 | 488,739 | 100 | 0.803272 | no | 0.4721 / 0.2113 |

The main configuration was selected as `protocol-v2-joint-256-seed42` using validation H=0.935432. Its later test outcomes did not choose this configuration. Deployment seeds use the same validation-only rule separately within each category. Full hashes, ties, size/type measurements and all-seed uncertainty records are in [controlled-study.json](controlled-study.json).

## MVTec: all training seeds and frozen operating thresholds

| Category | Seed | Pretest selected | Image AUROC [95% image CI] | Native pixel AP | Defects detected | Good parts falsely flagged | Recall / FPR | Target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metal_nut | 42 | no | 0.7693 [0.674, 0.853] | 0.1484 | 53/93 | 2/22 | 57.0% [0.468, 0.666] / 9.1% [0.025, 0.278] | **failed** |
| metal_nut | 43 | no | 0.7884 [0.705, 0.864] | 0.2030 | 58/93 | 2/22 | 62.4% [0.522, 0.715] / 9.1% [0.025, 0.278] | **failed** |
| metal_nut | 44 | yes | 0.7278 [0.607, 0.847] | 0.1672 | 36/93 | 3/22 | 38.7% [0.294, 0.489] / 13.6% [0.047, 0.333] | **failed** |
| screw | 42 | yes | 0.8660 [0.793, 0.925] | 0.1470 | 50/119 | 2/41 | 42.0% [0.335, 0.510] / 4.9% [0.013, 0.161] | **failed** |
| screw | 43 | no | 0.8461 [0.770, 0.911] | 0.0885 | 61/119 | 4/41 | 51.3% [0.424, 0.601] / 9.8% [0.039, 0.225] | **failed** |
| screw | 44 | no | 0.7210 [0.629, 0.810] | 0.0367 | 73/119 | 7/41 | 61.3% [0.524, 0.696] / 17.1% [0.085, 0.313] | **failed** |
| transistor | 42 | yes | 0.4408 [0.320, 0.566] | 0.0817 | 8/40 | 6/60 | 20.0% [0.105, 0.348] / 10.0% [0.047, 0.201] | **failed** |
| transistor | 43 | no | 0.5650 [0.434, 0.678] | 0.0789 | 10/40 | 7/60 | 25.0% [0.142, 0.402] / 11.7% [0.058, 0.222] | **failed** |
| transistor | 44 | no | 0.6138 [0.488, 0.734] | 0.0904 | 12/40 | 5/60 | 30.0% [0.181, 0.454] / 8.3% [0.036, 0.181] | **failed** |

The target requires recall ≥90% **and** normal false alarms ≤10%, using each checkpoint’s unchanged 90th-percentile normal-calibration threshold. Nonselected seeds remain descriptive evidence; they cannot rescue a failed selected checkpoint. Test point estimates do not establish factory reliability. Defective prevalence and the small normal sample affect precision and false-alarm uncertainty.

## Training-seed variation and image uncertainty

Each row reports the mean ± sample standard deviation across seeds 42/43/44. This is descriptive training variability, not a confidence interval; image-bootstrap and Wilson rate intervals in the JSON instead describe finite-image sampling uncertainty.

| Category | AUROC | Native pixel AP | Defect recall | Normal false-alarm rate |
| --- | --- | --- | --- | --- |
| metal_nut | 0.762 ± 0.031 | 0.173 ± 0.028 | 0.527 ± 0.124 | 0.106 ± 0.026 |
| screw | 0.811 ± 0.079 | 0.091 ± 0.055 | 0.515 ± 0.097 | 0.106 ± 0.061 |
| transistor | 0.540 ± 0.089 | 0.084 ± 0.006 | 0.250 ± 0.050 | 0.100 ± 0.017 |
| kolektor_surface | 0.972 ± 0.005 | 0.789 ± 0.031 | 0.948 ± 0.005 | 0.094 ± 0.008 |

## Native defect size and type: validation-selected checkpoints

| Category / annotated type | Detected / defective images | Recall |
| --- | --- | --- |
| metal_nut/bent | 13/25 | 52.0% |
| metal_nut/color | 11/22 | 50.0% |
| metal_nut/flip | 10/23 | 43.5% |
| metal_nut/scratch | 2/23 | 8.7% |
| screw/manipulated_front | 3/24 | 12.5% |
| screw/scratch_head | 13/24 | 54.2% |
| screw/scratch_neck | 12/25 | 48.0% |
| screw/thread_side | 7/23 | 30.4% |
| screw/thread_top | 15/23 | 65.2% |
| transistor/bent_lead | 2/10 | 20.0% |
| transistor/cut_lead | 1/10 | 10.0% |
| transistor/damaged_case | 2/10 | 20.0% |
| transistor/misplaced | 3/10 | 30.0% |
| kolektor_surface/defective | 105/110 | 95.5% |

| Category / native-mask area bin | Detected / defective images | Recall [95% Wilson interval] |
| --- | --- | --- |
| metal_nut / tiny_below_0.1_percent | 0/0 | n/a n/a |
| metal_nut / small_0.1_to_1_percent | 0/7 | 0.000 [0.000, 0.354] |
| metal_nut / medium_1_to_5_percent | 24/50 | 0.480 [0.348, 0.615] |
| metal_nut / large_at_least_5_percent | 12/36 | 0.333 [0.202, 0.497] |
| screw / tiny_below_0.1_percent | 1/1 | 1.000 [0.207, 1.000] |
| screw / small_0.1_to_1_percent | 48/117 | 0.410 [0.325, 0.501] |
| screw / medium_1_to_5_percent | 1/1 | 1.000 [0.207, 1.000] |
| screw / large_at_least_5_percent | 0/0 | n/a n/a |
| transistor / tiny_below_0.1_percent | 0/0 | n/a n/a |
| transistor / small_0.1_to_1_percent | 0/6 | 0.000 [0.000, 0.390] |
| transistor / medium_1_to_5_percent | 3/21 | 0.143 [0.050, 0.346] |
| transistor / large_at_least_5_percent | 5/13 | 0.385 [0.177, 0.645] |
| kolektor_surface / tiny_below_0.1_percent | 0/1 | 0.000 [0.000, 0.793] |
| kolektor_surface / small_0.1_to_1_percent | 33/37 | 0.892 [0.753, 0.957] |
| kolektor_surface / medium_1_to_5_percent | 53/53 | 1.000 [0.932, 1.000] |
| kolektor_surface / large_at_least_5_percent | 19/19 | 1.000 [0.832, 1.000] |

Defect-size analyses use original annotation areas. The JSON retains every area bin and its recall interval; zero-image bins carry no recall estimate. A small defect can lose visual evidence during input resizing even though its original mask remains intact for evaluation.

## Paired perturbations and timing

| Category | Perturbation | Recall | False alarms | Decisions changed |
| --- | --- | --- | --- | --- |
| metal_nut | original | 38.7% | 13.6% | 0 |
| metal_nut | gaussian_blur_radius_1 | 16.1% | 0.0% | 30 |
| metal_nut | brightness_0.8 | 33.3% | 9.1% | 12 |
| metal_nut | brightness_1.2 | 43.0% | 9.1% | 13 |
| metal_nut | jpeg_quality_60 | 50.5% | 31.8% | 25 |
| screw | original | 42.0% | 4.9% | 0 |
| screw | gaussian_blur_radius_1 | 25.2% | 2.4% | 27 |
| screw | brightness_0.8 | 21.0% | 2.4% | 28 |
| screw | brightness_1.2 | 65.5% | 17.1% | 39 |
| screw | jpeg_quality_60 | 33.6% | 2.4% | 13 |
| transistor | original | 20.0% | 10.0% | 0 |
| transistor | gaussian_blur_radius_1 | 7.5% | 6.7% | 15 |
| transistor | brightness_0.8 | 45.0% | 43.3% | 34 |
| transistor | brightness_1.2 | 10.0% | 3.3% | 12 |
| transistor | jpeg_quality_60 | 15.0% | 11.7% | 15 |
| kolektor_surface | original | 95.5% | 10.0% | 0 |
| kolektor_surface | gaussian_blur_radius_1 | 94.5% | 8.5% | 38 |
| kolektor_surface | brightness_0.8 | 92.7% | 8.6% | 75 |
| kolektor_surface | brightness_1.2 | 93.6% | 15.5% | 70 |
| kolektor_surface | jpeg_quality_60 | 96.4% | 31.4% | 243 |

The same test images receive predeclared blur, brightness and JPEG perturbations, with no threshold adjustment or retraining. These paired image-decision checks do not establish generalization to new cameras; pixel AP is not computed for the stress runs.

| Category | Device | Warm median / p95 (ms) | Measurements |
| --- | --- | --- | --- |
| metal_nut | cpu | 108.90 / 138.26 | 32 |
| metal_nut | mps | 13.32 / 19.40 | 32 |
| screw | cpu | 124.04 / 239.67 | 32 |
| screw | mps | 19.47 / 24.85 | 32 |
| transistor | cpu | 103.61 / 118.38 | 32 |
| transistor | mps | 11.26 / 12.98 | 32 |
| kolektor_surface | cpu | 98.94 / 100.71 | 32 |
| kolektor_surface | mps | 13.74 / 14.39 | 32 |

Timing uses the selected frozen model and a normal calibration image after scheduler serialization. The benchmark cannot enforce external system idleness. These warm forward-and-score measurements exclude checkpoint loading, decoding, preprocessing, transfers, rendering and service overhead.

## Conditional KSDD2 track

The predeclared fallback ran because no validation-selected MVTec category met both targets. KSDD2 trains on **real labeled defects** and selects a model/seed by real validation metrics. Its results belong to a separate supervised task; they are not an anomaly-detection ablation or evidence of MVTec transfer. Dataset choice was conditional on the frozen MVTec test target results, while KSDD2 model and seed selection remained validation-only.

| Category | Seed | Pretest selected | Image AUROC [95% image CI] | Native pixel AP | Defects detected | Good parts falsely flagged | Recall / FPR | Target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KSDD2 | 42 | no | 0.9731 [0.947, 0.992] | 0.7958 | 104/110 | 75/894 | 94.5% [0.886, 0.975] / 8.4% [0.067, 0.104] | met |
| KSDD2 | 43 | no | 0.9660 [0.939, 0.987] | 0.7551 | 104/110 | 87/894 | 94.5% [0.886, 0.975] / 9.7% [0.080, 0.119] | met |
| KSDD2 | 44 | yes | 0.9754 [0.953, 0.992] | 0.8155 | 105/110 | 89/894 | 95.5% [0.898, 0.980] / 10.0% [0.082, 0.121] | met |

## Selected galleries, attribution and provenance

Galleries retain existing deterministic FP/FN/TP/TN selections. Original images, fixed-scale predicted activation and ground truth occupy separate panels; activations are not calibrated probabilities or pixel decisions. No original dataset images were reopened to build this report.

- [metal_nut gallery index](controlled-study-galleries/metal_nut/index.json) and [CC BY-NC-SA 4.0 attribution](controlled-study-galleries/metal_nut/ATTRIBUTION.md).
- [screw gallery index](controlled-study-galleries/screw/index.json) and [CC BY-NC-SA 4.0 attribution](controlled-study-galleries/screw/ATTRIBUTION.md).
- [transistor gallery index](controlled-study-galleries/transistor/index.json) and [CC BY-NC-SA 4.0 attribution](controlled-study-galleries/transistor/ATTRIBUTION.md).
- [kolektor_surface gallery index](controlled-study-galleries/kolektor_surface/index.json) and [CC BY-NC-SA 4.0 attribution](controlled-study-galleries/kolektor_surface/ATTRIBUTION.md).

Public evidence uses relative paths and SHA-256 checksums. Checkpoints and raw datasets stay outside the source repository. All limitations, failed targets and nonselected seeds remain in the record; this report does not change the released demo model or claim production readiness.
