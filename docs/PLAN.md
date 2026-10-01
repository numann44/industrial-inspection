# Implementation and experiment plan

## Objective

Train a randomly initialized, compact visual anomaly detector that produces a defect score, an empirically calibrated good/defective decision and a spatial anomaly map. Begin with metal nuts, then expand to screws and transistors after the first category is understood. These categories require separate checkpoints initially; this is not yet a universal part recognizer.

## Ownership

| Workstream | Owner | Deliverable |
| --- | --- | --- |
| Authoritative data access | Data audit agent | Source, working download, checksum, counts and license verification |
| Data integrity and evaluation | Data/evaluation agent | Decode and mask audit, immutable splits, metrics, error groups and visual outputs |
| Model and training | Model/training agent | Compact reconstruction and segmentation networks, procedural defects, training, calibration and checkpoints |
| Integration and experiment review | Main agent with Numan | Environment, reproducibility, executed checks, interpretation and public repository preparation |

## Milestones and acceptance gates

1. **Data ready.** Download the complete archive, match the upstream SHA-256, safely extract every category, decode every image and mask, and compare actual counts to the official paper. Resolve missing masks or cross-split duplicates before training. Archive and raw images stay outside Git.
2. **Training pipeline works.** Start at 128 pixels and 16 base channels. Verify real gradient updates, finite losses, synthetic image/mask alignment, deterministic splits and independent calibration data. Run a short MPS pilot and record measured epoch time. A pilot checkpoint is not a final quality claim.
3. **First model trained.** Train on normal metal-nut images with procedural corruptions. Select a checkpoint using fixed synthetic validation examples only. Set the decision threshold using the separate normal calibration subset. Save configuration, split digest, logs, device, elapsed time and checkpoint together.
4. **Experiments completed.** Compare the trained detector to a normal-image reconstruction baseline and a pretrained feature baseline. Predeclare a small set of comparisons: patch-only versus patch-plus-scratch synthesis, segmentation with versus without reconstruction, and 128 versus 256 resolution. Hold the data split and evaluation procedure fixed. Use three training seeds for shortlisted configurations if compute permits. Synthetic validation chooses settings; final real-defect test results do not choose them.
5. **Final evaluation released.** Evaluate each frozen configuration on original, untouched test sets. Report image AUROC, pixel AP, precision/recall and confusion counts at the preselected threshold, false alarms, recall by defect group, and synchronized warm inference latency. Add uncertainty estimates where sample sizes permit and explicitly identify small groups. If a pilot's test set is inspected during development, disclose it as exploratory; use an untouched category or other external set for an additional final generalization check.
6. **Usable inspection demo.** Allow category selection and image upload; show the original image, anomaly overlay, score, threshold, good/defective decision and inference time. The score is not a calibrated probability. Export useful inspection results with checkpoint identity.
7. **GitHub portfolio release.** Publish English engineering documentation, model/data provenance, reproducible commands, real experiment tables, failure analysis and a brief demo. Do not claim deployment validation, state-of-the-art performance or a new research algorithm without evidence.

## Data protocol

Original `train/good` images are split deterministically into 70% training, 15% synthetic validation and 15% normal calibration. Original test membership is preserved. Test masks are available only for evaluation. For metal nuts this should produce 154 training, 33 validation and 33 calibration images; test should contain 22 good and 93 defective images. Check these against actual files.

Calibration uses the empirical 95th percentile of normal-image scores. With only 33 normal samples, this does not guarantee a 5% future false-alarm rate. Report the quantile rule, sample count and observed test behavior. Do not tune it on test labels.

Early engineering pilots may be evaluated for debugging, but they are exploratory runs. Keep this distinction in saved reports and avoid presenting repeated test inspection as a blind final benchmark.

## Method and scope

The starting model is inspired by DRAEM, not a reproduction. A compact randomly initialized reconstruction network repairs procedurally corrupted normal images. A second network segments corruption from the corrupted image and reconstruction. We initially use reconstruction L1 and segmentation BCE/Dice losses. External texture datasets and pretrained weights are not required for this model.

Procedural patches and scratches can be unrepresentative of real defects. Investigate background shortcuts, foreground coverage, small-defect loss during resizing and overly broad anomaly maps. Include foreground-constrained versus unrestricted synthesis as a predeclared later experiment. Better synthesis and higher resolution are experimental changes, not guaranteed improvements.

Final pixel evaluation uses original-resolution ground-truth masks and bilinearly upsampled model maps, so resolution comparisons keep the same target labels. Report masks that vanish when downsampled to the model input resolution. The split/content digest is independent of machine paths and input resolution; audited file hashes are rechecked before training and evaluation on only the splits each operation uses.

## Sources

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Official paper and dataset counts](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf)
- [Maintained download URL and checksum](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)
- [DRAEM paper](https://arxiv.org/abs/2108.07610)
- [DRAEM implementation](https://github.com/VitjanZ/DRAEM)
