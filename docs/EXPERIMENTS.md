# Experiment protocol v2

## Question and target

Can a compact, randomly initialized model identify defective industrial images and locate suspicious regions using normal training photographs plus procedural corruptions? The engineering target is **defect recall ≥0.90 and normal false-alarm rate ≤0.10**, assessed separately for each supported category. A separate labeled surface-defect track is available if the normal-only study fails to meet this target. This is a measured target, not a promised outcome.

## Existing evidence

The three legacy metal-nut runs are exploratory. Their original thresholds and weights are retained. The foreground run changes resolution, learning rate, duration, normal probability and corruption families together; it cannot isolate a causal foreground effect. The legacy reconstruction comparison also differs in architecture, supervision and budget. See [measured results](results/EXPERIMENT_REPORT.md).

Metal-nut test outcomes have already informed development. Screw and transistor predictions remain sealed until configurations and deployment checkpoints are selected without their test labels. Category-specific retraining is not cross-category inference by a single model.

## Data and common selection bank

- Preserve official MVTec train/test membership. Split each category's original normal training images into 70% training, 15% validation and 15% calibration using data seed 42.
- Keep this split unchanged across training seeds 42, 43, 44. Archive hashes and portable split hashes accompany every run.
- Build a canonical 256-pixel synthetic validation bank exclusively from original validation normals. For each source save one clean case, three legacy corruptions and five foreground corruptions. Float 32 images and binary masks are immutable and hash verified.
- Metal-nut bank:33 source images, 297 cases, bank SHA256 `032a8148af58b23ca2f2757ab80c075782cae4f18606968fac3980cfe22f5aaf`.
- A 128-pixel candidate sees resized versions of the same bank; predictions are restored to the same canonical 256 masks for comparison.
- Rank main candidates by the harmonic mean of synthetic image AUROC and pixel AP. Within a run evaluate the bank every five epochs and at the final epoch. Cross-candidate exact ties use CPU inference time under the same measurement conditions, followed by stable configuration identity if still equal.
- Synthetic success is a model-selection proxy, not evidence of real-defect recall.

## Bounded initial matrix

All runs use 100 epochs, 16 base channels, batch 8, Adam learning rate 0.0003, normal probability 0.25 and training seed 42 unless that field is the comparison under study. Normal-image caching preserves decoded tensors; CPU thread count 1 and all code/environment identities are saved.

| Configuration | Controlled change | Interpretation |
| --- | --- | --- |
| Joint 256 | Reference | Reconstruction plus segmentation; foreground-restricted synthesis |
| Joint 128 | Input resolution | Same original bank masks and data partitions |
| Unrestricted 256 | Placement restriction off | Same implemented corruption families; report actual corrupted area distributions |
| No-scratch 256 | Scratch family disabled | Remaining families renormalized |
| Segmentation 256 | Reconstruction network removed | Architecture comparison; changed parameter count disclosed |
| Reconstruction 256 | Normal denoising reference | Same split, resolution and epoch budget; different supervision and capacity |

Select the main configuration among the five joint/segmentation variants. The reconstruction and pretrained baselines are reported separately. Repeat the chosen configuration with seeds 43 and 44, retaining seed 42. Select a deployment seed using validation evidence before test evaluation; report results for all seeds, not just the selected one.

Apply the frozen selected configuration to screws and transistors with separate category training, validation banks and calibration subsets. Changes based on category normal validation are documented before opening its test predictions.

## Threshold policy

For v2, use the empirical 90 th percentile of a frozen checkpoint's separate normal calibration scores with NumPy's linear quantile interpolation. Predict defective when score >= threshold. Score the strongest 1% of valid anomaly-map pixels. Do not select hyperparameters, seeds or epochs from calibration performance, and never tune the threshold on test labels. Legacy v1 retains its 95 th-percentile thresholds.

With few normal calibration/test images, observed false alarms can differ substantially from the nominal 10%. Report counts and Wilson intervals. The result is not a calibrated probability.

## Reference and conditional supervised track

PatchCore uses ImageNet-pretrained WideResNet50-2, layer 2/3 features, patch size 3, 1024-dimensional embeddings, a 1% approximate greedy coreset and one nearest neighbor. Fit the memory only on our normal training partition. Use the same separate normal calibration policy and native test masks. The official implementation is retained with Apache-2.0 notices; an exact PyTorch squared-L 2 backend replaces FAISS for macOS runtime compatibility. Square 256 input and restricted training data differ from the original published benchmark. No paper performance is copied into our result table.

If no selected MVTec category meets the target, execute the separately documented [KolektorSDD2 supervised protocol](SUPERVISED_PROTOCOL.md). Its real-defect labels and preserved official test membership are explicit. Do not mix its metrics with MVTec or claim that it establishes recognition of unseen defect families.

## Frozen evaluation and interpretation

Report image AUROC/AP, class prevalence, confusion counts, recall, false alarms, precision, native-mask pixel AP, per-defect recall and predeclared defect-area bins. Use image-level stratified bootstrap for AUROC and Wilson rate intervals. Training-seed spread and test-sample uncertainty are different quantities.

Gallery selection includes worst false positives/negatives and rank-spaced correct cases. Panels show the original, fixed-scale prediction and ground-truth annotation separately, with dataset attribution. Float maps remain available independently of display images.

After primary evaluation, run paired mild blur, exposure and JPEG stress tests with the same frozen threshold. Benchmark CPU/MPS with no competing training, recording warm p 50/p 95 and full inspection time separately. Stress results do not establish factory robustness.

## Completion evidence

A successful release needs reproducible configs and checkpoints, saved selection decisions made before test access, results from every declared run, failure analysis, a working public CPU demo, a clean installation and passing hosted CI. If measured quality misses the target, the release remains explicitly experimental and the model-quality objective remains unmet.
