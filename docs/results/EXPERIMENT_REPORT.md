# Frozen experiment evidence

These metal-nut results are **exploratory development measurements**. The category's test set has been inspected during development; these are not blind final benchmark results. All weights and thresholds were frozen before each reported evaluation.

| Run | Input size | Trained / selected epoch | Image AUROC [95% image bootstrap interval] | Native-mask pixel AP |
| --- | --- | --- | --- | --- |
| metal_nut-pilot-001 | 128 | 30 / 30 | 0.655 [0.552, 0.753] | 0.213 |
| metal_nut-reconstruction-001 | 128 | 50 / 45 | 0.626 [0.481, 0.754] | 0.217 |
| metal_nut-foreground-001 | 256 | 100 / 100 | 0.769 [0.674, 0.853] | 0.148 |

| Run | Normal calibration quantile | Defects detected / available | Good parts falsely flagged / available | Precision |
| --- | --- | --- | --- | --- |
| metal_nut-pilot-001 | 0.95 | 37 / 93 | 2 / 22 | 0.949 |
| metal_nut-reconstruction-001 | 0.95 | 10 / 93 | 1 / 22 | 0.909 |
| metal_nut-foreground-001 | 0.95 | 51 / 93 | 1 / 22 | 0.981 |

Defective-image prevalence is 80.9%; image average precision and positive predictive value depend on this unusually defect-heavy test mix. An uninformative image ranking has an AP reference approximately equal to this prevalence. The normal calibration sample size is small, so empirical quantiles do not guarantee future false-alarm rates. Full rate intervals, native-mask area groups, checkpoint checksums and portable prediction records are in [experiment-metrics.json](experiment-metrics.json).

![Per-method learning objectives](experiment-learning-curves.png)

![Ranking and frozen decisions](experiment-ranking-and-decisions.png)

![Defect recall](experiment-defect-recall.png)

## What the measurements show

The highest image-AUROC point estimate belongs to `metal_nut-foreground-001` (0.769); the highest native-mask pixel-AP point estimate belongs to `metal_nut-reconstruction-001` (0.217). Image ranking and defect localization are different outcomes. The confidence intervals and fixed-threshold counts should be considered together; these test comparisons do not choose the deployed checkpoint.

- `metal_nut-pilot-001` misses 56 defective parts and flags 2 good parts. Its lowest-recall annotated group is `bent`: 1/25 detected. The corresponding failure-gallery panels allow direct comparison of predicted activations with annotated defects.
- `metal_nut-reconstruction-001` misses 83 defective parts and flags 1 good parts. Its lowest-recall annotated group is `scratch`: 0/23 detected. The corresponding failure-gallery panels allow direct comparison of predicted activations with annotated defects.
- `metal_nut-foreground-001` misses 42 defective parts and flags 1 good parts. Its lowest-recall annotated group is `scratch`: 2/23 detected. The corresponding failure-gallery panels allow direct comparison of predicted activations with annotated defects.

High precision alone cannot compensate for missed defective parts. In this defect-heavy test set, inspect recall and false alarms alongside precision. A normal operating threshold remains an empirical calibration rule rather than an accuracy guarantee.

## Interpretation limits

These pilots differ in objective, network capacity, corruption generator, input resolution, learning rate and training budget. They provide descriptive whole-method comparisons and do not isolate the causal effect of any single component. Lower training/validation loss does not establish better real-defect detection. Pixel AP uses original masks and restored model maps; resizing can remove small input evidence. Image bootstrap intervals do not capture variation across training seeds or deployment domain shift.

## Balanced failure galleries

Gallery selection is deterministic: highest-scoring false positives, lowest-scoring false negatives, and rank-spaced successful cases. Prediction activations and ground truth occupy separate panels. Display ranges are fixed per checkpoint at twice its frozen image threshold; colors are not calibrated probabilities or pixel decisions.

- [metal_nut-pilot-001 gallery index](galleries/metal_nut-pilot-001/index.json)

metal_nut-pilot-001: FP, good, score 0.0648754.

![FP comparison panels](galleries/metal_nut-pilot-001/FP_0079_metal_nut_good.png)

metal_nut-pilot-001: FN, flip, score 0.00144224.

![FN comparison panels](galleries/metal_nut-pilot-001/FN_0051_metal_nut_flip.png)

metal_nut-pilot-001: TP, scratch, score 0.0411085.

![TP comparison panels](galleries/metal_nut-pilot-001/TP_0098_metal_nut_scratch.png)

metal_nut-pilot-001: TN, good, score 0.00136155.

![TN comparison panels](galleries/metal_nut-pilot-001/TN_0077_metal_nut_good.png)
- [metal_nut-reconstruction-001 gallery index](galleries/metal_nut-reconstruction-001/index.json)

metal_nut-reconstruction-001: FP, good, score 0.0413511.

![FP comparison panels](galleries/metal_nut-reconstruction-001/FP_0082_metal_nut_good.png)

metal_nut-reconstruction-001: FN, scratch, score 0.0253718.

![FN comparison panels](galleries/metal_nut-reconstruction-001/FN_0112_metal_nut_scratch.png)

metal_nut-reconstruction-001: TP, bent, score 0.0398381.

![TP comparison panels](galleries/metal_nut-reconstruction-001/TP_0000_metal_nut_bent.png)

metal_nut-reconstruction-001: TN, good, score 0.0260615.

![TN comparison panels](galleries/metal_nut-reconstruction-001/TN_0087_metal_nut_good.png)
- [metal_nut-foreground-001 gallery index](galleries/metal_nut-foreground-001/index.json)

metal_nut-foreground-001: FP, good, score 6.40301e-05.

![FP comparison panels](galleries/metal_nut-foreground-001/FP_0082_metal_nut_good.png)

metal_nut-foreground-001: FN, scratch, score 2.30573e-05.

![FN comparison panels](galleries/metal_nut-foreground-001/FN_0092_metal_nut_scratch.png)

metal_nut-foreground-001: TP, bent, score 2.84105e-05.

![TP comparison panels](galleries/metal_nut-foreground-001/TP_0005_metal_nut_bent.png)

metal_nut-foreground-001: TN, good, score 2.2604e-05.

![TN comparison panels](galleries/metal_nut-foreground-001/TN_0088_metal_nut_good.png)

Derivative gallery images retain their dataset's CC BY-NC-SA 4.0 license and individual attribution files. Raw datasets and checkpoints remain outside Git; release assets must carry checksums and model provenance.
