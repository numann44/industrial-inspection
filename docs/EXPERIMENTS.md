# Planned experiments and interpretation

This is a small, predeclared experiment set rather than an unrestricted search on real test labels. Keep original test data untouched during optimization and use synthetic validation for configuration selection. Record exploratory test inspections explicitly.

## Fixed protocol

- Initial category: metal_nut. Start at 128 pixels, 16 base channels, Adam, batch size 8 and seed 42.
- Training supervision: original normal training images and known procedural corruption masks.
- Original normal images partitioned 70/15/15 into train, synthetic validation and normal calibration.
- Best epoch selected by fixed synthetic-validation loss. The 95th normal calibration-score percentile determines the image decision threshold.
- No pretrained weights in the main model. No texture dataset dependency in the starting generator.
- Report each frozen run, including unsuccessful variants. Compare on the same original-resolution test masks.

## Comparisons

| Comparison | Question | Controlled conditions |
| --- | --- | --- |
| Reconstruction-only baseline vs joint reconstruction/segmentation | Does learning corruption masks improve real-defect localization? | Same normal split, image size, synthesis budget and calibration rule |
| Pretrained feature baseline vs from-scratch model | What accuracy/runtime tradeoff does from-scratch learning achieve? | Same category, test images and normal calibration protocol; disclose pretrained data |
| Appearance/texture patches vs patches plus scratches | Does adding thin structures help real scratches without increasing normal alarms? | Same model, training epochs, splits and seeds |
| Unrestricted vs foreground-constrained synthesis | Is the generator teaching background shortcuts? | Same corruption families and approximate altered-area distribution |
| 128 vs 256 pixels | Does retaining fine structure improve small-defect detection? | Original-resolution pixel targets, same split; report additional memory and time |

Initially run one seed to validate the pipeline. Repeat shortlisted settings at seeds 42, 43 and 44; separate variability across training seeds from uncertainty caused by the finite test set. A reasonable result can include a baseline outperforming our trained model.

## Metrics and useful analysis

Image AUROC measures ranking across good/defective images. It does not establish the usefulness of a particular operating threshold. At the frozen threshold, report defective parts detected/missed, good parts falsely flagged, precision and recall. Show raw counts alongside percentages.

Pixel average precision measures spatial ranking. Use the original masks and upsampled predicted maps. Input resizing can still destroy fine image evidence, even though target masks are preserved. Compare performance by defect type and by annotated defect-area fraction.

For each category, inspect the highest-scoring good images, lowest-scoring defective images, broad maps and missing thin defects. Keep annotated masks separate from predictions. Relate failures to lighting/reflections, object boundaries, size and corruption coverage only where the examples support that explanation.

Measure warm single-image inference p50/p95 on the same device and resolution. Report batch throughput separately. Include the scope of timing: preprocessing, transfer, model inference or the whole inspection path. Avoid converting batch timings into unsupported interactive latency claims.

Later robustness experiments can apply mild blur, exposure changes and compression to copies of held-out images. Keep original and perturbed results paired. These are stress tests on this dataset, not proof of factory deployment reliability.

## Completion evidence

A final release requires trained checkpoints, experiment configurations, split identifiers, curves, a reproducible evaluation, failure examples and a working image-inspection demo. The repo should explain why design choices were made and what the measurements show. No target accuracy is promised before seeing real results.
