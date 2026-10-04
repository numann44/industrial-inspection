# Bounded acquisition-robustness screen

## Motivation and scope

The frozen v0.1.0 surface model meets the original KolektorSDD2 test's recall and false-alarm point targets, but fails brightness and JPEG stress tests. A [validation-only diagnostic](results/KSDD2_VALIDATION_STRESS_PLANNING.json) reproduced the mechanism: on 313 normal validation sources, false alarms were 37 for clean images, 53 at brightness ×1.2 and 102 at JPEG quality 60, using the existing threshold unchanged. This is reused validation evidence, not another independent test.

The hypothesis is that exposure to acquisition variation during supervised training reduces score shifts under brighter or compressed images without degrading clean-image discrimination or localization. This is a separate two-run experiment; it does not reopen the failed [MVTec synthesis screen](results/STUDY_V3_SCREEN.md).

## Matched comparison

| Setting | Both runs |
| --- | --- |
| Candidates | Fresh control; acquisition augmentation |
| Initialization | Random, seed 42; small U-Net, 16 base channels |
| Training sources | Existing 1,631-image split, including real defect masks |
| Input | Aspect-preserving 640-high × 256-wide letterbox; padding excluded |
| Optimization | Balanced batch 8, Adam, learning rate 0.0003, positive-weight-3 BCE + Dice |
| Budget | At most 100 epochs each, exactly two runs |
| Selection frequency | Every five epochs and the final epoch |
| Early stopping | 15 training epochs without selection improvement (three unsuccessful checks) |

Control receives the original image. The augmented candidate receives an unchanged image with probability 0.5; otherwise brightness is sampled uniformly from 0.75–1.25, followed by JPEG quality sampled uniformly from integers 60–95. Both operations occur at native resolution before letterboxing. Masks and geometry are unchanged. This compares the combined acquisition policy; it cannot isolate the contribution of brightness versus compression.

## Frozen validation and decision rule

The bank contains clean, brightness ×0.8, brightness ×1.2 and JPEG-quality-60 versions of each of the same 350 validation sources. These are **350 correlated source groups, not 1,400 independent observations**. Source hashes and prepared image/mask/valid-area tensor hashes pin the bank contents.

Each run selects one checkpoint by the harmonic mean of **pooled cross-condition image AUROC and exact pooled model-resolution pixel AP**. Pooling includes comparisons between conditions, so condition-wide score offsets cannot hide behind separate AUROCs. Ties retain the earlier checkpoint. Pixel AP excludes padding and handles tied float32 activations exactly; it is not native-resolution final pixel AP.

The augmented candidate passes only if its pooled harmonic score exceeds the fresh control by at least 0.01 and no individual condition's harmonic score regresses by more than 0.01. All condition scores come from each run's single pooled-selected checkpoint. A paired bootstrap of image AUROC retains all four variants together for each sampled source. Its descriptive reused-validation interval is not a pixel-AP interval and does not change the gate.

## Calibration, evaluation and stop conditions

Both runs first produce immutable **uncalibrated** validation-selected weights. The two-run gate is then recorded. A failed gate ends the experiment without opening calibration pixels or starting more seeds. A passed gate freezes both selections and permits one q90 calibration for each on the 313 original clean normal calibration sources. Thresholds cannot feed back into checkpoint selection. Positive calibration images remain unopened.

This screen never evaluates original test images. Only after a passed gate and frozen thresholds can a separately recorded exploratory comparison measure recall, false alarms, confidence intervals, native pixel AP, errors and acquisition robustness. The original KSDD2 test informed this new hypothesis, so **any reused test results for the new method are development-inspected/exploratory**. The earlier frozen v0.1.0 report remains unchanged; no new independent same-category holdout is available.

## Execution and recovery

Implementation lives in `experiments/ksdd2_robustness`, separate from the preserved v2/v3 sources. Source, environment, configuration, split and bank identities are frozen before training. The shared `outputs/mps-study.lock` serializes local heavy compute. Epoch-boundary snapshots retain optimizer and random states; recovery must use the same environment and compatible `last.pt`. Completed run directories cannot be overwritten.

Before launch, the complete local suite passed **155 tests**, including 16 focused screen checks. These exercise exact AP against scikit-learn, grouped validation, unchanged mask geometry, actual interrupted/resumed CPU weights and optimizer equality, exclusion of calibration/test pixels from selection, epoch-based stopping, a two-run budget, and one-time gated calibration with crash recovery. The screen received a separate code review. These checks establish implementation behavior, not a model-quality result.

The active model registry stays on v0.1.0 until a new model has measured evidence sufficient for a documented promotion. Finishing the two-run budget is not itself evidence that the quality target has been achieved.

The two-run screen launched locally on October 4, 2026 at 03:36 UTC from commit `60ceddf`. The control's first epoch completed and its resumable snapshot was written. The [launch record](results/KSDD2_ROBUSTNESS_LAUNCH.json) preserves source/configuration/split/bank identities and a dated running-state snapshot. This is execution evidence; no model-quality improvement has been measured yet.
