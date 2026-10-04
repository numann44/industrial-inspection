# Bounded learned-decision experiment

**Status: declared design under implementation and review; no training or quality result yet.** This is the final two-run architecture comparison in the current KSDD2 follow-up search. It does not extend the closed MVTec synthesis budget or repeat the completed acquisition screen.

## Why change the image decision?

The [completed acquisition comparison](results/KSDD2_ROBUSTNESS_RESULTS.md) improved pooled validation image AUROC and pixel AP, but its augmented model still failed the original-test false-alarm target. Its validation JPEG image AUROC also declined. The [saved-score diagnostic](results/KSDD2_DECISION_PLANNING.json) computes standardized partial AUROC over false-positive rates from 0 to 0.1 without reading new images, changing earlier checkpoints or recalibrating any threshold:

| Previously selected validation model | Pooled partial AUROC | JPEG partial AUROC | Pooled pixel AP |
| --- | ---: | ---: | ---: |
| Acquisition-screen control | 0.931728 | 0.937784 | 0.718344 |
| Acquisition augmentation | 0.937409 | 0.932194 | 0.754238 |

These are diagnostic results on reused validation data, not an independent confirmation. They show that the previous harmonic-mean selection can reward localization gains while image discrimination worsens in a relevant operating region. They do not establish the cause of false alarms.

The hypothesis is that explicitly learning an image-level decision from segmentation features can improve low-false-alarm discrimination compared with averaging the highest 1% of pixel responses. The Kolektor authors' [paper](https://arxiv.org/html/2104.06064) and [official model code](https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/master/models.py) provide a precedent for combining segmentation features with a classification network while stopping classification gradients from entering the segmentation network. Our compact U-Net and decision head are a lightweight adaptation, not a reproduction of that architecture or its reported results.

## Exactly two matched fresh trainings

Both runs start from seed 42 and random weights. They share the same audited train/validation/calibration membership, acquisition augmentation, base-16 U-Net, 640-high × 256-wide letterboxing, balanced batch size 8, Adam learning rate 0.0003 and segmentation loss (positive BCE weight 3 plus Dice). The acquisition policy keeps 50% of images unchanged; otherwise native RGB receives brightness U(0.75, 1.25) followed by JPEG quality 60–95. Masks keep their original geometry.

- **Control:** image score is the mean of the highest 1% of valid sigmoid segmentation responses.
- **Decision head:** concatenate detached bottleneck features and downsampled detached segmentation logits; apply three small convolutional blocks; combine masked global mean/max features with valid segmentation mean/max statistics; output a raw image logit. Add image-level BCE using existing training labels. Classification gradients never change the U-Net. The image logit is a ranking score, not a calibrated defect probability.

Head initialization must not change backbone initialization or augmentation/sampling randomness. Padding is excluded from the head's pooling and score statistics. This compares learned scoring with fixed scoring; it adds parameters and supervised computation, which will be reported explicitly.

The cap is 100 epochs per run, validation every five epochs, and stopping after 15 stale **training epochs**. Actual stopping epochs may differ. No extra seeds, restart selection, learning-rate search, threshold search or continuation jobs are allowed. Local heavy computation stays serialized by the project lock.

## Validation selection and the frozen gate

Reuse the identical checksum-pinned bank of 350 original validation images, each with clean, brightness 0.8, brightness 1.2 and JPEG-60 variants. The 1,400 variants are correlated observations from 350 sources. No original test or calibration pixels enter this screen.

For **both** new runs, select one checkpoint by highest pooled standardized partial AUROC over FPR 0–0.1, as defined by [`roc_auc_score(..., max_fpr=0.1)`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). Exact ties retain the earliest checkpoint. The old runs are not reselected under this new rule. Retain full AUROC and exact model-resolution pixel AP for analysis.

Let R be the selected pooled partial AUROC. Before calibration, freeze both selected weight hashes and apply all of these guards:

1. Candidate R strictly improves and reduces `1 - R` by at least 10% relative to the fresh control.
2. No individual condition's partial AUROC declines by more than 0.01.
3. Neither pooled nor any individual condition's pixel AP declines by more than 0.01.

All guards use each run's single selected checkpoint. No mixing of condition-specific winners. A control at R=1 leaves no room to pass the strictly positive gain rule. The descriptive paired bootstrap resamples original image groups, retains all four conditions and is excluded from the gate. Reused validation, few defective sources and one training seed limit interpretation.

## Calibration, evaluation and stopping boundaries

Gate failure closes this architecture search without reading calibration pixels or training extra candidates. On a pass, each selected model receives exactly one q90 threshold from the same 313 original clean normal calibration sources. The 37 positive calibration images stay unopened. Write the global gate before calibration and bind each calibrated asset to its selected checkpoint and gate digest. Never change old or new thresholds from observed test failures.

The screen runner stops after the gate and, if permitted, calibration. A later separately frozen evaluation may compare native-resolution localization, recall/FPR with uncertainty, defect sizes, fixed perturbations, error galleries and timing. All reused KSDD2 tests for these new models remain **development-inspected / exploratory**. A better validation score does not guarantee recall ≥90%, FPR ≤10%, robustness or independent success. Deployment requires a separate evidence review; v0.1.0 stays active during this comparison.

Regardless of the outcome, new independent same-surface reliability claims require untouched acquisition sessions or a documented nonoverlapping dataset containing both normal and defective examples. No available unused positive-only subset establishes both recall and normal false-alarm performance. If this head fails the frozen gate, further architecture/seed/epoch searches on the same evidence stop.

## Implementation and reproducibility

Implementation is isolated in `experiments/ksdd2_decision`; it imports preserved data and exact-AP helpers without modifying earlier experiment sources. Before any training, freeze source/configuration/environment/split/bank identities and both allowed output paths. Compatible run-owned epoch snapshots must retain optimizer, global, sampler and augmentation RNG states. A completed or unrelated run cannot be overwritten.

Required prelaunch checks cover gradient detachment, identical backbone initialization, padding exclusion, scoring and checkpoint identity, real interrupted-versus-continuous CPU training, data separation, finite metrics, the two-run budget and one-time gated calibration. Launch evidence will record the verified source, actual parameter counts and checks once implementation is reviewed; this document alone is not execution evidence.
