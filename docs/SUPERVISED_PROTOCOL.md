# KolektorSDD2 supervised experiment protocol

This track learns from actual labeled defects. It is a separate task from the normal-only MVTec anomaly detector, with separate data, model identities and result tables.

## Fixed data and model

- Original test: 1,004 images; unchanged membership.
- Seed-42 training split: 1,459 normal and 172 defective images.
- Validation: 313 normal and 37 defective images.
- Calibration: 313 normal and 37 defective images; only normal examples are used, and positive calibration examples remain unused.
- Exact/near pixel-similarity groups stay together. No product or production-batch identity is supplied, so no independence claim is made at those levels.
- Randomly initialized single U-Net, 16 base channels; no pretrained weights.
- Letterbox height 640 × width 256, bilinear RGB and nearest-neighbor masks. The decoded data audit found no positive mask that vanished at this geometry.
- Training uses BCE with positive-pixel weight 3 plus Dice, excluding padding. Every eight-image batch contains four normal and four defective examples, sampled with replacement.
- No geometric or appearance augmentation is added in this initial supervised configuration. Balanced sampling changes the training class mixture; natural prevalence is retained in validation and test.

## Selection and calibration

Adam learning rate is 0.0003. Train at most 100 epochs, with patience 15 on the harmonic mean of validation image AUROC and validation pixel average precision. Pixel AP used for selection is at model resolution and excludes letterbox padding. This selection score is not a final benchmark result.

Freeze the highest-scoring validation checkpoint before normal calibration. Image score is the mean of the highest 1% sigmoid map pixels within the valid image area. The decision threshold is the normal calibration-score 90th empirical percentile, with linear interpolation and `score >= threshold`. This empirical rule does not guarantee a 10% future false-alarm rate. The score is not a calibrated image-defect probability.

Repeat training seeds 42, 43 and 44 while keeping the split seed fixed. Retain all runs. A deployment candidate can be selected by the declared validation criterion; never select the best seed using test performance.

## Commands and recovery

```bash
python scripts/prepare_ksdd2.py
python -m inspection.ksdd2 --root data/ksdd2
python -m inspection.supervised --output runs/ksdd2-seed42 --training-seed 42
```

Each completed epoch atomically saves `last.pt`, with optimizer and random-generator states. To resume an interrupted run:

```bash
python -m inspection.supervised --output runs/ksdd2-seed42 --training-seed 42 \
  --resume runs/ksdd2-seed42/last.pt
```

Resume requires identical declared configuration, data content/split, relevant source hashes, dependency versions and thread count. `--stop-after-epoch 1` provides a deliberate epoch-boundary pause for profiling; this is not a new training configuration. Exact CPU resume is tested; accelerator kernels may be nondeterministic.

After training, `checkpoint.pt` holds the selected weights and calibrated operating threshold. `summary.json`, `history.json` and `config.json` record provenance and selection. No training job starts merely by preparing the data.

## Final evaluation

Freeze all candidate checkpoints and thresholds before observing official test predictions. Restore maps from letterbox geometry and evaluate original masks. Report AUROC, image/pixel AP, recall, false-alarm counts/rates, precision, confusion counts, per-seed variability and timing scope. The official test's defective prevalence is approximately 11%; accuracy alone can obscure missed defects.

After the first test inspection, later tuning on those observations is exploratory and must be labeled accordingly. Results do not establish transfer to unseen product types, previously unseen defect families, new cameras or production batches.

## Completed study result

The declared three-seed study completed. Seed 44, epoch 60, was selected by validation before test evaluation. It detects 105/110 defects (95.45%) with 89/894 normal false alarms (9.955%) and native pixel AP 0.815463. These point estimates meet the declared target, but confidence intervals cross both target boundaries and brightness/JPEG perturbations expose excess false alarms. The [controlled report](results/CONTROLLED_STUDY.md) preserves every seed; the [model card](MODEL_CARD.md) gives the selected model’s uncertainty, scope and limitations. This outcome does not alter the protocol above or establish success on the separate MVTec categories.
