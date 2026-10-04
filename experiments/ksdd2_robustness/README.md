# KSDD2 acquisition-robustness screen

This isolated experiment implements **exactly two fresh supervised training
runs**: `control` and `acquisition_aug`, both initialized with seed42. It follows
the closed, failed normal-only v3 synthesis screen. It does not modify v2/v3
sources, splits, selected weights or calibration thresholds. The KSDD2 task
uses real defect labels and remains separate from normal-only MVTec learning.

The existing validation-only diagnostic confirms sensitivity to brightness and
JPEG artifacts. The hypothesis is that training on a mixture of original and
degraded acquisitions improves robustness while retaining clean localization.
This compares a combined acquisition policy, not the separate effects of its
brightness and compression ingredients.

## Matched runs and frozen data

Both runs use the original audited train/validation/calibration membership,
randomly initialized base16 U-Net, balanced batches of8, Adam learning rate0.0003,
positive BCE weight3 plus Dice, and aspect-preserving 640-high ×256-wide
letterboxing. Their maximum budget is100 epochs. Architecture, mask geometry,
sampling, loss, optimizer, validation cadence and stopping rule are identical;
actual epoch counts may differ under early stopping.

The sole policy change in `acquisition_aug` is:

- Probability0.5: keep original RGB image.
- Otherwise: brightness uniformly sampled in[0.75,1.25], then JPEG integer quality
  uniformly sampled from60 through95, inclusive.
- Apply both operations at original resolution **before letterboxing**. Masks
  receive only the unchanged geometric preprocessing, never brightness/JPEG.

Training augmentation uses a separate deterministic epoch generator, including
distinct draws when balanced sampling repeats an image. DataLoader workers=0
preserve its sequence. Checkpoints record optimizer, global/sampler/augmentation
RNG state. Exact uninterrupted-versus-resumed CPU training is tested; MPS kernels
can still be nondeterministic.

The validation bank contains the existing350 source images with exactly four
deterministic conditions each: clean, brightness0.8, brightness1.2 and JPEG60.
Every image/mask source and derived image/mask/valid-area tensor is checksum-pinned.
Derived tensors are generated on demand and checked against the frozen bank;
there is no multi-gigabyte image-tensor cache. These are **350 correlated image
groups**, not1,400 independent observations. No test pixels or masks are opened.

## Selection, stopping and gate

Checkpoint selection uses the harmonic mean of **pooled cross-condition image
AUROC and exact pooled pixel AP**, with letterbox padding excluded. Pooling
allows condition-dependent score shifts to affect ranking; perfect AUROC within
each condition alone can conceal such shifts. Pixel AP is at model resolution,
not native-resolution localization quality.

Validation runs every5 epochs and at the final epoch. Equal scores retain the
earliest eligible checkpoint. Patience is15 **training epochs**, meaning three
successive failed five-epoch validation checks, not fifteen checks. Each run's
single pooled-selected checkpoint supplies every per-condition metric; no
separate condition-winning checkpoints are mixed together.

After both runs finish, freeze a global gate before reading calibration images:

- Candidate pooled H gain must be at least0.01 over the fresh control.
- Clean H regression must be no more than0.01.
- Every individual condition H regression must be no more than0.01.

Failure closes the experiment immediately, without calibration or extra training.
On passing, each already frozen checkpoint receives its original clean-normal
calibration q90 threshold once. Only the313 normal calibration images are read;
unused positive calibration images remain unopened. Thresholds never feed back
into candidate selection. Calibrated weights then await **exploratory** evaluation;
this runner does not evaluate original tests or change the public demo.

## Exact AP and uncertainty

Predicted pixel scores are streamed into per-condition positive/negative float32
files. Each file is sorted in place through a memory map. Exact average precision
is then summed only at distinct positive-score thresholds, counting all positive
and negative scores greater than or equal to each threshold. Negative-only
thresholds add zero recall. Ties are included in full; no score quantization or
full pixel-level int64 argsort is used. Pooled AP combines positive thresholds
and queries all four sorted negative arrays, avoiding a second pooled pixel file.
Only positives are concatenated in memory. No-positive AP is explicitly0;
nonfinite predictions are rejected. Tests compare against scikit-learn including
ties, sparse positives and all-positive samples. Temporary files are removed
after each validation; budget a few GB of local scratch disk.

The paired bootstrap reports a percentile95% interval for the candidate-minus-
control **pooled image AUROC** difference. It resamples original image groups
stratified by label, retaining all four variants together. It does not provide a
pixel AP interval, alter the gate or establish independent test reliability.

## Execution

From the repository root, after review:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.ksdd2_robustness.runner --prepare --device mps
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.ksdd2_robustness.runner --execute
```

Preparation freezes source hashes (including this README), environment, data
identities, validation bank, rules and exactly two run paths. Outputs live under
`outputs/ksdd2-robustness/`, outside the source tree, so declaration hashes do
not depend on themselves. Execution takes the shared project MPS lock, rejects
incompatible changes, resumes each run's own atomic `last.pt`, and refuses to
overwrite unrelated or incompatible completed runs. Do not edit frozen sources
after preparation.

Focused tests:

```bash
.venv/bin/python -m pytest tests/test_ksdd2_robustness.py
```

The final state is `closed_failed_gate` or `awaiting_exploratory_evaluation`.
There is no conditional seed expansion, continuation training, automatic test
evaluation, publication or quality-success claim in this screen.

## Independent-evidence limitation

Prior KSDD2 test/stress failures motivated this policy. Any reused KSDD2 tests
are consequently **development-inspected/exploratory for the new method**.
Original v2 measurements retain their historical status for their original
weights; that status cannot be transferred to the new weights. No new independent
same-category holdout is currently available. Unused positive calibration
examples alone cannot establish both recall and normal false-alarm performance.
New independent parts/acquisition sessions or a documented nonoverlapping dataset
would be required for a new independent reliability claim.
