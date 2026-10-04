# Detached image decision screen

This is one bounded, from-scratch supervised KolektorSDD2 experiment. It compares
exactly two fresh seed-42 runs using the same acquisition augmentation, audited
split, optimizer, initialization of the shared U-Net, and validation images.
It is separate from normal-only MVTec training. Existing results remain unchanged.

The hypothesis is that a learned image classifier can use segmentation features
and their spatial context to reject misleading local activations better than the
fixed mean of the highest 1% segmentation activations. Previous JPEG validation
regression and exploratory test failures motivated the study. Benign-texture
confusion is a hypothesis, not a visually established error taxonomy.

The architecture is inspired by the separate decision network and gradient
stopping in [Božič et al.](https://arxiv.org/html/2104.06064) and their
[official implementation](https://github.com/vicoslab/mixed-segdec-net-comind2021).
This compact U-Net adaptation is not a reproduction, and the publication's
reported performance does not predict this experiment's operating-point results.

## Fixed comparison

| Setting | Control | Candidate |
|---|---|---|
| Candidate key | `control` | `decision_head` |
| Image score | Top 1% sigmoid segmentation activations | Raw learned classification logit |
| Segmentation model | Base-16 U-Net, 488,705 parameters | Identical U-Net |
| Added parameters | 0 | 25,747; total 514,452 |
| Segmentation objective | BCE with positive weight 3 plus Dice | Identical |
| Image objective | None | Image-label BCE, weight 1 |
| Initialization | Seed 42, random | Backbone seed 42; isolated head seed 9,000,042 |

Both use batch 8 with the existing balanced sampler, Adam at 0.0003, and native
RGB acquisition augmentation: 50% unchanged; otherwise brightness U(.75,1.25)
then JPEG integer quality 60–95 before 640×256 letterboxing. The imported frozen
augmentation implementation is explicitly configured as `acquisition_aug` for
both runs, preserving the original per-epoch augmentation RNG sequence.

The candidate receives detached 128-channel bottleneck features and a detached,
valid-area average of segmentation logits at the bottleneck resolution. Three
3×3 convolution/GroupNorm/SiLU blocks use channels 129→16→16→32, with 2× max
pooling before each. Inputs are cropped to the valid bottleneck rectangle;
invalid positions cannot enter pooling statistics and are zeroed between blocks.
Masked mean/max of the final 32 feature channels and the full model-resolution
segmentation logit mean/max form 66 features for one linear output. All pathways
from classification to segmentation are detached. The shared backbone retains
its original fixed letterbox semantics; the head does not remove that context.
The head's CPU initialization restores global RNG, without reseeding MPS.

The candidate's score is an unbounded logit, not a probability. Its model kind is
`supervised_segmentation_detached_decision_v1`; its score identity is checkpointed
and checked on reuse/calibration. **The shared InspectionEngine does not support
this candidate yet.** Any later evaluation/demo needs an explicit adapter.

## Selection, gate and stopping

The existing 350 validation images × four conditions are copied as the identical
content-pinned bank, never regenerated. These are 350 correlated image groups,
not 1,400 independent images. Optimization and all selection avoid calibration
and original test pixels. Pixel AP uses exact float32 disk-backed spools,
excluding padding, at model resolution.

The sole checkpoint ranking is pooled standardized partial image AUROC over FPR
[0,.1], using `sklearn.metrics.roc_auc_score(..., max_fpr=.1)`. Evaluate every five
epochs and at the cap of 100 epochs. Exact ties retain the earliest checkpoint.
Three failed five-epoch checks stop a run after 15 stale training epochs. Both
runs share the same maximum budget and stopping rule; realized epochs can differ.

Let R be pooled partial AUROC. The candidate passes only with strictly positive
gain and at least a 10% reduction in 1−R, no individual condition's partial AUROC
regression above .01, and no pooled or per-condition pixel AP regression above
.01. Every guard uses each run's single selected checkpoint. No harmonic-mean
gain requirement can hide a classification regression behind localization gains.
If control R=1, there is no permitted positive improvement and the gate closes.
The fixed gate is a practical experiment criterion, not statistical significance.

A descriptive paired bootstrap resamples original validation images, stratified
by label, with all four conditions kept together (1,000 draws; seed 5,000,042).
Its partial-AUROC interval is never used for gating and does not establish an
independent test result or pixel-level uncertainty.

Both selected checkpoint SHA-256s and the global gate are written before any
calibration. Failed gate: zero calibration reads and no more training. Passed
gate: each selected model is calibrated once on the original **313 clean normal
calibration images**, q90 with linear interpolation. Positive calibration images
and all calibration masks stay unopened. Reusing a calibrated checkpoint checks
its model/score, weights, threshold, quantile and saved identity without refitting.
There is no threshold search, seed search, continuation, test evaluation or
publication in this runner.

All reused original KSDD2 tests remain **exploratory**. No fresh independent
holdout is available, and known product/batch groups are not supplied. Better
validation ranking does not guarantee a <=10% false-alarm rate at the final
clean-normal q90 threshold or resilience to unseen acquisition conditions.

## Commands and traceability

After code review and commit, from the project root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.ksdd2_decision.runner --prepare --device mps
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.ksdd2_decision.runner --execute
```

Default output: `outputs/ksdd2-decision`. The declaration pins all new experiment
files, imported robustness/shared inspection code, scheduler source, package and
thread environment, configuration, manifest bytes, split, original/copied bank
bytes/digest, planning analysis and previous robustness evidence. Operational
outputs cannot reside inside frozen sources. Preparation/execution acquire the
shared `outputs/mps-study.lock`; heavy jobs remain serialized.

Every completed epoch atomically saves last/best state, optimizer, global RNG,
sampler and augmentation RNG. Resume only accepts this run's compatible last
checkpoint. Existing directories, completed summaries, declarations, gates and
calibrated assets are not silently overwritten. Exact interrupted CPU resume is
tested within one environment; accelerator nondeterminism remains a limitation.

Previous two runs required about 2.6 hours for 95 total epochs. New runtime depends
on early stopping and the measured head overhead. A cold synthetic step includes
compilation effects and is not an epoch-speed benchmark or completion estimate.
The maximum remains two 100-epoch runs, serialized locally. No paid compute.
