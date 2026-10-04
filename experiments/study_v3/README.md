# Follow-up synthesis screen: study v3

This directory is isolated from the completed v2 implementation. It implements
**three screening runs**, an immutable validation bank, exact-compatible resume,
a validation gate and a frozen continuation manifest. It does **not** implement
the six conditional continuation runs or any test evaluation. A passed gate
means the continuation executor needs implementation/review, not that the whole
study or its quality target is complete. The hard cap is **nine new runs**.

## Hypothesis and controlled changes

The v2 synthetic validation score did not predict real-defect localization well.
Strong artificial seams and original-foreground-clipped warps are candidate
causes. These are hypotheses, not conclusions established by the previous test.

| Run | Synthetic training signal |
| --- | --- |
| control | Unchanged v2 foreground corruption families and parameters |
| A | Replace appearance/texture modes with normal-donor texture, locally mean-matched, irregular feathered support and blend strengths 0.15–0.75 |
| B | A plus replace the existing warp mode with tapered local displacement that may cross the original silhouette |

All three use the same family frequencies, original split, random seed 42,
joint model, 256-pixel input, base16, batch8, Adam at 0.0003, 100 epochs and
25% clean-image probability. A changes a synthesis package; the effect of each
ingredient is not individually isolated. Donor substitution retains the nominal
support area range; realized changed-pixel area can differ. All recipient and
donor training images come from the original normal training split. A failed
foreground heuristic returns an unchanged sample rather than using test-derived
part masks. The heuristic's uniform-background assumption remains a limitation.

The control is **newly trained**, with the same new validation selection rule
as A/B. The existing v2 weight remains a historical reference only. This avoids
comparing a new-bank-selected candidate with an old-bank-selected control as if
their selection budgets were identical.

## Frozen validation, selection and calibration

`--prepare` copies every original v2 bank image/mask NPY byte-for-byte and adds
one subtle patch, one boundary warp and one hard translated insertion per
validation normal. The hard insertion operator is absent from training, although
it is still synthetic rather than independently acquired defect evidence.
Validation donors come exclusively from validation normals; training donors
come exclusively from training normals. No calibration or real test pixels
enter bank generation or model selection.

Each of 11 anomaly families is compared against the same clean validation
controls. Its H is the harmonic mean of image AUROC and pixel AP. The arithmetic
mean of these family H values chooses checkpoints every five epochs and at the
last epoch. Equal scores retain the earliest eligible epoch. Old and new family
means are also retained. Pixel AP includes that family's clean control pixels;
families have equal rank weight regardless of their anomalous area.

A/B pass the gate only if the family macro H improves over the fresh control by
at least 0.01, with no more than 0.02 regression on the old-family mean. The
highest eligible macro H wins; exact ties choose A. These rules are frozen
before running. Scores from real tests cannot enter the gate.

After each candidate's checkpoint is fixed by validation, its threshold is the
unchanged protocol's 90th percentile of original normal calibration scores.
Previous v2 thresholds are untouched. The calibration quantile is not a future
false-alarm guarantee. The three-run screen does not inspect original tests.

## Commands

Run from the repository root, using the existing environment:

```bash
.venv/bin/python -m experiments.study_v3.runner --prepare --device mps
.venv/bin/python -m experiments.study_v3.runner --execute
```

Preparation requires completed v2 evidence, hashes every v3 file (including this
README) and the shared Python modules, and refuses a changed existing declaration.
The bank is constructed in a staging directory and then renamed. A leftover
`.building` directory after failure must be inspected/preserved before retry.

Execution takes the project's `outputs/mps-study.lock`, trains sequentially,
and resumes only each run's own `last.pt`. It validates source/config/bank/split,
package/Python/device identity and restores optimizer plus all saved RNG states.
Atomic epoch checkpoints prevent lost completed epochs. Completed checkpoint
reuse is verified; a nonempty unresumable run refuses overwrite. CPU exact
resume is covered by a tiny synthetic-data test; MPS kernels can be nondeterministic.

Run the focused tests without starting the study:

```bash
.venv/bin/python -m pytest tests/test_study_v3.py
```

The screen writes `screen-result.json` and `continuation.json`. If the gate
fails, stop this cycle. If it passes, the six permitted continuation jobs are:
winner metal_nut seeds43/44; winner screw, transistor and cable seed42; and the
unchanged control on cable seed42. Their executor is intentionally not supplied
here. The maximum is three screening runs plus six continuation runs; additional
searches require a separately justified declaration, not silent budget expansion.

## Evidence boundaries

The metal_nut, screw and transistor original tests are now development-inspected
for this follow-up and must remain exploratory. Resplitting already evaluated
test images cannot restore independence. Cable is only a prospective separate
category comparison: verify that its tests were not used in prior method
decisions, retain the existing audited normal split, and freeze both cable
models/thresholds before any test read. Separate cable training weights do not
establish that the metal_nut checkpoint generalizes to cable.

No new independent same-category holdout is currently available. A new physical
collection would need independent specimens/acquisition sessions and frozen
grouped splits. The KSDD2 supervised task remains separate and unchanged.
Passing a synthetic gate proves neither 90% real-defect recall nor factory
reliability. Report failed hypotheses, finite-sample uncertainty and resource
costs alongside positive results.
