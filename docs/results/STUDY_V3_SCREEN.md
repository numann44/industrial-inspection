# Study v3: a completed negative synthesis screen

**Neither proposed synthesis change passed the predeclared validation gate.** The fresh control and candidates A/B each completed 100 epochs. No candidate was promoted, none of the six conditional continuation runs started, and the published **v0.1.0 demo remains unchanged**. The next step is diagnosis of the validation failures; this report does not declare or report another training job as running.

The screen ran sequentially on local MPS from `2026-10-04T01:29:00.587800+00:00` to `2026-10-04T02:17:54.229853+00:00`. It read **zero real test images** and produced no real-test predictions or new real-defect success claim. Exact metrics, per-family values, configuration, calibration and hashes are retained in [study-v3-screen.json](study-v3-screen.json).

## Hypothesis and fair comparison

The v2 synthetic validation score transferred poorly to real-defect results. This follow-up tested whether subtler corruption and deformations that cross the original silhouette could improve a broader synthetic validation objective. These were hypotheses, not established explanations of the v2 failures.

| Run | Training change |
| --- | --- |
| Fresh control | Unchanged v2 foreground corruption families and parameters |
| A | Appearance/texture modes replaced with normal-donor texture, local mean matching, feathered irregular support and blend strengths 0.15–0.75 |
| B | A plus tapered local displacement replacing the warp mode; its support may cross the original silhouette |

All runs use the original metal-nut training split, training seed 42, joint architecture, 256-pixel input, base width 16, batch size 8, Adam at 0.0003, 100 epochs and 25% clean-image probability. Recipients and donors for training come exclusively from the original normal training split. The control was freshly trained and selected with the same new bank and checkpoint budget as A/B; the old v2 checkpoint was only a historical reference.

This comparison tests synthesis packages, not isolated attribution to feathering, donor choice or contrast. All three finish their declared budgets: **300 new training epochs across three runs**, without extra seeds or epochs after a failed gate. The aggregate recorded run time is 2,928.55 seconds (48.81 minutes), excluding the small scheduler overhead and bank preparation; this is a local execution observation, not a general runtime guarantee.

## Frozen bank and selection rule

The bank retains the original eight synthetic anomaly families byte-for-byte and adds three: subtle patch, boundary warp and a hard translated insertion absent from the training dispatcher. Every anomaly family has 33 examples and is compared with the same 33 clean validation controls. All sources and donors are validation normals, separate from training and calibration. There are **396 bank records**: 33 clean and 363 synthetic anomalous images. These repeated source images do not constitute 396 independent physical observations.

For each of the 11 anomaly families, the score is the harmonic mean H of image AUROC and pixel AP, including clean control pixels in the latter. The arithmetic mean of these 11 H values is the checkpoint objective. Checkpoints are considered every five epochs and at the final epoch; equal scores retain the earlier eligible checkpoint. Old-family and new-family means are reported separately. Neither real test outcomes nor calibration scores select checkpoints.

## Selected-checkpoint validation results

Values below are the saved full-precision scores for each run's validation-selected checkpoint, not necessarily its final-epoch model.

| Run | Completed epochs | Selected epoch | Family-macro H | Old-family mean H | New-family mean H |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 100 | 100 | 0.8971445541893278 | 0.9389902315818857 | 0.7855560811425074 |
| A | 100 | 100 | 0.7626782874899303 | 0.7687686852345004 | 0.7464372268377438 |
| B | 100 | 95 | 0.7425017629318905 | 0.7193207140898681 | 0.8043178931772834 |

The gate requires **at least +0.01 family-macro H** over the fresh control and **no more than 0.02 regression in old-family mean H**. Thus a candidate needed macro H ≥0.9071445541893278 and old-family H ≥0.9189902315818857. Neither candidate satisfied either condition.

| Candidate | Macro H change vs control | Old-family H change | New-family H change | Gate |
| --- | ---: | ---: | ---: | --- |
| A | −0.13446626669939754 | −0.1702215463473853 | −0.039118854304763584 | Failed |
| B | −0.1546427912574373 | −0.21966951749201757 | +0.018761812034776004 | Failed |

The saved outcome is `stopped_no_validation_gain`, with `selected: null` and an empty `continuation_jobs` list. The original cap was nine new runs, comprising this screen plus at most six conditional jobs. **Only three ran.** The six conditional jobs are not queued; their trigger failed.

## What the negative result shows

A improves the subtle-patch family's H from 0.686092 to 0.777144, but loses on the other new families and on every old family. B improves boundary-warp H from 0.778244 to 0.811855 and withheld-translation H from 0.892331 to 0.931220; its aggregate gains on the three new families are outweighed by broad old-family regression. The strongest regressions include the legacy appearance/texture families. The complete per-family AUROC, pixel AP and H values are retained in the JSON, rather than selecting only the families that improved.

These observations support rejecting A/B under this declared screen. They do not establish that these synthesis ideas can never help under another method, explain the exact cause of the regressions, or measure their performance on real defects. This is one training seed and one fixed synthetic bank; no new test comparison or across-seed uncertainty estimate was performed.

The family failures and calibration distributions are evidence for further diagnosis. Additional training requires a new justified, bounded declaration; the failed gate is not bypassed by searching more seeds, quietly extending epochs or opening tests to pick a winner.

## Calibration and retained artifacts

Each selected checkpoint was calibrated afterward on the same **33 separate normal calibration images**, using the unchanged 90th-percentile rule with linear interpolation and `score >= threshold`. No existing v2 threshold was altered.

| Run | Frozen threshold | Checkpoint SHA-256 |
| --- | ---: | --- |
| Control | 4.510770959313959e-05 | `f6affb94c2c1b67be0360b37d9e3a46f18ec2c8bad87ee2f452d421b5e9f0038` |
| A | 0.7825649619102478 | `3a6ee3752a1e74cc0aca5eed09e0f78f30d3ff64d07eb19ae2e327fb66c24f0e` |
| B | 0.5168538331985474 | `b871477da383666baa0587665244e4b50820ca720796aeb4047f5a2ef21a67f1` |

These are uncalibrated activation scales and empirical operating thresholds, not defect probabilities or guarantees of future false-alarm rates. The threshold differences are not evidence of real-test recall. Run summaries, normal calibration scores and checkpoint identities are preserved without promoting any model to the live registry.

## Provenance and evidence boundaries

| Identity | Digest |
| --- | --- |
| Frozen declaration digest | `19d20b3348a0b522235149f8faf7e7406abf45ec727330438a2422f284c3d723` |
| Validation bank digest | `372054ed54c9bba433dfa79ffc77824711c8d5cd5ea019d55a6f2e53927dc4bc` |
| Audited original split digest | `2df9587b2f8d3f7f389a7ef2a1e9e13cf21ea6b6738a1d9023e7daf1d59de5ad` |
| Screen-result digest recorded by continuation | `78b6a1bb940d816275f9557417b2c5be068064e1db5b93b8a16d1bf0cc1d0a91` |

The source/protocol snapshot was frozen at `b158f56233af492efd33a9ab6e0371aff1e5f292`. Control records that Git revision; A/B record `0f81d02a03c0d8db282ebe35c894e4477d791f8a`, which adds only publication documentation and evidence. All **27 frozen source/protocol file hashes match across the declaration and all three run summaries**. The JSON retains the full hash map, package/Python identities, clean-worktree flags and original local artifact hashes.

Public evidence replaces absolute machine-specific repository prefixes with relative paths. Original declaration/artifact digests identify the unmodified local records; they are not presented as digests of the path-normalized copies. A separate portable-evidence digest covers the published JSON payload. No home-directory path is published.

The original metal-nut, screw and transistor tests are development-inspected/exploratory for v3 because their v2 failure evidence motivated the follow-up. Their held-out status within the completed v2 study is retained in its historical report. No v3 test images were read. Cable remained prospective: no continuation run or new cable test evaluation occurred. No fresh independent same-category holdout is currently available, and synthetic validation cannot establish new independent real-defect success.

The [v0.1.0 study](CONTROLLED_STUDY.md), [model card](../MODEL_CARD.md) and [hosted deployment](../DEPLOYMENT.md) remain unchanged by this negative screen. The surface model's separate real-defect-supervised result does not resolve the failed normal-only MVTec targets.
