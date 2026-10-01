# Initial synthesis audit and exploratory model interpretation

This is a descriptive audit of **normal training images only**, plus interpretation of the already produced pilot report. The audit script reads no validation, calibration or real-defect test images/masks. It does not establish why a particular real defect was missed.

## Reproduce the normal-data audit

```bash
python scripts/audit_synthesis.py --manifest data/manifest.json \
  --output docs/results/synthesis-baseline-audit.json
```

The audit uses the 154 normal metal-nut training records in manifest order, 128-pixel bilinear resizing matching the training loader, and CPU procedural synthesis. It forces one scratch, appearance patch and texture patch per image, producing 462 examples. The seed is `420000 + image_index * 3 + mode_index`. Clean-return probability is deliberately bypassed; the numbers describe generated corruptions rather than the full mixture of clean/corrupt training examples.

The JSON report records the portable split digest, source-file hash, dependency versions, seed, interpolation method, sampling rules and quantiles. Source images are checked against their audited SHA-256 hashes before use.

## Measurements

Brightness here means arithmetic mean RGB in `[0,1]`, not perceptual luminance. Candidate foreground is the descriptive heuristic `mean RGB > 0.12`; it is not ground-truth part segmentation. The central hole remains background under this rule.

| Measurement | 5th percentile | Median | 95th percentile |
| --- | ---: | ---: | ---: |
| Sampled border brightness | 0.0928 | 0.0967 | 0.1033 |
| Sampled candidate foreground brightness | 0.1908 | 0.3595 | 0.5752 |
| Candidate foreground fraction of image | 48.86% | 49.10% | 49.38% |
| Synthetic mask fraction outside candidate foreground | 0.00% | 20.53% | 53.44% |
| Synthetic mask fraction of full image | 0.93% | 4.81% | 14.18% |
| Absolute mean RGB change inside synthetic mask | 0.0399 | 0.1680 | 0.3725 |

The object occupies approximately half the image under this heuristic. Unrestricted corruption frequently changes the background: the median generated mask has about one fifth of its pixels outside the candidate surface. This supports investigating foreground-constrained synthesis. It does **not** prove that background corruption caused the pilot's failures, nor establish that foreground-constrained synthesis will improve real-defect performance.

The legacy generator uses colored stripes and appearance/texture patches. It does not explicitly model general part deformation. Restricting all changed pixels to the original surface also cannot represent outward growth of an object silhouette. A local edge warp can approximate some inward changes, but is not a physical simulation of bending. Thin scratches, subtle lighting changes and shifted surface textures are hypotheses to test, with their own realism limitations.

## Pilot results require prevalence context

The initial 128-pixel, 30-epoch pilot's already generated report contains:

| Measurement | Pilot result |
| --- | ---: |
| Image AUROC | 0.655 |
| Defective parts detected | 37 / 93 (39.8%) |
| Defective parts missed | 56 / 93 |
| Normal parts falsely flagged | 2 / 22 (9.1%) |
| Precision among flagged parts | 37 / 39 (94.9%) |
| Image average precision | 0.910 |
| Pixel average precision on original masks | 0.213 |

Defective images make up 93 of the 115 test images (80.9%). High precision and image average precision therefore coexist with weak detection recall; they must not be presented as 94.9% accuracy or as evidence of a reliable inspection system. Pixel-positive prevalence is approximately 11.7%, which provides context for pixel AP without replacing localization evaluation.

Detection at the frozen threshold is uneven across groups: bent `1/25`, scratch `4/23`, color `13/22`, and flip `19/23`. This establishes different observed recall across groups. The small group sizes and absence of a controlled comparison prevent attributing those differences to one design choice.

## How to interpret the next variant

The proposed 256-pixel, 100-epoch foreground variant changes resolution, training duration, learning rate, clean-example probability, corruption placement and corruption families. It is a **combined exploratory variant**, not a causal foreground-only ablation. Report its result alongside the original pilot and any reconstruction baseline, including unsuccessful outcomes.

The initial real-defect test report has already been inspected. Later changes informed by those results remain exploratory on this category; calling configurations predeclared before the next training run does not restore a previously unseen test set. Additional categories can provide further evidence under a frozen development protocol, while category-specific training must be clearly distinguished from transfer of the same checkpoint.

A causal foreground comparison would hold resolution, training schedule, corruption families, altered-area distribution, split, score and calibration procedure fixed. A reconstruction baseline comparison describes whole methods when capacity, losses, corruption and training budget differ; it cannot isolate the effect of segmentation supervision.
