# Audited pretrained PatchCore reference

This reference answers a practical question: how much stronger can defect detection be with pretrained visual features and a normal-part memory bank? It is separate from the models trained from random weights. Its exploratory metal-nut results do not select the scratch-model configuration or the released demo checkpoint.

| Measurement | Frozen result |
| --- | --- |
| Image AUROC | 0.999511; 95% image bootstrap interval [0.997067, 1.000000] |
| Native-mask pixel average precision | 0.885310 |
| Defective parts detected | 93 / 93; recall 100%; Wilson interval [96.0%, 100%] |
| Good parts falsely flagged | 4 / 22; false-alarm rate 18.2%; Wilson interval [7.3%, 38.5%] |
| Frozen image threshold | 3.774648666381836 squared feature distance |
| Calibration exceedances | 4 / 33 normal calibration images |
| Recall ≥90%, false alarms ≤10% | **Failed: false-alarm target not met** |
| CPU fitting, calibration and saving | 285.478 seconds; two CPU threads |
| CPU forward latency | Median 618.4 ms; p95 1128.1 ms, 32 single-image measurements |

No threshold was changed using test labels. The narrow ranking-error margin does not imply perfect image ranking: AUROC is 0.999511, not 1. The 22 normal test images provide limited evidence about future false alarms. Latency was measured during the wider study, without an idle-system guarantee; it excludes image decoding, preprocessing and response generation. It is not a production service latency claim.

## Fixed method and split

The adapter uses the [Amazon Science PatchCore implementation](https://github.com/amazon-science/patchcore-inspection/tree/fcaa92f124fb1ad74a7acf56726decd4b27cbcad), pinned at `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`. Its feature extractor is torchvision's WideResNet50-2 with `IMAGENET1K_V1` weights. Images receive the checkpoint's 256 × 256 square preprocessing and ImageNet normalization. Features from `layer2` and `layer3` use a patch size of 3 and 1024-dimensional embedding; approximate greedy coreset selection retains 1% of normal feature patches, producing 1,576 memory patches from 154 normal images.

The test split remains the original MVTec AD metal-nut test split: 93 defective and 22 normal images. Only the audited training normals build the memory bank. The separate 33 normal calibration images determine a fixed linear-interpolated 90th-percentile threshold. The 33 validation normals do not enter this fixed reference's memory fitting or calibration. The 70/15/15 normal partition and 256-square preprocessing differ from published-paper settings; this is a reference implementation under our protocol, not a reproduction of the paper's benchmark numbers.

Nearest-neighbor search uses exact float32 squared Euclidean distances in PyTorch on CPU, replacing the FAISS backend to avoid duplicate OpenMP runtimes on macOS. The upstream `common.py` has only a lazy FAISS import adjustment; the other runtime modules remain unchanged. Image scoring preserves the official maximum unsmoothed patch distance. Localization maps use the upstream bilinear interpolation and Gaussian smoothing with sigma 4. Native-mask pixel AP compares restored maps with original annotations, without shrinking the ground-truth defects. Image scores are distances, not probabilities.

## Implementation and artifact provenance

Adapter, vendor sources, notices and package versions were hashed **before model initialization and fitting**, then verified unchanged before checkpoint saving. The initialized backbone tensor state was separately hashed and confirmed unchanged throughout fitting and calibration. The backbone hash describes the state after the upstream feature-dimension probe; it is not a checksum of the downloaded weights archive. The official weights URL is retained in the provenance record.

- Checkpoint SHA-256: `cdf0778d3b9e5bed4ed366b0da26f98111aba13a13e2d8b2de1c86764a7d6448`.
- Adapter SHA-256: `496a7002c4da7b6713bb793eaee422e1c83cf8ea1d7365f3c1d29e1975284472`.
- Initialized backbone tensor-state SHA-256: `42dcd4d10f75567fb74f8ac882e8976202b12dc699e45b01c56886c53aecb710`.
- Portable split digest: `2df9587b2f8d3f7f389a7ef2a1e9e13cf21ea6b6738a1d9023e7daf1d59de5ad`.
- [Portable provenance and full metric record](patchcore-reference-provenance.json).
- [Comparison with the three scratch-model pilots](EXPERIMENT_REPORT.md).
- [Balanced activation / ground-truth gallery](galleries/metal_nut-patchcore-002/index.json).

The earlier `metal_nut-patchcore-001` directory is preserved as an **excluded engineering attempt**. Its adapter changed while fitting and its end-only provenance could not establish the executed source version. It contributes no metrics to the evidence tables. The fresh `-002` run is the sole audited reference reported here.

The exact-search test checks 300 queries against directly computed squared distances across chunk boundaries; a separate test verifies exported/restored memory-bank prediction parity without a pretrained download. Four drift tests ensure source, adapter, vendor and package changes are rejected. The shared evaluator preserves the reference's official score rather than substituting the scratch models' top-1% pixel score.

## License and scope

The vendored runtime and notices retain Apache-2.0. [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) images and derivative gallery panels retain CC BY-NC-SA 4.0 attribution. Checkpoints and raw datasets stay outside the source repository. This result is exploratory category-specific research, with ImageNet pretraining and no claim about unseen production domains, multiple training seeds, or blind final benchmarking.
