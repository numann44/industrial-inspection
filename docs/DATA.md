# Dataset availability, provenance and license

MVTec AD is available as a complete 5,264,982,680-byte XZ archive. The download was verified with a successful HTTP response and a real byte-range GET before beginning the full local download. The current official website uses a download form; the maintained Anomalib project publishes the working archive link and SHA-256 used by our preparation script.

Dataset attribution: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. Original data copyright: MVTec Software GmbH, 2019.

Expected archive SHA-256:

```text
cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d
```

Expected original dataset counts:

| Category | Normal training | Normal test | Defective test | Total images |
| --- | ---: | ---: | ---: | ---: |
| metal_nut | 220 | 22 | 93 | 335 |
| screw | 320 | 41 | 119 | 480 |
| transistor | 213 | 60 | 40 | 313 |
| All 15 categories | 3,629 | 467 | 1,258 | 5,354 |

All categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper. Ground-truth masks accompany defective test images. The preparation and audit reports contain measured counts, not only these expectations.

No additional dataset is required for initial training: artificial defects are generated procedurally using only training images. Real defect images and masks remain in the original test split.

## License

The dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It is intended here for a noncommercial learning and portfolio experiment. Credit MVTec and the dataset authors, include the license link and indicate changes when publishing transformed sample images. Published sample images and adaptations retain the dataset license. Source-code licensing is separate. Commercial deployment requires separately checking data rights; this repository is not such a deployment.

Raw data, the compressed archive and local manifests containing machine-specific paths are ignored by Git. A sanitized summary can be committed under `docs/results/` after verification.

## Separate supervised dataset: KolektorSDD2

KolektorSDD2 is a separate real-defect supervision track. It is not merged into normal-only MVTec training and its results are not directly interchangeable with anomaly-detection results.

The official 853,126,555-byte ZIP was downloaded from `https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip`. All 6,679 archive members passed ZIP CRC verification before safe extraction. The locally computed SHA-256 is `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`; this is a recorded download identity, not a publisher-provided checksum.

All 3,335 canonical images and 3,335 masks decode successfully. Masks are aligned binary grayscale images and image labels are derived from nonzero mask pixels. The official training set contains 2,085 normal and 246 defective images; its official test set contains 894 normal and 110 defective images. Two extra PNGs, `train/10301 (copy).png` and `train/10301_GT (copy).png`, were decoded and verified byte-identical to their canonical counterparts before exclusion. No `.pyb` pickle file is executed.

Deterministic seed-42, approximately 70/15/15 stratification of official training images yields:

| Split | Normal | Defective | Total |
| --- | ---: | ---: | ---: |
| Training | 1,459 | 172 | 1,631 |
| Validation | 313 | 37 | 350 |
| Calibration | 313 | 37 | 350 |
| Original test | 894 | 110 | 1,004 |

Only the 313 normal calibration images fit the operating threshold; positive calibration examples are held out and unused. Validation has real defects and can select supervised checkpoints. Official test membership stays unchanged.

Exact decoded-pixel duplicate checks found no canonical pairs. A conservative near-duplicate check also found no pairs under 64-bit dHash Hamming distance ≤2 combined with 64×64 bilinear RGB mean absolute difference ≤0.004. Similarity-connected groups would be assigned together; training images related to official test images would be quarantined. These checks do not establish physical product, camera-session or production-batch independence: the provided metadata documents image identity and annotation availability, without product/group identifiers.

Supervised preprocessing preserves aspect ratio by letterboxing to height 640 × width 256. Padding is excluded from losses, image scoring and validation pixel metrics. Final maps are restored to original dimensions and evaluated against original masks. [Measured preparation evidence](results/ksdd2-data-verification.json) records actual counts, image sizes, duplicate checks and portable split identity.

Dataset attribution: Jakob Božič, Domen Tabernik and Danijel Skočaj, *Mixed supervision for surface-defect detection: from weakly to fully supervised learning*, Computers in Industry, 2021. Images and annotations were provided by Kolektor Group. The dataset uses CC BY-NC-SA 4.0; apply attribution, noncommercial and share-alike requirements to published data derivatives.

- [Official KolektorSDD2 page](https://www.vicos.si/resources/kolektorsdd2/)
- [Official split format documentation](https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/master/splits/KSDD2/README.md)
- [Official dataset loader](https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/master/data/input_ksdd2.py)

## MVTec sources

- [Official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Official paper, Table 1](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf)
- [Download definition with SHA-256](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)
