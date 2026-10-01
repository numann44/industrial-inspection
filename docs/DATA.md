# Dataset availability, provenance and license

MVTec AD is available as a complete 5,264,982,680-byte XZ archive. The download was verified with a successful HTTP response and a real byte-range GET before beginning the full local download. The current official website uses a download form; the maintained Anomalib project publishes the working archive link and SHA-256 used by our preparation script.

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

## Sources

- [Official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Official paper, Table 1](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf)
- [Download definition with SHA-256](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py)
