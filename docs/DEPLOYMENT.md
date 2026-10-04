# CPU demo deployment

**Live app:** [numan-industrial-inspection.streamlit.app](https://numan-industrial-inspection.streamlit.app/) · **Published artifacts:** [v0.1.0](https://github.com/numann44/industrial-inspection/releases/tag/v0.1.0)

The hosted Streamlit app serves the four validation-selected v0.1.0 checkpoints: KolektorSDD2 surface, metal nut, screw and transistor. The release is pinned to commit `e76ff78`. Each remotely downloaded weight was checked against the local SHA-256. The datasets and pretrained PatchCore reference are not required to serve the app.

## Verified hosted behavior

All four task selections were exercised on the actual hosted service with their packaged examples. The displayed category, checkpoint identity, frozen threshold and decision were checked. A real Chrome file upload of `assets/examples/kolektor_surface_01_defective.png` returned a defective decision, and all four result exports were downloaded: JSON, overlay PNG, heatmap PNG and raw float32 NPZ.

The uploaded surface example's JSON score agrees exactly with local CPU inference at **0.9999679327011108**. The downloaded float32 map's maximum absolute difference is **1.55e-6**; downloaded rendered images agree exactly. These checks validate the hosted inference/export path on the tested input, not bitwise identity for every possible image or device. [Portable deployment evidence](results/DEPLOYMENT_VERIFICATION.json) records the concrete inputs, model hashes and validation/error-check scope.

![Actual hosted app with selected surface-model measurements and limitations](images/live-study-v0.1.0.png)

Actual hosted screenshot captured October 4, 2026. The page shows 105 detected defects, five misses, 89 normal false alarms and 805 accepted normal images. The rounded 10.0% false-alarm display corresponds to **89/894 = 9.955%**. Dataset target status, confidence-interval limits and brightness/JPEG failures remain visible.

Malformed-file and 11.5 MB oversized-file checks also passed. The app returned `Cannot decode this image; upload a valid PNG, JPEG or WebP` and `Error: File must be 10.0MB or smaller.`, respectively. Packaged examples and the measured-performance panel were restored afterward.

This verification used browser sessions on the development computer against a remote Linux service. It does not establish verification from a second physical client device.

## Release and source checks

The release-tag and main-branch Linux workflows passed:

- [Release-tag workflow, successful attempt 2](https://github.com/numann44/industrial-inspection/actions/runs/37167492944/attempts/2).
- [Main-branch workflow](https://github.com/numann44/industrial-inspection/actions/runs/37167626497).

The first release-tag attempt ran before its model assets were available; the successful retry verifies the published assets. The preview follow-up passed **139 local tests** and [its Linux workflow](https://github.com/numann44/industrial-inspection/actions/runs/37173629613). That count is not attributed retroactively to the earlier release-tag workflow.

The separate acquisition-evaluation source at `05d6660487fe4891bd3432a35925b98c59c6c194` passed **164 local tests** and [Linux CI](https://github.com/numann44/industrial-inspection/actions/runs/37185709663). These verify the later evaluation and artifact code; they do not change the four deployed checkpoint identities or establish a model-quality improvement.

## Community Cloud configuration

| Field | Value |
| --- | --- |
| Repository | `numann44/industrial-inspection` |
| Branch | `main` |
| Main file path | `app.py` |
| Python | `3.12` |
| Secrets | None |
| Hosting | Free Streamlit Community Cloud, CPU |

The root `requirements.txt` installs CPU PyTorch on Linux. Follow the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) to reproduce the hosting configuration. `artifacts/models.json` is authoritative for the active model choices and their project-owned GitHub release URLs.

## Artifacts, visualization and privacy

A first request downloads the selected weight, validates its SHA-256 and stores it in `.cache/inspection`; later requests reuse the cached engine. Invalid checksums stop loading. The application does not accept uploaded checkpoint files.

The original v0.1.0 checkpoint-relative color range can saturate for the surface model's extremely low image threshold. The follow-up app renderer uses a fixed 0–1 sigmoid activation range for supervised surface previews, preserving both the original checkpoint range and the explicit preview range in exported JSON. It changes no weights, image scores, thresholds or float32 maps; the historical CLI/engine rendering remains reproducible. Normal-only previews retain their checkpoint scale. **Exported float32 maps are authoritative for underlying activation values**; display images are previews. Scores and color intensities are not defect probabilities, and model activations are not certified defect boundaries.

The [hosted preview check](results/PREVIEW_VERIFICATION.json) downloaded all four exports again after this change. The score, threshold, decision and raw maps match the prior hosted result exactly; the new overlay and heatmap match the local preview exactly on this tested example.

![Actual hosted surface example and fixed-range activation overlay](images/live-surface-overlay.png)

Actual hosted screenshot captured October 4, 2026, showing KolektorSDD2 `test/20042.png`. It is a demonstration example, not an independent evaluation; see [image attribution](images/ATTRIBUTION.md).

Users explicitly choose the matching task; the application does not identify arbitrary parts. PNG, JPEG and WebP uploads are limited to 10 MiB and 20 million decoded pixels. Uploaded images are decoded and processed in memory and are never saved by the application. The application does not log image bytes or filenames; hosting-provider access logging is outside its control.

## Historical deployment and continuing quality work

The [v0.1.0-alpha.1 release](https://github.com/numann44/industrial-inspection/releases/tag/v0.1.0-alpha.1) and its exploratory metal-nut pilot remain historical evidence. Its earlier hosted normal-example score difference was `3.64e-12`, and its uploaded failure-example map difference was below `2e-10`. Those numbers belong to the alpha checkpoint, not the new surface model; the original [alpha deployment record](results/DEPLOYMENT_VERIFICATION_ALPHA.json) is retained. Historical screenshots `live-inspection.png` and `live-demo.png` are identified in [their attribution](images/ATTRIBUTION.md).

The completed v0.1.0 publication does not mark all quality goals complete. The surface model passes dataset point estimates only; the three selected MVTec models fail the target. The bounded [v3 synthesis screen](results/STUDY_V3_SCREEN.md) completed three runs and failed its validation-only gate. No conditional continuation jobs were started. Published model weights remain those of v0.1.0.

The [two-run acquisition-robustness comparison](results/KSDD2_ROBUSTNESS_RESULTS.md) is also complete. Control detects **104/110** defects with **98/894** normal false alarms and native pixel AP **0.800701**; augmentation detects **106/110** with **115/894** false alarms and pixel AP **0.811400**. Its validation gate passed, but **both original-test false-alarm targets failed**. Brightness ×1.2 false alarms improved from 207 to 131, while JPEG-quality-60 false alarms worsened from 153 to 185, each out of 894 normals. These results are exploratory reused-test evidence. [Portable records](results/KSDD2_ROBUSTNESS_RESULTS.json) and [24 attributed examples](results/KSDD2_ROBUSTNESS_RESULTS.md#auditable-examples-and-errors) document the comparison; neither new model is in the active registry or live demo.

A separate [exactly two-run learned image-decision direction](KSDD2_DECISION_SCREEN.md) is planned and under implementation. It has not launched and does not change the serving contract or published weights. New evaluation code, completed training and published reports are distinct from a documented model-promotion decision.
