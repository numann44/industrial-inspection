# CPU demo deployment

The tested entrypoint is `app.py`, served by Streamlit. The deployed app needs the small checksum-pinned model release and attributed sample images; it does not need either training dataset or the pretrained reference model.

## Community Cloud configuration

| Field | Value |
| --- | --- |
| Repository | `numann44/industrial-inspection` |
| Branch | `main` |
| Main file path | `app.py` |
| Python | `3.12` |
| Secrets | None |

After signing in to Community Cloud, create an app from the repository and set Python 3.12 in Advanced settings. The root `requirements.txt` installs CPU PyTorch on Linux. Follow the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for the hosting controls. No public app URL is claimed until a deployment has been opened and tested independently.

## Artifacts and privacy

`artifacts/models.json` pins the model's SHA-256 and project-owned GitHub release URL. A first request downloads and verifies the file into `.cache/inspection`; later requests reuse the cached engine. Invalid checksums stop loading. Uploaded images are decoded and processed in memory, never saved by the application. PNG, JPEG and WebP uploads are limited to 10 MiB and 20 million decoded pixels. The application does not log image bytes or filenames; hosting-provider access logging is outside the application.

The checkpoint display range is fixed, and anomaly scores are not probabilities. A user chooses the category-specific checkpoint; the app does not identify arbitrary parts. Model status and measured failure cases stay visible.

## Release acceptance

- Clean Linux installation and tests pass in GitHub Actions, including a freshly downloaded release and real CPU demo inference.
- An independent browser opens the public URL, selects a packaged example and uploads a valid photo.
- JSON, overlay PNG, heatmap PNG and float32 map exports agree with the shared engine and frozen checkpoint.
- Malformed and oversized images produce readable errors.
- The public URL and deployment commit are recorded only after those checks; a running local app is not described as a hosted deployment.

The preliminary artifact release is `v0.1.0-alpha.1`. It retains the existing exploratory pilot and its unmet recall target. The final `v0.1.0` release remains gated on the completed controlled study, honest outcome reporting and the public demo checks.
