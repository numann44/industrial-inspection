"""Download, verify and safely extract the complete MVTec AD archive.

Only a verified archive is extracted. Raw data is local and excluded from Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

URL = "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"
SIZE = 5264982680
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path) -> str:
    if path.stat().st_size != SIZE:
        raise ValueError(f"Unexpected archive size: {path.stat().st_size}; expected {SIZE}")
    digest = sha256_file(path)
    if digest != SHA256:
        raise ValueError(f"SHA-256 mismatch: {digest}; expected {SHA256}")
    return digest


def extract_archive(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, mode="r|xz") as source:
        for member in source:
            # Python's data filter rejects traversal, unsafe links and special files.
            source.extract(member, destination, filter="data")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    archive = args.data_dir / "downloads" / "mvtec_anomaly_detection.tar.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_download and (not archive.exists() or archive.stat().st_size != SIZE):
        subprocess.run([
            "curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
            "--connect-timeout", "30", "--output", str(archive), URL,
        ], check=True)
    print("Verifying complete archive SHA-256...", flush=True)
    digest = verify_archive(archive)
    print(f"Verified {digest}", flush=True)
    destination = args.data_dir / "mvtec_ad"
    members = None
    if not args.verify_only:
        print("Extracting all 15 categories...", flush=True)
        members = extract_archive(archive, destination)
        missing = [name for name in CATEGORIES if not (destination / name / "train" / "good").is_dir()]
        if missing:
            raise ValueError(f"Missing categories after extraction: {missing}")
    report = {
        "dataset": "MVTec AD", "source_url": URL, "size_bytes": SIZE,
        "sha256": digest, "archive_verified": True,
        "extracted": not args.verify_only, "archive_members": members,
        "categories": list(CATEGORIES),
        "checksum_source": "https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/data/datamodules/image/mvtecad.py",
        "license": "CC BY-NC-SA 4.0",
    }
    (args.data_dir / "download-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
