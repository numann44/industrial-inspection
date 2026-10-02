"""Download, integrity-check and safely extract the official KolektorSDD2 ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"
SIZE = 853126555
SOURCE = "https://www.vicos.si/resources/kolektorsdd2/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(archive: Path, destination: Path) -> int:
    """Validate all paths/types and CRCs before extracting any member."""
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if len(infos) > 10000 or sum(info.file_size for info in infos) > 4 * 1024**3:
            raise ValueError("Unexpectedly large KSDD2 archive")
        names = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or not path.parts or info.filename in names
                    or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            names.add(info.filename)
        corrupt = source.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC verification failed: {corrupt}")
        destination.mkdir(parents=True, exist_ok=True)
        resolved = destination.resolve()
        for info in infos:
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            if not target.resolve().is_relative_to(resolved):
                raise ValueError(f"ZIP destination escapes dataset root: {target}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
        return len(infos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-sha256", help="Optional locally recorded checksum for repeat verification")
    args = parser.parse_args()
    archive = args.data_dir / "downloads" / "KolektorSDD2.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_download and (not archive.exists() or archive.stat().st_size != SIZE):
        subprocess.run(["curl", "--fail", "--location", "--continue-at", "-", "--retry", "3",
                        "--connect-timeout", "30", "--output", str(archive), URL], check=True)
    if archive.stat().st_size != SIZE:
        raise ValueError(f"Archive size differs from verified official download: expected {SIZE}")
    print("Computing local SHA-256; no publisher-provided checksum is asserted...", flush=True)
    checksum = sha256_file(archive)
    if args.expected_sha256 and checksum != args.expected_sha256:
        raise ValueError("Archive differs from expected locally recorded SHA-256")
    members = None
    if not args.verify_only:
        print("Checking ZIP CRCs and extracting safely...", flush=True)
        members = extract_archive(archive, args.data_dir / "ksdd2")
    report = {"dataset": "KolektorSDD2", "source_page": SOURCE, "source_url": URL,
              "size_bytes": SIZE, "sha256": checksum,
              "checksum_origin": "locally computed from official HTTPS download; not publisher-supplied",
              "extracted": not args.verify_only, "archive_members": members,
              "zip_crc_verified": not args.verify_only, "license": "CC BY-NC-SA 4.0",
              "pickle_files_executed": False}
    output = args.data_dir / "ksdd2-download-verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
