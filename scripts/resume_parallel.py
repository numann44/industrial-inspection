"""Resume the verified MVTec archive endpoint using four bounded range requests.

Useful when a single connection is slow. Run prepare_dataset.py afterwards to
verify SHA-256 before extracting. Existing partial archive bytes are preserved.
"""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import time
from pathlib import Path

from prepare_dataset import SIZE, URL


def main() -> None:
    archive = Path("data/downloads/mvtec_anomaly_detection.tar.xz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.touch()
    start = archive.stat().st_size
    if start == SIZE:
        print("Archive already has its expected size; verify its SHA-256 next.")
        return
    if start > SIZE:
        raise ValueError("Existing archive is larger than expected")
    directory = archive.parent / "ranges"
    directory.mkdir(exist_ok=True)
    plan = {"prefix_bytes": start, "size_bytes": SIZE, "url": URL, "jobs": 4}
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Range plan differs; inspect existing parts before continuing")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    chunk_size = (SIZE - start + 3) // 4

    def fetch(index: int) -> Path:
        first = start + index * chunk_size
        last = min(SIZE - 1, first + chunk_size - 1)
        length = last - first + 1
        part = directory / f"part-{index}.bin"
        header = directory / f"part-{index}.headers"
        expected_range = f"content-range: bytes {first}-{last}/{SIZE}"
        if part.exists() and part.stat().st_size == length and header.exists():
            if expected_range in header.read_text().lower():
                return part
        subprocess.run([
            "curl", "--silent", "--show-error", "--fail", "--location",
            "--connect-timeout", "30", "--retry", "3", "--range", f"{first}-{last}",
            "--max-filesize", str(length), "--dump-header", str(header),
            "--output", str(part), URL,
        ], check=True)
        content_range = header.read_text().lower()
        if expected_range not in content_range or part.stat().st_size != length:
            raise ValueError(f"Invalid range response for chunk {index}")
        return part

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, index) for index in range(4)]
        while not all(future.done() for future in futures):
            downloaded = sum(path.stat().st_size for path in directory.glob("part-*.bin"))
            print(json.dumps({"downloaded_bytes": start + downloaded, "total_bytes": SIZE,
                              "percent": round(100 * (start + downloaded) / SIZE, 1)}), flush=True)
            time.sleep(10)
        parts = [future.result() for future in futures]
    # Build atomically so failed concatenation never destroys the resumable prefix.
    assembled = archive.with_suffix(".assembled")
    with assembled.open("wb") as destination:
        with archive.open("rb") as prefix:
            shutil.copyfileobj(prefix, destination, 8 * 1024 * 1024)
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    if assembled.stat().st_size != SIZE:
        raise ValueError("Assembled archive has the wrong size")
    assembled.replace(archive)
    for part in parts:
        part.unlink()
    for header in directory.glob("*.headers"):
        header.unlink()
    plan_path.unlink()
    directory.rmdir()
    print("Complete archive assembled. Run prepare_dataset.py --skip-download to verify and extract.", flush=True)


if __name__ == "__main__":
    main()
