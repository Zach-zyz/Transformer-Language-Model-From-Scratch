"""Download the pinned TinyStories text files with resume and hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "source_manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {actual_size}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path}: SHA-256 mismatch; expected {expected_sha256}, found {actual_sha256}"
        )


def copy_response(response: BinaryIO, destination: BinaryIO, expected_size: int) -> None:
    written = destination.tell()
    next_report = written + 256 * 1024 * 1024
    while chunk := response.read(CHUNK_SIZE):
        destination.write(chunk)
        written += len(chunk)
        if written >= next_report:
            print(f"  downloaded {written:,}/{expected_size:,} bytes", flush=True)
            next_report += 256 * 1024 * 1024


def open_download(url: str, start: int = 0):
    headers = {"User-Agent": "cse8803-hw1-downloader/1"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=120)


def download_file(entry: dict, output_root: Path) -> Path:
    destination = output_root / entry["local_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(entry["size"])
    expected_sha256 = str(entry["sha256"])
    if destination.exists():
        try:
            verify_file(
                destination,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except ValueError as error:
            raise ValueError(
                f"{error}. Remove the corrupt final file before retrying."
            ) from error
        print(f"verified {destination}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_size:
        raise ValueError(f"{partial}: partial file is larger than the pinned source")
    if start == expected_size:
        verify_file(
            partial,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        os.replace(partial, destination)
        print(f"verified {destination}")
        return destination

    print(f"downloading {entry['url']}")
    try:
        response = open_download(entry["url"], start)
    except urllib.error.HTTPError as error:
        if start and error.code == 416:
            start = 0
            partial.unlink(missing_ok=True)
            response = open_download(entry["url"])
        else:
            raise

    with response:
        status = getattr(response, "status", response.getcode())
        if start and status != 206:
            print("  server ignored Range; restarting the partial download")
            start = 0
            mode = "wb"
        else:
            mode = "ab" if start else "wb"
        with partial.open(mode) as handle:
            copy_response(response, handle, expected_size)

    verify_file(partial, expected_size=expected_size, expected_sha256=expected_sha256)
    os.replace(partial, destination)
    print(f"verified {destination}")
    return destination


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported source manifest schema")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--only", choices=("train", "validation"), action="append")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    selected = set(args.only or ())
    entries = [
        entry for entry in manifest["files"] if not selected or entry["id"] in selected
    ]
    plan = {
        "dataset": manifest["dataset"],
        "files": [
            {
                "id": entry["id"],
                "destination": str((args.output_root / entry["local_path"]).resolve()),
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
            for entry in entries
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    failures = []
    for entry in entries:
        destination = args.output_root / entry["local_path"]
        try:
            if args.verify_only:
                verify_file(
                    destination,
                    expected_size=int(entry["size"]),
                    expected_sha256=str(entry["sha256"]),
                )
                print(f"verified {destination}")
            else:
                download_file(entry, args.output_root)
        except (OSError, ValueError, urllib.error.URLError) as error:
            failures.append(str(error))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
