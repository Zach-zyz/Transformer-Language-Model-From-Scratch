"""Regenerate fixed spans and pack TinyStories with a student tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from cs8803_hw1.tokenizer import ByteBPETokenizer, SPECIAL_TOKEN, SPECIAL_TOKEN_ID


CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(path: Path, expected: dict) -> None:
    if path.stat().st_size != int(expected["size"]):
        raise ValueError(f"size mismatch for {path}")
    actual = sha256_file(path)
    if actual != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def extract_first_documents(
    source: Path,
    destination: Path,
    *,
    documents: int,
    boundary: bytes = b"<|endoftext|>\n",
) -> None:
    if documents <= 0:
        raise ValueError("documents must be positive")
    buffer = bytearray()
    found = 0
    with source.open("rb") as handle:
        while found < documents and (chunk := handle.read(CHUNK_SIZE)):
            buffer.extend(chunk)
            search_from = 0
            while found < documents:
                index = buffer.find(boundary, search_from)
                if index < 0:
                    break
                found += 1
                search_from = index + len(boundary)
                if found == documents:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(bytes(buffer[:search_from]))
                    return
    raise ValueError(f"{source} contains fewer than {documents} complete documents")


def iter_encoded_chunks(
    source: Path,
    tokenizer: ByteBPETokenizer,
    *,
    max_documents: int | None = None,
) -> Iterator[tuple[list[int], int]]:
    pending = ""
    documents = 0
    with source.open("r", encoding="utf-8", newline="") as handle:
        while chunk := handle.read(1024 * 1024):
            pending += chunk
            while SPECIAL_TOKEN in pending:
                before, pending = pending.split(SPECIAL_TOKEN, 1)
                token_ids = tokenizer.encode(before)
                token_ids.append(SPECIAL_TOKEN_ID)
                documents += 1
                yield token_ids, documents
                if max_documents is not None and documents >= max_documents:
                    return
    if pending:
        yield tokenizer.encode(pending), documents


def pack_file(
    source: Path,
    destination: Path,
    tokenizer: ByteBPETokenizer,
    *,
    max_documents: int | None = None,
    source_label: str | None = None,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    token_count = 0
    document_count = 0
    with temporary.open("wb") as handle:
        for token_ids, document_count in iter_encoded_chunks(
            source,
            tokenizer,
            max_documents=max_documents,
        ):
            values = np.asarray(token_ids, dtype="<i8")
            values.tofile(handle)
            token_count += len(values)
    os.replace(temporary, destination)
    return {
        "path": destination.name,
        "dtype": "<i8",
        "tokens": token_count,
        "documents": document_count,
        "size": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source": source_label or source.name,
        "source_size": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "partial": max_documents is not None,
    }


def source_entry(source_manifest: dict, source_id: str) -> dict:
    return next(entry for entry in source_manifest["files"] if entry["id"] == source_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "source_manifest.json")
    parser.add_argument(
        "--preparation-manifest",
        type=Path,
        default=ROOT / "preparation_manifest.json",
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "packed")
    parser.add_argument("--regenerate-fixed-spans", action="store_true")
    parser.add_argument("--derive-only", action="store_true")
    parser.add_argument("--max-train-documents", type=int)
    parser.add_argument("--max-validation-documents", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_manifest = load_json(args.source_manifest)
    preparation = load_json(args.preparation_manifest)
    plan = {
        "raw_root": str(args.raw_root.resolve()),
        "tokenizer": None if args.tokenizer is None else str(args.tokenizer.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "regenerate_fixed_spans": args.regenerate_fixed_spans,
        "packed_dtype": preparation["packed"]["dtype"],
        "outputs": preparation["packed"]["outputs"],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if args.regenerate_fixed_spans:
        for derivative in preparation["derivatives"].values():
            source = source_entry(source_manifest, derivative["source_id"])
            source_path = args.raw_root / source["local_path"]
            verify_file(source_path, source)
            destination = ROOT / derivative["output"]
            extract_first_documents(
                source_path,
                destination,
                documents=int(derivative["documents"]),
            )
            verify_file(destination, derivative)
            print(f"regenerated {destination}")
    else:
        for derivative in preparation["derivatives"].values():
            verify_file(ROOT / derivative["output"], derivative)

    if args.derive_only:
        return
    if args.tokenizer is None:
        raise SystemExit("--tokenizer is required unless --derive-only is used")

    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    outputs = preparation["packed"]["outputs"]
    sources = {
        "train": args.raw_root / source_entry(source_manifest, "train")["local_path"],
        "validation": args.raw_root
        / source_entry(source_manifest, "validation")["local_path"],
        "validation_span": ROOT
        / preparation["derivatives"]["validation_span"]["output"],
    }
    source_labels = {
        "train": source_entry(source_manifest, "train")["path"],
        "validation": source_entry(source_manifest, "validation")["path"],
        "validation_span": preparation["derivatives"]["validation_span"]["output"],
    }
    for source_id in ("train", "validation"):
        verify_file(
            sources[source_id],
            source_entry(source_manifest, source_id),
        )

    packed = {
        "schema_version": 1,
        "tokenizer": {
            "path": args.tokenizer.name,
            "sha256": sha256_file(args.tokenizer),
            "vocabulary_size": len(tokenizer.vocab) + 1,
        },
        "files": {},
    }
    limits = {
        "train": args.max_train_documents,
        "validation": args.max_validation_documents,
        "validation_span": None,
    }
    for name, source in sources.items():
        output_path = args.output_dir / outputs[name]
        print(f"packing {source} -> {output_path}")
        packed["files"][name] = pack_file(
            source,
            output_path,
            tokenizer,
            max_documents=limits[name],
            source_label=source_labels[name],
        )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(packed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
