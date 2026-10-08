"""Pack the pinned optional OpenWebText shard with a submitted tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tokenizer import ByteBPETokenizer, SPECIAL_TOKEN_ID
from data.download_data import load_manifest, sha256_file, verify_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "openwebtext_source_manifest.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "openwebtext" / "raw" / "train-00000-of-00080.parquet",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "openwebtext" / "packed",
    )
    parser.add_argument("--validation-documents", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "input": str(args.input.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "validation_documents": args.validation_documents,
        "split": "last N rows are validation; all earlier rows are training",
        "packed_dtype": "<i8",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.validation_documents <= 0:
        raise ValueError("--validation-documents must be positive")

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "OpenWebText preparation requires pyarrow; install it in a separate "
            "bonus environment with `python -m pip install pyarrow==23.0.0`"
        ) from error

    source_manifest = load_manifest(args.source_manifest)
    source_entry = source_manifest["files"][0]
    verify_file(
        args.input,
        expected_size=int(source_entry["size"]),
        expected_sha256=str(source_entry["sha256"]),
    )
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    parquet_file = parquet.ParquetFile(args.input)
    total_documents = int(parquet_file.metadata.num_rows)
    if args.validation_documents >= total_documents:
        raise ValueError("validation split must leave at least one training document")
    validation_start = total_documents - args.validation_documents

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.bin",
        "validation_span": output_dir / "validation_span.bin",
    }
    temporary = {
        name: path.with_name(path.name + ".tmp") for name, path in paths.items()
    }
    handles = {name: path.open("wb") for name, path in temporary.items()}
    statistics = {
        "train": {"documents": 0, "tokens": 0, "source_size": 0},
        "validation_span": {"documents": 0, "tokens": 0, "source_size": 0},
    }
    row_index = 0
    try:
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=256):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    raise ValueError(f"row {row_index} does not contain text")
                split = "train" if row_index < validation_start else "validation_span"
                token_ids = tokenizer.encode(text)
                token_ids.append(SPECIAL_TOKEN_ID)
                np.asarray(token_ids, dtype="<i8").tofile(handles[split])
                statistics[split]["documents"] += 1
                statistics[split]["tokens"] += len(token_ids)
                statistics[split]["source_size"] += len(text.encode("utf-8"))
                row_index += 1
    finally:
        for handle in handles.values():
            handle.close()
    if row_index != total_documents:
        raise ValueError(
            f"expected {total_documents} parquet rows, processed {row_index}"
        )

    for name, path in paths.items():
        os.replace(temporary[name], path)
        statistics[name].update(
            {
                "path": path.name,
                "dtype": "<i8",
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": source_manifest["dataset"],
        "source": {
            "path": source_entry["path"],
            "size": source_entry["size"],
            "sha256": source_entry["sha256"],
            "rows": total_documents,
        },
        "tokenizer": {
            "path": args.tokenizer.name,
            "sha256": sha256_file(args.tokenizer),
            "vocabulary_size": len(tokenizer.vocab) + 1,
        },
        "files": statistics,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
