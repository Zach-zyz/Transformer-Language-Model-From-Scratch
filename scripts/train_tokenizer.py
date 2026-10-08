"""Train the required tokenizer on the fixed shard and verify its checksum."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import ByteBPETokenizer
from release_utils import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "tokenizer_train.txt",
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--skip-canonical-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixed_shard = ROOT / "data" / "tokenizer_train.txt"
    artifact_manifest_path = ROOT / "artifacts" / "manifest.json"
    plan = {
        "input": str(args.input.resolve()),
        "vocab_size": args.vocab_size,
        "output": str(args.output.resolve()),
        "metrics": None if args.metrics is None else str(args.metrics.resolve()),
        "canonical_check": (
            not args.skip_canonical_check
            and args.input.resolve() == fixed_shard.resolve()
            and args.vocab_size == 8192
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.vocab_size < 257:
        raise ValueError("--vocab-size must include all bytes and the special token")

    text = args.input.read_text(encoding="utf-8")
    started_at = time.perf_counter()
    tokenizer = ByteBPETokenizer.train(text, vocab_size=args.vocab_size)
    training_seconds = time.perf_counter() - started_at
    actual_vocabulary_size = len(tokenizer.vocab) + 1

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    tokenizer.save(temporary)
    output_sha256 = sha256_file(temporary)
    os.replace(temporary, output)

    canonical_sha256 = None
    canonical_match = None
    should_check = (
        not args.skip_canonical_check
        and args.input.resolve() == fixed_shard.resolve()
        and args.vocab_size == 8192
    )
    if should_check:
        manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        canonical_sha256 = manifest["files"]["canonical_tokenizer.json"]["sha256"]
        canonical_match = output_sha256 == canonical_sha256

    result = {
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "input_sha256": sha256_file(args.input),
        "requested_vocabulary_size": args.vocab_size,
        "actual_vocabulary_size": actual_vocabulary_size,
        "merge_count": len(tokenizer.merges),
        "training_seconds": training_seconds,
        "output": str(output),
        "output_sha256": output_sha256,
        "canonical_sha256": canonical_sha256,
        "canonical_match": canonical_match,
    }
    if args.metrics is not None:
        write_json(args.metrics, result)
    print(json.dumps(result, indent=2, sort_keys=True))

    if actual_vocabulary_size != args.vocab_size:
        raise SystemExit(
            "training stopped before the requested vocabulary size; "
            "this is expected only for undersized custom corpora"
        )
    if canonical_match is False:
        raise SystemExit(
            "tokenizer checksum does not match the canonical fixed-shard tokenizer"
        )


if __name__ == "__main__":
    main()
