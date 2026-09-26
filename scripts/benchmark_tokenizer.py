"""Benchmark tokenizer training/encoding and produce vocabulary-prefix analysis."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs8803_hw1.tokenizer import ByteBPETokenizer
from release_utils import sha256_file, write_json


VOCABULARY_SIZES = (512, 1024, 2048, 4096, 8192)


def tokenizer_prefix(tokenizer: ByteBPETokenizer, vocabulary_size: int) -> ByteBPETokenizer:
    merge_count = max(0, vocabulary_size - 257)
    merges = tokenizer.merges[:merge_count]
    maximum_id = 256 + len(merges)
    vocab = {
        token_id: value
        for token_id, value in tokenizer.vocab.items()
        if token_id <= maximum_id
    }
    return ByteBPETokenizer(vocab=vocab, merges=merges)


def render_svg(rows: list[dict], path: Path) -> None:
    width, height = 720, 420
    left, right, top, bottom = 80, 30, 30, 70
    xs = [row["vocabulary_size"] for row in rows]
    ys = [row["bytes_per_token"] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    def point(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_min) / (x_max - x_min) * (width - left - right)
        py = top + (y_max - y) / (y_max - y_min) * (height - top - bottom)
        return px, py

    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(x, y) for x, y in zip(xs, ys)))
    labels = []
    for row in rows:
        x, y = point(row["vocabulary_size"], row["bytes_per_token"])
        labels.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#166534"/>'
            f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" '
            f'font-size="12">{row["bytes_per_token"]:.3f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<polyline points="{points}" fill="none" stroke="#166534" stroke-width="2"/>
{''.join(labels)}
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14">Vocabulary size</text>
<text x="20" y="{height/2}" text-anchor="middle" font-size="14"
 transform="rotate(-90 20 {height/2})">Bytes per token</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "canonical_tokenizer.json")
    parser.add_argument("--text", type=Path, default=ROOT / "data" / "validation_span.txt")
    parser.add_argument("--train-shard", type=Path, default=ROOT / "data" / "tokenizer_train.txt")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save-trained-tokenizer", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "tokenizer": str(args.tokenizer.resolve()),
        "text": str(args.text.resolve()),
        "train_shard": str(args.train_shard.resolve()),
        "train": args.train,
        "vocabulary_sizes": VOCABULARY_SIZES,
        "repeats": args.repeats,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    training_seconds = None
    if args.train:
        shard = args.train_shard.read_text(encoding="utf-8")
        started_at = time.perf_counter()
        tokenizer = ByteBPETokenizer.train(shard, vocab_size=8192)
        training_seconds = time.perf_counter() - started_at
        if len(tokenizer.vocab) + 1 != 8192:
            raise ValueError("fixed shard did not produce the required vocabulary")
        if args.save_trained_tokenizer is not None:
            tokenizer.save(args.save_trained_tokenizer)
    else:
        tokenizer = ByteBPETokenizer.load(args.tokenizer)

    text = args.text.read_text(encoding="utf-8")
    raw_bytes = len(text.encode("utf-8"))
    words = len(text.split())
    rows = []
    for vocabulary_size in VOCABULARY_SIZES:
        candidate = tokenizer_prefix(tokenizer, vocabulary_size)
        candidate.encode(text, {"<|endoftext|>"})
        timings = []
        token_ids = []
        for _ in range(args.repeats):
            started_at = time.perf_counter()
            token_ids = candidate.encode(text, {"<|endoftext|>"})
            timings.append(time.perf_counter() - started_at)
        median_seconds = statistics.median(timings)
        rows.append(
            {
                "vocabulary_size": vocabulary_size,
                "tokens": len(token_ids),
                "bytes_per_token": raw_bytes / len(token_ids),
                "tokens_per_word": len(token_ids) / words,
                "encoding_seconds_median": median_seconds,
                "encoding_megabytes_per_second": (
                    raw_bytes / 1_000_000 / max(median_seconds, 1e-12)
                ),
            }
        )
    result = {
        "tokenizer_sha256": sha256_file(args.tokenizer)
        if not args.train
        else None,
        "training_shard_bytes": args.train_shard.stat().st_size,
        "training_shard_sha256": sha256_file(args.train_shard),
        "training_seconds": training_seconds,
        "validation_bytes": raw_bytes,
        "validation_sha256": sha256_file(args.text),
        "rows": rows,
    }
    write_json(args.output, result)
    render_svg(rows, args.plot)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
