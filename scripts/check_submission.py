"""Independently validate the HW1 submission directory and code archive."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path, PurePosixPath

import yaml


REQUIRED_BUNDLE_FILES = {
    "code.zip",
    "writeup.pdf",
    "ai_disclosure.md",
    "run_config.yaml",
    "metrics.json",
}
REQUIRED_CODE_FILES = {
    "cs8803_hw1/tokenizer.py",
    "cs8803_hw1/model.py",
    "cs8803_hw1/optimizer.py",
    "cs8803_hw1/train.py",
    "cs8803_hw1/generate.py",
    "cs8803_hw1/evaluate.py",
}
FORBIDDEN_CODE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data",
    "artifacts",
    "instructor",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".pyc"}


def require_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics field {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"metrics field {label} has an invalid value")
    return number


def validate_cache_rows(rows: object, label: str) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"metrics field {label} must be a list")
    by_batch = {}
    for row in rows:
        if not isinstance(row, dict) or "batch_size" not in row:
            raise ValueError(f"metrics field {label} contains an invalid row")
        by_batch[int(row["batch_size"])] = row
    for batch_size in (1, 32):
        if batch_size not in by_batch:
            raise ValueError(f"metrics field {label} is missing batch size {batch_size}")
        row = by_batch[batch_size]
        require_number(row.get("analytical_cache_bytes"), f"{label}[{batch_size}].cache", positive=True)
        require_number(row.get("cached_tokens_per_second"), f"{label}[{batch_size}].throughput", positive=True)
        require_number(row.get("speedup"), f"{label}[{batch_size}].speedup", positive=True)


def validate_metrics(metrics: object) -> None:
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must contain a mapping")
    for model_name, expected_steps in (("gqa", 3000), ("mha", 1000)):
        model = metrics.get(model_name)
        if not isinstance(model, dict):
            raise ValueError(f"metrics.json is missing {model_name}")
        if int(model.get("completed_steps", -1)) != expected_steps:
            raise ValueError(
                f"metrics field {model_name}.completed_steps must equal {expected_steps}"
            )
        require_number(model.get("parameter_count"), f"{model_name}.parameter_count", positive=True)
        step_1000 = model.get("step_1000_validation")
        if not isinstance(step_1000, dict):
            raise ValueError(f"metrics.json is missing {model_name}.step_1000_validation")
        require_number(
            step_1000.get("validation_mean_nll"),
            f"{model_name}.step_1000_validation.validation_mean_nll",
            positive=True,
        )
        require_number(
            step_1000.get("validation_perplexity"),
            f"{model_name}.step_1000_validation.validation_perplexity",
            positive=True,
        )
        validate_cache_rows(model.get("cache_rows"), f"{model_name}.cache_rows")

    evaluation = metrics["gqa"].get("metrics")
    if not isinstance(evaluation, dict):
        raise ValueError("metrics.json is missing gqa.metrics")
    for field in ("mean_nll", "perplexity", "bits_per_byte", "target_tokens"):
        require_number(evaluation.get(field), f"gqa.metrics.{field}", positive=True)


def validate_archive(path: Path, allow_incomplete: bool) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_CODE_FILES - names
        if missing:
            raise ValueError(f"code.zip is missing: {sorted(missing)}")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive path: {name}")
            if any(part in FORBIDDEN_CODE_PARTS for part in relative.parts):
                raise ValueError(f"forbidden archive path: {name}")
            if relative.suffix in FORBIDDEN_SUFFIXES:
                raise ValueError(f"forbidden archive file type: {name}")
        if not allow_incomplete:
            for name in REQUIRED_CODE_FILES:
                text = archive.read(name).decode("utf-8")
                if "NotImplementedError" in text:
                    raise ValueError(f"graded implementation is incomplete: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.submission_dir.resolve()
    present = {path.name for path in root.iterdir()}
    missing = REQUIRED_BUNDLE_FILES - present
    if missing:
        raise ValueError(f"submission is missing: {sorted(missing)}")
    unexpected = present - REQUIRED_BUNDLE_FILES
    if unexpected:
        raise ValueError(f"submission contains unexpected entries: {sorted(unexpected)}")
    for name in REQUIRED_BUNDLE_FILES:
        if not (root / name).is_file():
            raise ValueError(f"submission entry is not a file: {name}")
    validate_archive(root / "code.zip", args.allow_incomplete)
    if not (root / "writeup.pdf").read_bytes().startswith(b"%PDF-"):
        raise ValueError("writeup.pdf is not a PDF")
    if not (root / "ai_disclosure.md").read_text(encoding="utf-8").strip():
        raise ValueError("ai_disclosure.md is empty")
    run_config = yaml.safe_load((root / "run_config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(run_config, dict):
        raise ValueError("run config must contain a mapping")
    validate_metrics(metrics)
    print("submission check passed")


if __name__ == "__main__":
    main()
