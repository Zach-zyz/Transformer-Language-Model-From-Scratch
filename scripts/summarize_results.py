"""Combine run, evaluation, and cache benchmark JSON into report-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .release_utils import write_json
except ImportError:
    from release_utils import write_json


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(payload)
    return records


def validation_at_step(records: list[dict], step: int) -> dict:
    matching = [record for record in records if int(record.get("step", -1)) == step]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one metrics record for step {step}")
    record = matching[0]
    required = ("validation_mean_nll", "validation_perplexity")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"step {step} metrics are missing {missing}")
    return {key: record[key] for key in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa-run", type=Path, required=True)
    parser.add_argument("--gqa-metrics", type=Path, required=True)
    parser.add_argument("--gqa-evaluation", type=Path, required=True)
    parser.add_argument("--gqa-cache", type=Path, required=True)
    parser.add_argument("--mha-run", type=Path, required=True)
    parser.add_argument("--mha-metrics", type=Path, required=True)
    parser.add_argument("--mha-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    gqa_run = load(args.gqa_run)
    gqa_metrics = load_jsonl(args.gqa_metrics)
    gqa_eval = load(args.gqa_evaluation)
    gqa_cache = load(args.gqa_cache)
    mha_run = load(args.mha_run)
    mha_metrics = load_jsonl(args.mha_metrics)
    mha_cache = load(args.mha_cache)
    summary = {
        "gqa": {
            "parameter_count": gqa_run["parameter_count"],
            "completed_steps": gqa_run["completed_steps"],
            "step_1000_validation": validation_at_step(gqa_metrics, 1000),
            "metrics": gqa_eval["metrics"],
            "cache_rows": gqa_cache["rows"],
        },
        "mha": {
            "parameter_count": mha_run["parameter_count"],
            "completed_steps": mha_run["completed_steps"],
            "step_1000_validation": validation_at_step(mha_metrics, 1000),
            "cache_rows": mha_cache["rows"],
        },
    }
    write_json(args.output_json, summary)

    lines = [
        "# HW1 Result Summary",
        "",
        "## Final GQA Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("mean_nll", "perplexity", "bits_per_byte", "target_tokens"):
        lines.append(f"| {key} | {gqa_eval['metrics'][key]} |")
    lines.extend(
        [
            "",
            "## Architecture Comparison",
            "",
            "| Model | Parameters | Completed steps |",
            "|---|---:|---:|",
            f"| GQA | {gqa_run['parameter_count']} | {gqa_run['completed_steps']} |",
            f"| MHA | {mha_run['parameter_count']} | {mha_run['completed_steps']} |",
            "",
            "## Step-1000 Validation Comparison",
            "",
            "| Model | Mean NLL | Perplexity |",
            "|---|---:|---:|",
            f"| GQA | {summary['gqa']['step_1000_validation']['validation_mean_nll']} | "
            f"{summary['gqa']['step_1000_validation']['validation_perplexity']} |",
            f"| MHA | {summary['mha']['step_1000_validation']['validation_mean_nll']} | "
            f"{summary['mha']['step_1000_validation']['validation_perplexity']} |",
            "",
            "## KV-Cache Benchmark",
            "",
            "| Model | Batch | Speedup | Cache bytes | Cached tokens/s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_name, payload in (("GQA", gqa_cache), ("MHA", mha_cache)):
        for row in payload["rows"]:
            lines.append(
                f"| {model_name} | {row['batch_size']} | {row['speedup']:.3f} | "
                f"{row['analytical_cache_bytes']} | {row['cached_tokens_per_second']:.2f} |"
            )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
