"""Train or resume the required TinyStories model from packed token arrays."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_token_ids
from src.model import TransformerConfig, TransformerLM
from src.train import (
    build_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    train_steps,
)
from src.utils import set_seed
try:
    from .release_utils import choose_device, load_yaml, sha256_file, write_json
except ImportError:
    from release_utils import choose_device, load_yaml, sha256_file, write_json


RUN_KEYS = {
    "steps",
    "micro_batch_size",
    "sequence_length",
    "accumulation_steps",
    "peak_lr",
    "min_lr",
    "warmup_steps",
    "weight_decay",
    "betas",
    "grad_clip",
    "precision",
    "seed",
    "checkpoint_every",
}


def validate_run_config(config: dict) -> None:
    missing = RUN_KEYS - set(config)
    if missing:
        raise ValueError(f"run configuration is missing: {sorted(missing)}")
    if int(config["steps"]) <= 0 or int(config["checkpoint_every"]) <= 0:
        raise ValueError("steps and checkpoint_every must be positive")
    schedule_total_steps = int(config.get("schedule_total_steps", config["steps"]))
    if schedule_total_steps < int(config["steps"]):
        raise ValueError("schedule_total_steps must be at least steps")
    effective_tokens = (
        int(config["micro_batch_size"])
        * int(config["sequence_length"])
        * int(config["accumulation_steps"])
    )
    if effective_tokens != 2**17:
        raise ValueError(
            f"required effective tokens per step is 2^17, found {effective_tokens}"
        )
    if config["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")


def validate_resume_metadata(
    extra: dict,
    *,
    model_config: dict,
    run_config: dict,
    packed_manifest_sha256: str,
) -> None:
    if extra.get("model_config") != model_config:
        raise ValueError("resume checkpoint model configuration does not match")
    if extra.get("run_config") != run_config:
        raise ValueError("resume checkpoint run configuration does not match")
    if extra.get("packed_manifest_sha256") != packed_manifest_sha256:
        raise ValueError("resume checkpoint packed-data manifest does not match")


def load_packed_tokens(manifest_path: Path, name: str) -> tuple[torch.Tensor, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][name]
    if entry["dtype"] != "<i8":
        raise ValueError("training CLI requires packed dtype <i8")
    path = manifest_path.parent / entry["path"]
    if path.stat().st_size != int(entry["size"]):
        raise ValueError(f"packed file size mismatch: {path}")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"packed file hash mismatch: {path}")
    tokens = torch.from_file(
        str(path),
        shared=False,
        size=int(entry["tokens"]),
        dtype=torch.int64,
    )
    return tokens, entry


def append_metrics(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"checkpoint_step_{step:04d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "small.yaml")
    parser.add_argument("--run-config", type=Path, default=ROOT / "configs" / "train_gqa.yaml")
    parser.add_argument(
        "--packed-manifest",
        type=Path,
        default=ROOT / "data" / "packed" / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_payload = load_yaml(args.model_config)
    run_payload = load_yaml(args.run_config)
    validate_run_config(run_payload)
    total_steps = int(run_payload["steps"])
    schedule_total_steps = int(
        run_payload.get("schedule_total_steps", run_payload["steps"])
    )
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        total_steps = min(total_steps, args.max_steps)
    effective_tokens = (
        int(run_payload["micro_batch_size"])
        * int(run_payload["sequence_length"])
        * int(run_payload["accumulation_steps"])
    )
    plan = {
        "model_config": str(args.model_config.resolve()),
        "run_config": str(args.run_config.resolve()),
        "packed_manifest": str(args.packed_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "optimizer_steps": total_steps,
        "schedule_total_steps": schedule_total_steps,
        "effective_tokens_per_step": effective_tokens,
        "total_training_tokens": total_steps * effective_tokens,
        "resume": None if args.resume is None else str(args.resume.resolve()),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None:
        if not args.overwrite:
            raise SystemExit("output directory is not empty; use --overwrite or --resume")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    device = choose_device(args.device)
    seed = int(run_payload["seed"])
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = TransformerConfig(**model_payload)
    model = TransformerLM(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_tokens, train_entry = load_packed_tokens(args.packed_manifest, "train")
    validation_tokens, validation_entry = load_packed_tokens(
        args.packed_manifest,
        "validation_span",
    )
    packed_manifest_sha256 = sha256_file(args.packed_manifest)
    if len(train_tokens) <= int(run_payload["sequence_length"]):
        raise ValueError("packed training data is too short")

    optimizer = build_optimizer(
        model,
        learning_rate=float(run_payload["peak_lr"]),
        weight_decay=float(run_payload["weight_decay"]),
        betas=tuple(run_payload["betas"]),
    )
    data_generator = torch.Generator().manual_seed(seed)
    use_fp16_scaler = run_payload["precision"] == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    start_step = 0
    if args.resume is not None:
        start_step, extra = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            data_generator=data_generator,
            scaler=scaler,
            map_location=device,
        )
        validate_resume_metadata(
            extra,
            model_config=model_payload,
            run_config=run_payload,
            packed_manifest_sha256=packed_manifest_sha256,
        )
    if start_step >= total_steps:
        raise ValueError("checkpoint is already at or beyond the requested final step")

    frozen = {
        "model_config": model_payload,
        "run_config": run_payload,
        "packed_manifest": str(args.packed_manifest.resolve()),
        "packed_manifest_sha256": packed_manifest_sha256,
        "parameter_count": parameter_count,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    write_json(output_dir / "frozen_config.json", frozen)
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume is None:
        metrics_path.write_text("", encoding="utf-8")

    current_step = start_step
    checkpoint_every = int(run_payload["checkpoint_every"])
    peak_memory_bytes = 0
    total_elapsed = 0.0
    while current_step < total_steps:
        chunk_steps = min(checkpoint_every, total_steps - current_step)
        result = train_steps(
            model,
            train_tokens,
            steps=chunk_steps,
            batch_size=int(run_payload["micro_batch_size"]),
            sequence_length=int(run_payload["sequence_length"]),
            learning_rate=float(run_payload["peak_lr"]),
            seed=seed,
            device=device,
            accumulation_steps=int(run_payload["accumulation_steps"]),
            warmup_steps=int(run_payload["warmup_steps"]),
            min_learning_rate=float(run_payload["min_lr"]),
            weight_decay=float(run_payload["weight_decay"]),
            betas=tuple(run_payload["betas"]),
            gradient_clip=float(run_payload["grad_clip"]),
            precision=str(run_payload["precision"]),
            optimizer=optimizer,
            data_generator=data_generator,
            scaler=scaler,
            start_step=current_step,
            schedule_total_steps=schedule_total_steps,
        )
        total_elapsed += result.elapsed_seconds
        peak_memory_bytes = max(peak_memory_bytes, result.peak_memory_bytes)
        first_step = current_step + 1
        records = [
            {
                "step": first_step + index,
                "training_loss": result.losses[index],
                "gradient_norm": result.gradient_norms[index],
                "learning_rate": result.learning_rates[index],
                "tokens_seen": (first_step + index) * effective_tokens,
                "chunk_tokens_per_second": result.tokens_per_second,
            }
            for index in range(chunk_steps)
        ]
        current_step += chunk_steps

        model.eval()
        validation = evaluate_token_ids(
            model,
            validation_tokens,
            sequence_length=int(run_payload["sequence_length"]),
            raw_byte_count=int(validation_entry["source_size"]),
        )
        records[-1]["validation_mean_nll"] = validation["mean_nll"]
        records[-1]["validation_perplexity"] = validation["perplexity"]
        append_metrics(metrics_path, records)

        path = checkpoint_path(output_dir, current_step)
        temporary = path.with_name(path.name + ".tmp")
        save_training_checkpoint(
            temporary,
            model=model,
            optimizer=optimizer,
            step=current_step,
            data_generator=data_generator,
            scaler=scaler,
            extra={
                "model_config": model_payload,
                "run_config": run_payload,
                "packed_manifest_sha256": packed_manifest_sha256,
            },
        )
        os.replace(temporary, path)
        write_json(
            output_dir / "latest_checkpoint.json",
            {"step": current_step, "path": str(path.resolve())},
        )
        print(
            json.dumps(
                {
                    "step": current_step,
                    "training_loss": records[-1]["training_loss"],
                    "validation_mean_nll": validation["mean_nll"],
                    "checkpoint": str(path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        **frozen,
        "completed_steps": current_step,
        "training_tokens_seen": current_step * effective_tokens,
        "elapsed_training_seconds": total_elapsed,
        "average_tokens_per_second": (
            (current_step - start_step) * effective_tokens / max(total_elapsed, 1e-12)
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "train_token_count": train_entry["tokens"],
        "validation_span_token_count": validation_entry["tokens"],
        "final_checkpoint": str(checkpoint_path(output_dir, current_step).resolve()),
        "finished_at_unix": time.time(),
    }
    write_json(output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
