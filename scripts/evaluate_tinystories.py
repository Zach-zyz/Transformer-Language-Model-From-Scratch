"""Evaluate a checkpoint on the fixed validation span and generate fixed prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs8803_hw1.evaluate import evaluate_token_ids
from cs8803_hw1.generate import generate
from cs8803_hw1.model import TransformerConfig, TransformerLM
from cs8803_hw1.tokenizer import ByteBPETokenizer
from release_utils import (
    choose_device,
    load_checkpoint_model_state,
    load_yaml,
    sha256_file,
    write_json,
)


def load_tokens(manifest_path: Path, name: str) -> tuple[torch.Tensor, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["files"][name]
    path = manifest_path.parent / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"packed file hash mismatch: {path}")
    return (
        torch.from_file(
            str(path),
            shared=False,
            size=int(entry["tokens"]),
            dtype=torch.int64,
        ),
        entry,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "small.yaml")
    parser.add_argument(
        "--packed-manifest",
        type=Path,
        default=ROOT / "data" / "packed" / "manifest.json",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs" / "evaluation.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "model_config": str(args.model_config.resolve()),
        "packed_manifest": str(args.packed_manifest.resolve()),
        "evaluation_config": str(args.evaluation_config.resolve()),
        "output": str(args.output.resolve()),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    device = choose_device(args.device)
    model_config = TransformerConfig(**load_yaml(args.model_config))
    model = TransformerLM(model_config).to(device)
    model.load_state_dict(load_checkpoint_model_state(args.checkpoint, device))
    model.eval()
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    validation_tokens, validation_entry = load_tokens(
        args.packed_manifest,
        "validation_span",
    )
    metrics = evaluate_token_ids(
        model,
        validation_tokens,
        sequence_length=model_config.max_seq_len,
        raw_byte_count=int(validation_entry["source_size"]),
    )

    generation_config = load_yaml(args.evaluation_config)
    samples = []
    for index, prompt in enumerate(generation_config["prompts"]):
        prompt_ids = torch.tensor(
            [tokenizer.encode(str(prompt))],
            dtype=torch.long,
            device=device,
        )
        output_ids = generate(
            model,
            prompt_ids,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            temperature=float(generation_config["temperature"]),
            top_k=int(generation_config["top_k"]),
            top_p=float(generation_config["top_p"]),
            seed=2026 + index,
            use_cache=True,
        )
        samples.append(
            {
                "prompt": prompt,
                "seed": 2026 + index,
                "token_ids": output_ids[0].cpu().tolist(),
                "text": tokenizer.decode(output_ids[0].cpu().tolist()),
            }
        )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "model_config": load_yaml(args.model_config),
        "metrics": metrics,
        "generation_config": generation_config,
        "samples": samples,
    }
    write_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
