"""Measure cached and non-cached greedy decoding for one model configuration."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs8803_hw1.generate import generate
from cs8803_hw1.model import TransformerConfig, TransformerLM
from release_utils import (
    choose_device,
    load_checkpoint_model_state,
    load_yaml,
    write_json,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_generation(
    model: TransformerLM,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    use_cache: bool,
    warmups: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmups):
        generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_cache=use_cache,
        )
    synchronize(prompt.device)
    timings = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_cache=use_cache,
        )
        synchronize(prompt.device)
        timings.append(time.perf_counter() - started_at)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, action="append", default=[])
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batch_sizes = args.batch_size or [1, 32]
    plan = {
        "model_config": str(args.model_config.resolve()),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "batch_sizes": batch_sizes,
        "prompt_length": args.prompt_length,
        "new_tokens": args.new_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "dtype": args.dtype,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    device = choose_device(args.device)
    config = TransformerConfig(**load_yaml(args.model_config))
    if args.prompt_length + args.new_tokens > config.max_seq_len:
        raise ValueError("prompt plus generated tokens exceeds max_seq_len")
    torch.manual_seed(2026)
    model = TransformerLM(config)
    if args.checkpoint is not None:
        model.load_state_dict(load_checkpoint_model_state(args.checkpoint, "cpu"))
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    model = model.to(device=device, dtype=dtype).eval()
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for batch_size in batch_sizes:
        prompt = (
            torch.arange(batch_size * args.prompt_length, device=device)
            .reshape(batch_size, args.prompt_length)
            .remainder(config.vocab_size)
        )
        cached = timed_generation(
            model,
            prompt,
            max_new_tokens=args.new_tokens,
            use_cache=True,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        uncached = timed_generation(
            model,
            prompt,
            max_new_tokens=args.new_tokens,
            use_cache=False,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        cached_median = statistics.median(cached)
        uncached_median = statistics.median(uncached)
        final_sequence = args.prompt_length + args.new_tokens
        cache_bytes = (
            2
            * config.n_layers
            * batch_size
            * final_sequence
            * config.n_kv_heads
            * (config.d_model // config.n_heads)
            * dtype_bytes
        )
        generated = batch_size * args.new_tokens
        rows.append(
            {
                "batch_size": batch_size,
                "cached_seconds_median": cached_median,
                "uncached_seconds_median": uncached_median,
                "cached_tokens_per_second": generated / cached_median,
                "uncached_tokens_per_second": generated / uncached_median,
                "speedup": uncached_median / cached_median,
                "analytical_cache_bytes": cache_bytes,
            }
        )
    result = {
        "model_config": load_yaml(args.model_config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "dtype": args.dtype,
        "prompt_length": args.prompt_length,
        "new_tokens": args.new_tokens,
        "rows": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
