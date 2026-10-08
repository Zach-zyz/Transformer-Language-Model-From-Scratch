"""Verify release artifacts and compare them against the submitted implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import TransformerConfig, TransformerLM
from src.tokenizer import ByteBPETokenizer
from release_utils import sha256_file


ARTIFACT_ROOT = ROOT / "artifacts"


def verify_files(manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = ARTIFACT_ROOT / name
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"artifact size mismatch: {path}")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {path}")


def check_tokenizer() -> None:
    tokenizer = ByteBPETokenizer.load(ARTIFACT_ROOT / "canonical_tokenizer.json")
    if len(tokenizer.vocab) + 1 != 8192:
        raise ValueError("canonical tokenizer does not contain 8192 tokens")
    sample = "Lily found a box.\n<|endoftext|>"
    token_ids = tokenizer.encode(sample, {"<|endoftext|>"})
    if tokenizer.decode(token_ids) != sample:
        raise ValueError("canonical tokenizer round trip failed")


def load_reference_model() -> tuple[TransformerLM, torch.Tensor]:
    checkpoint = torch.load(
        ARTIFACT_ROOT / "tiny_reference_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = TransformerLM(TransformerConfig(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["model"])
    with np.load(ARTIFACT_ROOT / "reference_logits.npz", allow_pickle=False) as arrays:
        input_ids = torch.from_numpy(arrays["input_ids"].copy()).long()
        expected = torch.from_numpy(arrays["full_logits"].copy())
    with torch.no_grad():
        actual = model(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    return model, input_ids


def check_cache(model: TransformerLM, input_ids: torch.Tensor) -> None:
    with np.load(ARTIFACT_ROOT / "cache_golden.npz", allow_pickle=False) as arrays:
        expected_full = torch.from_numpy(arrays["full_logits"].copy())
        expected_token = torch.from_numpy(arrays["token_cached_logits"].copy())
        expected_chunked = torch.from_numpy(arrays["chunked_logits"].copy())

    with torch.no_grad():
        full = model(input_ids).logits
        cache = None
        token_pieces = []
        for index in range(input_ids.shape[1]):
            output = model(input_ids[:, index : index + 1], cache=cache, use_cache=True)
            cache = output.cache
            token_pieces.append(output.logits)
        token_cached = torch.cat(token_pieces, dim=1)
        split = max(1, input_ids.shape[1] // 2)
        first = model(input_ids[:, :split], use_cache=True)
        second = model(input_ids[:, split:], cache=first.cache, use_cache=True)
        chunked = torch.cat((first.logits, second.logits), dim=1)

    for actual, expected in (
        (full, expected_full),
        (token_cached, expected_token),
        (chunked, expected_chunked),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        action="append",
        choices=("files", "tokenizer", "model", "cache"),
        default=[],
    )
    args = parser.parse_args()

    selected = set(args.component or ("files", "tokenizer", "model", "cache"))
    manifest = json.loads((ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    completed = []
    if "files" in selected:
        verify_files(manifest)
        completed.append("files")
    if "tokenizer" in selected:
        check_tokenizer()
        completed.append("tokenizer")
    model = None
    input_ids = None
    if selected & {"model", "cache"}:
        model, input_ids = load_reference_model()
    if "model" in selected:
        completed.append("model")
    if "cache" in selected:
        assert model is not None and input_ids is not None
        check_cache(model, input_ids)
        completed.append("cache")
    print(json.dumps({"passed": completed}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
