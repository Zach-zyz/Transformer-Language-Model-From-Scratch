from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cs8803_hw1.model import TransformerConfig, TransformerLM
from cs8803_hw1.tokenizer import ByteBPETokenizer, SPECIAL_TOKEN
from cs8803_hw1.train import cosine_lr
from scripts.summarize_results import validation_at_step
from scripts.train_tinystories import validate_resume_metadata
from data.download_data import download_file


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tinystories_sources_are_immutably_pinned() -> None:
    manifest = json.loads(
        (ROOT / "data" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset"] == {
        "license": "cdla-sharing-1.0",
        "repository": "roneneldan/TinyStories",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    }
    files = {entry["id"]: entry for entry in manifest["files"]}
    assert files["train"]["size"] == 1_924_281_556
    assert files["train"]["sha256"] == (
        "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"
    )
    assert files["validation"]["size"] == 19_447_282
    assert files["validation"]["sha256"] == (
        "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
    )
    for entry in files.values():
        assert manifest["dataset"]["revision"] in entry["url"]


def test_openwebtext_bonus_source_is_pinned() -> None:
    manifest = json.loads(
        (ROOT / "data" / "openwebtext_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset"] == {
        "license": "cc0-1.0",
        "repository": "Skylion007/openwebtext",
        "revision": "79d93d786212f7344586290adb811d4ae6a1762c",
    }
    [entry] = manifest["files"]
    assert entry["size"] == 302_848_326
    assert entry["sha256"] == (
        "caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e"
    )


def test_openwebtext_packed_reference_manifest_is_complete() -> None:
    manifest = json.loads(
        (ROOT / "data" / "openwebtext_packed_reference_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["tokenizer"]["sha256"] == (
        "2150eb89992a0dc1aadf79cbbb071673ab82b607e70f85e530f7639002b675e2"
    )
    assert manifest["files"]["train"]["documents"] == 98_173
    assert manifest["files"]["validation_span"]["documents"] == 2_000
    assert manifest["files"]["train"]["tokens"] > 100_000_000
    assert manifest["files"]["validation_span"]["tokens"] > 1_000_000


def test_tinystories_packed_reference_manifest_is_complete() -> None:
    manifest = json.loads(
        (ROOT / "data" / "packed_reference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["tokenizer"]["sha256"] == (
        "2150eb89992a0dc1aadf79cbbb071673ab82b607e70f85e530f7639002b675e2"
    )
    assert manifest["files"]["train"]["tokens"] == 459_227_569
    assert manifest["files"]["validation"]["tokens"] == 4_607_239
    assert manifest["files"]["validation_span"]["tokens"] == 9_161
    assert {
        name: payload["source"]
        for name, payload in manifest["files"].items()
    } == {
        "train": "TinyStories-train.txt",
        "validation": "TinyStories-valid.txt",
        "validation_span": "validation_span.txt",
    }
    assert manifest["files"]["train"]["sha256"] == (
        "0754e1f547145bc30d696d8596deb9b0d48b027a5aed8510224d7f4743b8d26b"
    )


def test_packed_manifests_use_portable_tokenizer_names() -> None:
    prepare_data_text = (ROOT / "data" / "prepare_data.py").read_text(encoding="utf-8")
    prepare_owt_text = (ROOT / "data" / "prepare_openwebtext.py").read_text(
        encoding="utf-8"
    )
    assert '"path": args.tokenizer.name' in prepare_data_text
    assert '"path": args.tokenizer.name' in prepare_owt_text
    assert '"path": str(args.tokenizer.resolve())' not in prepare_data_text
    assert '"path": str(args.tokenizer.resolve())' not in prepare_owt_text


def test_fixed_derivatives_match_manifest_and_document_count() -> None:
    manifest = json.loads(
        (ROOT / "data" / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["derivatives"].values():
        path = ROOT / "data" / entry["output"]
        assert path.stat().st_size == entry["size"]
        assert sha256(path) == entry["sha256"]
        assert path.read_text(encoding="utf-8").count(SPECIAL_TOKEN) == entry["documents"]


def test_toy_packed_arrays_match_manifest() -> None:
    manifest = json.loads(
        (ROOT / "data" / "toy_manifest.json").read_text(encoding="utf-8")
    )
    for name, entry in manifest["files"].items():
        path = ROOT / "data" / name
        assert path.stat().st_size == entry["size"]
        assert sha256(path) == entry["sha256"]
        values = np.fromfile(path, dtype="<i8")
        assert len(values) == entry["tokens"]
        assert len(values) > 32


def test_completed_partial_download_is_verified_and_promoted(tmp_path: Path) -> None:
    payload = b"pinned test payload\n"
    entry = {
        "local_path": "raw/sample.txt",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://invalid.example/sample.txt",
    }
    partial = tmp_path / "raw" / "sample.txt.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload)
    destination = download_file(entry, tmp_path)
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_artifact_manifest_and_canonical_tokenizer() -> None:
    manifest = json.loads(
        (ROOT / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    for name, entry in manifest["files"].items():
        path = ROOT / "artifacts" / name
        assert path.stat().st_size == entry["size"]
        assert sha256(path) == entry["sha256"]
    tokenizer = ByteBPETokenizer.load(ROOT / "artifacts" / "canonical_tokenizer.json")
    assert len(tokenizer.vocab) + 1 == 8192
    ranges = json.loads(
        (ROOT / "artifacts" / "reference_ranges.json").read_text(encoding="utf-8")
    )
    assert ranges["tokenizer_training"]["student_limit_seconds"] > 0
    assert ranges["data_packing"]["projected_full_train_seconds"] > 0
    training_reference = json.loads(
        (ROOT / "artifacts" / "training_reference.json").read_text(encoding="utf-8")
    )
    assert training_reference["gqa"]["completed_steps"] == 3000
    assert training_reference["mha"]["completed_steps"] == 1000
    assert training_reference["gqa"]["evaluation"]["bits_per_byte"] > 0
    assert training_reference["mha"]["evaluation"]["bits_per_byte"] > 0
    bonus = json.loads(
        (ROOT / "artifacts" / "bonus_baselines.json").read_text(encoding="utf-8")
    )
    values = [
        bonus["tiers"][tier]["bits_per_byte"]
        for tier in ("B1", "B2", "B3")
    ]
    assert values[0] > values[1] > values[2] > 0


def test_reference_logits_and_cache_goldens() -> None:
    checkpoint = torch.load(
        ROOT / "artifacts" / "tiny_reference_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = TransformerLM(TransformerConfig(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["model"])
    with np.load(ROOT / "artifacts" / "cache_golden.npz", allow_pickle=False) as arrays:
        input_ids = torch.from_numpy(arrays["input_ids"].copy()).long()
        expected_full = torch.from_numpy(arrays["full_logits"].copy())
        expected_token = torch.from_numpy(arrays["token_cached_logits"].copy())
        expected_chunked = torch.from_numpy(arrays["chunked_logits"].copy())

    with torch.no_grad():
        full = model(input_ids).logits
        cache = None
        pieces = []
        for index in range(input_ids.shape[1]):
            output = model(input_ids[:, index : index + 1], cache=cache, use_cache=True)
            cache = output.cache
            pieces.append(output.logits)
        token_cached = torch.cat(pieces, dim=1)
        split = max(1, input_ids.shape[1] // 2)
        first = model(input_ids[:, :split], use_cache=True)
        second = model(input_ids[:, split:], cache=first.cache, use_cache=True)
        chunked = torch.cat((first.logits, second.logits), dim=1)

    torch.testing.assert_close(full, expected_full, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(token_cached, expected_token, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(chunked, expected_chunked, atol=2e-5, rtol=1e-5)


def test_mha_uses_first_thousand_steps_of_gqa_schedule() -> None:
    gqa = yaml.safe_load((ROOT / "configs" / "train_gqa.yaml").read_text())
    mha = yaml.safe_load((ROOT / "configs" / "train_mha.yaml").read_text())
    assert mha["steps"] == 1000
    assert mha["schedule_total_steps"] == gqa["steps"] == 3000
    assert cosine_lr(
        1000,
        mha["schedule_total_steps"],
        mha["warmup_steps"],
        mha["peak_lr"],
        mha["min_lr"],
    ) == cosine_lr(
        1000,
        gqa["steps"],
        gqa["warmup_steps"],
        gqa["peak_lr"],
        gqa["min_lr"],
    )


def test_resume_metadata_binds_packed_data() -> None:
    expected = {
        "model_config": {"d_model": 32},
        "run_config": {"steps": 10},
        "packed_manifest_sha256": "a" * 64,
    }
    validate_resume_metadata(
        expected,
        model_config=expected["model_config"],
        run_config=expected["run_config"],
        packed_manifest_sha256=expected["packed_manifest_sha256"],
    )
    with pytest.raises(ValueError, match="packed-data manifest"):
        validate_resume_metadata(
            expected,
            model_config=expected["model_config"],
            run_config=expected["run_config"],
            packed_manifest_sha256="b" * 64,
        )


def test_validation_at_step_requires_reported_validation() -> None:
    record = validation_at_step(
        [
            {"step": 900, "training_loss": 1.0},
            {
                "step": 1000,
                "validation_mean_nll": 2.0,
                "validation_perplexity": 7.389,
            },
        ],
        1000,
    )
    assert record == {
        "validation_mean_nll": 2.0,
        "validation_perplexity": 7.389,
    }
