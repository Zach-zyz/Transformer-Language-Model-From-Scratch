# Reference Artifacts

These small artifacts isolate tokenizer, model-forward, and KV-cache failures. `manifest.json`
records the exact byte size and SHA-256 digest of each file.

- `canonical_tokenizer.json`: required 8192-token tokenizer trained on the fixed 64-document shard.
- `tiny_reference_checkpoint.pt`: deterministic eight-step tiny-model checkpoint and configuration.
- `reference_logits.npz`: fixed input IDs and full-forward logits from the tiny checkpoint.
- `cache_golden.npz`: full, token-at-a-time cached, and chunked-cache logits for the same input.
- `reference_ranges.json`: measured tokenizer and packing ranges on the named course runner.
- `training_reference.json`: measured required GQA/MHA training, evaluation, memory, and cache
  ranges from the named course runner.
- `bonus_baselines.json`: frozen B1/B2/B3 OpenWebText bits-per-byte tiers and input hashes.

Verify file integrity without executing student components:

```bash
python scripts/check_artifacts.py --component files
```

After implementing the corresponding APIs:

```bash
python scripts/check_artifacts.py --component tokenizer
python scripts/check_artifacts.py --component model
python scripts/check_artifacts.py --component cache
```

The artifacts are debugging references only. The required TinyStories run must use the tokenizer and
model implementation submitted by the student. Runtime ranges are machine-specific comparison data,
not cross-hardware guarantees.
