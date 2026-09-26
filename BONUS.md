# Optional OpenWebText Bonus Workflow

The bonus uses one pinned CC0 OpenWebText Parquet shard. It contains 100,173 documents and is about
303 MB. The last 2,000 documents are the fixed validation span; the remaining documents are
training data.

Download and verify the shard:

```bash
bash data/download_openwebtext.sh --dry-run
bash data/download_openwebtext.sh
bash data/download_openwebtext.sh --verify-only
```

Pack it with the tokenizer used by the bonus model. `pyarrow` is needed only for this optional step:

```bash
python -m venv .bonus-venv
source .bonus-venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pyarrow==23.0.0

python data/prepare_openwebtext.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output-dir data/openwebtext/packed
```

The output manifest uses the same `train` and `validation_span` names accepted by
`scripts/train_tinystories.py` and `scripts/evaluate_tinystories.py`. Create a separate run
configuration and stay within the handout's 100M-parameter and two-billion-token caps.
Compare your generated counts, sizes, and hashes with
`data/openwebtext_packed_reference_manifest.json`.

The published staff tiers use:

- B1: `configs/bonus_b1_model.yaml` with `configs/bonus_b1_train.yaml`;
- B2: `configs/small.yaml` with `configs/bonus_b2_train.yaml`;
- B3: `configs/small.yaml` with `configs/bonus_b3_train.yaml`.

The frozen B1/B2/B3 bits-per-byte values and their exact staff configs are published in
`artifacts/bonus_baselines.json`.
