# TinyStories Data

`source_manifest.json` pins the original TinyStories train and validation text files to one dataset
revision, byte size, and SHA-256 digest. The data is licensed under `cdla-sharing-1.0`; the upstream
dataset repository is `roneneldan/TinyStories`.

Download or verify the raw files:

```bash
bash data/download_data.sh
bash data/download_data.sh --verify-only
```

Downloads use `.part` files and HTTP Range requests. A completed file is renamed into place only
after its byte size and SHA-256 digest match the manifest.

The repository includes two document-aligned, checksummed derivatives:

- `tokenizer_train.txt`: first 64 complete training documents, 46,906 bytes;
- `validation_span.txt`: first 64 complete validation documents, 44,843 bytes.

After implementing the tokenizer, pack the full data and fixed validation span:

```bash
python data/prepare_data.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output-dir data/packed
```

The packed arrays use little-endian signed 64-bit token IDs so the public training API can consume a
memory-mapped array without converting the full dataset in RAM. `data/packed/manifest.json` records
the tokenizer hash, token counts, source hashes, and packed-file hashes.
`packed_reference_manifest.json` records the expected full-pack values for the canonical tokenizer
without distributing the multi-gigabyte arrays.

Inspect the full path without reading raw data:

```bash
python data/prepare_data.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output-dir data/packed \
  --dry-run
```

For a bounded integration check, add both `--max-train-documents 64` and
`--max-validation-documents 64`. The final required run must use the full pinned train and
validation files.
