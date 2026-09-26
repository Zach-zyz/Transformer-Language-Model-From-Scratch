# CSE 8803 HW1 Student Workflow

Read [HANDOUT.md](HANDOUT.md) before editing code. It is the authoritative assignment contract.
This README orders the provided commands; it does not replace the requirements or rubric.

## 1. Set Up

If you completed Tutorial 0, reuse the same Conda environment. Activate the command that matches
where you are working:

```bash
# Laptop
conda activate cse8803

# ICE
module load anaconda3
conda activate ~/scratch/conda-envs/cse8803

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest --collect-only -q
```

If you do not have that environment, create a repository-local virtual environment instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest --collect-only -q
```

The portable requirements use lower bounds so the assignment works across CPU, Apple Silicon, and
different CUDA systems. To reproduce the staff reference software versions, use Python 3.10.20,
install the platform-appropriate PyTorch 2.12.0 wheel using the official PyTorch installation
instructions, then run:

```bash
python -m pip install -r requirements.txt -c constraints-tested.txt
```

The measured DMServe2 image used PyTorch 2.12.0+cu130, NumPy 2.2.6, pytest 9.1.0, and PyYAML 6.0.3.
The CUDA wheel itself is accelerator-specific and is not forced onto CPU or Apple Silicon machines.

The graded bodies under `cs8803_hw1/` intentionally raise `NotImplementedError`. Do not change their
public signatures. The included data spans and artifacts are small enough to keep in Git; raw and
packed TinyStories data are deliberately excluded.

`data/toy_train.bin` and `data/toy_val.bin` are canonical-tokenizer int64 arrays for quickly testing
the packed-data and evaluation paths without downloading the full dataset.

## 2. Implement in Dependency Order

Use the public tests while working through these stages:

1. `tokenizer.py`: deterministic training, special-token policy, encode/decode, save/load.
2. `model.py`: RMSNorm, RoPE, GQA/MHA attention, SwiGLU, Transformer blocks, tied output weights.
3. `optimizer.py` and `train.py`: stable loss, AdamW, schedule, accumulation, precision, checkpoints.
4. `generate.py` and `evaluate.py`: sampling, cached generation, PPL, and bits per byte.

Run focused tests after each stage, then the full suite:

```bash
python -m pytest -q tests/test_tokenizer.py
python -m pytest -q tests/test_model.py
python -m pytest -q tests/test_optimizer.py tests/test_training.py
python -m pytest -q tests/test_generation.py tests/test_evaluate.py
python -m pytest -q
```

The full **8192-token vocabulary** contains 256 byte tokens, one reserved special token, and 7935
learned merge tokens. Training it and packing the multi-gigabyte dataset require a local-update
implementation rather than repeated full rescans. Read
[PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) after making a correct tiny-corpus oracle and before
optimizing it.

## 3. Use the Reference Artifacts

Start with `artifacts/README.md`, which catalogs every staff-provided reference file. The canonical
tokenizer, tiny checkpoint, logits, and cache goldens are for fault isolation;
`reference_ranges.json` and `training_reference.json` provide comparison measurements; and
`bonus_baselines.json` defines the optional bonus tiers. They do not replace your tokenizer or model
in the required run.

```bash
# Verify only the provided files and hashes.
python scripts/check_artifacts.py --component files

# After implementing the corresponding components:
python scripts/check_artifacts.py --component tokenizer
python scripts/check_artifacts.py --component model
python scripts/check_artifacts.py --component cache
```

If a downstream component is blocked, use `artifacts/canonical_tokenizer.json` or
`artifacts/tiny_reference_checkpoint.pt` temporarily. Return to your own tokenizer and submitted
model for the required training runs.

## 4. Run the Smoke Pipeline

```bash
python scripts/run_smoke.py
```

This CPU/MPS-friendly command trains on toy text, checks that loss falls, compares cached and
non-cached decoding, and generates a sample.

## 5. Train and Benchmark Your Tokenizer

The fixed shard is the first 64 complete documents from the pinned original TinyStories train file:
46,906 bytes with SHA-256
`2afe3baf6ba281da49924cfdf9edf19730465fd442a1d149bba04994c44704a3`.

```bash
mkdir -p outputs/tokenizer
python scripts/train_tokenizer.py \
  --output outputs/tokenizer/tokenizer.json \
  --metrics outputs/tokenizer/training.json

python scripts/benchmark_tokenizer.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output outputs/tokenizer/benchmark.json \
  --plot outputs/tokenizer/vocabulary.svg
```

`train_tokenizer.py` checks the full 8192-token checksum. The benchmark trains only once and evaluates
prefixes at vocabulary sizes 512, 1024, 2048, 4096, and 8192. The published tokenizer-training limit
is **0.853 seconds** on the named course runner. The exact value and runner/software description are
in `artifacts/reference_ranges.json` under
`tokenizer_training.student_limit_seconds`. Record that limit and your measured result in the report.

## 6. Download and Pack TinyStories

The release pins the original TinyStories train and validation files to revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, exact byte sizes, and SHA-256 digests.

```bash
# Inspect destinations, sizes, and hashes without downloading.
bash data/download_data.sh --dry-run

# Resumable download followed by mandatory size/hash verification.
bash data/download_data.sh
bash data/download_data.sh --verify-only

# Inspect the full packing plan.
python data/prepare_data.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output-dir data/packed \
  --dry-run

# Pack train, validation, and the fixed validation span.
python data/prepare_data.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output-dir data/packed
```

The arrays are little-endian signed 64-bit token IDs. `data/packed/manifest.json` records the
tokenizer hash, source hashes, token counts, byte sizes, and packed-file hashes. For a quick
end-to-end data-path check before the full pack:

```bash
python data/prepare_data.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output-dir /tmp/cse8803-hw1-packed-smoke \
  --max-train-documents 64 \
  --max-validation-documents 64
```

After the full pack, compare your generated manifest with
`data/packed_reference_manifest.json`. Matching tokenizer and source hashes must produce the
published token counts, sizes, and packed-file SHA-256 values.

## 7. Dry-Run the Required Experiments

Dry runs validate paths and frozen configurations without allocating a model or reading packed data.

```bash
python scripts/train_tinystories.py \
  --model-config configs/small.yaml \
  --run-config configs/train_gqa.yaml \
  --output-dir runs/gqa \
  --dry-run

python scripts/train_tinystories.py \
  --model-config configs/small_mha.yaml \
  --run-config configs/train_mha.yaml \
  --output-dir runs/mha \
  --dry-run

python scripts/evaluate_tinystories.py \
  --checkpoint runs/gqa/checkpoints/checkpoint_step_3000.pt \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output runs/gqa/evaluation.json \
  --dry-run
```

## 8. Run GQA and MHA Training

Use one GPU. The supplied configurations enforce `2^17` effective tokens per optimizer step.

```bash
python scripts/train_tinystories.py \
  --model-config configs/small.yaml \
  --run-config configs/train_gqa.yaml \
  --packed-manifest data/packed/manifest.json \
  --output-dir runs/gqa

python scripts/train_tinystories.py \
  --model-config configs/small_mha.yaml \
  --run-config configs/train_mha.yaml \
  --packed-manifest data/packed/manifest.json \
  --output-dir runs/mha
```

Resume from a checkpoint without changing the model, run configuration, or packed manifest:

```bash
python scripts/train_tinystories.py \
  --model-config configs/small.yaml \
  --run-config configs/train_gqa.yaml \
  --packed-manifest data/packed/manifest.json \
  --output-dir runs/gqa \
  --resume runs/gqa/checkpoints/checkpoint_step_1000.pt
```

Each run writes a frozen configuration, JSONL metrics, periodic exact-resume checkpoints, a latest
checkpoint pointer, and a final summary.

## 9. Evaluate and Benchmark the Final Models

```bash
python scripts/evaluate_tinystories.py \
  --checkpoint runs/gqa/checkpoints/checkpoint_step_3000.pt \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --model-config configs/small.yaml \
  --packed-manifest data/packed/manifest.json \
  --output runs/gqa/evaluation.json

python scripts/benchmark_kv_cache.py \
  --model-config configs/small.yaml \
  --checkpoint runs/gqa/checkpoints/checkpoint_step_3000.pt \
  --dtype bf16 \
  --output runs/gqa/cache_benchmark.json

python scripts/benchmark_kv_cache.py \
  --model-config configs/small_mha.yaml \
  --checkpoint runs/mha/checkpoints/checkpoint_step_1000.pt \
  --dtype bf16 \
  --output runs/mha/cache_benchmark.json
```

The cache benchmark uses batch sizes 1 and 32 by default and reports analytical cache bytes, cached
and uncached throughput, and measured speedup.

Create report-ready JSON and Markdown tables:

```bash
python scripts/summarize_results.py \
  --gqa-run runs/gqa/run_summary.json \
  --gqa-metrics runs/gqa/metrics.jsonl \
  --gqa-evaluation runs/gqa/evaluation.json \
  --gqa-cache runs/gqa/cache_benchmark.json \
  --mha-run runs/mha/run_summary.json \
  --mha-metrics runs/mha/metrics.jsonl \
  --mha-cache runs/mha/cache_benchmark.json \
  --output-json runs/metrics.json \
  --output-markdown runs/results.md
```

Use [templates/report_checklist.md](templates/report_checklist.md) to audit the report evidence.

## 10. Package and Check the Submission

Complete [templates/ai_disclosure.md](templates/ai_disclosure.md), export the report to PDF, then run:

```bash
python scripts/make_submission.py \
  --writeup writeup.pdf \
  --ai-disclosure ai_disclosure.md \
  --run-config configs/train_gqa.yaml \
  --metrics runs/metrics.json \
  --output-dir submission

python scripts/check_submission.py submission
```

The checker requires exactly the handout deliverables, rejects unsafe archive paths, and rejects raw
data, packed arrays, checkpoints, NumPy artifacts, bytecode, and incomplete graded bodies.
`metrics.json` must be the complete output of `scripts/summarize_results.py`, including the required
GQA/MHA step-1000 comparison, final GQA evaluation, and batch-size 1/32 cache rows.
