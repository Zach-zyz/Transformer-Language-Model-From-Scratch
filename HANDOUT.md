# Assignment 1: Build and Train a Transformer Language Model from Scratch

**Course**: CSE 8803 - Language Models and Language Agents (Fall 2026, Georgia Tech)  
**Instructor**: Prof. Chao Zhang  
**Released**: Tuesday, September 1, 2026  
**Final due date**: Tuesday, September 22, 2026, 11:59 PM ET  
**Weight**: 15% of the final course grade  
**Format**: Individual  
**Expected successful-run compute**: 1-3 modern GPU-hours; plan for up to 6 GPU-hours including debugging

> This file is the authoritative HW1 handout. The student-facing HTML page summarizes and links to
> this specification. If a stale copy appears elsewhere, this file governs.

## 1. Learning Goals

By the end of this assignment, you should be able to:

1. Explain and implement byte-level BPE tokenization.
2. Implement every core operation in a modern decoder-only Transformer.
3. Train a small language model from random initialization and diagnose failures.
4. Implement autoregressive generation, perplexity evaluation, and KV caching.
5. Run a controlled architecture comparison and interpret a negative or noisy result.

The required path is deliberately one coherent pipeline:

```text
raw text
  -> your tokenizer
  -> your Transformer
  -> your loss and optimizer
  -> your training loop
  -> your generated text and evaluation
```

The goal is understanding, not leaderboard-scale training. OpenWebText training and additional
performance optimization are bonus work.

## 2. What We Provide

The GitHub Classroom repository contains:

```text
hw1-starter/
  cs8803_hw1/
    tokenizer.py
    model.py
    optimizer.py
    train.py
    generate.py
    evaluate.py
    utils.py
  tests/
    test_tokenizer.py
    test_model.py
    test_optimizer.py
    test_training.py
    test_generation.py
    test_evaluate.py
  configs/
    smoke.yaml          # 1-2M parameters; CPU/MPS friendly
    small.yaml          # required ~29.37M model
  data/
    download_data.sh
    tokenizer_train.txt # fixed TinyStories shard
    toy_train.bin
    toy_val.bin
  artifacts/
    canonical_tokenizer.json
    tiny_reference_checkpoint.pt
    reference_logits.npz
    cache_golden.npz
    reference_ranges.json
    training_reference.json
  scripts/
    benchmark_tokenizer.py
    benchmark_kv_cache.py
    evaluate_tinystories.py
    make_submission.py
  requirements.txt
  README.md
```

The reference artifacts are for fault isolation. The main files are:

- `canonical_tokenizer.json`: the expected tokenizer for checking training, encoding, and data packing;
- `tiny_reference_checkpoint.pt`, `reference_logits.npz`, and `cache_golden.npz`: a tiny model and
  expected full-forward and cached-decoding outputs;
- `reference_ranges.json`: the tokenizer runtime limit and packing measurements on the named course
  runner;
- `training_reference.json`: staff measurements for the required GQA and MHA runs.

You may use these files to test downstream components while debugging, but the final required
TinyStories run must use the tokenizer and model code you submit. See `artifacts/README.md` for the
complete artifact list and verification commands.

### 2.1 Data

- **Tokenizer training shard**: a fixed, checksummed TinyStories text shard.
- **TinyStories train/validation**: required end-to-end training data.
- **OpenWebText subset**: optional bonus/leaderboard data.

Run:

```bash
bash data/download_data.sh
```

The download script verifies file hashes and refuses silently truncated files.

## 3. Rules

### 3.1 Prohibited for the Required Implementation

You may not use:

- `torch.nn.Transformer*`
- `torch.nn.MultiheadAttention`
- pre-built or fused attention as your submitted core attention implementation
- Hugging Face model or tokenizer classes
- `tiktoken`, SentencePiece, or another tokenizer implementation
- `torch.optim.AdamW` as your submitted optimizer
- pretrained model weights
- copied implementations from public repositories

`torch.nn.functional.scaled_dot_product_attention` and `torch.optim.AdamW` may be used only as
correctness references in tests. After your own implementation passes, fused attention may be used
for optional leaderboard experiments.

### 3.2 Permitted

You may use:

- `torch.nn.Module`, `Parameter`, `ModuleList`, `Linear`, `Embedding`, and `Dropout`
- tensor operations such as `matmul`, `einsum`, `softmax`, `exp`, `log`, `rsqrt`, and masking
- `torch.amp` for mixed precision
- NumPy and standard Python libraries
- TensorBoard or Weights & Biases for logging

### 3.3 Collaboration and AI Tools

This is an individual assignment.

You may discuss concepts, tensor shapes, mathematical derivations, and general debugging strategies.
You may not share code, inspect another student's code, or exchange trained checkpoints.

AI coding tools are allowed, but you must:

1. disclose the tools and the tasks for which you used them;
2. understand every submitted line;
3. be able to substantiate the submitted implementation and reported evidence.

There is no routine oral checkoff. Staff may request a brief code walkthrough only when needed to
resolve anomalous or conflicting submission evidence or an academic-integrity concern; this is not
a separately graded component.

## 4. Required Configuration

All graded runs use:

| Parameter | Value |
|---|---:|
| Vocabulary size | 8192 |
| Sequence length | 512 |
| `d_model` | 512 |
| Layers | 8 |
| Query heads | 8 |
| KV heads | 4 |
| Head dimension | 64 |
| SwiGLU hidden dimension | 1536 |
| Dropout | 0.0 |
| Biases | none |
| Weight tying | token embedding and output projection |
| Parameter count | approximately 29.37M |

### 4.1 Initialization

Use exactly:

| Parameter | Initialization |
|---|---|
| Token embedding, Q/K/V, FFN `W1` and `W3` | `Normal(0, 0.02)` |
| Attention output projection and FFN `W2` | `Normal(0, 0.02 / sqrt(2 * n_layers))` |
| RMSNorm scale | ones |
| Linear biases | absent |
| RoPE tables | non-persistent buffers, never persistent state or parameters |

The provided reference checkpoint intentionally omits the derived RoPE tables. Register them with
`persistent=False` so strict checkpoint loading works and the tables are regenerated from the model
configuration. The starter tests include a fixed-seed initial-logit and initial-loss check.

## 5. Part 1: Byte-Level BPE Tokenizer - 15 Points

### 5.1 Training - 5 Points

Implement byte-level BPE training:

1. Initialize tokens 0-255 as single bytes.
2. Reserve token ID 256 for `<|endoftext|>`; it counts toward the 8192-token vocabulary.
3. Treat `<|endoftext|>` as a document boundary and never include it in merge-pair counts.
4. Count adjacent token pairs within documents.
5. Merge the most frequent pair.
6. Break frequency ties by ascending `(left_token_id, right_token_id)`.
7. Assign merge token IDs 257 through 8191 in merge order.

The required tokenizer uses no regex pre-tokenization. This makes the algorithm and output fully
specified. Your merge list must match the canonical checksum. On a tiny test corpus, training stops
cleanly if no adjacent pair remains before the requested vocabulary size; the fixed course shard is
guaranteed to reach all 8192 tokens.

The benchmark runs on the fixed course runner and shard. The published student runtime limit is
**0.853 seconds** on the named course runner, three times the staff median. The exact machine-readable
value and runner/software description are in
`artifacts/reference_ranges.json` under
`tokenizer_training.student_limit_seconds`.

### 5.2 Encoding and Decoding - 6 Points

Implement:

```python
encode(text: str, allowed_special: set[str] | None = None) -> list[int]
decode(token_ids: list[int]) -> str
```

Requirements:

- `decode(encode(text)) == text` for every valid UTF-8 string.
- Apply merges in learned priority order.
- Reject disallowed special-token text rather than silently interpreting it.
- Support empty strings, repeated special tokens, and arbitrary Unicode.
- Produce identical output whether a file is encoded at once or as document-aligned chunks.

For arbitrary model-generated token sequences, decode bytes with UTF-8 replacement rather than
crashing on an incomplete or invalid byte sequence. Exact round-trip equality is required for every
valid UTF-8 input.

### 5.3 Vocabulary Analysis - 4 Points

Train once to 8192 tokens, including the reserved special token. Use prefixes of the learned merge
list to evaluate total vocabulary sizes:

```text
512, 1024, 2048, 4096, 8192
```

Report on the fixed validation text:

- bytes per token;
- tokens per whitespace-delimited word;
- encoding throughput;
- one plot of vocabulary size versus bytes per token.

Do not retrain five independent tokenizers.

## 6. Part 2: Modern Transformer - 35 Points

### 6.1 Causal Attention and GQA - 8 Points

Implement scaled dot-product causal attention and multi-head attention with:

- model input shape `(batch, sequence, d_model)`;
- projected attention shape `(batch, heads, sequence, head_dim)`;
- numerically stable softmax;
- no explicit loop over heads;
- `n_heads % n_kv_heads == 0`;
- MHA when `n_kv_heads == n_heads`;
- GQA when `n_kv_heads < n_heads`.

Your implementation must match the PyTorch reference within the tolerance specified by the tests.

### 6.2 Rotary Position Embeddings - 5 Points

Implement RoPE with:

- precomputed sin/cos tables;
- rotation applied to Q and K, not V;
- explicit position offsets for cached decoding;
- even head dimension validation.

The invariant tested is precise: for fixed unrotated vectors, the rotated Q/K dot product depends on
relative position, not on a vague claim that full attention patterns are identical.

### 6.3 RMSNorm and SwiGLU - 5 Points

Implement:

```text
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
SwiGLU(x)  = (silu(x W1) * (x W3)) W2
```

Use `eps=1e-6` and `d_ff=1536`.

### 6.4 Full Transformer LM - 17 Points

Assemble:

```text
token embedding
  -> 8 x [RMSNorm -> GQA -> residual
          RMSNorm -> SwiGLU -> residual]
  -> RMSNorm
  -> tied output projection
```

Requirements:

- pre-norm residual architecture;
- causal language-model logits;
- tied embedding/output weights;
- exact configuration and initialization from Section 4;
- support for full-sequence and incremental cached forward passes;
- parameter count within 1% of the reference count.

## 7. Part 3: Training and Numerics - 20 Points

### 7.1 Cross-Entropy and AdamW - 7 Points

Implement numerically stable cross-entropy:

```text
logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
```

Implement AdamW as a subclass of `torch.optim.Optimizer` with:

- first and second moments;
- bias correction;
- decoupled weight decay;
- parameter groups;
- no decay for RMSNorm scale vectors;
- serializable optimizer state.

The tests compare several consecutive updates, not only the first step.

### 7.2 Schedule, Accumulation, and Mixed Precision - 6 Points

Implement:

- linear warmup over 5% of optimizer steps;
- cosine decay to `0.1 * peak_lr`;
- gradient accumulation with loss divided by `accumulation_steps`;
- gradient clipping at norm 1.0;
- `torch.amp.autocast`.

You may use bf16 on supported GPUs. If using fp16, use `torch.amp.GradScaler` and unscale gradients
before clipping.

At fixed effective batch size, changing the number of micro-batches should produce equivalent
updates within floating-point tolerance.

### 7.3 Data, Logging, and Checkpointing - 7 Points

Implement:

- deterministic random-window sampling from packed token arrays using a dedicated generator;
- checkpoint save and resume for model, optimizer, scaler, global step, RNG states, and data-generator
  state; the cosine schedule is stateless and is reconstructed from the global step and frozen config;
- training/validation loss;
- learning rate;
- pre-clipping gradient norm;
- tokens per second;
- peak GPU memory.

A resumed run must match an uninterrupted run for the next several optimizer steps within the
published tolerance.

## 8. Part 4: Generation and Evaluation - 10 Points

### 8.1 Sampling - 4 Points

Implement greedy, temperature, top-k, and top-p sampling. Define `temperature=0` as greedy decoding.
Top-k must retain exactly `k` token IDs; equal logits are ordered by lower token ID. The starter
tests cover filtering boundaries, ties, and invalid arguments.

### 8.2 Perplexity and Bits per Byte - 2 Points

Report:

- token-level perplexity for your fixed required tokenizer;
- bits per byte on the fixed raw validation span.

The canonical evaluator scores each target byte/token exactly once. It may use overlapping context
windows, but overlapping targets are never counted twice.

### 8.3 KV Cache - 4 Points

Implement incremental decoding with per-layer K/V caches.

Required correctness:

- cached and non-cached logits match token by token;
- position offsets are correct;
- cache shape supports MHA and GQA;
- cache growth does not recompute prior K/V tensors.

Run the provided benchmark and report speedup and cache memory. There is no hard `5x` grade
threshold because kernel launch overhead and GPU type can dominate at this model size.

The release repo includes cached/non-cached golden logits, single-token and chunked-cache tests, and
an executable benchmark. These materials are the required prerequisite for this part; the Sep 22
efficient-inference lecture is reinforcement, not the first specification of the cache API.

## 9. Part 5: Experiments and Report - 20 Points

### 9.1 Required TinyStories Run - 12 Points

Train the required GQA model with:

| Hyperparameter | Value |
|---|---:|
| Optimizer steps | 3000 |
| Effective tokens per step | `2^17` |
| Sequence length | 512 |
| Peak learning rate | `3e-4` |
| Warmup | 150 steps |
| Minimum learning rate | `3e-5` |
| Weight decay | 0.1 |
| Betas | `(0.9, 0.95)` |
| Gradient clip | 1.0 |
| Seed | 2026 |

Report:

- training and validation loss curves;
- actual learning-rate curve;
- gradient norm and throughput;
- final token PPL and bits per byte;
- four generated samples using the fixed prompts and decoding settings in the starter repo;
- two concrete failure modes.

The staff solution's measured reference is published in
`artifacts/training_reference.json`. Results outside the range do not automatically receive zero;
they require a technically sound diagnosis.

### 9.2 Short MHA vs. GQA Comparison - 4 Points

Compare:

- MHA: 8 query heads, 8 KV heads;
- GQA: 8 query heads, 4 KV heads.

Use the first 1000 steps of your required GQA run and one additional 1000-step MHA run with identical
seed, data order, initialization policy, and hyperparameters.

Report:

- parameter counts;
- validation loss at step 1000;
- analytical KV-cache size;
- measured decoding throughput at batch sizes 1 and 32;
- a short interpretation that treats “no measurable speed difference” as a valid result.

### 9.3 Written Report - 4 Points

Submit a concise report, at most 6 main-text pages excluding references and appendix. In addition to
the required plots and tables, answer:

1. Why is token PPL not comparable across arbitrary tokenizers, and how does bits per byte help?
2. Derive the KV-cache memory formula for MHA, GQA, and MQA.
3. Explain why a causal mask permits parallel training without future-token leakage.
4. Explain why AdamW differs from L2 regularization under Adam.
5. Diagnose one real bug you encountered using evidence from tests or logs.

## 10. Grading

| Component | Points |
|---|---:|
| Part 1: Tokenizer | 15 |
| Part 2: Transformer | 35 |
| Part 3: Training and numerics | 20 |
| Part 4: Generation and evaluation | 10 |
| Part 5: Experiments and report | 20 |
| **Total** | **100** |

Each task has explicit correctness tests and rubric anchors. There is no blanket
“30% tests / 40% manual / 30% results” rule across unrelated parts.

Hidden tests extend the public tests with edge cases; they do not introduce undocumented APIs or
performance requirements.

## 11. Bonus Leaderboard - Up to 3 Points

Optional OpenWebText experiments are ranked by **bits per byte**, not token PPL.

Rules:

- use only the provided training data;
- train from random initialization;
- at most 100M parameters;
- at most 2 billion training tokens;
- single-GPU inference;
- submit a reproducible config and checkpoint.

The release publishes three fixed baseline tiers in `artifacts/bonus_baselines.json`:

- beat baseline B1: +1;
- beat baseline B2: +2;
- beat baseline B3: +3.

The bonus is not based on class percentile.

## 12. Compute and Contingency

The required protocol is designed for approximately 1-3 successful modern GPU-hours:

| Run | Approximate purpose |
|---|---|
| Smoke config | CPU/MPS correctness and overfit |
| GQA 3000-step TinyStories | required main run |
| MHA 1000-step TinyStories | required short comparison |

Plan for up to 6 GPU-hours to include debugging and one restart. Queue time is not GPU time.

Staff publish reference timings for each supported GPU class. If an ICE outage removes a material
fraction of the assignment window, staff will reduce the required training steps or extend the
deadline; students will not be graded on inaccessible compute.

See [Tutorial 0](TUTORIAL0.md) for ICE setup. Use one GPU and checkpoint regularly.

## 13. Submission

Submit through Gradescope:

1. `code.zip`, produced by `scripts/make_submission.py`;
2. `writeup.pdf`;
3. `ai_disclosure.md`;
4. the final run config and machine-readable metrics JSON.

Open the HW1 assignment in Canvas for the current ZIP, deadline, and Gradescope launch link.

## 14. Required Background

- Vaswani et al., “Attention Is All You Need”
- Sennrich et al., “Neural Machine Translation of Rare Words with Subword Units”
- Su et al., “RoFormer”
- Zhang and Sennrich, “Root Mean Square Layer Normalization”
- Shazeer, “GLU Variants Improve Transformer”
- Ainslie et al., “GQA”
- Loshchilov and Hutter, “Decoupled Weight Decay Regularization”

Course Lectures 2-4 and the starter README contain the required implementation guidance.

## 15. FAQ

**Can I use the canonical tokenizer or tiny reference checkpoint?**  
Yes, for debugging downstream components. Your final required run must use your submitted tokenizer
and model implementation.

**Can my required model use MHA instead of GQA?**  
No. The required model uses four KV heads. MHA is used only for the short comparison.

**Can I use fused attention?**  
Not for the graded core implementation. It is permitted for optional leaderboard experiments after
your own attention passes all tests.

**What if my loss is outside the staff reference range?**  
You still receive correctness credit. Use component tests, the smoke-overfit test, initial logits,
gradient norms, and checkpoint comparisons to diagnose the difference.

**Do I need a 5x KV-cache speedup?**  
No. Correct cached decoding and a reproducible benchmark are required. The measured speedup is an
experimental result.

**Why is the leaderboard metric bits per byte?**  
It normalizes the total negative log-likelihood by raw bytes, so systems using different tokenizers
can be compared more meaningfully.

---

*Last updated: August 15, 2026*
