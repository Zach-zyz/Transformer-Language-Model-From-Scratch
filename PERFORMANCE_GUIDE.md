# HW1 Tokenizer Performance Guide

This guide describes algorithmic constraints, not an implementation. Your tokenizer remains graded
code that you must write and explain.

## Why the Direct Algorithm Is Too Slow

A correct first version often does this for every merge:

1. scan every document to count all adjacent pairs;
2. select the best pair;
3. scan every document again to replace that pair.

It is useful as a tiny-corpus oracle, but it repeats nearly the same work thousands of times.
Likewise, encoding by rescanning the entire token sequence once for every learned merge is not
viable for packing the full TinyStories data.

Keep the direct version for differential tests on small random corpora. Use a local-update algorithm
for the fixed 8192-token run and full-data encoding.

## Scalable Training Structure

One workable design maintains:

- each document as a mutable linked sequence of token nodes;
- a count for every currently adjacent token pair;
- the live occurrences of each pair;
- a priority structure ordered by `(-frequency, left_id, right_id)`;
- a version or lazy-invalid marker so stale priority entries can be discarded.

When merging one pair, only adjacencies touching an occurrence of that pair can change. Remove the
old left/current/right pair contributions, splice in the new token node, then add the new local
pair contributions. Do not recount unrelated documents.

Important invariants:

- occurrences never cross `<|endoftext|>` document boundaries;
- overlapping occurrences are handled left to right without reusing a consumed node;
- the selected pair obeys the exact frequency and ascending-ID tie break;
- stale occurrence and priority entries are validated before use.

Compare the optimized merge list against your direct oracle on many tiny corpora before trusting
performance measurements.

## Scalable Encoding Structure

For a fixed merge-rank table, maintain a mutable linked token sequence and a priority queue of
currently adjacent mergeable pairs. Order candidates by learned merge rank and sequence position.
After applying one merge, invalidate stale candidates and add only the new left and right
adjacencies.

This produces the same result as applying every learned merge in order while avoiding a full
sequence scan for each merge. Stream the corpus in document-aligned chunks; never split or merge
across the special-token boundary.

## Measure Before Full Packing

Use the supplied commands:

```bash
python scripts/train_tokenizer.py \
  --output outputs/tokenizer/tokenizer.json \
  --metrics outputs/tokenizer/training.json

python scripts/benchmark_tokenizer.py \
  --tokenizer outputs/tokenizer/tokenizer.json \
  --output outputs/tokenizer/benchmark.json \
  --plot outputs/tokenizer/vocabulary.svg
```

Read `artifacts/reference_ranges.json` for the named runner, software versions, staff timing, and
packing throughput. Before starting a multi-gigabyte pack, time encoding on `validation_span.txt`
and a document-aligned 1 MiB sample. Extrapolate from measured bytes per second.

## Debugging Order

1. Make the direct oracle correct on tiny examples.
2. Differential-test optimized training against the oracle.
3. Differential-test ranked encoding against sequential merge application.
4. Verify the canonical tokenizer SHA-256.
5. Benchmark the fixed validation span.
6. Run a 64-document packing smoke test.
7. Start the full pack only after the projection is reasonable.

If an optimized implementation disagrees with the oracle, minimize the failing corpus and inspect
overlapping occurrences, stale heap entries, document boundaries, and tie-breaking first.
