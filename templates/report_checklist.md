# HW1 Report Checklist

The main text is at most six pages, excluding references and appendix.

## Required Evidence

- [ ] Tokenizer vocabulary-size versus bytes-per-token plot.
- [ ] Bytes per token, tokens per word, and encoding throughput for all five vocabulary sizes.
- [ ] GQA training loss, validation loss, learning rate, gradient norm, and throughput.
- [ ] Final GQA token perplexity and bits per byte.
- [ ] Four fixed-prompt samples with the released decoding settings.
- [ ] Two concrete model failure modes.
- [ ] GQA and MHA parameter counts and validation loss at step 1000.
- [ ] Analytical KV-cache sizes for MHA and GQA.
- [ ] Batch-size 1 and 32 cached-decoding throughput.
- [ ] Reproducible run configuration, machine description, and checkpoint step.

## Required Explanations

- [ ] Token PPL comparability and why bits per byte helps.
- [ ] KV-cache memory derivation for MHA, GQA, and MQA.
- [ ] Why causal masking permits parallel training.
- [ ] AdamW versus L2 regularization under Adam.
- [ ] One real implementation bug diagnosed from tests or logs.

## Submission

- [ ] `writeup.pdf`
- [ ] `ai_disclosure.md`
- [ ] `run_config.yaml`
- [ ] `metrics.json`
- [ ] `code.zip`
- [ ] `python scripts/check_submission.py submission` passes.
