# Changelog

All notable changes to `julia-system-one` are documented here.

## [1.0.2] — 2026-09-27

### Added: measured performance, and a model identity in /health

- **Benchmarks for this model**, measured through the published package on
  every platform we ship a binary for, plus warm-engine numbers from one
  machine. The README previously carried the other checkpoint's figures.
- **`/health` reports which model is loaded**: the encoder named by
  `rl_agent_config.json` and the sha256 of the model file. It used to return a
  hardcoded name, which proves nothing — a deployment could be serving a
  different checkpoint and it would still say so.

### Fixed: the published-package check

`verify-published.yml` failed on all eight platforms, for three reasons that
had nothing to do with the package:

- the chunk count was hardcoded to 13 (this model ships 6);
- it read the manifest from the repository, but the job installs from the
  registry and never checks the repository out;
- it gated on exact answer labels. This model is INT8 with dynamic
  quantization, and ONNX Runtime's kernels differ per architecture: the same
  commit scored 5/10 in one run and 8/10 in the next, on the same platform. It
  now asserts what a build check can assert — the binary starts, loads the
  model, answers every prompt, returns a valid label with probabilities
  summing to 1, and repeats an answer when a prompt is repeated.

## [1.0.1] — 2026-09-27

### Fixed: /health returned a hardcoded model name

It now reports the encoder and the model's sha256, so a deployment can confirm
which checkpoint it serves. 1.0.0 is on the registry without this.

## [1.0.0] — 2026-09-27

First release: the Julia-1 decision model, running in Node.js, Bun and the
browser with no Python in the loop.

### The model

- **Julia-1** ([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)),
  144.3M parameters, a fine-tune of
  [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small).
  Exported to ONNX and quantized to INT8: **147.6 MB**, against 324 MB for the
  mmBERT-base checkpoint this runtime was first built for.
- Distributed as **6 chunk packages** on npm, reassembled and checksum-verified
  on first use. The checkpoint's bytes never depend on a git repository.

### The runtime

- **No runtime dependencies.** `dependencies` is empty. The inference server is
  a self-contained Rust binary that ships with the package; the tokenizer is a
  pure-JS BPE implementation; the browser path is a pure-Rust wasm build.
- **Node.js, Bun and the browser.** The right binary for the machine is
  selected by npm through `os`/`cpu`, so an install pulls 8–27 MB of binary
  rather than every platform's.
- **The CPU execution provider is pinned.** Left to itself, ONNX Runtime
  registers whichever provider its prebuilt was built with, and `ort-sys`
  ships the DirectML build for Windows x64 — the same model answered
  differently there (3/10 on the check against 10/10 on Linux). Windows now
  links Microsoft's official CPU build.
- **The wasm engine compiles asynchronously**, because Chrome refuses to
  compile more than 8 MB on the main thread and this engine is ~13 MB.

### What is specific to Julia-1

- The tokenizer is byte-for-byte the same file as the checkpoint this runtime
  was first built for (sha256 identical), so the BPE implementation needed no
  changes.
- The ONNX contract is the same five inputs and one `logits` output.
- The exported graph leaves its dimensions as **expressions** — `6*batch`,
  `batch*tokens`, `(tokens//batch)` — where the earlier checkpoints had
  concrete names. Each is resolved with batch fixed at 1; an unrecognised
  expression is left alone deliberately, because a wrong substitution would
  produce wrong numbers silently while an unresolved one fails loudly at plan
  time.

### Known differences

- **The wasm backend can answer differently from the native one.** Julia-1's
  graph uses dynamic quantization (98 `DynamicQuantizeLinear`/`MatMulInteger`
  pairs), which ONNX Runtime and tract implement differently. Where they
  disagree, the native path matches the reference implementation (Python +
  ONNX Runtime) and the wasm path does not. Prefer `native` where it is
  available.
- **Accuracy is the model's, not this package's.** Julia-1 reports 71.5%
  macro accuracy across 52 MASSIVE locales, 86.8% on English, 73.15% on the
  typed-decision suite. Long-document accuracy is not measured for this model
  yet.
