# Changelog

All notable changes to `julia-system-one` are documented here.

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
