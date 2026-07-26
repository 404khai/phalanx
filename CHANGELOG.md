# Changelog

All notable changes to Phalanx Runtime are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 4 tokenizer: [`tokenizer`](src/tokenizer/) module loading vocab /
  specials from GGUF metadata, with encode (greedy / BPE) and decode
  (`▁` / `<0xXX>`), plus nested [`TokenizerError`](src/tokenizer/error.rs).
- Educational tokenizer notes in [`docs/tokenizer.md`](docs/tokenizer.md).
- Phase 3 GGUF parser: [`gguf`](src/gguf/) module with streaming header,
  metadata KV (all value types), tensor info directory, alignment /
  `data_offset`, and nested [`GgufError`](src/gguf/error.rs).
- Educational GGUF notes in [`docs/gguf.md`](docs/gguf.md).
- Phase 2 math foundation: [`tensor`](src/tensor/) module with `DType`, `Shape`,
  `Tensor`, and nested [`TensorError`](src/tensor/error.rs).
- Contiguous row-major `f32` storage with stride / multi-index offset helpers.
- Reference kernels: element-wise `add`/`sub`/`mul`/`div`, `scale`, `matmul`,
  `transpose`, `sum`.
- Criterion microbenchmarks in [`benches/tensor_ops.rs`](benches/tensor_ops.rs).
- Phase 1 repository foundation: Cargo library + binary crate.
- Typed [`PhalanxError`](src/errors/mod.rs) surface via `thiserror`.
- Structured logging bootstrap via `tracing` / `tracing-subscriber`.
- Project documentation: README, architecture notes, implementation notes.
- Formatting (`rustfmt.toml`) and Clippy lint configuration.
- Smoke tests covering the public crate API.

## [0.1.0] - TBD

Initial development version; not yet published.
