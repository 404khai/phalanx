# Changelog

All notable changes to Phalanx Runtime are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 8 rotary embeddings: [`layers::Rope`](src/layers/rope.rs) with
  precomputed cos/sin caches, Llama adjacent-pair rotation, partial rotary
  dims, linear scaling, and docs in [`docs/rope.md`](docs/rope.md).
- Phase 7 embedding layer: [`layers`](src/layers/) module with
  [`EmbeddingTable`](src/layers/embedding.rs) gather from
  `token_embd.weight`, ggml layout reinterpret to `[vocab, embd]`, and nested
  [`LayersError`](src/layers/error.rs).
- Educational embedding notes in [`docs/embeddings.md`](docs/embeddings.md).
- Phase 6 model configuration: [`model`](src/model/) module with
  [`Architecture`](src/model/architecture.rs), validated
  [`ModelConfig`](src/model/config.rs) (attention / RoPE hparams) from GGUF
  `{arch}.*` metadata, and nested [`ModelError`](src/model/error.rs).
- Educational model notes in [`docs/model.md`](docs/model.md).
- Phase 5 weight loading: [`weights`](src/weights/) module with read-only
  `memmap2` mapping, [`QuantMeta`](src/weights/quant.rs) block layouts,
  payload bounds checks, and dense `f32`/`f16` materialization into
  [`Tensor`](src/tensor/). Nested [`WeightsError`](src/weights/error.rs).
- Educational weights notes in [`docs/weights.md`](docs/weights.md).
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
