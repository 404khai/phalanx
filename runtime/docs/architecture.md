# Architecture — Phase 6

This document tracks architectural intent as Phalanx grows. Update it every
phase; the README embeds a summary diagram for quick reading.

## Goals

- **Correctness first** — wrong logits are worse than slow logits.
- **Readable systems code** — senior engineers should navigate without tribal knowledge.
- **Incremental completeness** — each phase ships a compiling, tested slice.
- **Educational clarity** — diagrams and notes explain *why*, not only *what*.

## Phase 6 component map

```mermaid
flowchart TB
    main["main.rs"]
    lib["lib.rs"]
    errors["errors::PhalanxError"]
    gguf["gguf::GgufFile"]
    tok["tokenizer::Tokenizer"]
    weights["weights::WeightSet"]
    model["model::ModelConfig"]
    tensor["tensor::Tensor"]

    main --> lib
    lib --> errors
    lib --> gguf
    lib --> tok
    lib --> weights
    lib --> model
    lib --> tensor
    tok --> gguf
    weights --> gguf
    model --> gguf
    weights --> tensor
    model -.-> weights
```

### Responsibilities

| Component | Responsibility | Non-goals (Phase 6) |
|---|---|---|
| `gguf` | Parse directory / metadata | Own the byte map |
| `weights` | mmap, bounds check, dense materialize | Full Q4_K dequant kernels |
| `tokenizer` | Vocab encode/decode | Chat templates |
| `model` | Architecture + validated hparams | Layer execution / weight binding |
| `tensor` | Contiguous f32 math | On-disk layout |

## Config load pipeline

```mermaid
flowchart LR
    Meta["GgufFile metadata"] --> Arch["general.architecture"]
    Arch --> Keys["{arch}.* hparams"]
    Keys --> Validate["shape / GQA / RoPE checks"]
    Validate --> Cfg["ModelConfig"]
```

## Boundary rules

1. **Library never depends on CLI concerns.**
2. **Typed errors stay in the library.**
3. **No empty domain modules.**
4. **`unsafe` only in `weights::storage`** for `memmap2::map`, with safety docs.
5. **Tokenizer reads only metadata** — weights module owns file bytes.
6. **Quantized payloads stay as `&[u8]`** until a kernel needs dequant.
7. **Layers read shapes from `ModelConfig`**, not raw metadata maps.

## Module ownership

| Module | Owns | Introduced |
|---|---|---|
| `tensor` | Contiguous buffers, shapes, ops | Phase 2 |
| `gguf` | Header, metadata, tensor info | Phase 3 |
| `tokenizer` | Vocab, specials, encode/decode | Phase 4 |
| `weights` | mmap, quant meta, materialize | Phase 5 |
| `model` | Architecture + hyperparameters | **Phase 6** |

## Tradeoffs recorded

### Llama-only vs multi-arch parser

| Option | Pros | Cons |
|---|---|---|
| **Llama-only (chosen)** | Tight Phase 6 scope; clear validation | Other GGUF archs rejected loudly |
| Generic key reader for all archs | Broader open | Fake “support” without kernels |

### Nested `ModelError` vs `PhalanxError::Config(String)`

| Option | Pros | Cons |
|---|---|---|
| **Nested typed error (chosen)** | Matchable; consistent with gguf/tokenizer | Another variant |
| String `Config` only | Fewer types | Callers scrape messages |

## Evolution policy

When a phase adds a subsystem:

1. Add the module with real types and tests.
2. Update Mermaid diagrams here and in the README.
3. Record the tradeoff that drove the design.
4. Extend `PhalanxError` with a typed nested error.
