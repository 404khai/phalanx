# Architecture — Phase 9

This document tracks architectural intent as Phalanx grows. Update it every
phase; the README embeds a summary diagram for quick reading.

## Goals

- **Correctness first** — wrong logits are worse than slow logits.
- **Readable systems code** — senior engineers should navigate without tribal knowledge.
- **Incremental completeness** — each phase ships a compiling, tested slice.
- **Educational clarity** — diagrams and notes explain *why*, not only *what*.
- **Odyssey parity** — Rule 6: claim Spec compliance only after `validate_*` PASSes.

## Phase 9 component map

```mermaid
flowchart TB
    main["main.rs"]
    lib["lib.rs"]
    errors["errors::PhalanxError"]
    gguf["gguf::GgufFile"]
    tok["tokenizer::Tokenizer"]
    weights["weights::WeightSet"]
    model["model::ModelConfig"]
    emb["layers::EmbeddingTable"]
    rope["layers::Rope"]
    rms["layers::RmsNorm"]
    tensor["tensor::Tensor"]

    main --> lib
    lib --> errors
    lib --> gguf
    lib --> tok
    lib --> weights
    lib --> model
    lib --> emb
    lib --> rope
    lib --> rms
    lib --> tensor
    tok --> gguf
    weights --> gguf
    model --> gguf
    emb --> weights
    emb --> model
    rope --> model
    rms --> model
    rms --> weights
    emb --> tensor
    rope --> tensor
    rms --> tensor
```

### Responsibilities

| Component | Responsibility | Non-goals (Phase 9) |
|---|---|---|
| `model` | Hparams including `rms_norm_eps` | Execute norms |
| `layers::RmsNorm` | γ ⊙ x / RMS(x) | Residual block wiring |
| `layers::Rope` | Cos/sin cache + Q/K rotate | Attention scores |
| `layers::EmbeddingTable` | Token gather | Positions |
| `weights` | mmap / materialize | Dequant kernels |

## RMSNorm pipeline

```mermaid
flowchart LR
    X["activations (…, D)"] --> RMS["RmsNorm::forward"]
    G["γ weight [D]"] --> RMS
    E["eps"] --> RMS
    RMS --> Y["normalized (…, D)"]
```

## Boundary rules

1. **Library never depends on CLI concerns.**
2. **Typed errors stay in the library.**
3. **No empty domain modules.**
4. **`unsafe` only in `weights::storage`** for `memmap2::map`, with safety docs.
5. **Tokenizer reads only metadata** — weights module owns file bytes.
6. **Quantized payloads stay as `&[u8]`** until a kernel needs dequant.
7. **Layers read shapes from `ModelConfig`**, not raw metadata maps.
8. **ggml dimension order is reinterpreted explicitly** at layer boundaries.
9. **RoPE does not touch V** — only Q/K (attention Phase 11).
10. **RMSNorm is not LayerNorm** — no mean centering; Spec-noncompliant otherwise.
11. **Cross-impl validators** (`validate_rope`, `validate_rmsnorm`) are part of the public contract.

## Module ownership

| Module | Owns | Introduced |
|---|---|---|
| `tensor` | Contiguous buffers, shapes, ops | Phase 2 |
| `gguf` | Header, metadata, tensor info | Phase 3 |
| `tokenizer` | Vocab, specials, encode/decode | Phase 4 |
| `weights` | mmap, quant meta, materialize | Phase 5 |
| `model` | Architecture + hyperparameters | Phase 6 |
| `layers` | Embedding + RoPE + RMSNorm | Phase 7–**9** |

## Tradeoffs recorded

### Float64 RMS reduction

| Option | Pros | Cons |
|---|---|---|
| **f64 Σx² then f32 RMS (chosen)** | Bit-match Odyssey; stable for D=768+ | Extra cast |
| Pure f32 reduction | Slightly faster | ~1e-6 drift vs Odyssey |

### Linear-only RoPE scaling (Phase 8)

| Option | Pros | Cons |
|---|---|---|
| **Linear only (chosen)** | Matches common GGUF exports; correct math | YaRN/NTK rejected |
| Stub all scaling types as no-ops | Broader open | Silently wrong long-context |

## Evolution policy

When a phase adds a subsystem:

1. Add the module with real types and tests.
2. Update Mermaid diagrams here and in the README.
3. Record the tradeoff that drove the design.
4. Extend `PhalanxError` with a typed nested error.
5. Add `validate_<component>` and land a PASS report before flipping Spec compliance.
