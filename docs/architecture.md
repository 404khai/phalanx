# Architecture — Phase 2

This document tracks architectural intent as Phalanx grows. Update it every
phase; the README embeds a summary diagram for quick reading.

## Goals

- **Correctness first** — wrong logits are worse than slow logits.
- **Readable systems code** — senior engineers should navigate without tribal knowledge.
- **Incremental completeness** — each phase ships a compiling, tested slice.
- **Educational clarity** — diagrams and notes explain *why*, not only *what*.

## Phase 2 component map

```mermaid
flowchart TB
    main["main.rs<br/>process edge"]
    lib["lib.rs<br/>crate root"]
    errors["errors::PhalanxError"]
    logging["utils::logging"]
    tensor["tensor::Tensor"]
    shape["tensor::Shape"]
    ops["tensor ops<br/>add/mul/matmul"]
    tracing["tracing subscriber"]

    main -->|"anyhow::Context"| lib
    lib --> errors
    lib --> logging
    lib --> tensor
    logging --> tracing
    main -->|init_logging| logging
    tensor --> shape
    tensor --> ops
    ops --> errors
    tensor --> errors
```

### Responsibilities

| Component | Responsibility | Non-goals (Phase 2) |
|---|---|---|
| `main.rs` | Process entry, logging init, banner | Arg parsing, generation loop |
| `lib.rs` | Public surface, version constants | Domain algorithms beyond tensors |
| `errors` | Typed library failures | CLI exit-code mapping |
| `utils::logging` | Global subscriber bootstrap | Metrics / profiling exporters |
| `tensor` | Contiguous f32 buffers, shape/strides, reference kernels | Quantized storage, broadcasting, BLAS |

## Tensor memory layout

Phalanx stores dense tensors as **contiguous row-major** `f32` buffers:

```text
shape [2, 3]          strides [3, 1]

logical                 physical
[[a00 a01 a02]          [a00 a01 a02 a10 a11 a12]
 [a10 a11 a12]]
```

Offset formula: `sum(index[i] * stride[i])`.

### Why row-major?

| Layout | Pros | Cons |
|---|---|---|
| **Row-major / C-order (chosen)** | Matches NumPy defaults; familiar for GGUF tooling notes | Column-heavy BLAS may prefer col-major |
| Column-major / Fortran | Natural for some BLAS APIs | Surprises Python-trained readers |

**Decision:** row-major contiguous for Phase 2. If a future BLAS backend requires col-major, convert at the boundary rather than infecting the core model.

## Boundary rules

1. **Library never depends on CLI concerns** — no `clap` / argv parsing inside `lib`.
2. **Typed errors stay in the library** — `anyhow` is for binaries and examples.
3. **No empty domain modules** — create `gguf`, `attention`, … only with real code.
4. **`unsafe` is forbidden** until a reviewed hot path (e.g. mmap, SIMD) needs it.
5. **Tensors stay contiguous** — strided views wait until KV-cache windows need them.

## Target architecture (later phases)

```mermaid
flowchart LR
    CLI --> Runtime
    Runtime --> Tokenizer
    Runtime --> Loader
    Loader --> GGUF
    Loader --> Weights
    Runtime --> Model
    Model --> Embed
    Model --> Blocks
    Blocks --> Attn
    Blocks --> FFN
    Attn --> KVCache
    Model --> Sampler
    Model --> Tensor
```

### Module ownership

| Module | Owns | Introduced |
|---|---|---|
| `tensor` | Contiguous buffers, dtype, shapes, ops | **Phase 2** |
| `gguf` | Header, metadata, tensor info | Phase 3 |
| `tokenizer` | Vocab, encode/decode | Phase 4 |
| `model` | Config + weight handles | Phase 6 |
| `attention` / `kv_cache` | Decode-critical path | Phases 11–12 |
| `sampling` | Logits → token | Phase 14 |
| `runtime` | Orchestration / streaming | Phases 13–15 |
| `cli` | User-facing commands | Phase 16 |

## Tradeoffs recorded

### Single crate vs workspace

| Option | Pros | Cons |
|---|---|---|
| **Single crate (chosen)** | Simple paths, one `Cargo.toml`, fast iteration | All code shares dependency set |
| Cargo workspace | Cleaner publish boundaries | Overhead too early for ~few modules |

**Decision:** stay single-crate until compile times or dependency isolation hurt.

### Owned `Tensor` vs `ndarray`

| Option | Pros | Cons |
|---|---|---|
| **Owned contiguous `Vec<f32>` (chosen)** | Teaches layout; tiny API; easy invariants | Manual broadcasting / views later |
| `ndarray` | Views, broadcasting, mature ops | Hides memory model; large dependency |

**Decision:** own the abstraction. Revisit only if maintenance cost exceeds benefit.

### Error & logging choices

Documented in [implementation-notes.md](implementation-notes.md).

## Evolution policy

When a phase adds a subsystem:

1. Add the module with real types and tests.
2. Update the Mermaid diagrams here and in the README.
3. Record the tradeoff that drove the design.
4. Extend `PhalanxError` with a typed variant (or nested error) — avoid stringly errors for expected failures.
