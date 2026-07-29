# Implementation Notes

Notes capture decisions that are easy to lose in commit history. Prefer
updating this file over leaving undocumented tribal knowledge.

---

## Phase 1 — Repository foundation

### Scope delivered

- Cargo package `phalanx` with library (`src/lib.rs`) and binary (`src/main.rs`)
- `errors` module: `PhalanxError`, `Result<T>`
- `utils::logging`: `LogConfig`, `init_logging`
- Tooling: `rustfmt.toml`, Clippy lints, `.cargo/config.toml` aliases
- Docs: README, architecture, this file, CHANGELOG, LICENSE
- Tests: unit tests in modules + `tests/smoke.rs`

### Decision log

1. **Library errors = `thiserror`, CLI = `anyhow`** — matchable API vs edge ergonomics.
2. **`tracing` for logging** — spans map to load / prefill / decode.
3. **Edition 2024 + `rust-version = "1.85"`** — latest stable, greenfield.
4. **`unsafe_code` lint** — started as `forbid`; Phase 5 lowered to `deny` so
   `weights::storage` can opt in for `memmap2` after review.
5. **No empty domain folders** — tree reflects reality.
6. **Commit `Cargo.lock`** — binary runtime needs reproducible builds.

---

## Phase 2 — Math foundation

### Scope delivered

- `tensor` module: `DType`, `Shape`, `Tensor`, `TensorError`
- Contiguous row-major `f32` storage with explicit strides helper
- Ops: `add` / `sub` / `mul` / `div`, `scale`, `matmul`, `transpose`, `sum`
- `PhalanxError::Tensor` nesting via `#[from]`
- Criterion bench `benches/tensor_ops.rs` (elemwise add, matmul, transpose)
- Unit + integration coverage for shape math and kernels

### Decision log

#### 1. Owned contiguous `Vec<f32>` (not `ndarray`)

**Pros:** Clear memory model for an educational runtime; trivial aliasing story.
**Cons:** No free broadcasting / advanced views.
**Reason:** Teaching layout matters as much as shipping ops. Phase 5 can swap
storage behind the façade.

#### 2. `DType` tag with only `F32` today

**Pros:** Call sites already thread a dtype; quantized variants slot in later.
**Cons:** Slight indirection while only one variant exists.
**Reason:** Avoid a painful API break when GGUF quants arrive.

#### 3. No broadcasting in Phase 2

**Pros:** Shape errors stay obvious; kernels stay short.
**Cons:** Some NumPy-style one-liners need explicit `scale` / expand later.
**Reason:** Attention / FFN paths use explicit shapes; broadcast bugs are subtle.

#### 4. Matmul is naïve \(O(n^3)\) ijk loops

**Pros:** Auditable reference; good correctness oracle for future kernels.
**Cons:** Not competitive with BLAS.
**Reason:** Phase 17/18 own performance work; wrong-fast is worthless.

#### 5. Transpose copies into a new contiguous buffer

**Pros:** Preserves the “always contiguous” invariant for every `Tensor`.
**Cons:** Extra bandwidth vs a strided view.
**Reason:** Defer strided tensors until KV-cache windows need them (Phase 12).

#### 6. Criterion 0.7 (not 0.8)

**Pros:** Honors `rust-version = "1.85"`.
**Cons:** Misses newest Criterion features.
**Reason:** Cargo rejected 0.8 (needs rustc 1.86). Bump MSRV intentionally later
if we want 0.8.

### Testing strategy (Phase 2)

| Layer | Location | Covers |
|---|---|---|
| Unit | `tensor::*` | strides, offsets, constructors, ops |
| Integration | `tests/smoke.rs` | public matmul + nested tensor errors |
| Bench | `benches/tensor_ops.rs` | baseline latency for add / matmul / transpose |

### Performance notes

Baseline only — numbers vary by machine. Use `cargo bench --bench tensor_ops`
to refresh locally. Expect matmul_256 to dominate; treat regressions after
kernel changes as signal, not absolute SLA yet.

### Follow-ups deferred

- Broadcasting / batched matmul → when attention needs them
- f16 / bf16 / quantized storage → Phase 5+
- SIMD / blocked matmul → Phase 17
- Strided views → Phase 12 (KV cache)
- CLI argument parsing (`clap`) → Phase 16
- Workspace split → when compile time / deps justify it

### References used this phase

- Golub & Van Loan, *Matrix Computations* (matmul structure)
- [NumPy C-order / row-major](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Criterion.rs](https://docs.rs/criterion)
- Prior Phase 1 references (`thiserror`, `anyhow`, `tracing`)

---

## Phase 3 — GGUF file parser

### Scope delivered

- `gguf` module: `GgufFile`, `GgufHeader`, `MetadataValue`, `TensorInfo`, `GgmlType`
- Streaming little-endian reader with running byte offset
- Metadata KV decode for all `gguf_metadata_value_type`s (incl. nested arrays)
- Tensor directory parse + alignment (`general.alignment` or default 32)
- `data_offset` computation; weight blob intentionally unread
- `PhalanxError::Gguf` nesting
- In-test `GgufBuilder` fixture writer + unit/integration tests
- Educational notes in `docs/gguf.md`

### Decision log

#### 1. Streaming `Read` instead of slurp / mmap

**Pros:** Inspect multi-GB checkpoints without pulling weights into RAM.
**Cons:** More cursor bookkeeping than `Cursor<&[u8]>` alone.
**Reason:** Phase 3’s job is the directory, not the payload.

#### 2. Hand-rolled parser (no crates.io `gguf`)

**Pros:** Matches the educational mission; typed `GgufError` we control.
**Cons:** We own format quirks.
**Reason:** Senior readers should see the byte layout, not a black box.

#### 3. Accept versions 2 and 3

**Pros:** Covers virtually all modern GGUF files.
**Cons:** Must not silently accept structural breaks in future versions.
**Reason:** Reject unknown versions loudly via `UnsupportedVersion`.

#### 4. Preserve unknown `ggml_type` as `GgmlType::Unknown`

**Pros:** Inspection still works when ggml adds formats.
**Cons:** Callers must handle `Unknown` before dequant (Phase 5).
**Reason:** Forward-compatible directory listing.

#### 5. Safety limits on strings / arrays / counts

**Pros:** Hostile headers cannot force absurd allocations.
**Cons:** Extremely exotic files might need limit bumps.
**Reason:** Parser robustness before trust.

### Testing strategy (Phase 3)

| Layer | Location | Covers |
|---|---|---|
| Unit | `gguf::*` | magic, version, alignment, tensor offsets, arrays |
| Integration | `tests/smoke.rs` | public `GgufFile` / `GgufError` re-exports |

### Follow-ups deferred

- `mmap` / dequant of `tensor_data` → Phase 5
- Big-endian GGUF → if/when real files require it
- CLI `inspect` subcommand → Phase 16

### References used this phase

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

## Phase 4 — Vocabulary & tokenizer

### Scope delivered

- `tokenizer` module: `Tokenizer`, `Vocabulary`, `SpecialTokens`, `TokenizerModel`
- Load from `tokenizer.ggml.*` via `Tokenizer::from_gguf`
- Decode with SentencePiece `▁` + `<0xXX>` byte pieces
- Encode via greedy longest-match or BPE merges
- `PhalanxError::Tokenizer` nesting
- Docs: `docs/tokenizer.md`

### Decision log

#### 1. Hand-rolled encode/decode (no HF `tokenizers`)

**Pros:** Teaches the data path; zero new deps; uses GGUF tables directly.
**Cons:** Not guaranteed bit-identical to every HF export.
**Reason:** Educational runtime first; golden tests can tighten later.

#### 2. Default decode skips specials + control

**Pros:** Display text matches what users expect from chat UIs.
**Cons:** Callers debugging ids must opt into raw decode.
**Reason:** `DecodeOptions` exposes both behaviours.

#### 3. Default encode prepends BOS

**Pros:** Matches common Llama prefill behaviour.
**Cons:** Some models omit BOS — disable via `EncodeOptions`.
**Reason:** Safe default for decoder-only chat checkpoints.

### Testing strategy (Phase 4)

| Layer | Location | Covers |
|---|---|---|
| Unit | `tokenizer::*` | from_gguf, greedy/BPE round-trip, missing keys |
| Integration | `tests/smoke.rs` | public encode/decode re-exports |

### Follow-ups deferred

- Chat template / Jinja (`tokenizer.chat_template`) → CLI/runtime phases
- Golden parity vs llama.cpp on real GGUF → later phases

### References used this phase

- [GGUF tokenizer metadata](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- SentencePiece (Kudo & Richardson, 2018)
- GPT-2 BPE (Radford et al.)

---

## Phase 5 — Weight loading

### Scope delivered

- `weights` module: `WeightSet`, `WeightStorage`, `WeightTensor`, `QuantMeta`
- Read-only `memmap2` mapping (sole `unsafe` island)
- Quant block metadata for dense + legacy Q + K-quants
- Payload bounds validation for every tensor at open
- Materialize `f32` / `f16` → `tensor::Tensor`
- `PhalanxError::Weights` nesting
- Docs: `docs/weights.md`

### Decision log

#### 1. Crate lint `unsafe_code = "deny"` (was `forbid`)

**Pros:** Still defaults to no unsafe; reviewed modules can opt in.
**Cons:** A careless `#[allow]` could slip in.
**Reason:** `forbid` cannot be overridden; mmap requires one `unsafe` call.

#### 2. Map the whole file, not only `tensor_data`

**Pros:** Simple absolute offsets; parse from the same bytes.
**Cons:** Slightly larger map than a data-only window.
**Reason:** Clarity over micro-optimization.

#### 3. Defer block dequant kernels

**Pros:** Phase stays focused; avoids half-baked Q4_K code.
**Cons:** Can't run quantized matmul yet.
**Reason:** Dequant belongs next to the kernels that consume it (Phase 7+).

#### 4. IQ / ternary / MX types unsupported for sizing

**Pros:** No wrong `type_size` guesses.
**Cons:** Those GGUF files won't open until sizes are verified.
**Reason:** Prefer loud `UnsupportedType` over silent corruption.

### Testing strategy (Phase 5)

| Layer | Location | Covers |
|---|---|---|
| Unit | `weights::*` | quant sizes, f32 fixture, mmap tmpfile, truncated reject |
| Integration | `tests/smoke.rs` | public `QuantMeta` export |

### Follow-ups deferred

- Q4_0 / Q4_K / Q8_0 dequant → layer kernels
- CLI `inspect` → Phase 16

### References used this phase

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp / ggml quant blocks](https://github.com/ggml-org/llama.cpp)
- [memmap2](https://docs.rs/memmap2)

---

## Phase 6 — Model configuration

### Scope delivered

- `model` module: `Architecture`, `ModelConfig`, attention / RoPE sub-configs
- Parse Llama `{arch}.*` hyperparameters from GGUF metadata
- Structural validation (GQA divisibility, head×dim = embd, RoPE parity, …)
- Defaults: `head_count_kv → head_count`, `rope.freq_base → 10000`
- Legacy `rope.scale` → linear `RopeScaling`
- `PhalanxError::Model` nesting
- Docs: `docs/model.md`

### Decision log

#### 1. Llama-only architecture enum

**Pros:** Honest scope; validation tuned to Llama invariants.
**Cons:** `qwen2` / others fail with `UnsupportedArchitecture`.
**Reason:** Phase 6 is “Llama architecture”; multi-arch lands when kernels do.

#### 2. Accept u32 **or** u64 counts

**Pros:** Matches real GGUF writers / GGUF spec (`uint64` counts).
**Cons:** Slightly wider parse path.
**Reason:** Rejecting u64 would break valid files.

#### 3. Config does not bind weights yet

**Pros:** Clear boundary; Phase 7 owns embedding + named tensors.
**Cons:** Two-step load for callers (`ModelConfig` + `WeightSet`).
**Reason:** Avoid half-wired layer graphs before embeddings exist.

### Testing strategy (Phase 6)

| Layer | Location | Covers |
|---|---|---|
| Unit | `model::*` | Llama 7B-style, GQA, u64 counts, bad GQA, legacy rope.scale |
| Integration | `tests/smoke.rs` | nested `ModelError` re-export |

### Follow-ups deferred

- Qwen2 / Phi / MoE architecture variants
- Cross-check `vocab_size` vs tokenizer length → when both load together

### References used this phase

- [LLaMA](https://arxiv.org/abs/2302.13971)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [RoFormer](https://arxiv.org/abs/2104.09864)
- llama.cpp `gguf-py/gguf/constants.py`

---

## Phase 7 — Embedding layer

### Scope delivered

- `layers` module: `EmbeddingTable`, `LayersError`, `TOKEN_EMBD_WEIGHT`
- Load `token_embd.weight`, validate against `ModelConfig`
- Reinterpret ggml `[n_embd, n_vocab]` bytes as row-major `[vocab, embd]`
- `forward` / `forward_one` gather
- Squeeze trailing unitary dims
- Docs: `docs/embeddings.md`

### Decision log

#### 1. New `layers` module (not under `model`)

**Pros:** Keeps hparams separate from kernels; RoPE/attn/FFN land nearby.
**Cons:** Extra top-level module.
**Reason:** Phase 8–11 are all layer kernels.

#### 2. Reinterpret shape instead of transpose-copy

**Pros:** Zero-copy on multi-GB embedding matrices.
**Cons:** Easy to get wrong without the ggml-order comment.
**Reason:** Educational runtime should teach the layout, not hide a silent copy.

#### 3. Dense-only embeddings in Phase 7

**Pros:** Uses existing `to_f32_tensor`; scope stays focused.
**Cons:** Quantized GGUF embeddings still fail materialize.
**Reason:** Dequant belongs with the first kernel that needs blocks at scale;
gather correctness is the Phase 7 deliverable.

### Testing strategy (Phase 7)

| Layer | Location | Covers |
|---|---|---|
| Unit | `layers::embedding` | GGUF fixture gather, trailing ones, OOR, missing weight |
| Integration | `tests/smoke.rs` | public gather + nested `LayersError` |

### Follow-ups deferred

- Quantized embedding dequant
- Tied `output.weight` / input embeddings

### References used this phase

- [LLaMA](https://arxiv.org/abs/2302.13971)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- llama.cpp `TOKEN_EMBD` naming

---

## Phase 8 — Rotary embeddings

### Scope delivered

- `layers::Rope` with cos/sin cache from `ModelConfig`
- Adjacent-pair Llama / RoFormer rotation on `[seq, head_dim]` and `[seq, heads, head_dim]`
- Partial rotary dims; linear position scaling
- `LayersError::{InvalidActivationShape, RopePositionOutOfRange}`
- Docs: `docs/rope.md`

### Decision log

#### 1. Precompute tables to `context_length`

**Pros:** Decode is a gather + mul/add; tables are inspectable.
**Cons:** Memory grows with context.
**Reason:** Educational clarity + typical inference pattern (llama.cpp-style).

#### 2. Adjacent pairs (not GPT-NeoX half-split)

**Pros:** Matches Llama / RoFormer reference.
**Cons:** Some HF ports use NeoX layout — document the choice.
**Reason:** GGUF Llama checkpoints expect this pairing.

#### 3. Reject YaRN/NTK instead of no-op

**Pros:** No silent long-context bugs.
**Cons:** Those GGUF files error until Phase later.
**Reason:** Prefer loud failure over wrong angles.

### Testing strategy (Phase 8)

| Layer | Location | Covers |
|---|---|---|
| Unit | `layers::rope` | pos0 identity, L2 norm, partial, linear scale, OOR, yarn reject |
| Integration | `tests/smoke.rs` | public norm-preserving rotate |

### Follow-ups deferred

- YaRN / NTK / sectioned RoPE
- Wire into attention (Phase 11)
- Complex-view / SIMD rotate kernels (Phase 17)

### References used this phase

- [RoFormer](https://arxiv.org/abs/2104.09864)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- llama.cpp RoPE + GGUF `rope.*` keys

---

## Phase 9 — RMSNorm

### Delivered

- `layers::RmsNorm` with Spec formula `γ ⊙ x / RMS(x)`
- `eps` from `ModelConfig::rms_norm_eps`
- GGUF γ helpers: `attn_norm_weight_name`, `ffn_norm_weight_name`, `OUTPUT_NORM_WEIGHT`
- Cross-impl binary `src/bin/validate_rmsnorm.rs`
- Docs: `docs/rmsnorm.md`

### Decision log

#### 1. No mean centering (RMS only)

**Pros:** Matches Odyssey Spec / LLaMA; cheaper than LayerNorm.
**Cons:** Diverges from original Transformer post-norm stacks.
**Reason:** Spec compliance is non-negotiable (Rule 6).

#### 2. Float32 sum-of-squares in the kernel

**Pros:** Matches Odyssey `normalization.rms` accumulation path.
**Cons:** Slightly more ops than a pure fp16 path.
**Reason:** Cross-impl parity at `1e-6` before chasing speed.

### Testing strategy (Phase 9)

| Layer | Location | Covers |
|---|---|---|
| Unit | `layers::rmsnorm` | unit RMS, γ scale, shape, eps reject |
| Integration | `tests/smoke.rs` | public unit-RMS check |
| Parity | `validate_rmsnorm` + Odyssey script | max/mean abs error |

### Follow-ups deferred

- Wire into decoder pre-norm residuals (Phase 13)
- SIMD / fused norm kernels (Phase 17)

### References used this phase

- [RMSNorm](https://arxiv.org/abs/1910.07467)
- Odyssey Spec `rmsnorm.md`
