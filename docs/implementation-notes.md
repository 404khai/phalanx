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
4. **Forbid `unsafe_code`** — force review before UB risk.
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
