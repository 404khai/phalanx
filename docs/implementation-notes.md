# Implementation Notes — Phase 1

Notes capture decisions that are easy to lose in commit history. Prefer
updating this file over leaving undocumented tribal knowledge.

## Scope delivered

- Cargo package `phalanx` with library (`src/lib.rs`) and binary (`src/main.rs`)
- `errors` module: `PhalanxError`, `Result<T>`
- `utils::logging`: `LogConfig`, `init_logging`
- Tooling: `rustfmt.toml`, Clippy lints, `.cargo/config.toml` aliases
- Docs: README, architecture, this file, CHANGELOG, LICENSE
- Tests: unit tests in modules + `tests/smoke.rs`

## Decision log

### 1. Library errors = `thiserror`, CLI = `anyhow`

**Pros (chosen mix):** Embedders match on `PhalanxError`; CLI gets `Context`.
**Cons:** Two error styles to teach newcomers.
**Rejected alternative:** `anyhow` in the library — faster to write, worse API.

### 2. `tracing` for logging

**Pros:** Spans map cleanly to load / prefill / decode; wide ecosystem support.
**Cons:** Heavier than `env_logger`.
**Rejected alternative:** Defer logging until Phase 16 — we want lifecycle
visibility while building early phases.

### 3. Edition 2024 + `rust-version = "1.85"`

**Pros:** Matches AGENTS.md “latest stable”; modern language defaults.
**Cons:** Contributors on older toolchains must upgrade.
**Reason:** Greenfield project; no compatibility debt yet.

### 4. Forbid `unsafe_code` at the crate lint level

**Pros:** Forces explicit discussion before introducing UB risk.
**Cons:** Will need a scoped `allow` when mmap / SIMD lands.
**Reason:** Correctness culture > micro-optimization at foundation time.

### 5. Do not create empty domain folders

**Pros:** Tree reflects reality; no fake APIs.
**Cons:** Diverges briefly from the full preferred tree in AGENTS.md.
**Reason:** AGENTS.md says “only create folders when needed.”

### 6. Commit `Cargo.lock`

Phalanx ships a binary runtime. Lockfile reproducibility matters for CLI
users and CI. (Pure libraries often omit the lockfile; we are not pure.)

## Testing strategy (Phase 1)

| Layer | Location | Covers |
|---|---|---|
| Unit | `errors`, `logging`, `lib` | Display, conversions, constants |
| Integration | `tests/smoke.rs` | Public re-exports across crate boundary |

Logging’s success path is **not** asserted in unit tests: installing a global
subscriber twice fails, and parallel tests would race. Invalid filter parsing
is tested because it fails before `set_global_default`.

## Performance notes

None yet — no kernels. Phase 2 will introduce Criterion (or built-in benches)
alongside the tensor core.

## Follow-ups deferred

- CLI argument parsing (`clap`) → Phase 16 (or earlier if inspection tools need it)
- Workspace split → when compile time / deps justify it
- `PhalanxError` variants for GGUF / tokenizer / tensor → with those phases
- JSON / file logging layers → if operators need them

## References used this phase

- [thiserror](https://docs.rs/thiserror)
- [anyhow](https://docs.rs/anyhow)
- [tracing](https://docs.rs/tracing)
- [The Cargo Book — package layout](https://doc.rust-lang.org/cargo/guide/project-layout.html)
