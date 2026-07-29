# Shared Cross-Implementation Validation Suite

This directory is the **long-term compatibility suite** between:

| Project | Role | Language |
| --- | --- | --- |
| [Odyssey](../odyssey/) | Training framework (Spec source) | Python / PyTorch |
| [Phalanx Runtime](../runtime/) | Reference inference runtime | Rust |

Every mathematical component that both projects implement must PASS here before claiming Spec compliance (Odyssey Principle 8 · Phalanx Rule 6).

---

## Layout

```text
validation/
├── README.md
├── _common.py              # shared f32 I/O + compare helpers
├── test_embeddings.py      # planned (Phase parity pending dedicated binary)
├── test_rope.py            # RoPE — live
├── test_rmsnorm.py         # RMSNorm — live
├── test_attention.py       # stub until both sides implement attention
└── test_swiglu.py          # stub until both sides implement SwiGLU
```

---

## Contract

Each live validator:

1. Generates **deterministic** random inputs (fixed seed).
2. Runs the **Odyssey** implementation.
3. Runs the **Phalanx** implementation (via `cargo run --bin validate_*`).
4. Compares outputs within a configurable tolerance (default float32 **`1e-6`**).
5. Prints a small report: **max error**, **mean error**, **PASS/FAIL**.

Underlying drivers (kept as the source of truth for I/O manifests):

- Odyssey: `odyssey/scripts/validate_<component>.py`
- Phalanx: `runtime/src/bin/validate_<component>.rs`

These wrappers call those drivers so CI / humans have one entry point under `validation/`.

---

## Running

From the monorepo root (`phalanx/`):

```bash
# Live components
python validation/test_rmsnorm.py
python validation/test_rope.py

# Full suite (skips stubs with a clear message)
python validation/test_rmsnorm.py && python validation/test_rope.py
```

Optional knobs are forwarded to the Odyssey scripts (`--seed`, `--tolerance`, …).

---

## Status

| Component | Odyssey | Phalanx | Suite |
| --- | --- | --- | --- |
| Embedding | ✓ | ✓ | planned (`test_embeddings.py` placeholder) |
| RoPE | ✓ | ✓ | ✓ `test_rope.py` |
| RMSNorm | ✓ | ✓ | ✓ `test_rmsnorm.py` |
| Attention | — | — | stub |
| SwiGLU | — | — | stub |
| KV Cache / Decoder / Sampling | — | — | future |

---

## Adding a component

1. Implement Odyssey + Phalanx to Spec.
2. Add `odyssey/scripts/validate_<name>.py` + `runtime/src/bin/validate_<name>.rs`.
3. Add `validation/test_<name>.py` that shells out (or imports) the Odyssey driver.
4. Flip the status row above only after PASS.
