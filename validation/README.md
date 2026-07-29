# Shared Cross-Implementation Validation Suite

This directory is the **long-term compatibility suite** between:

| Project | Role | Language |
| --- | --- | --- |
| [Odyssey](../odyssey/) | Training framework (Spec source) | Python / PyTorch |
| [Phalanx Runtime](../runtime/) | Reference inference runtime | Rust |

*(When this folder lives inside `runtime/validation/`, Phalanx is `..` and Odyssey is the sibling checkout — `_common.py` auto-detects both layouts.)*

Every mathematical component that both projects implement must PASS here before claiming Spec compliance (Odyssey Principle 8 · Phalanx Rule 6).

---

## Layout

```text
validation/
├── README.md
├── _common.py
├── test_embeddings.py      # placeholder
├── test_rope.py            # live
├── test_rmsnorm.py         # live
├── test_swiglu.py          # live
└── test_attention.py       # stub
```

---

## Contract

Each live validator:

1. Generates **deterministic** random inputs (fixed seed).
2. Runs the **Odyssey** implementation.
3. Runs the **Phalanx** implementation (via `cargo run --bin validate_*`).
4. Compares outputs within a configurable tolerance.
5. Prints **max error**, **mean error**, **PASS/FAIL**.

Default float32 tolerance: `1e-6` (SwiGLU documents `1e-3` for GEMM accum order).

---

## Running

```bash
python validation/test_rmsnorm.py
python validation/test_rope.py
python validation/test_swiglu.py
```

---

## Status

| Component | Odyssey | Phalanx | Suite |
| --- | --- | --- | --- |
| Embedding | ✓ | ✓ | planned (`test_embeddings.py` placeholder) |
| RoPE | ✓ | ✓ | ✓ `test_rope.py` |
| RMSNorm | ✓ | ✓ | ✓ `test_rmsnorm.py` |
| SwiGLU | ✓ | ✓ | ✓ `test_swiglu.py` |
| Attention | — | — | stub |
| KV Cache / Decoder / Sampling | — | — | future |

---

## Adding a component

1. Implement Odyssey + Phalanx to Spec.
2. Add `odyssey/scripts/validate_<name>.py` + `runtime/src/bin/validate_<name>.rs`.
3. Add `validation/test_<name>.py` that shells out to the Odyssey driver.
4. Flip the status row above only after PASS.
