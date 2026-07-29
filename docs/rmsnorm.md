# RMSNorm

Llama-style root-mean-square normalization for residual-stream activations.

**Spec:** [Odyssey `spec/rmsnorm.md`](../../odyssey/spec/rmsnorm.md)  
**Phase:** 9  
**Cross-check:** Odyssey `scripts/validate_rmsnorm.py` ↔ `cargo run --bin validate_rmsnorm`

---

## Role

```text
x  →  RMSNorm(γ, ε)  →  sub-layer
```

Used as `attn_norm`, `ffn_norm`, and final `output_norm` once the decoder lands.

## Mathematics

\[
\mathrm{RMS}(x)=\sqrt{\mathrm{mean}(x^2)+\varepsilon}
\qquad
y=\gamma\odot\frac{x}{\mathrm{RMS}(x)}
\]

No mean centering. No bias. LayerNorm here is **non-compliant**.

## API

```rust
use phalanx::{RmsNorm, Tensor};

let norm = RmsNorm::ones(/* hidden */ 768, /* eps */ 1e-6)?;
let y = norm.forward(&x)?; // shape preserved; last dim = 768
```

Load from GGUF:

```rust
let attn = RmsNorm::from_weights(&weights, &attn_norm_weight_name(0), &config)?;
let ffn  = RmsNorm::from_weights(&weights, &ffn_norm_weight_name(0), &config)?;
let out  = RmsNorm::from_weights(&weights, OUTPUT_NORM_WEIGHT, &config)?;
```

## Scope

| In | Out |
| --- | --- |
| γ length `embedding_length` | Same activation rank/shape |
| `eps = config.rms_norm_eps` | Float32 kernel (Phase 2 tensor) |
| Rank ≥ 1, last dim = `D` | |

## References

- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
- Odyssey Spec v1.0.0 `rmsnorm.md`
