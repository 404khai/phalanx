# SwiGLU Feed-Forward

Llama-style gated FFN for residual-stream activations.

**Spec:** [Odyssey `spec/feedforward.md`](../../odyssey/spec/feedforward.md)  
**Phase:** 10  
**Cross-check:** Odyssey `scripts/validate_swiglu.py` ↔ `cargo run --bin validate_swiglu`

---

## Formula

\[
\mathrm{FFN}(x)=\bigl(\mathrm{SiLU}(x W_1^\top)\odot(x W_3^\top)\bigr)W_2^\top
\]

## API

```rust
use phalanx::{SwiGlu, Tensor};

let ffn = SwiGlu::from_tensors(w_gate, w_up, w_down)?;
let y = ffn.forward(&x)?; // last dim = hidden
```

GGUF load:

```rust
let ffn = SwiGlu::from_weights(&weights, /*layer*/ 0, &config)?;
```

## Weight shapes

| Tensor | Shape |
| --- | --- |
| `w_gate` / `w_up` | `[I, D]` |
| `w_down` | `[D, I]` |
| Activations | `(..., D)` → `(..., D)` |
