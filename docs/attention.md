# Attention (GQA / MHA)

Causal multi-head / grouped-query self-attention for residual-stream activations.

**Spec:** [Odyssey `spec/attention.md`](../../odyssey/spec/attention.md)  
**Phase:** 11  
**Cross-check:** Odyssey `scripts/validate_attention.py` ↔ `cargo run --bin validate_attention`

---

## Formula

\[
Q = x W_Q^\top,\quad K = x W_K^\top,\quad V = x W_V^\top
\]

\[
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
\]

Optional [`Rope`](rope.md) rotates Q/K after the head reshape (Spec requirement).

GQA: each KV head serves `H / H_kv` query heads.

## API

```rust
use phalanx::{Attention, Rope, Tensor};

let attn = Attention::from_tensors(w_q, w_k, w_v, w_o, num_heads, num_kv_heads, head_dim)?;
let y = attn.forward(&x, Some(&rope), /*position_offset*/ 0)?; // last dim = hidden
```

GGUF load:

```rust
let attn = Attention::from_weights(&weights, /*layer*/ 0, &config)?;
```

## Weight shapes

| Tensor | Shape |
| --- | --- |
| `w_q` | `[H·d, D]` |
| `w_k`, `w_v` | `[H_kv·d, D]` |
| `w_o` | `[D, H·d]` |
| Activations | `(B,S,D)` or `(S,D)` → same |

No projection biases.

## Validation

Default abs tolerance **`1e-3`** (GEMM accum order; mean ≪ `1e-6`), matching SwiGLU.
