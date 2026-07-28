# Odyssey Spec Compliance

**Runtime role:** Reference inference implementation for Odyssey  
**Target specification:** [Odyssey Spec `1.0.0`](../../odyssey/spec/README.md)  
**Declared support:** `supported_odyssey_spec = ["1.0.0"]`

This matrix is the roadmap to full Odyssey compatibility. It does **not** authorize divergence from Spec shapes or names.

---

## Compliance Matrix

| Spec area | Status | Runtime location | Notes |
| --- | --- | --- | --- |
| Architecture metadata load | ✓ Partial | `model::ModelConfig` | Llama GGUF keys; Odyssey KV not yet required |
| Tensor shape invariants | ✓ Partial | `ModelConfig::validate` | Head/GQA/RoPE checks |
| Tokenizer API parity | ✗ | `tokenizer::Tokenizer` | GGUF path only; no `odyssey-bpe` dir loader yet |
| Weight naming (GGUF map) | ✓ Partial | `token_embd.weight` | Only embedding bound |
| Embedding gather | ✓ | `layers::EmbeddingTable` | Matches Spec `(V,D)` logical table |
| RoPE | ✓ | `layers::Rope` | θ, partial rotary, linear scaling |
| RMSNorm | ✗ | — | Phase 9 |
| Attention (causal / GQA) | ✗ | — | Phase 11 |
| SwiGLU FFN | ✗ | — | Phase 10 |
| Residual pre-norm block | ✗ | — | Phase 13 |
| Final norm + LM head | ✗ | — | Phase 13 |
| KV cache | ✗ | — | Phase 12 |
| Sampling | ✗ | — | Phase 14 |
| Full decoder forward | ✗ | — | Phase 13 |
| Spec version negotiation | ✗ | — | Read `odyssey.spec.version` KV |
| Numeric parity tests vs Odyssey | ✗ | — | After shared fixtures |

Legend: ✓ done · ✗ not done · Partial = present but not full Spec surface.

---

## Diagram

```mermaid
flowchart TD
    Spec[Odyssey Spec 1.0.0]
    Done[Done: Emb + RoPE + Config + GGUF IO]
    Next[Next: RMSNorm → FFN → Attn → KV → Decoder → Sample]
    Spec --> Done --> Next
```

---

## Non-Goals for This Document

- Implementing missing layers (tracked by runtime phases)
- Changing Odyssey Spec unilaterally

When a layer lands, flip its row to ✓ and link the module + doc.
