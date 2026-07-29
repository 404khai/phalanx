# Architecture Mapping — Odyssey ↔ Phalanx

Crosswalk between Odyssey Specification components and Phalanx Runtime modules.

Normative Spec: [`odyssey/spec/`](../../odyssey/spec/README.md) `1.0.0`.

---

## System Mapping

```mermaid
flowchart TB
    subgraph odyssey [Odyssey Spec / Train]
        SArch[architecture.md]
        STok[tokenizer.md]
        SEmb[tok_embeddings]
        SRope[rope.md]
        SNorm[rmsnorm.md]
        SAttn[attention.md]
        SFfn[feedforward.md]
        SKv[kv_cache.md]
        SSamp[sampling.md]
    end

    subgraph phalanx [Phalanx Runtime]
        MCfg[model::ModelConfig]
        TTok[tokenizer::Tokenizer]
        LEmb[layers::EmbeddingTable]
        LRope[layers::Rope]
        LNorm[RMSNorm future]
        LAttn[Attention future]
        LFfn[FFN future]
        LKv[KV cache future]
        LSamp[Sampler future]
    end

    SArch --> MCfg
    STok --> TTok
    SEmb --> LEmb
    SRope --> LRope
    SNorm --> LNorm
    SAttn --> LAttn
    SFfn --> LFfn
    SKv --> LKv
    SSamp --> LSamp
```

---

## Component Table

| Odyssey Spec | Logical weight / concept | GGUF | Phalanx today |
| --- | --- | --- | --- |
| Architecture / config | hparams | `llama.*` keys | `ModelConfig::from_gguf` |
| Tokenizer | `odyssey-bpe` | `tokenizer.ggml.*` | GGUF tokenizer only |
| Embedding | `tok_embeddings.weight` | `token_embd.weight` | `EmbeddingTable` ✓ |
| RoPE | positions on Q/K | `llama.rope.*` | `Rope` ✓ |
| RMSNorm | `*.attention_norm` / `ffn_norm` / `norm` | `blk.*.attn_norm` / `ffn_norm` / `output_norm` | `RmsNorm` ✓ |
| SwiGLU | `w1/w3/w2` | `blk.*.ffn_gate/up/down` | `SwiGlu` ✓ |
| Attention | `wq/wk/wv/wo` | `blk.*.attn_*` | Planned |
| SwiGLU | `w1/w3/w2` | `ffn_gate/up/down` | Planned |
| KV cache | runtime state | — | Planned |
| LM head | `output.weight` | `output.weight` | Planned |
| Sampler | — | — | Planned |

---

## Forward Pass Mapping

| Spec stage | Phalanx execution |
| --- | --- |
| Tokenizer encode | `Tokenizer::encode` |
| Embedding | `EmbeddingTable::forward` |
| RoPE | `Rope::forward` on Q/K |
| RMSNorm | `RmsNorm::forward` on residual stream |
| SwiGLU | `SwiGlu::forward` |
| Attn / block residuals | Not wired |
| Logits / sample | Not wired |

---

## Weight Loading Pipeline

```mermaid
flowchart LR
    GGUF[GGUF file]
    WS[WeightSet mmap]
    Cfg[ModelConfig]
    Emb[EmbeddingTable]
    GGUF --> WS
    GGUF --> Cfg
    WS --> Emb
    Cfg --> Emb
```

---

## Gaps to Close (ordered)

1. Require / read `odyssey.spec.version`
2. Native load of `odyssey-bpe` artifacts (parity with Python)
3. RMSNorm → SwiGLU → Attention → KV → Decoder → Sampling
4. Golden parity tests against Odyssey tensors

---

## References

- [spec-compliance.md](spec-compliance.md)
- [compatibility.md](compatibility.md)
- [embeddings.md](embeddings.md), [rope.md](rope.md), [model.md](model.md)
