# Token embeddings — Phase 7

The first decoder stage maps token ids to dense activations via
[`EmbeddingTable`](../src/layers/embedding.rs).

## Role in the forward pass

```text
token ids ──► EmbeddingTable::forward ──► [seq, n_embd]
                                              │
                                              ▼
                                    (Phase 8) RoPE on Q/K …
```

Prefill gathers many ids at once; decode gathers one. Both share the same
row-major table.

## GGUF tensor

| Item | Value |
|---|---|
| Name | `token_embd.weight` |
| GGUF dims | `[n_embd, n_vocab]` (ggml order) |
| Runtime table | `[n_vocab, n_embd]` (row-major gather) |

ggml stores `ne[0]` (here `n_embd`) as the contiguous axis, so the on-disk
byte order already matches vocab-major rows. After `f32`/`f16` materialization
Phalanx **reinterprets** the buffer (no copy) before gather.

## Loading

```rust
use phalanx::{EmbeddingTable, ModelConfig, WeightSet};

fn load(path: &str) -> phalanx::Result<EmbeddingTable> {
    let weights = WeightSet::open_mmap(path)?;
    let config = ModelConfig::from_gguf(weights.gguf())?;
    EmbeddingTable::from_weights(&weights, &config)
}
```

Validation:

- Rank-2 after squeezing trailing `1` dims
- `n_embd` matches `config.embedding_length`
- Optional `config.vocab_size` matches `n_vocab` when present

## Scope

| Feature | Status |
|---|---|
| Dense `f32` / `f16` gather | ✅ |
| Config shape checks | ✅ |
| Quantized embedding dequant | ❌ (uses existing materialize path) |
| Tied output / input embeddings | ❌ later |
| Position embeddings (absolute) | ❌ Llama uses `RoPE` ([docs/rope.md](rope.md)) |

## References

- [LLaMA](https://arxiv.org/abs/2302.13971)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- llama.cpp `TOKEN_EMBD` → `token_embd.weight`
