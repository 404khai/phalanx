# Model configuration — Phase 6

Phalanx loads transformer hyperparameters from GGUF metadata into a validated
[`ModelConfig`](../src/model/config.rs). Layer kernels never re-parse ad-hoc
keys; they ask the config for sizes (`embedding_length`, `head_dim`, GQA
groups, RoPE θ, RMSNorm ε).

## Why config is separate from weights

| Concern | Module |
|---|---|
| Container bytes / tensor directory | `gguf` |
| mmap + quant layouts | `weights` |
| Vocab / encode / decode | `tokenizer` |
| **Shapes & hparams** | **`model`** |

Weights tell you *what tensors exist*; config tells you *how they wire into a
Llama-style decoder*. Binding named tensors to layers is Phase 7+.

## GGUF key layout

Architecture comes from `general.architecture`. Hyperparameters use the
prefix `{arch}.…`:

| Key | Role |
|---|---|
| `llama.block_count` | Transformer layers (`n_layer`) |
| `llama.context_length` | Max sequence length |
| `llama.embedding_length` | Hidden size (`n_embd`) |
| `llama.feed_forward_length` | FFN width (`n_ff`) |
| `llama.attention.head_count` | Query heads |
| `llama.attention.head_count_kv` | KV heads (GQA); defaults to head count |
| `llama.attention.layer_norm_rms_epsilon` | RMSNorm ε |
| `llama.rope.dimension_count` | Rotary dims |
| `llama.rope.freq_base` | θ (default `10000` if omitted) |

Optional: `vocab_size`, `attention.key_length` / `value_length`, RoPE scaling
(`rope.scaling.type` / `factor`, or legacy `rope.scale`).

## Validation rules (selected)

- Positive sizes for layers, context, embd, FFN, heads
- `head_count % head_count_kv == 0` (GQA)
- `head_count * key_length == embedding_length`
- `rope.dimension_count` even and ≤ `key_length`
- Finite, positive RMSNorm ε and RoPE θ

## Scope

| Feature | Status |
|---|---|
| Llama (`general.architecture = "llama"`) | ✅ |
| GQA / MHA sizing helpers | ✅ |
| RoPE scaling metadata capture | ✅ (kernels later) |
| Qwen2 / Phi / MoE architectures | ❌ later |
| Named weight → layer binding | ✅ Phase 7 (`token_embd.weight`) |

## Example

```rust
use phalanx::{GgufFile, ModelConfig};

fn load_config(path: &str) -> phalanx::Result<ModelConfig> {
    let file = GgufFile::from_path(path)?;
    ModelConfig::from_gguf(&file)
}
```

## References

- [LLaMA](https://arxiv.org/abs/2302.13971)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)
- llama.cpp `gguf-py/gguf/constants.py`
