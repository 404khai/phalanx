# Rotary embeddings (`RoPE`) — Phase 8

Phalanx applies Llama-style rotary positional embeddings via
[`Rope`](../src/layers/rope.rs).

## Role in the forward pass

```text
Q, K  [seq, heads, head_dim]
        │
        ▼
   Rope::forward(…, position_offset)
        │
        ▼
Q′, K′  (V untouched)
        │
        ▼
   (Phase 11) attention
```

Prefill uses `position_offset = 0` with `seq = prompt_len`. Decode uses
`position_offset = past_len` with `seq = 1`.

## Math (adjacent pairs)

For rotary width `d` (even) and base θ (`rope.freq_base`, default 10_000):

```text
θ_i = θ ^ (-2i / d)          i = 0 .. d/2 - 1

[x'₀]   [ cos(mθᵢ)  -sin(mθᵢ) ] [x₀]
[x'₁] = [ sin(mθᵢ)   cos(mθᵢ) ] [x₁]
```

with `(x₀, x₁) = (x_{2i}, x_{2i+1})` at absolute position `m`.

Rotation is orthogonal → each head vector keeps its L2 norm (useful test oracle).

## Partial RoPE

If `rope.dimension_count < head_dim`, only the first `dimension_count` features
rotate; the tail is copied. Llama usually sets them equal.

## Linear scaling

GGUF `rope.scaling.type = linear` (or legacy `rope.scale`) divides positions:

`m' = m / factor` before looking up angles. YaRN / NTK are rejected until
implemented.

## API

```rust
use phalanx::{ModelConfig, Rope, WeightSet};

fn build_rope(path: &str) -> phalanx::Result<Rope> {
    let weights = WeightSet::open_mmap(path)?;
    let config = ModelConfig::from_gguf(weights.gguf())?;
    Rope::from_config(&config)
}
```

Accepted activation shapes: `[seq, head_dim]` or `[seq, n_heads, head_dim]`.

## Scope

| Feature | Status |
|---|---|
| Cos/sin cache from `ModelConfig` | ✅ |
| Adjacent-pair Llama rotation | ✅ |
| Partial rotary dims | ✅ |
| Linear position scale | ✅ |
| YaRN / NTK / sectioned RoPE | ❌ later |
| Apply inside attention module | ✓ Phase 11 (`Attention::forward`) |

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- llama.cpp RoPE kernels / GGUF `rope.*` keys
