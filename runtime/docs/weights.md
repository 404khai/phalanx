# Weights — Educational Notes (Phase 5)

Phase 5 connects the GGUF **directory** (Phase 3) to the **bytes on disk**.

## Why mmap

| Approach | Pros | Cons |
|---|---|---|
| **`mmap` whole file (chosen)** | No multi-GB anonymous copy; OS demand-paging | Needs a reviewed `unsafe` call into `memmap2` |
| `read()` into `Vec<u8>` | Pure safe Rust | Impractical for large checkpoints |
| Map only `tensor_data` | Slightly smaller map | More offset bookkeeping |

Phalanx keeps a single read-only map of the entire GGUF file. Tensor views are
slices `&map[absolute_offset .. absolute_offset + nbytes]`.

## Quantization metadata

Most GGUF weights are **block quantized**: a fixed number of elements share a
scale. The loader stores:

- `block_size` — elements per block (`32` for Q4_0/Q8_0, `256` for Q4_K, …)
- `type_size` — bytes per block (`18` for Q4_0, `144` for Q4_K, …)

```text
nbytes = (numel / block_size) * type_size
```

`numel` must be a multiple of `block_size` or the file is rejected.

Dense types (`f32`, `f16`, …) use `block_size = 1`.

## What materializes today

| On-disk type | `to_f32_tensor()` |
|---|---|
| `f32` | ✅ copy / reinterpret LE floats |
| `f16` | ✅ expand via `half` |
| Q4_0 / Q4_K / Q8_0 / … | ❌ `DequantNotImplemented` (Phase 7+) |

Views are still available for quantized tensors (`WeightTensor::data`) so
later kernels can dequant on the fly without a second file read.

## API sketch

```rust
use phalanx::WeightSet;

let weights = WeightSet::open_mmap("model.gguf")?;
let view = weights.tensor("token_embd.weight")?;
println!("{} bytes, quant={:?}", view.data.len(), view.quant);

// Dense only for now:
// let t = view.to_f32_tensor()?;
```

## Safety note

`weights::storage` is the only module allowed to contain `unsafe`, solely to
call `memmap2::MmapOptions::map`. Model files must not be truncated or
rewritten while a [`WeightSet`](../src/weights/set.rs) is alive.
