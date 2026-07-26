# GGUF — Educational Notes (Phase 3)

GGUF (**G**PT-**G**enerated **U**nified **F**ormat) is the on-disk container
Phalanx targets for local LLM inference. This note explains *why* it exists and
what Phase 3 already parses.

## Why GGUF exists

| Problem with training checkpoints | GGUF answer |
|---|---|
| Multi-file sharded `state_dict` + config JSON | Single-file deployment |
| Python/`pickle` coupling | Language-agnostic binary |
| Quantization as an afterthought | `ggml_type` per tensor + metadata |
| Slow cold start | Designed for `mmap` of weight blobs |

It succeeds earlier GGML/GGMF/GGJT layouts. Authoritative spec:

<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

## File layout

```text
[ magic "GGUF" | version | tensor_count | metadata_kv_count ]
[ metadata key-value × metadata_kv_count ]
[ tensor info × tensor_count ]
[ padding to alignment ]
[ tensor_data … ]          ← not loaded in Phase 3
```

- **Endianness:** little-endian by default (Phase 3 assumption).
- **Alignment:** `general.alignment` metadata, else **32** bytes.
- **Tensor offsets:** relative to `tensor_data`, not file start.
  Absolute offset = `data_offset + tensor.offset`.

## What Phalanx parses today

| Section | Status |
|---|---|
| Magic / version | ✅ validate (`2` or `3`) |
| Metadata KV (all value types + nested arrays) | ✅ |
| Tensor info directory | ✅ |
| Weight bytes / mmap / quant meta | ✅ Phase 5 (`weights` module) |
| Block dequant → f32 | ❌ later (layer kernels) |
| Vocabulary / tokenizer | ✅ Phase 4 (`tokenizer` module) |
| Model hyperparameters | ✅ Phase 6 (`model` module) |

## Common metadata keys

- `general.architecture` — e.g. `llama`, `qwen2`
- `general.name` — display name
- `general.alignment` — data alignment override
- `{arch}.block_count`, `{arch}.embedding_length`, … — hyperparameters
- `tokenizer.ggml.*` — vocabulary (see [tokenizer.md](tokenizer.md))

## Safety limits

The parser caps string / array / count sizes (see `gguf::types::limits`) so a
hostile header cannot force unbounded allocations before validation completes.
