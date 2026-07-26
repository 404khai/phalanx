# Tokenizer — Educational Notes (Phase 4)

Phalanx loads tokenizer tables from GGUF metadata and exposes encode / decode
without depending on Hugging Face `tokenizers` or SentencePiece C++.

## Why vocabulary lives in GGUF

| Approach | Pros | Cons |
|---|---|---|
| **Embedded `tokenizer.ggml.*` (chosen)** | Single-file deploy; ids match training | Large metadata arrays |
| Sidecar `tokenizer.json` | Familiar HF layout | Easy to mismatch with weights |
| External crate only | Fast to ship | Opaque; weaker teaching value |

## Metadata keys

| Key | Role |
|---|---|
| `tokenizer.ggml.model` | Family: `llama`, `gpt2`, `replit`, `rwkv` |
| `tokenizer.ggml.tokens` | Piece strings indexed by token id |
| `tokenizer.ggml.scores` | Optional SentencePiece scores |
| `tokenizer.ggml.token_type` | normal / control / byte / … |
| `tokenizer.ggml.merges` | Optional BPE rules (`"left right"`) |
| `tokenizer.ggml.*_token_id` | bos / eos / unk / sep / pad |

Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

## Decode rules

1. Optionally skip special + control ids (default on).
2. Expand `<0xNN>` byte pieces to raw bytes.
3. Replace SentencePiece `▁` (U+2581) with ASCII space.
4. UTF-8-decode the concatenated bytes.

## Encode rules (Phase 4)

- **With merges** (typical `gpt2`): iterative BPE using merge rank order.
- **Without merges** (typical `llama`): greedy longest-match after mapping
  spaces to `▁`, with `<0xXX>` byte fallback when present.

This is an educational reference path — golden parity tests against llama.cpp
can tighten edge cases once real checkpoints are exercised (Phase 5+).

## API sketch

```rust
use phalanx::{EncodeOptions, GgufFile, Tokenizer};

let gguf = GgufFile::from_path("model.gguf")?;
let tok = Tokenizer::from_gguf(&gguf)?;
let ids = tok.encode("Hello", EncodeOptions::default())?;
let text = tok.decode(&ids)?;
```
