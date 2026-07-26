//! Vocabulary loading and tokenization for GGUF models.
//!
//! # Why this sits above `gguf`
//!
//! GGUF stores tokenizer tables as metadata arrays (`tokenizer.ggml.*`).
//! Phase 3 only *parses* those values; this module turns them into an encode /
//! decode engine the runtime can use for prompts and generated text.
//!
//! # Scope (Phase 4)
//!
//! | Capability | Status |
//! |---|---|
//! | Load vocab / scores / types / merges from [`crate::GgufFile`] | ✅ |
//! | Special token ids (bos/eos/unk/sep/pad) | ✅ |
//! | Decode ids → text (`▁` + `<0xXX>` rules) | ✅ |
//! | Encode text → ids (greedy or BPE merges) | ✅ |
//! | Full HF `SentencePiece` parity | deferred (golden tests later) |
//!
//! Spec keys: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

mod decode;
mod encode;
mod engine;
mod error;
mod keys;
mod special;
mod vocab;

pub use decode::DecodeOptions;
pub use encode::MergeRule;
pub use engine::{EncodeOptions, Tokenizer, TokenizerModel};
pub use error::TokenizerError;
pub use special::SpecialTokens;
pub use vocab::{TokenType, Vocabulary};
