//! Standardized GGUF tokenizer metadata keys.
//!
//! Source: [GGUF specification — Tokenizer](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

/// Tokenizer family string (`llama`, `gpt2`, …).
pub const MODEL: &str = "tokenizer.ggml.model";
/// Vocabulary pieces indexed by token id.
pub const TOKENS: &str = "tokenizer.ggml.tokens";
/// Optional per-token scores (`SentencePiece`).
pub const SCORES: &str = "tokenizer.ggml.scores";
/// Optional per-token type tags.
pub const TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
/// Optional BPE merge rules (`"left right"`).
pub const MERGES: &str = "tokenizer.ggml.merges";
/// Optional post-training added tokens.
pub const ADDED_TOKENS: &str = "tokenizer.ggml.added_tokens";

/// Beginning-of-sequence token id.
pub const BOS_TOKEN_ID: &str = "tokenizer.ggml.bos_token_id";
/// End-of-sequence token id.
pub const EOS_TOKEN_ID: &str = "tokenizer.ggml.eos_token_id";
/// Unknown-token id.
pub const UNKNOWN_TOKEN_ID: &str = "tokenizer.ggml.unknown_token_id";
/// Separator token id.
pub const SEPARATOR_TOKEN_ID: &str = "tokenizer.ggml.separator_token_id";
/// Padding token id.
pub const PADDING_TOKEN_ID: &str = "tokenizer.ggml.padding_token_id";
