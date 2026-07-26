//! GGUF metadata key suffixes for transformer hyperparameters.
//!
//! GGUF stores architecture-specific keys as `{arch}.{suffix}` where `arch`
//! comes from `general.architecture` (e.g. `llama.block_count`).
//!
//! Source: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! and llama.cpp `gguf-py/gguf/constants.py`.

/// Build `{arch}.{suffix}` for a metadata lookup.
#[must_use]
pub fn keyed(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

/// Number of transformer blocks (`n_layer`).
pub const BLOCK_COUNT: &str = "block_count";
/// Maximum context length the model was trained / converted for.
pub const CONTEXT_LENGTH: &str = "context_length";
/// Hidden / embedding dimension (`n_embd`).
pub const EMBEDDING_LENGTH: &str = "embedding_length";
/// Intermediate `FFN` width (`n_ff` / `SwiGLU` intermediate size).
pub const FEED_FORWARD_LENGTH: &str = "feed_forward_length";
/// Optional vocabulary size (also implied by tokenizer tables).
pub const VOCAB_SIZE: &str = "vocab_size";

/// Attention head count (`n_head`).
pub const ATTENTION_HEAD_COUNT: &str = "attention.head_count";
/// `KV` head count for `GQA` (`n_head_kv`); defaults to head count when absent.
pub const ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";
/// Optional per-head key dimension override.
pub const ATTENTION_KEY_LENGTH: &str = "attention.key_length";
/// Optional per-head value dimension override.
pub const ATTENTION_VALUE_LENGTH: &str = "attention.value_length";
/// `RMSNorm` epsilon.
pub const ATTENTION_LAYER_NORM_RMS_EPSILON: &str = "attention.layer_norm_rms_epsilon";

/// Rotary embedding dimension count.
pub const ROPE_DIMENSION_COUNT: &str = "rope.dimension_count";
/// `RoPE` base frequency θ (`rope_theta`).
pub const ROPE_FREQ_BASE: &str = "rope.freq_base";
/// Optional `RoPE` scaling algorithm name (`linear`, `yarn`, …).
pub const ROPE_SCALING_TYPE: &str = "rope.scaling.type";
/// Optional `RoPE` scaling factor.
pub const ROPE_SCALING_FACTOR: &str = "rope.scaling.factor";
/// Legacy linear scale key used by some older Llama GGUF exports.
pub const ROPE_SCALE: &str = "rope.scale";
