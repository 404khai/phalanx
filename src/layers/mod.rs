//! Decoder layer kernels and weight-bound tables.
//!
//! Phase 7–9: embedding, `RoPE`, `RMSNorm`. Phase 10 adds [`SwiGlu`].
//!
//! # Module map
//!
//! - [`embedding`] — [`EmbeddingTable`] gather from `token_embd.weight`
//! - [`rope`] — rotary positional embeddings for Q/K
//! - [`rmsnorm`] — Llama-style `RMSNorm` (γ ⊙ x / RMS(x))
//! - [`swiglu`] — Llama-style `SwiGLU` feed-forward
//! - [`error`] — [`LayersError`]

mod embedding;
mod error;
mod rmsnorm;
mod rope;
mod swiglu;

pub use embedding::{EmbeddingTable, TOKEN_EMBD_WEIGHT};
pub use error::LayersError;
pub use rmsnorm::{
    ATTN_NORM_WEIGHT_PREFIX, ATTN_NORM_WEIGHT_SUFFIX, FFN_NORM_WEIGHT_SUFFIX, OUTPUT_NORM_WEIGHT,
    RmsNorm, attn_norm_weight_name, ffn_norm_weight_name,
};
pub use rope::Rope;
pub use swiglu::{
    FFN_DOWN_WEIGHT_SUFFIX, FFN_GATE_WEIGHT_SUFFIX, FFN_UP_WEIGHT_SUFFIX, FFN_WEIGHT_PREFIX,
    SwiGlu, ffn_down_weight_name, ffn_gate_weight_name, ffn_up_weight_name,
};
