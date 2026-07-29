//! Decoder layer kernels and weight-bound tables.
//!
//! Phase 7 introduced [`EmbeddingTable`]. Phase 8 adds [`Rope`]. Phase 9
//! adds [`RmsNorm`] ([RMSNorm](https://arxiv.org/abs/1910.07467)).
//!
//! # Module map
//!
//! - [`embedding`] — [`EmbeddingTable`] gather from `token_embd.weight`
//! - [`rope`] — rotary positional embeddings for Q/K
//! - [`rmsnorm`] — Llama-style `RMSNorm` (γ ⊙ x / RMS(x))
//! - [`error`] — [`LayersError`]

mod embedding;
mod error;
mod rmsnorm;
mod rope;

pub use embedding::{EmbeddingTable, TOKEN_EMBD_WEIGHT};
pub use error::LayersError;
pub use rmsnorm::{
    ATTN_NORM_WEIGHT_PREFIX, ATTN_NORM_WEIGHT_SUFFIX, FFN_NORM_WEIGHT_SUFFIX, OUTPUT_NORM_WEIGHT,
    RmsNorm, attn_norm_weight_name, ffn_norm_weight_name,
};
pub use rope::Rope;
