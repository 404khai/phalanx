//! Decoder layer kernels and weight-bound tables.
//!
//! Phase 7 introduced [`EmbeddingTable`]. Phase 8 adds [`Rope`]
//! ([RoFormer](https://arxiv.org/abs/2104.09864)). Later phases add `RMSNorm`,
//! attention, and `FFN`.
//!
//! # Module map
//!
//! - [`embedding`] — [`EmbeddingTable`] gather from `token_embd.weight`
//! - [`rope`] — rotary positional embeddings for Q/K
//! - [`error`] — [`LayersError`]

mod embedding;
mod error;
mod rope;

pub use embedding::{EmbeddingTable, TOKEN_EMBD_WEIGHT};
pub use error::LayersError;
pub use rope::Rope;
