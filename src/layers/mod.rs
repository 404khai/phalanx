//! Decoder layer kernels and weight-bound tables.
//!
//! Phase 7 introduces the token [`EmbeddingTable`]. Later phases add
//! [`RoPE`](https://arxiv.org/abs/2104.09864), `RMSNorm`, attention, and `FFN`
//! beside it under this module.
//!
//! # Module map
//!
//! - [`embedding`] — [`EmbeddingTable`] gather from `token_embd.weight`
//! - [`error`] — [`LayersError`]

mod embedding;
mod error;

pub use embedding::{EmbeddingTable, TOKEN_EMBD_WEIGHT};
pub use error::LayersError;
