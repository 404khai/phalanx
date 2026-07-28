//! Typed failures for transformer layer kernels.
//!
//! Nested under [`crate::PhalanxError::Layers`] so callers can match layer
//! problems without scraping display strings.

use thiserror::Error;

/// Errors from loading or executing decoder layers.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LayersError {
    /// Required weight tensor was not found in the GGUF directory.
    #[error("layer weight '{name}' not found")]
    MissingWeight {
        /// Expected tensor name, e.g. `token_embd.weight`.
        name: &'static str,
    },

    /// On-disk tensor rank / dims are incompatible with the layer.
    #[error("invalid shape for weight '{name}': {reason}")]
    InvalidWeightShape {
        /// Tensor name.
        name: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// Weight dims disagree with [`crate::model::ModelConfig`].
    #[error("weight '{name}' does not match model config: {reason}")]
    ConfigMismatch {
        /// Tensor name.
        name: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// Token id is outside the embedding table.
    #[error("token id {id} out of range for vocabulary size {vocab_size}")]
    TokenOutOfRange {
        /// Requested id.
        id: u32,
        /// Rows in the embedding table.
        vocab_size: usize,
    },

    /// Activation tensor has the wrong rank or sizes for a kernel.
    #[error("invalid activation shape for {op}: {reason}")]
    InvalidActivationShape {
        /// Kernel name (`rope`, …).
        op: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// Absolute position exceeds the precomputed `RoPE` cache.
    #[error("RoPE position {position} exceeds cache length {max_position}")]
    RopePositionOutOfRange {
        /// Requested absolute position.
        position: usize,
        /// Cached positions (`0 .. max_position`).
        max_position: usize,
    },
}
