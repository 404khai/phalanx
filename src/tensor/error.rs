//! Typed failures for shape checks and tensor kernels.
//!
//! Kept separate from [`crate::PhalanxError`] so tensor call sites can match
//! layout problems without string parsing, while the crate root still exposes
//! a single error enum via `#[from]`.

use thiserror::Error;

/// Errors produced by tensor construction and operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    /// Shape failed validation (empty rank, etc.).
    #[error("invalid shape: {reason}")]
    InvalidShape {
        /// Human-readable explanation.
        reason: String,
    },

    /// Buffer length does not match `shape.numel()`.
    #[error("data length mismatch: shape {shape} needs {expected} elements, got {got}")]
    DataLengthMismatch {
        /// Shape that was requested.
        shape: String,
        /// `shape.numel()`.
        expected: usize,
        /// Caller-provided element count.
        got: usize,
    },

    /// Two operands disagree on shape where equality is required.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Expected shape display.
        expected: String,
        /// Actual shape display.
        got: String,
    },

    /// Operation requires a specific rank (e.g. matmul needs rank 2).
    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch {
        /// Required rank.
        expected: usize,
        /// Actual rank.
        got: usize,
    },

    /// Multi-index refers past a dimension.
    #[error("index {index} out of bounds for axis {axis} of size {dim}")]
    IndexOutOfBounds {
        /// Axis that was indexed.
        axis: usize,
        /// Requested index.
        index: usize,
        /// Size of that axis.
        dim: usize,
    },

    /// Matrix-multiply inner dimensions are incompatible.
    #[error("matmul incompatible shapes: {lhs} × {rhs} (inner dims must match)")]
    MatMulIncompatible {
        /// Left-hand shape display.
        lhs: String,
        /// Right-hand shape display.
        rhs: String,
    },
}
