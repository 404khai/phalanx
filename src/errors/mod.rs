//! Typed error surface for the Phalanx library API.
//!
//! Library code returns [`PhalanxError`] so callers can match on failure
//! kinds (I/O vs configuration vs tensor vs future GGUF/model errors). The
//! CLI binary wraps these with [`anyhow`] for ergonomic context chaining at
//! the edge.
//!
//! # Design tradeoff
//!
//! | Approach | Pros | Cons |
//! |---|---|---|
//! | `thiserror` enums (chosen) | Stable matchable API, zero-cost Display | Variants grow as subsystems land |
//! | `anyhow` everywhere | Fast to write | Callers cannot match typed failures |
//! | `snafu` | Rich context macros | Extra dependency / learning cost |
//!
//! We use `thiserror` in the library and `anyhow` only in `main` / examples.

use thiserror::Error;

use crate::gguf::GgufError;
use crate::tensor::TensorError;
use crate::tokenizer::TokenizerError;
use crate::weights::WeightsError;

/// Convenient alias for fallible library operations.
pub type Result<T> = std::result::Result<T, PhalanxError>;

/// Top-level error type returned by Phalanx library APIs.
///
/// Subsystem errors nest behind dedicated variants (e.g. [`PhalanxError::Tensor`])
/// so callers can match broadly or down to the detailed cause.
#[derive(Debug, Error)]
pub enum PhalanxError {
    /// Filesystem or other I/O failure (model paths, mmap, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid runtime or model configuration supplied by the caller.
    #[error("configuration error: {0}")]
    Config(String),

    /// Shape / layout / kernel failure from the tensor subsystem.
    #[error(transparent)]
    Tensor(#[from] TensorError),

    /// GGUF container parse / validation failure.
    #[error(transparent)]
    Gguf(#[from] GgufError),

    /// Vocabulary / encode / decode failure.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),

    /// Weight mmap / bounds / materialization failure.
    #[error(transparent)]
    Weights(#[from] WeightsError),

    /// Unexpected internal invariant violation.
    ///
    /// Prefer typed variants for expected failure modes; use this only when
    /// the failure is a programming bug or an unclassified edge case.
    #[error("internal error: {0}")]
    Internal(String),
}

impl PhalanxError {
    /// Build a [`PhalanxError::Config`] from anything displayable.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    /// Build a [`PhalanxError::Internal`] from anything displayable.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_display_includes_message() {
        let err = PhalanxError::config("missing model path");
        assert_eq!(err.to_string(), "configuration error: missing model path");
    }

    #[test]
    fn io_error_converts_via_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: PhalanxError = io_err.into();
        assert!(matches!(err, PhalanxError::Io(_)));
        assert!(err.to_string().contains("gone"));
    }
}
