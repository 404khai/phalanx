//! Typed failures for model architecture / hyperparameter loading.
//!
//! Nested under [`crate::PhalanxError::Model`] so callers can match config
//! problems without scraping display strings.

use thiserror::Error;

/// Errors from parsing or validating a transformer [`super::ModelConfig`].
#[derive(Debug, Error, PartialEq)]
pub enum ModelError {
    /// `general.architecture` is missing from GGUF metadata.
    #[error("missing general.architecture metadata key")]
    MissingArchitecture,

    /// Architecture string is known but not yet supported by this runtime.
    #[error("unsupported model architecture '{architecture}'")]
    UnsupportedArchitecture {
        /// Value of `general.architecture`.
        architecture: String,
    },

    /// Required `{arch}.*` hyperparameter key is absent.
    #[error("missing model metadata key '{key}'")]
    MissingKey {
        /// Full metadata key, e.g. `llama.block_count`.
        key: String,
    },

    /// Metadata value has the wrong type for a known key.
    #[error("invalid type for model key '{key}': {reason}")]
    InvalidType {
        /// Full metadata key being read.
        key: String,
        /// Human-readable explanation.
        reason: String,
    },

    /// A hyperparameter is present but fails structural validation.
    ///
    /// Examples: zero layer count, head count that does not divide embedding
    /// length, or `GQA` where `head_count` is not a multiple of `head_count_kv`.
    #[error("invalid model configuration: {reason}")]
    InvalidConfig {
        /// Validation failure explanation.
        reason: String,
    },
}

impl ModelError {
    /// Convenience builder for [`ModelError::InvalidConfig`].
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::InvalidConfig {
            reason: reason.into(),
        }
    }
}
