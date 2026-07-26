//! Typed failures for vocabulary loading and tokenization.
//!
//! Nested under [`crate::PhalanxError::Tokenizer`] so callers can match
//! tokenizer problems without scraping display strings.

use thiserror::Error;

/// Errors from loading a vocabulary or encoding / decoding text.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TokenizerError {
    /// Required GGUF metadata key is missing.
    #[error("missing tokenizer metadata key '{key}'")]
    MissingKey {
        /// Absent key, e.g. `tokenizer.ggml.tokens`.
        key: &'static str,
    },

    /// Metadata value has the wrong type for a known key.
    #[error("invalid type for tokenizer key '{key}': {reason}")]
    InvalidType {
        /// Metadata key being read.
        key: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// Parallel arrays disagree on length (tokens vs scores vs types).
    #[error("tokenizer array length mismatch for '{key}': expected {expected}, got {got}")]
    LengthMismatch {
        /// Which array failed the check.
        key: &'static str,
        /// Length implied by `tokenizer.ggml.tokens`.
        expected: usize,
        /// Actual length of this array.
        got: usize,
    },

    /// Token id is outside the loaded vocabulary.
    #[error("token id {id} out of range for vocabulary size {vocab_size}")]
    UnknownTokenId {
        /// Requested id.
        id: u32,
        /// `vocab.len()`.
        vocab_size: usize,
    },

    /// Encode could not map a span of text to any vocabulary piece.
    #[error("failed to encode text near {context:?}: no matching vocabulary piece")]
    EncodeFailure {
        /// Short snippet around the failure for debugging.
        context: String,
    },

    /// Special-token id points outside the vocabulary.
    #[error("special token '{name}' id {id} out of range for vocabulary size {vocab_size}")]
    InvalidSpecialToken {
        /// Special role name (`bos`, `eos`, …).
        name: &'static str,
        /// Declared id.
        id: u32,
        /// Vocabulary length.
        vocab_size: usize,
    },

    /// Concatenated byte pieces did not form valid UTF-8.
    #[error("decoded token bytes are not valid UTF-8: {reason}")]
    InvalidUtf8 {
        /// Underlying UTF-8 error display.
        reason: String,
    },
}
