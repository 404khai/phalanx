//! Typed failures for GGUF header / metadata / tensor-info parsing.
//!
//! Kept separate from [`crate::PhalanxError`] so callers can match parse
//! failures without string scraping, while the crate root still exposes a
//! single error enum via `#[from]`.

use thiserror::Error;

/// Errors produced while validating or decoding a GGUF container.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GgufError {
    /// Magic bytes were not `GGUF`.
    #[error("invalid GGUF magic: expected {expected:?}, got {got:?}")]
    InvalidMagic {
        /// Expected magic (`b"GGUF"`).
        expected: [u8; 4],
        /// Bytes actually read.
        got: [u8; 4],
    },

    /// Format version is not supported by this reader.
    #[error("unsupported GGUF version {version} (supported: {supported:?})")]
    UnsupportedVersion {
        /// Version field from the file.
        version: u32,
        /// Versions this build accepts.
        supported: &'static [u32],
    },

    /// Unexpected end of input while reading a field.
    #[error("unexpected end of GGUF stream while reading {context}")]
    UnexpectedEof {
        /// Field or section being decoded.
        context: &'static str,
    },

    /// A length / count exceeded a safety limit.
    #[error("GGUF {context} length {got} exceeds limit {limit}")]
    LimitExceeded {
        /// What was being sized.
        context: &'static str,
        /// Declared length.
        got: u64,
        /// Configured maximum.
        limit: u64,
    },

    /// Metadata or tensor field failed a format rule.
    #[error("invalid GGUF {context}: {reason}")]
    Invalid {
        /// Field or section name.
        context: &'static str,
        /// Human-readable explanation.
        reason: String,
    },

    /// Unknown `gguf_metadata_value_type` discriminant.
    #[error("unknown GGUF metadata value type id {type_id}")]
    UnknownValueType {
        /// Raw type tag from the file.
        type_id: u32,
    },

    /// Boolean metadata byte was not `0` or `1`.
    #[error("invalid GGUF bool value {value} (must be 0 or 1)")]
    InvalidBool {
        /// Raw byte.
        value: u8,
    },

    /// UTF-8 decoding failed for a GGUF string.
    #[error("invalid UTF-8 in GGUF {context}: {reason}")]
    InvalidUtf8 {
        /// Key, tensor name, or string value.
        context: &'static str,
        /// Underlying error display.
        reason: String,
    },
}
