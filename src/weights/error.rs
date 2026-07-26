//! Typed failures for weight loading, mmap, and materialization.
//!
//! Nested under [`crate::PhalanxError::Weights`] so callers can match load
//! failures without scraping display strings.

use thiserror::Error;

use crate::gguf::GgmlType;

/// Errors from mapping a GGUF file and resolving tensor payloads.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WeightsError {
    /// Named tensor was not present in the GGUF directory.
    #[error("weight tensor '{name}' not found in GGUF")]
    TensorNotFound {
        /// Requested tensor name.
        name: String,
    },

    /// Element count overflowed or was zero when sizing a payload.
    #[error("invalid element count for tensor '{name}'")]
    InvalidElementCount {
        /// Tensor name.
        name: String,
    },

    /// `ggml_type` has no known block layout in this build.
    #[error("unsupported ggml type for tensor '{name}': {ggml_type}")]
    UnsupportedType {
        /// Tensor name.
        name: String,
        /// Type tag.
        ggml_type: GgmlType,
    },

    /// Element count is not a multiple of the quantization block size.
    #[error(
        "tensor '{name}' numel {numel} is not a multiple of block size {block_size} for {ggml_type}"
    )]
    MisalignedElements {
        /// Tensor name.
        name: String,
        /// Product of dimensions.
        numel: u64,
        /// Elements per quant block.
        block_size: u64,
        /// Type tag.
        ggml_type: GgmlType,
    },

    /// Declared payload extends past the end of the mapped file.
    #[error(
        "tensor '{name}' payload [{start}..{end}) exceeds file length {file_len} (type {ggml_type})"
    )]
    OutOfBounds {
        /// Tensor name.
        name: String,
        /// Absolute start offset.
        start: u64,
        /// Exclusive end offset.
        end: u64,
        /// Mapped file length.
        file_len: u64,
        /// Type tag.
        ggml_type: GgmlType,
    },

    /// Materialization was requested for a type that is not yet dequantized.
    #[error("cannot materialize tensor '{name}' as f32 from {ggml_type} (dequant not implemented)")]
    DequantNotImplemented {
        /// Tensor name.
        name: String,
        /// Type tag.
        ggml_type: GgmlType,
    },

    /// Payload byte length did not match the float element expectation.
    #[error("tensor '{name}' f32/f16 payload has unexpected byte length {got}")]
    InvalidFloatPayload {
        /// Tensor name.
        name: String,
        /// Byte length observed.
        got: usize,
    },

    /// Memory-map failed.
    #[error("failed to memory-map GGUF file: {reason}")]
    Mmap {
        /// Human-readable `memmap2` / OS error.
        reason: String,
    },
}
