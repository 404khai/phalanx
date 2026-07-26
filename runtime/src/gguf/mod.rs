//! GGUF container parser — header, metadata, and tensor directory.
//!
//! # Why GGUF exists
//!
//! Training frameworks (`PyTorch`, etc.) ship checkpoints that are awkward for
//! local inference: multi-file sharded state dicts, Python pickles, and little
//! quantization metadata. GGUF (successor to GGML/GGMF/GGJT) packs **weights +
//! typed metadata + tensor layout** into one `mmap`-friendly binary so an
//! executor can load a model with minimal code and no sidecar JSON.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
//!
//! # Phase 3 scope
//!
//! ```text
//! ┌──────────────┬─────────────────┬──────────────┬────────────┐
//! │ magic/ver/n  │ metadata KV×N   │ tensor info×M│  padding   │ tensor_data …
//! └──────────────┴─────────────────┴──────────────┴────────────┘
//!  ▲ parsed here ▲ parsed here      ▲ parsed here  ▲ data_offset
//!                                                  weights → Phase 5
//! ```
//!
//! # Module map
//!
//! - [`types`] — magic, versions, `ggml_type`, alignment helpers
//! - [`value`] — decoded metadata values
//! - [`tensor_info`] — tensor directory records
//! - [`file`] — [`GgufFile`] parse entrypoints
//! - [`error`] — [`GgufError`]

mod error;
mod file;
mod reader;
mod tensor_info;
mod types;
mod value;

pub use error::GgufError;
pub use file::{GgufFile, GgufHeader};
pub use tensor_info::TensorInfo;
pub use types::{
    ALIGNMENT_KEY, ARCHITECTURE_KEY, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION_V2,
    GGUF_VERSION_V3, GgmlType, MetadataValueType, NAME_KEY, SUPPORTED_VERSIONS, align_offset,
};
pub use value::{MetadataArray, MetadataEntry, MetadataValue};

/// Synthetic GGUF writer shared by unit tests across modules.
#[cfg(test)]
pub use file::test_support;
