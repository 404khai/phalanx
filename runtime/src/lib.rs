//! # Phalanx Runtime
//!
//! A high-performance, educational inference runtime for decoder-only
//! language models, beginning with GGUF-format weights.
//!
//! Phase 6 adds [`model::ModelConfig`]: validated Llama-style transformer
//! hyperparameters parsed from GGUF `{arch}.*` metadata.
//!
//! # Crate layout
//!
//! - [`errors`] — typed [`errors::PhalanxError`] for library APIs
//! - [`tensor`] — shapes, f32 storage, element-wise ops, matmul
//! - [`gguf`] — GGUF container parse
//! - [`tokenizer`] — vocab, special tokens, encode / decode
//! - [`weights`] — mmap + quant metadata + dense materialization
//! - [`model`] — architecture + transformer config
//! - [`utils`] — cross-cutting helpers (logging today; more later)

#![doc(html_root_url = "https://docs.rs/phalanx/0.1.0")]

pub mod errors;
pub mod gguf;
pub mod model;
pub mod tensor;
pub mod tokenizer;
pub mod utils;
pub mod weights;

pub use errors::{PhalanxError, Result};
pub use gguf::{GgmlType, GgufError, GgufFile, GgufHeader, MetadataValue, TensorInfo};
pub use model::{Architecture, AttentionConfig, ModelConfig, ModelError, RopeConfig, RopeScaling};
pub use tensor::{DType, Shape, Tensor, TensorError};
pub use tokenizer::{
    DecodeOptions, EncodeOptions, SpecialTokens, Tokenizer, TokenizerError, TokenizerModel,
    Vocabulary,
};
pub use utils::{LogConfig, init_logging};
pub use weights::{QuantMeta, WeightSet, WeightStorage, WeightTensor, WeightsError};

/// Library version string, matching `Cargo.toml`.
///
/// Exposed so the CLI and future introspection APIs can report a single
/// source of truth without re-parsing package metadata at runtime.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Human-readable runtime name used in banners and logs.
pub const RUNTIME_NAME: &str = "Phalanx Runtime";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_semver_like() {
        // Keep the public constant honest: packaging mistakes should fail CI.
        assert!(!VERSION.is_empty());
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn runtime_name_is_stable() {
        assert_eq!(RUNTIME_NAME, "Phalanx Runtime");
    }
}
