//! Model architecture and transformer hyperparameters.
//!
//! Phase 6 parses Llama-style GGUF metadata into a validated [`ModelConfig`]
//! that later phases (embeddings, `RoPE`, `RMSNorm`, attention, `FFN`, decoder)
//! will consume as the single source of truth for tensor shapes.
//!
//! # Module map
//!
//! - [`architecture`] — [`Architecture`] enum (`llama` today)
//! - [`config`] — [`ModelConfig`] + attention / `RoPE` sub-configs
//! - [`keys`] — `{arch}.*` metadata suffixes
//! - [`error`] — [`ModelError`]

mod architecture;
mod config;
mod error;
pub(crate) mod keys;

pub use architecture::Architecture;
pub use config::{AttentionConfig, DEFAULT_ROPE_FREQ_BASE, ModelConfig, RopeConfig, RopeScaling};
pub use error::ModelError;
pub use keys::keyed;
