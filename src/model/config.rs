//! Transformer hyperparameters loaded from GGUF metadata.
//!
//! # Why a dedicated config type
//!
//! Layer kernels (`RoPE`, `RMSNorm`, attention, `FFN`) need a single validated
//! source of truth for `n_embd`, head counts, `RoPE` θ, etc. Reading ad-hoc
//! metadata keys inside every layer invites drift; [`ModelConfig`] parses once
//! and exposes derived sizes (`head_dim`, `GQA` groups) that later phases bind
//! weights against.
//!
//! # References
//!
//! - [LLaMA](https://arxiv.org/abs/2302.13971)
//! - [GGUF LLM keys](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! - llama.cpp `gguf-py/gguf/constants.py` (`Keys.LLM` / `Keys.Attention` / `Keys.Rope`)

use tracing::debug;

use super::architecture::Architecture;
use super::error::ModelError;
use super::keys::{self, keyed};
use crate::errors::Result;
use crate::gguf::{GgufFile, MetadataValue, NAME_KEY};

/// Default `RoPE` base frequency when `rope.freq_base` is omitted.
///
/// Matches the original `RoFormer` / Llama theta (`10000`).
pub const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;

/// Attention-head hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    /// Query head count (`n_head`).
    pub head_count: u32,
    /// Key/value head count (`n_head_kv`); equals `head_count` when not `GQA`.
    pub head_count_kv: u32,
    /// Per-head key dimension (`d_k`).
    pub key_length: u32,
    /// Per-head value dimension (`d_v`).
    pub value_length: u32,
}

impl AttentionConfig {
    /// `true` when the model uses grouped-query attention.
    #[must_use]
    pub const fn is_gqa(self) -> bool {
        self.head_count_kv != self.head_count
    }

    /// Number of query heads that share one KV head (`head_count / head_count_kv`).
    #[must_use]
    pub const fn gqa_groups(self) -> u32 {
        // Validated at construction: head_count_kv divides head_count.
        self.head_count / self.head_count_kv
    }
}

/// Optional `RoPE` context-extension scaling declared in GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeScaling {
    /// Algorithm name (`linear`, `yarn`, …).
    pub scaling_type: String,
    /// Multiplicative scale factor when present.
    pub factor: Option<f32>,
}

/// Rotary positional embedding hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    /// Number of rotary dimensions (usually equal to `key_length` for Llama).
    pub dimension_count: u32,
    /// Base frequency θ.
    pub freq_base: f32,
    /// Optional long-context scaling metadata.
    pub scaling: Option<RopeScaling>,
}

/// Validated decoder-only transformer configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Architecture family.
    pub architecture: Architecture,
    /// Optional `general.name` display string.
    pub name: Option<String>,
    /// Optional `{arch}.vocab_size` (tokenizer length is authoritative when absent).
    pub vocab_size: Option<u32>,
    /// Maximum context length.
    pub context_length: u32,
    /// Hidden size / embedding length (`n_embd`).
    pub embedding_length: u32,
    /// `FFN` intermediate size (`n_ff`).
    pub feed_forward_length: u32,
    /// Number of transformer blocks (`n_layer`).
    pub block_count: u32,
    /// Attention head layout.
    pub attention: AttentionConfig,
    /// `RoPE` parameters for [`crate::layers::Rope`].
    pub rope: RopeConfig,
    /// `RMSNorm` ε (`attention.layer_norm_rms_epsilon`).
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    /// Load and validate hyperparameters from a parsed [`GgufFile`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] when architecture is missing/unsupported, required
    /// keys are absent, types are wrong, or structural invariants fail.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch_name = file.architecture().ok_or(ModelError::MissingArchitecture)?;
        let architecture =
            Architecture::parse(arch_name).ok_or_else(|| ModelError::UnsupportedArchitecture {
                architecture: arch_name.to_owned(),
            })?;

        let name = file
            .get(NAME_KEY)
            .and_then(MetadataValue::as_str)
            .map(str::to_owned);

        let block_count = required_u32(file, arch_name, keys::BLOCK_COUNT)?;
        let context_length = required_u32(file, arch_name, keys::CONTEXT_LENGTH)?;
        let embedding_length = required_u32(file, arch_name, keys::EMBEDDING_LENGTH)?;
        let feed_forward_length = required_u32(file, arch_name, keys::FEED_FORWARD_LENGTH)?;
        let vocab_size = optional_u32(file, arch_name, keys::VOCAB_SIZE)?;

        let head_count = required_u32(file, arch_name, keys::ATTENTION_HEAD_COUNT)?;
        let head_count_kv =
            optional_u32(file, arch_name, keys::ATTENTION_HEAD_COUNT_KV)?.unwrap_or(head_count);

        // Spec default: d_k = d_v = n_embd / n_head when overrides are absent.
        let default_head_dim = embedding_length
            .checked_div(head_count)
            .ok_or_else(|| ModelError::invalid("attention.head_count must be non-zero"))?;
        let key_length =
            optional_u32(file, arch_name, keys::ATTENTION_KEY_LENGTH)?.unwrap_or(default_head_dim);
        let value_length = optional_u32(file, arch_name, keys::ATTENTION_VALUE_LENGTH)?
            .unwrap_or(default_head_dim);

        let rms_norm_eps = required_f32(file, arch_name, keys::ATTENTION_LAYER_NORM_RMS_EPSILON)?;

        let rope_dimension_count = required_u32(file, arch_name, keys::ROPE_DIMENSION_COUNT)?;
        let rope_freq_base =
            optional_f32(file, arch_name, keys::ROPE_FREQ_BASE)?.unwrap_or(DEFAULT_ROPE_FREQ_BASE);

        let rope_scaling = load_rope_scaling(file, arch_name)?;

        let config = Self {
            architecture,
            name,
            vocab_size,
            context_length,
            embedding_length,
            feed_forward_length,
            block_count,
            attention: AttentionConfig {
                head_count,
                head_count_kv,
                key_length,
                value_length,
            },
            rope: RopeConfig {
                dimension_count: rope_dimension_count,
                freq_base: rope_freq_base,
                scaling: rope_scaling,
            },
            rms_norm_eps,
        };

        config.validate()?;

        debug!(
            architecture = %config.architecture,
            name = ?config.name,
            layers = config.block_count,
            embd = config.embedding_length,
            heads = config.attention.head_count,
            kv_heads = config.attention.head_count_kv,
            "loaded model config from GGUF metadata"
        );

        Ok(config)
    }

    /// Construct a config from already-validated parts (unit tests / non-GGUF).
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] when structural checks fail.
    pub fn from_parts(config: Self) -> Result<Self> {
        config.validate()?;
        Ok(config)
    }

    /// Query head dimension (`key_length`; usually `n_embd / n_head`).
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.attention.key_length
    }

    /// Total attention projection width for Q (`head_count * key_length`).
    #[must_use]
    pub const fn query_dim(&self) -> u32 {
        self.attention.head_count * self.attention.key_length
    }

    /// Total KV projection width per cache side (`head_count_kv * key_length`).
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.attention.head_count_kv * self.attention.key_length
    }

    fn validate(&self) -> Result<()> {
        if self.block_count == 0 {
            return Err(ModelError::invalid("block_count must be > 0").into());
        }
        if self.context_length == 0 {
            return Err(ModelError::invalid("context_length must be > 0").into());
        }
        if self.embedding_length == 0 {
            return Err(ModelError::invalid("embedding_length must be > 0").into());
        }
        if self.feed_forward_length == 0 {
            return Err(ModelError::invalid("feed_forward_length must be > 0").into());
        }
        if self.attention.head_count == 0 {
            return Err(ModelError::invalid("attention.head_count must be > 0").into());
        }
        if self.attention.head_count_kv == 0 {
            return Err(ModelError::invalid("attention.head_count_kv must be > 0").into());
        }
        if self.attention.key_length == 0 || self.attention.value_length == 0 {
            return Err(ModelError::invalid("attention key/value length must be > 0").into());
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(ModelError::invalid(
                "attention.layer_norm_rms_epsilon must be a finite value > 0",
            )
            .into());
        }
        if !self.rope.freq_base.is_finite() || self.rope.freq_base <= 0.0 {
            return Err(ModelError::invalid("rope.freq_base must be a finite value > 0").into());
        }
        if self.rope.dimension_count == 0 {
            return Err(ModelError::invalid("rope.dimension_count must be > 0").into());
        }
        if self.rope.dimension_count > self.attention.key_length {
            return Err(ModelError::invalid(format!(
                "rope.dimension_count ({}) exceeds key_length ({})",
                self.rope.dimension_count, self.attention.key_length
            ))
            .into());
        }
        if self.rope.dimension_count % 2 != 0 {
            // RoPE pairs dimensions into (cos, sin) couples.
            return Err(ModelError::invalid(
                "rope.dimension_count must be even (paired rotary dimensions)",
            )
            .into());
        }
        if self.attention.head_count % self.attention.head_count_kv != 0 {
            return Err(ModelError::invalid(format!(
                "attention.head_count ({}) must be a multiple of head_count_kv ({})",
                self.attention.head_count, self.attention.head_count_kv
            ))
            .into());
        }
        // When writers omit key_length, embedding must divide evenly by heads.
        let q_width = self
            .attention
            .head_count
            .checked_mul(self.attention.key_length)
            .ok_or_else(|| ModelError::invalid("query width overflowed u32"))?;
        if q_width != self.embedding_length {
            return Err(ModelError::invalid(format!(
                "head_count * key_length ({q_width}) != embedding_length ({})",
                self.embedding_length
            ))
            .into());
        }
        if let Some(vocab) = self.vocab_size
            && vocab == 0
        {
            return Err(ModelError::invalid("vocab_size must be > 0 when present").into());
        }
        if let Some(scale) = &self.rope.scaling
            && let Some(factor) = scale.factor
            && (!factor.is_finite() || factor <= 0.0)
        {
            return Err(ModelError::invalid(
                "rope.scaling.factor must be a finite value > 0 when present",
            )
            .into());
        }
        Ok(())
    }
}

fn load_rope_scaling(file: &GgufFile, arch: &str) -> Result<Option<RopeScaling>> {
    let scaling_type = optional_str(file, arch, keys::ROPE_SCALING_TYPE)?;
    let factor = match optional_f32(file, arch, keys::ROPE_SCALING_FACTOR)? {
        Some(f) => Some(f),
        // Older Llama exports used `rope.scale` for linear context extension.
        None => optional_f32(file, arch, keys::ROPE_SCALE)?,
    };

    match (scaling_type, factor) {
        (None, None) => Ok(None),
        (Some(scaling_type), factor) => Ok(Some(RopeScaling {
            scaling_type,
            factor,
        })),
        (None, Some(factor)) => Ok(Some(RopeScaling {
            // Legacy key implies linear scaling when type is omitted.
            scaling_type: "linear".into(),
            factor: Some(factor),
        })),
    }
}

fn required_u32(file: &GgufFile, arch: &str, suffix: &str) -> Result<u32> {
    let key = keyed(arch, suffix);
    let value = file
        .get(&key)
        .ok_or_else(|| ModelError::MissingKey { key: key.clone() })?;
    u32_from_value(value, &key)
}

fn optional_u32(file: &GgufFile, arch: &str, suffix: &str) -> Result<Option<u32>> {
    let key = keyed(arch, suffix);
    match file.get(&key) {
        None => Ok(None),
        Some(value) => Ok(Some(u32_from_value(value, &key)?)),
    }
}

fn required_f32(file: &GgufFile, arch: &str, suffix: &str) -> Result<f32> {
    let key = keyed(arch, suffix);
    let value = file
        .get(&key)
        .ok_or_else(|| ModelError::MissingKey { key: key.clone() })?;
    f32_from_value(value, &key)
}

fn optional_f32(file: &GgufFile, arch: &str, suffix: &str) -> Result<Option<f32>> {
    let key = keyed(arch, suffix);
    match file.get(&key) {
        None => Ok(None),
        Some(value) => Ok(Some(f32_from_value(value, &key)?)),
    }
}

fn optional_str(file: &GgufFile, arch: &str, suffix: &str) -> Result<Option<String>> {
    let key = keyed(arch, suffix);
    match file.get(&key) {
        None => Ok(None),
        Some(value) => match value.as_str() {
            Some(s) => Ok(Some(s.to_owned())),
            None => Err(ModelError::InvalidType {
                key,
                reason: format!("expected string, got {:?}", value.value_type()),
            }
            .into()),
        },
    }
}

fn u32_from_value(value: &MetadataValue, key: &str) -> Result<u32> {
    // Writers use u32 or u64 for counts; accept both and reject overflow.
    let v = value.as_u64().ok_or_else(|| ModelError::InvalidType {
        key: key.to_owned(),
        reason: format!("expected u32/u64, got {:?}", value.value_type()),
    })?;
    u32::try_from(v).map_err(|_| {
        ModelError::InvalidType {
            key: key.to_owned(),
            reason: format!("value {v} does not fit in u32"),
        }
        .into()
    })
}

fn f32_from_value(value: &MetadataValue, key: &str) -> Result<f32> {
    match value {
        MetadataValue::F32(v) => Ok(*v),
        // Rare: some converters store eps / θ as f64; values fit comfortably in f32.
        MetadataValue::F64(v) => {
            #[allow(clippy::cast_possible_truncation)]
            let narrowed = *v as f32;
            Ok(narrowed)
        }
        other => Err(ModelError::InvalidType {
            key: key.to_owned(),
            reason: format!("expected f32/f64, got {:?}", other.value_type()),
        }
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::gguf::test_support::GgufBuilder;

    fn llama_fixture() -> GgufBuilder {
        GgufBuilder::new()
            .architecture("llama")
            .name("toy-llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 32)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 2048)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), 4096)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), 11_008)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 32)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT_KV), 32)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 128)
            .meta_f32(&keyed("llama", keys::ROPE_FREQ_BASE), 10_000.0)
    }

    #[test]
    fn from_gguf_loads_llama_7b_style_hparams() {
        let bytes = llama_fixture().build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();

        assert_eq!(cfg.architecture, Architecture::Llama);
        assert_eq!(cfg.name.as_deref(), Some("toy-llama"));
        assert_eq!(cfg.block_count, 32);
        assert_eq!(cfg.embedding_length, 4096);
        assert_eq!(cfg.feed_forward_length, 11_008);
        assert_eq!(cfg.attention.head_count, 32);
        assert_eq!(cfg.attention.head_count_kv, 32);
        assert_eq!(cfg.head_dim(), 128);
        assert!(!cfg.attention.is_gqa());
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-12);
        assert!((cfg.rope.freq_base - 10_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gqa_defaults_and_groups() {
        // Llama-3 8B style: 32 query heads, 8 KV heads; omit rope.freq_base → default θ.
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 32)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 8192)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), 4096)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), 14_336)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 32)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT_KV), 8)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 128)
            .build();

        let file = GgufFile::from_bytes(&bytes).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        assert!(cfg.attention.is_gqa());
        assert_eq!(cfg.attention.gqa_groups(), 4);
        assert_eq!(cfg.kv_dim(), 8 * 128);
        assert!((cfg.rope.freq_base - DEFAULT_ROPE_FREQ_BASE).abs() < f32::EPSILON);
    }

    #[test]
    fn accepts_u64_counts() {
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .meta_u64(&keyed("llama", keys::BLOCK_COUNT), 2)
            .meta_u64(&keyed("llama", keys::CONTEXT_LENGTH), 512)
            .meta_u64(&keyed("llama", keys::EMBEDDING_LENGTH), 256)
            .meta_u64(&keyed("llama", keys::FEED_FORWARD_LENGTH), 512)
            .meta_u64(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 4)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-6,
            )
            .meta_u64(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 64)
            .build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        assert_eq!(cfg.block_count, 2);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn rejects_unsupported_architecture() {
        let bytes = GgufBuilder::new().architecture("qwen2").build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let err = ModelConfig::from_gguf(&file).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Model(ModelError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn rejects_missing_architecture() {
        let bytes = GgufBuilder::new()
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 1)
            .build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let err = ModelConfig::from_gguf(&file).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Model(ModelError::MissingArchitecture)
        ));
    }

    #[test]
    fn rejects_bad_gqa_ratio() {
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 2)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 128)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), 256)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), 512)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 8)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT_KV), 3)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 32)
            .build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let err = ModelConfig::from_gguf(&file).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Model(ModelError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn legacy_rope_scale_becomes_linear_scaling() {
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 2)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 4096)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), 256)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), 512)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 4)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 64)
            .meta_f32(&keyed("llama", keys::ROPE_SCALE), 2.0)
            .build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        let scaling = cfg.rope.scaling.expect("scaling");
        assert_eq!(scaling.scaling_type, "linear");
        assert_eq!(scaling.factor, Some(2.0));
    }
}
