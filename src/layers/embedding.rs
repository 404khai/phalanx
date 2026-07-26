//! Token embedding lookup (`token_embd.weight`).
//!
//! # What this layer does
//!
//! Maps each token id to a dense activation row of length `n_embd`. Prefill
//! gathers a sequence; decode gathers one id. Output shape is
//! `[seq_len, embedding_length]`.
//!
//! # GGUF / ggml layout (why we reshape)
//!
//! Llama GGUF stores `token_embd.weight` with dimensions
//! `[n_embd, n_vocab]` in **ggml order**: `ne[0] = n_embd` is the *innermost*
//! (contiguous) axis. Bytes are therefore laid out as:
//!
//! ```text
//! for token in 0..n_vocab:
//!     for dim in 0..n_embd:
//!         data[token * n_embd + dim]
//! ```
//!
//! That is identical to a row-major `[n_vocab, n_embd]` table. Our Phase 2
//! [`Tensor`] is row-major with shape taken from the GGUF dimension list, so
//! a naïve `[n_embd, n_vocab]` view would mis-index gathers. After dense
//! materialization we **reinterpret** the same buffer as `[n_vocab, n_embd]`
//! (no copy) and gather rows.
//!
//! References:
//! - [LLaMA](https://arxiv.org/abs/2302.13971) (embedding + transformer stack)
//! - [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! - llama.cpp tensor name `TOKEN_EMBD` → `token_embd.weight`

use tracing::debug;

use super::error::LayersError;
use crate::errors::Result;
use crate::model::ModelConfig;
use crate::tensor::{Shape, Tensor};
use crate::weights::WeightSet;

/// Canonical GGUF name for the token embedding matrix.
pub const TOKEN_EMBD_WEIGHT: &str = "token_embd.weight";

/// Dense token embedding table ready for gather.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingTable {
    /// Row-major `[vocab_size, embedding_length]` weight matrix.
    weight: Tensor,
    vocab_size: usize,
    embedding_length: usize,
}

impl EmbeddingTable {
    /// Load `token_embd.weight` from a [`WeightSet`] and validate against config.
    ///
    /// Dense `f32` / `f16` payloads are materialized via
    /// [`crate::weights::WeightTensor::to_f32_tensor`]. Quantized embeddings
    /// still return [`crate::weights::WeightsError::DequantNotImplemented`]
    /// until block dequant lands.
    ///
    /// # Errors
    ///
    /// Missing tensor, bad rank/dims, config mismatch, or materialization errors.
    pub fn from_weights(weights: &WeightSet, config: &ModelConfig) -> Result<Self> {
        let view = match weights.tensor(TOKEN_EMBD_WEIGHT) {
            Ok(v) => v,
            Err(crate::PhalanxError::Weights(crate::weights::WeightsError::TensorNotFound {
                ..
            })) => {
                return Err(LayersError::MissingWeight {
                    name: TOKEN_EMBD_WEIGHT,
                }
                .into());
            }
            Err(other) => return Err(other),
        };

        let dims = squeeze_trailing_ones(&view.info.dimensions);
        if dims.len() != 2 {
            return Err(LayersError::InvalidWeightShape {
                name: TOKEN_EMBD_WEIGHT,
                reason: format!(
                    "expected rank-2 [n_embd, n_vocab], got {:?}",
                    view.info.dimensions
                ),
            }
            .into());
        }

        let n_embd = usize_dim(dims[0], TOKEN_EMBD_WEIGHT)?;
        let n_vocab = usize_dim(dims[1], TOKEN_EMBD_WEIGHT)?;
        let expected_embd =
            usize::try_from(config.embedding_length).map_err(|_| LayersError::ConfigMismatch {
                name: TOKEN_EMBD_WEIGHT,
                reason: format!(
                    "embedding_length {} does not fit usize",
                    config.embedding_length
                ),
            })?;

        if n_embd != expected_embd {
            return Err(LayersError::ConfigMismatch {
                name: TOKEN_EMBD_WEIGHT,
                reason: format!(
                    "tensor n_embd {n_embd} != config.embedding_length {expected_embd}"
                ),
            }
            .into());
        }

        if let Some(declared) = config.vocab_size {
            let declared = usize::try_from(declared).map_err(|_| LayersError::ConfigMismatch {
                name: TOKEN_EMBD_WEIGHT,
                reason: format!("vocab_size {declared} does not fit usize"),
            })?;
            if declared != n_vocab {
                return Err(LayersError::ConfigMismatch {
                    name: TOKEN_EMBD_WEIGHT,
                    reason: format!("tensor n_vocab {n_vocab} != config.vocab_size {declared}"),
                }
                .into());
            }
        }

        // Bytes are ggml-ordered; reshape metadata to row-major [vocab, embd].
        let weight = view.to_f32_tensor()?.into_shape([n_vocab, n_embd])?;

        debug!(
            vocab_size = n_vocab,
            embedding_length = n_embd,
            "loaded token embedding table"
        );

        Ok(Self {
            weight,
            vocab_size: n_vocab,
            embedding_length: n_embd,
        })
    }

    /// Build from an already row-major `[vocab, embd]` matrix (unit tests).
    ///
    /// # Errors
    ///
    /// Returns [`LayersError::InvalidWeightShape`] when rank/dims are wrong.
    pub fn from_tensor(weight: Tensor) -> Result<Self> {
        let shape = weight.shape().as_slice();
        if shape.len() != 2 {
            return Err(LayersError::InvalidWeightShape {
                name: TOKEN_EMBD_WEIGHT,
                reason: format!("expected rank-2 [vocab, embd], got {shape:?}"),
            }
            .into());
        }
        let vocab_size = shape[0];
        let embedding_length = shape[1];
        if vocab_size == 0 || embedding_length == 0 {
            return Err(LayersError::InvalidWeightShape {
                name: TOKEN_EMBD_WEIGHT,
                reason: "vocab_size and embedding_length must be > 0".into(),
            }
            .into());
        }
        Ok(Self {
            weight,
            vocab_size,
            embedding_length,
        })
    }

    /// Vocabulary rows in the table.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden size (`n_embd`).
    #[must_use]
    pub const fn embedding_length(&self) -> usize {
        self.embedding_length
    }

    /// Borrow the `[vocab, embd]` weight matrix.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Gather one token → shape `[embedding_length]`.
    ///
    /// # Errors
    ///
    /// Returns [`LayersError::TokenOutOfRange`] for unknown ids.
    pub fn forward_one(&self, token_id: u32) -> Result<Tensor> {
        let row = self.row_slice(token_id)?;
        Tensor::from_slice(row, Shape::new([self.embedding_length])?)
    }

    /// Gather a sequence → shape `[seq_len, embedding_length]`.
    ///
    /// # Errors
    ///
    /// Returns [`LayersError::TokenOutOfRange`] when any id is invalid.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Tensor> {
        let embd = self.embedding_length;
        let mut out = vec![0.0; token_ids.len().saturating_mul(embd)];
        for (step, &token_id) in token_ids.iter().enumerate() {
            let row = self.row_slice(token_id)?;
            let dest = &mut out[step * embd..step * embd + embd];
            dest.copy_from_slice(row);
        }
        Tensor::from_vec(out, Shape::new([token_ids.len(), embd])?)
    }

    fn row_slice(&self, token_id: u32) -> Result<&[f32]> {
        let id = usize::try_from(token_id).map_err(|_| LayersError::TokenOutOfRange {
            id: token_id,
            vocab_size: self.vocab_size,
        })?;
        if id >= self.vocab_size {
            return Err(LayersError::TokenOutOfRange {
                id: token_id,
                vocab_size: self.vocab_size,
            }
            .into());
        }
        let embd = self.embedding_length;
        Ok(&self.weight.as_slice()[id * embd..id * embd + embd])
    }
}

/// Drop trailing unitary dims some converters append (`[e, v, 1, 1]` → `[e, v]`).
fn squeeze_trailing_ones(dims: &[u64]) -> Vec<u64> {
    let mut end = dims.len();
    while end > 2 && dims[end - 1] == 1 {
        end -= 1;
    }
    dims[..end].to_vec()
}

fn usize_dim(value: u64, name: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        LayersError::InvalidWeightShape {
            name,
            reason: format!("dimension {value} does not fit usize"),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_support::GgufBuilder;
    use crate::gguf::{DEFAULT_ALIGNMENT, GgmlType, TensorInfo, align_offset};
    use crate::model::keys::{self, keyed};
    use crate::model::{AttentionConfig, ModelConfig, RopeConfig};
    use crate::weights::WeightSet;

    fn tiny_config(vocab: u32, embd: u32) -> ModelConfig {
        ModelConfig::from_parts(ModelConfig {
            architecture: crate::model::Architecture::Llama,
            name: Some("toy".into()),
            vocab_size: Some(vocab),
            context_length: 128,
            embedding_length: embd,
            feed_forward_length: embd * 2,
            block_count: 1,
            attention: AttentionConfig {
                head_count: 2,
                head_count_kv: 2,
                key_length: embd / 2,
                value_length: embd / 2,
            },
            rope: RopeConfig {
                dimension_count: embd / 2,
                freq_base: 10_000.0,
                scaling: None,
            },
            rms_norm_eps: 1e-5,
        })
        .unwrap()
    }

    /// GGUF with `token_embd.weight` dims `[n_embd, n_vocab]` and ggml-ordered bytes.
    fn fixture_embedding_gguf(vocab: u32, embd: u32) -> Vec<u8> {
        let n_vocab = vocab as usize;
        let n_embd = embd as usize;
        // data[token * embd + dim] = token as f32 + dim as f32 / 100
        let mut payload = Vec::with_capacity(n_vocab * n_embd * 4);
        for token in 0..n_vocab {
            for dim in 0..n_embd {
                // Fixture dims are tiny; `u16 → f32` is exact.
                let v = f32::from(u16::try_from(token).unwrap())
                    + f32::from(u16::try_from(dim).unwrap()) / 100.0;
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }

        let header = GgufBuilder::new()
            .architecture("llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 1)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 128)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), embd)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), embd * 2)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 2)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), embd / 2)
            .meta_u32(&keyed("llama", keys::VOCAB_SIZE), vocab)
            .tensor(TensorInfo {
                name: TOKEN_EMBD_WEIGHT.into(),
                // ggml / GGUF order: [n_embd, n_vocab]
                dimensions: vec![u64::from(embd), u64::from(vocab)],
                ggml_type: GgmlType::F32,
                offset: 0,
            })
            .build();

        let data_offset =
            usize::try_from(align_offset(header.len() as u64, DEFAULT_ALIGNMENT)).unwrap();
        let mut bytes = header;
        bytes.resize(data_offset, 0);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn from_weights_gathers_correct_rows() {
        let bytes = fixture_embedding_gguf(4, 4);
        let weights = WeightSet::from_bytes(bytes).unwrap();
        let config = ModelConfig::from_gguf(weights.gguf()).unwrap();
        let table = EmbeddingTable::from_weights(&weights, &config).unwrap();

        assert_eq!(table.vocab_size(), 4);
        assert_eq!(table.embedding_length(), 4);
        assert_eq!(table.weight().shape().as_slice(), &[4, 4]);

        let row1 = table.forward_one(1).unwrap();
        assert_eq!(row1.shape().as_slice(), &[4]);
        assert!((row1.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((row1.as_slice()[1] - 1.01).abs() < 1e-6);

        let batch = table.forward(&[0, 2, 3]).unwrap();
        assert_eq!(batch.shape().as_slice(), &[3, 4]);
        assert!((batch.get(&[0, 0]).unwrap() - 0.0).abs() < 1e-6);
        assert!((batch.get(&[1, 0]).unwrap() - 2.0).abs() < 1e-6);
        assert!((batch.get(&[2, 2]).unwrap() - 3.02).abs() < 1e-6);
    }

    #[test]
    fn squeezes_trailing_ones() {
        let config = tiny_config(2, 4);
        let weight = Tensor::from_vec(
            vec![
                0.0, 0.01, 0.02, 0.03, // token 0
                1.0, 1.01, 1.02, 1.03, // token 1
            ],
            Shape::new([2, 4]).unwrap(),
        )
        .unwrap();
        // Direct table path still works; trailing-ones covered via from_weights
        // by building a custom TensorInfo in a mini fixture.
        let mut payload = Vec::new();
        for v in weight.as_slice() {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let header = GgufBuilder::new()
            .architecture("llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 1)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 128)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), 4)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), 8)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 2)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 2)
            .tensor(TensorInfo {
                name: TOKEN_EMBD_WEIGHT.into(),
                dimensions: vec![4, 2, 1, 1],
                ggml_type: GgmlType::F32,
                offset: 0,
            })
            .build();
        let data_offset =
            usize::try_from(align_offset(header.len() as u64, DEFAULT_ALIGNMENT)).unwrap();
        let mut bytes = header;
        bytes.resize(data_offset, 0);
        bytes.extend_from_slice(&payload);

        let weights = WeightSet::from_bytes(bytes).unwrap();
        let table = EmbeddingTable::from_weights(&weights, &config).unwrap();
        assert!((table.forward_one(1).unwrap().as_slice()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_token_errors() {
        let table = EmbeddingTable::from_tensor(Tensor::zeros([2, 3]).unwrap()).unwrap();
        let err = table.forward(&[0, 2]).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::TokenOutOfRange { id: 2, .. })
        ));
    }

    #[test]
    fn missing_weight_errors() {
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .meta_u32(&keyed("llama", keys::BLOCK_COUNT), 1)
            .meta_u32(&keyed("llama", keys::CONTEXT_LENGTH), 128)
            .meta_u32(&keyed("llama", keys::EMBEDDING_LENGTH), 4)
            .meta_u32(&keyed("llama", keys::FEED_FORWARD_LENGTH), 8)
            .meta_u32(&keyed("llama", keys::ATTENTION_HEAD_COUNT), 2)
            .meta_f32(
                &keyed("llama", keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
                1e-5,
            )
            .meta_u32(&keyed("llama", keys::ROPE_DIMENSION_COUNT), 2)
            .build();
        // No tensor payload — WeightSet allows empty tensor lists.
        let weights = WeightSet::from_bytes(bytes).unwrap();
        let config = ModelConfig::from_gguf(weights.gguf()).unwrap();
        let err = EmbeddingTable::from_weights(&weights, &config).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::MissingWeight {
                name: TOKEN_EMBD_WEIGHT
            })
        ));
    }

    #[test]
    fn empty_sequence_yields_zero_rows() {
        let table = EmbeddingTable::from_tensor(Tensor::ones([3, 2]).unwrap()).unwrap();
        let out = table.forward(&[]).unwrap();
        assert_eq!(out.shape().as_slice(), &[0, 2]);
        assert_eq!(out.numel(), 0);
    }
}
