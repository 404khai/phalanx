//! Root Mean Square Layer Normalization (`RMSNorm`).
//!
//! # Why `RMSNorm`
//!
//! `LayerNorm` centers and scales. `RMSNorm` scales by the root-mean-square
//! only — no mean subtraction — which is cheaper and matches Llama / Odyssey
//! Spec v1.0.0. Using `LayerNorm` here is **non-compliant**.
//!
//! Paper: [RMSNorm](https://arxiv.org/abs/1910.07467).
//!
//! # Formula
//!
//! ```text
//! RMS(x) = sqrt( mean(x_i²) + ε )
//! y      = γ ⊙ (x / RMS(x))
//! ```
//!
//! `ε` comes from [`ModelConfig::rms_norm_eps`]. `γ` is a length-`D` weight
//! vector (`attn_norm` / `ffn_norm` / `output_norm` in GGUF).
//!
//! # Layout
//!
//! Activations are `(..., D)` where `D = embedding_length`. Rank ≥ 1.
//! Output shape matches input.

use tracing::debug;

use super::error::LayersError;
use crate::errors::Result;
use crate::model::ModelConfig;
use crate::tensor::{Shape, Tensor};
use crate::weights::WeightSet;

/// GGUF name for final output `RMSNorm` γ (`norm.weight` in Odyssey).
pub const OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";

/// GGUF name template prefix for per-block attention `RMSNorm` γ.
pub const ATTN_NORM_WEIGHT_PREFIX: &str = "blk.";
/// GGUF suffix for attention `RMSNorm` γ.
pub const ATTN_NORM_WEIGHT_SUFFIX: &str = ".attn_norm.weight";
/// GGUF suffix for FFN `RMSNorm` γ.
pub const FFN_NORM_WEIGHT_SUFFIX: &str = ".ffn_norm.weight";

/// Build GGUF tensor name for layer `i` attention `RMSNorm`.
#[must_use]
pub fn attn_norm_weight_name(layer: usize) -> String {
    format!("{ATTN_NORM_WEIGHT_PREFIX}{layer}{ATTN_NORM_WEIGHT_SUFFIX}")
}

/// Build GGUF tensor name for layer `i` FFN `RMSNorm`.
#[must_use]
pub fn ffn_norm_weight_name(layer: usize) -> String {
    format!("{ATTN_NORM_WEIGHT_PREFIX}{layer}{FFN_NORM_WEIGHT_SUFFIX}")
}

/// Llama-style `RMSNorm`: γ ⊙ x / RMS(x).
#[derive(Debug, Clone, PartialEq)]
pub struct RmsNorm {
    /// Learnable scale γ, shape `[hidden_size]`.
    weight: Tensor,
    /// Stabilizer ε (`rms_norm_eps`).
    eps: f32,
    hidden_size: usize,
}

impl RmsNorm {
    /// Load a named γ tensor from [`WeightSet`] and bind `config.rms_norm_eps`.
    ///
    /// # Errors
    ///
    /// Missing tensor, bad shape, or config mismatch.
    pub fn from_weights(weights: &WeightSet, name: &str, config: &ModelConfig) -> Result<Self> {
        let view = match weights.tensor(name) {
            Ok(v) => v,
            Err(crate::PhalanxError::Weights(crate::weights::WeightsError::TensorNotFound {
                ..
            })) => {
                // `MissingWeight` needs `&'static str`; dynamic `blk.{i}.*` names
                // report through `InvalidWeightShape` instead.
                if name == OUTPUT_NORM_WEIGHT {
                    return Err(LayersError::MissingWeight {
                        name: OUTPUT_NORM_WEIGHT,
                    }
                    .into());
                }
                return Err(LayersError::InvalidWeightShape {
                    name: OUTPUT_NORM_WEIGHT,
                    reason: format!("tensor '{name}' not found in weight set"),
                }
                .into());
            }
            Err(other) => return Err(other),
        };

        let dims = squeeze_trailing_ones(&view.info.dimensions);
        if dims.len() != 1 {
            return Err(LayersError::InvalidWeightShape {
                name: OUTPUT_NORM_WEIGHT,
                reason: format!(
                    "expected rank-1 [hidden] for '{name}', got {:?}",
                    view.info.dimensions
                ),
            }
            .into());
        }

        let hidden = usize_dim(dims[0], name)?;
        let expected =
            usize::try_from(config.embedding_length).map_err(|_| LayersError::ConfigMismatch {
                name: OUTPUT_NORM_WEIGHT,
                reason: format!(
                    "embedding_length {} does not fit usize",
                    config.embedding_length
                ),
            })?;
        if hidden != expected {
            return Err(LayersError::ConfigMismatch {
                name: OUTPUT_NORM_WEIGHT,
                reason: format!("γ length {hidden} != embedding_length {expected}"),
            }
            .into());
        }

        let weight = view.to_f32_tensor()?.into_shape([hidden])?;
        Self::from_tensor(weight, config.rms_norm_eps)
    }

    /// Build from an already materialised `[hidden]` γ and explicit ε.
    ///
    /// Preferred by unit tests and the Odyssey cross-validation binary.
    ///
    /// # Errors
    ///
    /// Bad γ rank/dims or non-positive ε.
    pub fn from_tensor(weight: Tensor, eps: f32) -> Result<Self> {
        let shape = weight.shape().as_slice();
        if shape.len() != 1 {
            return Err(LayersError::InvalidWeightShape {
                name: OUTPUT_NORM_WEIGHT,
                reason: format!("expected rank-1 [hidden], got {shape:?}"),
            }
            .into());
        }
        let hidden_size = shape[0];
        if hidden_size == 0 {
            return Err(LayersError::InvalidWeightShape {
                name: OUTPUT_NORM_WEIGHT,
                reason: "hidden_size must be > 0".into(),
            }
            .into());
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(LayersError::ConfigMismatch {
                name: OUTPUT_NORM_WEIGHT,
                reason: format!("rms_norm_eps must be finite and > 0, got {eps}"),
            }
            .into());
        }

        debug!(hidden_size, eps, "built RmsNorm");
        Ok(Self {
            weight,
            eps,
            hidden_size,
        })
    }

    /// Convenience: ones γ of length `hidden_size` (training-style init).
    ///
    /// # Errors
    ///
    /// Propagates [`from_tensor`](Self::from_tensor) validation errors.
    pub fn ones(hidden_size: usize, eps: f32) -> Result<Self> {
        let weight = Tensor::ones([hidden_size])?;
        Self::from_tensor(weight, eps)
    }

    /// Hidden / embedding length `D`.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Stabilizer ε.
    #[must_use]
    pub const fn eps(&self) -> f32 {
        self.eps
    }

    /// Scale parameter γ.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Apply `RMSNorm` to activations.
    ///
    /// Accepted shapes: any rank ≥ 1 whose **last** dim equals `hidden_size`.
    ///
    /// # Errors
    ///
    /// Shape mismatches.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().as_slice();
        if shape.is_empty() {
            return Err(LayersError::InvalidActivationShape {
                op: "rmsnorm",
                reason: "expected rank >= 1".into(),
            }
            .into());
        }
        // Checked non-empty above.
        let dim = shape[shape.len() - 1];
        if dim != self.hidden_size {
            return Err(LayersError::InvalidActivationShape {
                op: "rmsnorm",
                reason: format!(
                    "last dim {dim} != configured hidden_size {}",
                    self.hidden_size
                ),
            }
            .into());
        }

        let data = input.as_slice();
        let mut out = vec![0.0f32; data.len()];
        let gamma = self.weight.as_slice();
        debug_assert_eq!(gamma.len(), self.hidden_size);

        // Process each last-dim vector independently.
        // Match Odyssey: y = (x / RMS(x)) * γ  (not fused x * inv_rms * γ)
        // to keep float32 rounding aligned for Principle 8 / Rule 6.
        let n_rows = data.len() / self.hidden_size;
        for row in 0..n_rows {
            let base = row * self.hidden_size;
            let src = &data[base..base + self.hidden_size];
            let dst = &mut out[base..base + self.hidden_size];

            let mut sum_sq = 0.0f64;
            for &v in src {
                let v64 = f64::from(v);
                sum_sq += v64 * v64;
            }
            #[allow(clippy::cast_precision_loss)]
            let mean_sq = sum_sq / self.hidden_size as f64;
            #[allow(clippy::cast_possible_truncation)]
            let rms = (mean_sq + f64::from(self.eps)).sqrt() as f32;
            for i in 0..self.hidden_size {
                dst[i] = (src[i] / rms) * gamma[i];
            }
        }

        Tensor::from_vec(out, Shape::new(shape.to_vec())?)
    }
}

fn squeeze_trailing_ones(dims: &[u64]) -> Vec<u64> {
    let mut out: Vec<u64> = dims.to_vec();
    while out.len() > 1 && out.last() == Some(&1) {
        out.pop();
    }
    out
}

fn usize_dim(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        LayersError::InvalidWeightShape {
            name: OUTPUT_NORM_WEIGHT,
            reason: format!("{name}: dimension {value} does not fit usize"),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ones_gamma_scales_by_inv_rms_only() {
        let norm = RmsNorm::ones(4, 1e-6).unwrap();
        let x = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], Shape::new([1, 4]).unwrap()).unwrap();
        let y = norm.forward(&x).unwrap();
        // mean(x²)=1 → RMS=sqrt(1+eps) ≈ 1 → output ≈ x
        for (a, b) in x.as_slice().iter().zip(y.as_slice()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn normalized_vector_has_unit_rms() {
        let norm = RmsNorm::ones(8, 1e-6).unwrap();
        let x = Tensor::from_vec(
            (0..8)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let v = i as f32 * 0.3 - 1.0;
                    v
                })
                .collect(),
            Shape::new([8]).unwrap(),
        )
        .unwrap();
        let y = norm.forward(&x).unwrap();
        let mean_sq: f32 = y.as_slice().iter().map(|v| v * v).sum::<f32>() / 8.0;
        assert!(
            (mean_sq.sqrt() - 1.0).abs() < 1e-4,
            "rms={}",
            mean_sq.sqrt()
        );
    }

    #[test]
    fn gamma_scales_output() {
        let gamma = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], Shape::new([4]).unwrap()).unwrap();
        let norm = RmsNorm::from_tensor(gamma, 1e-6).unwrap();
        let ones_norm = RmsNorm::ones(4, 1e-6).unwrap();
        let x = Tensor::from_vec(vec![0.5, -0.25, 1.0, 0.0], Shape::new([1, 4]).unwrap()).unwrap();
        let y = norm.forward(&x).unwrap();
        let y1 = ones_norm.forward(&x).unwrap();
        for (a, b) in y.as_slice().iter().zip(y1.as_slice()) {
            assert!((a - 2.0 * b).abs() < 1e-5, "{a} vs 2*{b}");
        }
    }

    #[test]
    fn preserves_batch_seq_shape() {
        let norm = RmsNorm::ones(4, 1e-6).unwrap();
        let x = Tensor::ones([2, 3, 4]).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape().as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn rejects_wrong_last_dim() {
        let norm = RmsNorm::ones(4, 1e-6).unwrap();
        let x = Tensor::ones([2, 3]).unwrap();
        let err = norm.forward(&x).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::InvalidActivationShape { .. })
        ));
    }

    #[test]
    fn rejects_non_positive_eps() {
        let w = Tensor::ones([4]).unwrap();
        let err = RmsNorm::from_tensor(w, 0.0).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::ConfigMismatch { .. })
        ));
    }

    #[test]
    fn weight_name_helpers() {
        assert_eq!(attn_norm_weight_name(0), "blk.0.attn_norm.weight");
        assert_eq!(ffn_norm_weight_name(3), "blk.3.ffn_norm.weight");
    }
}
