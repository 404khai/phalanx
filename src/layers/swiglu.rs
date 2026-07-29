//! LLaMA-style `SwiGLU` feed-forward network (`FFN`).
//!
//! # Formula (Odyssey Spec v1.0.0)
//!
//! ```text
//! Swish(z) = z · σ(z)
//! FFN(x)   = (Swish(x W1ᵀ) ⊙ (x W3ᵀ)) W2ᵀ
//! ```
//!
//! | Weight | Role | Shape | GGUF |
//! |--------|------|-------|------|
//! | `w1`   | Gate | `(I, D)` | `ffn_gate` |
//! | `w3`   | Up   | `(I, D)` | `ffn_up` |
//! | `w2`   | Down | `(D, I)` | `ffn_down` |
//!
//! No biases. Advertising `swiglu` while running ReLU/GeLU is **non-compliant**.

use tracing::debug;

use super::error::LayersError;
use crate::errors::Result;
use crate::model::ModelConfig;
use crate::tensor::Tensor;
use crate::weights::WeightSet;

/// GGUF suffix for gate projection (`w1`).
pub const FFN_GATE_WEIGHT_SUFFIX: &str = ".ffn_gate.weight";
/// GGUF suffix for up projection (`w3`).
pub const FFN_UP_WEIGHT_SUFFIX: &str = ".ffn_up.weight";
/// GGUF suffix for down projection (`w2`).
pub const FFN_DOWN_WEIGHT_SUFFIX: &str = ".ffn_down.weight";
/// Prefix shared with other block tensors.
pub const FFN_WEIGHT_PREFIX: &str = "blk.";

/// Build GGUF name for layer `i` gate weight.
#[must_use]
pub fn ffn_gate_weight_name(layer: usize) -> String {
    format!("{FFN_WEIGHT_PREFIX}{layer}{FFN_GATE_WEIGHT_SUFFIX}")
}

/// Build GGUF name for layer `i` up weight.
#[must_use]
pub fn ffn_up_weight_name(layer: usize) -> String {
    format!("{FFN_WEIGHT_PREFIX}{layer}{FFN_UP_WEIGHT_SUFFIX}")
}

/// Build GGUF name for layer `i` down weight.
#[must_use]
pub fn ffn_down_weight_name(layer: usize) -> String {
    format!("{FFN_WEIGHT_PREFIX}{layer}{FFN_DOWN_WEIGHT_SUFFIX}")
}

/// `SwiGLU` feed-forward: gate × up → down.
#[derive(Debug, Clone, PartialEq)]
pub struct SwiGlu {
    /// Gate projection `w1`, shape `[intermediate, hidden]`.
    w_gate: Tensor,
    /// Up projection `w3`, shape `[intermediate, hidden]`.
    w_up: Tensor,
    /// Down projection `w2`, shape `[hidden, intermediate]`.
    w_down: Tensor,
    hidden_size: usize,
    intermediate_size: usize,
}

impl SwiGlu {
    /// Load `ffn_gate` / `ffn_up` / `ffn_down` for one block from a [`WeightSet`].
    ///
    /// # Errors
    ///
    /// Missing tensors, bad shapes, or config mismatch.
    pub fn from_weights(weights: &WeightSet, layer: usize, config: &ModelConfig) -> Result<Self> {
        let gate = load_matrix(
            weights,
            &ffn_gate_weight_name(layer),
            /*rows*/ config.feed_forward_length,
            /*cols*/ config.embedding_length,
        )?;
        let up = load_matrix(
            weights,
            &ffn_up_weight_name(layer),
            config.feed_forward_length,
            config.embedding_length,
        )?;
        let down = load_matrix(
            weights,
            &ffn_down_weight_name(layer),
            config.embedding_length,
            config.feed_forward_length,
        )?;
        Self::from_tensors(gate, up, down)
    }

    /// Build from already materialised Spec-shaped weights (validators / tests).
    ///
    /// # Errors
    ///
    /// Rank/dim mismatches between the three matrices.
    pub fn from_tensors(w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Result<Self> {
        let g = w_gate.shape().as_slice();
        let u = w_up.shape().as_slice();
        let d = w_down.shape().as_slice();
        if g.len() != 2 || u.len() != 2 || d.len() != 2 {
            return Err(LayersError::InvalidWeightShape {
                name: "ffn_gate.weight",
                reason: format!("expected rank-2 weights, got gate={g:?} up={u:?} down={d:?}"),
            }
            .into());
        }
        if g != u {
            return Err(LayersError::InvalidWeightShape {
                name: "ffn_gate.weight",
                reason: format!("gate shape {g:?} != up shape {u:?}"),
            }
            .into());
        }
        let intermediate = g[0];
        let hidden = g[1];
        if intermediate == 0 || hidden == 0 {
            return Err(LayersError::InvalidWeightShape {
                name: "ffn_gate.weight",
                reason: "hidden and intermediate sizes must be > 0".into(),
            }
            .into());
        }
        if d[0] != hidden || d[1] != intermediate {
            return Err(LayersError::InvalidWeightShape {
                name: "ffn_down.weight",
                reason: format!("down expected [{hidden}, {intermediate}], got {d:?}"),
            }
            .into());
        }

        debug!(hidden, intermediate, "built SwiGlu");
        Ok(Self {
            w_gate,
            w_up,
            w_down,
            hidden_size: hidden,
            intermediate_size: intermediate,
        })
    }

    /// Hidden / embedding length `D`.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Intermediate / FFN width `I`.
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Gate weight `w1` (`[I, D]`).
    #[must_use]
    pub fn w_gate(&self) -> &Tensor {
        &self.w_gate
    }

    /// Up weight `w3` (`[I, D]`).
    #[must_use]
    pub fn w_up(&self) -> &Tensor {
        &self.w_up
    }

    /// Down weight `w2` (`[D, I]`).
    #[must_use]
    pub fn w_down(&self) -> &Tensor {
        &self.w_down
    }

    /// Apply `SwiGLU` FFN.
    ///
    /// Accepted shapes: any rank ≥ 2 whose last dim equals `hidden_size`
    /// (typically `[seq, D]` or `[batch, seq, D]`).
    ///
    /// # Errors
    ///
    /// Shape mismatches or matmul failures.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().as_slice();
        if shape.len() < 2 {
            return Err(LayersError::InvalidActivationShape {
                op: "swiglu",
                reason: format!("expected rank >= 2 with last dim D, got {shape:?}"),
            }
            .into());
        }
        let dim = shape[shape.len() - 1];
        if dim != self.hidden_size {
            return Err(LayersError::InvalidActivationShape {
                op: "swiglu",
                reason: format!(
                    "last dim {dim} != configured hidden_size {}",
                    self.hidden_size
                ),
            }
            .into());
        }

        let rows: usize = shape[..shape.len() - 1].iter().product();
        let flat = input.reshape([rows, self.hidden_size])?;

        // y = x @ Wᵀ  with W stored as (out, in)
        let gate = linear(&flat, &self.w_gate)?;
        let up = linear(&flat, &self.w_up)?;
        let gated = silu(&gate)?.mul(&up)?;
        let out_flat = linear(&gated, &self.w_down)?;

        out_flat.into_shape(shape.to_vec())
    }
}

/// `nn.Linear`-style product: `x @ Wᵀ` for `W` shaped `[out, in]`.
fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let weight_t = weight.transpose()?;
    x.matmul(&weight_t)
}

/// `SiLU` / `Swish`: `x · σ(x)` with `σ(x) = 1 / (1 + e^{-x})`.
fn silu(x: &Tensor) -> Result<Tensor> {
    let data: Vec<f32> = x
        .as_slice()
        .iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
        .collect();
    Tensor::from_vec(data, x.shape().clone())
}

fn load_matrix(
    weights: &WeightSet,
    name: &str,
    expected_rows: u32,
    expected_cols: u32,
) -> Result<Tensor> {
    let view = match weights.tensor(name) {
        Ok(v) => v,
        Err(crate::PhalanxError::Weights(crate::weights::WeightsError::TensorNotFound {
            ..
        })) => {
            return Err(LayersError::InvalidWeightShape {
                name: "ffn_gate.weight",
                reason: format!("tensor '{name}' not found in weight set"),
            }
            .into());
        }
        Err(other) => return Err(other),
    };

    let dims = squeeze_trailing_ones(&view.info.dimensions);
    if dims.len() != 2 {
        return Err(LayersError::InvalidWeightShape {
            name: "ffn_gate.weight",
            reason: format!(
                "expected rank-2 for '{name}', got {:?}",
                view.info.dimensions
            ),
        }
        .into());
    }

    // ggml stores ne[0] as innermost → bytes are row-major [ne1, ne0] = [rows, cols]
    // when ne0=cols (in_features) and ne1=rows (out_features), matching Spec (out, in).
    let ne0 = usize_dim(dims[0], name)?;
    let ne1 = usize_dim(dims[1], name)?;
    let rows = usize::try_from(expected_rows).map_err(|_| LayersError::ConfigMismatch {
        name: "ffn_gate.weight",
        reason: format!("expected_rows {expected_rows} does not fit usize"),
    })?;
    let cols = usize::try_from(expected_cols).map_err(|_| LayersError::ConfigMismatch {
        name: "ffn_gate.weight",
        reason: format!("expected_cols {expected_cols} does not fit usize"),
    })?;

    // GGUF Llama FFN: dimensions reported as [n_embd, n_ff] for gate/up (ne0=D, ne1=I)
    // → reinterpret as [I, D]. Down is [n_ff, n_embd] → [D, I].
    if ne0 == cols && ne1 == rows {
        return view.to_f32_tensor()?.into_shape([rows, cols]);
    }
    if ne0 == rows && ne1 == cols {
        // Already logical (out, in).
        return view.to_f32_tensor()?.into_shape([rows, cols]);
    }
    Err(LayersError::ConfigMismatch {
        name: "ffn_gate.weight",
        reason: format!("'{name}' dims {dims:?} incompatible with expected [{rows}, {cols}]"),
    }
    .into())
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
            name: "ffn_gate.weight",
            reason: format!("{name}: dimension {value} does not fit usize"),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    fn toy_weights(hidden: usize, intermediate: usize) -> (Tensor, Tensor, Tensor) {
        #[allow(clippy::cast_precision_loss)]
        let gate = Tensor::from_vec(
            (0..intermediate * hidden)
                .map(|i| (i % 7) as f32 * 0.01 - 0.03)
                .collect(),
            Shape::new([intermediate, hidden]).unwrap(),
        )
        .unwrap();
        #[allow(clippy::cast_precision_loss)]
        let up = Tensor::from_vec(
            (0..intermediate * hidden)
                .map(|i| (i % 5) as f32 * 0.02 - 0.04)
                .collect(),
            Shape::new([intermediate, hidden]).unwrap(),
        )
        .unwrap();
        #[allow(clippy::cast_precision_loss)]
        let down = Tensor::from_vec(
            (0..hidden * intermediate)
                .map(|i| (i % 3) as f32 * 0.01 - 0.01)
                .collect(),
            Shape::new([hidden, intermediate]).unwrap(),
        )
        .unwrap();
        (gate, up, down)
    }

    #[test]
    fn preserves_batch_seq_shape() {
        let (gate, up, down) = toy_weights(8, 16);
        let ffn = SwiGlu::from_tensors(gate, up, down).unwrap();
        let input = Tensor::ones([2, 4, 8]).unwrap();
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.shape().as_slice(), &[2, 4, 8]);
    }

    #[test]
    fn silu_matches_formula() {
        let input = Tensor::from_vec(vec![-2.0, 0.0, 1.0], Shape::new([1, 3]).unwrap()).unwrap();
        let output = silu(&input).unwrap();
        let expected: Vec<f32> = input
            .as_slice()
            .iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        for (got, want) in output.as_slice().iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_wrong_last_dim() {
        let (gate, up, down) = toy_weights(8, 16);
        let ffn = SwiGlu::from_tensors(gate, up, down).unwrap();
        let err = ffn.forward(&Tensor::ones([2, 4]).unwrap()).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::InvalidActivationShape { .. })
        ));
    }

    #[test]
    fn weight_name_helpers() {
        assert_eq!(ffn_gate_weight_name(0), "blk.0.ffn_gate.weight");
        assert_eq!(ffn_up_weight_name(2), "blk.2.ffn_up.weight");
        assert_eq!(ffn_down_weight_name(1), "blk.1.ffn_down.weight");
    }
}
