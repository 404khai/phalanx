//! Rotary positional embeddings (`RoPE`).
//!
//! # Why `RoPE`
//!
//! Absolute position embeddings add a fixed vector per index. `RoPE` instead
//! **rotates** pairs of features in Q/K by an angle that grows with position,
//! so relative offsets appear as relative rotations in attention scores.
//!
//! Paper: [RoFormer](https://arxiv.org/abs/2104.09864). Llama uses the same
//! adjacent-pair rotation with θ = `rope.freq_base` (default `10000`).
//!
//! # Layout
//!
//! For rotary dimension `d` (even) and position `m`:
//!
//! ```text
//! θ_i = freq_base ^ (-2i / d)     i = 0 .. d/2 - 1
//!
//! [x_{2i}']     [ cos(m θ_i)  -sin(m θ_i) ] [x_{2i}]
//! [x_{2i+1}'] = [ sin(m θ_i)   cos(m θ_i) ] [x_{2i+1}]
//! ```
//!
//! Activations are `[seq, n_heads, head_dim]` (or `[seq, head_dim]`). Only the
//! first `rope.dimension_count` features rotate; any remaining head dims pass
//! through (partial `RoPE`).
//!
//! # Scaling
//!
//! When GGUF declares linear `rope.scaling` / legacy `rope.scale`, positions
//! are divided by the factor before looking up angles (`m' = m / factor`).
//! `YaRN` and `NTK` variants are deferred.

use tracing::debug;

use super::error::LayersError;
use crate::errors::Result;
use crate::model::{ModelConfig, RopeConfig};
use crate::tensor::{Shape, Tensor};

/// Precomputed rotary embedding tables and apply kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    /// Features rotated per head (`rope.dimension_count`, even).
    rotary_dim: usize,
    /// Full per-head width (`attention.key_length`); may exceed `rotary_dim`.
    head_dim: usize,
    /// Base frequency θ.
    freq_base: f32,
    /// Linear position scale (`1.0` = none). Effective pos = `m / scale`.
    scale: f32,
    /// Cached absolute positions: `0 .. max_position`.
    max_position: usize,
    /// `cos[pos * n_pairs + pair]`.
    cos: Vec<f32>,
    /// `sin[pos * n_pairs + pair]`.
    sin: Vec<f32>,
}

impl Rope {
    /// Build cos/sin caches from a validated [`ModelConfig`].
    ///
    /// Cache length is `context_length` positions.
    ///
    /// # Errors
    ///
    /// Returns [`LayersError`] when rope dims are invalid or allocations overflow.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        let max_position = usize::try_from(config.context_length).map_err(|_| {
            LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!(
                    "context_length {} does not fit usize",
                    config.context_length
                ),
            }
        })?;
        let head_dim = usize::try_from(config.attention.key_length).map_err(|_| {
            LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!(
                    "key_length {} does not fit usize",
                    config.attention.key_length
                ),
            }
        })?;
        Self::from_rope_config(&config.rope, head_dim, max_position)
    }

    /// Build from explicit [`RopeConfig`] + head width + cache length.
    ///
    /// # Errors
    ///
    /// Returns [`LayersError`] on invalid dims / empty cache.
    pub fn from_rope_config(
        rope: &RopeConfig,
        head_dim: usize,
        max_position: usize,
    ) -> Result<Self> {
        let rotary_dim = usize::try_from(rope.dimension_count).map_err(|_| {
            LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!(
                    "rope.dimension_count {} does not fit usize",
                    rope.dimension_count
                ),
            }
        })?;
        if max_position == 0 {
            return Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: "max_position (context length) must be > 0".into(),
            }
            .into());
        }
        if head_dim == 0 {
            return Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: "head_dim must be > 0".into(),
            }
            .into());
        }
        if rotary_dim == 0 || rotary_dim % 2 != 0 {
            return Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!("rope.dimension_count must be even and > 0, got {rotary_dim}"),
            }
            .into());
        }
        if rotary_dim > head_dim {
            return Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!(
                    "rope.dimension_count ({rotary_dim}) exceeds head_dim ({head_dim})"
                ),
            }
            .into());
        }
        if !rope.freq_base.is_finite() || rope.freq_base <= 0.0 {
            return Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: "rope.freq_base must be finite and > 0".into(),
            }
            .into());
        }

        let scale = linear_scale(rope)?;
        let n_pairs = rotary_dim / 2;
        let mut cos = vec![0.0; max_position.saturating_mul(n_pairs)];
        let mut sin = vec![0.0; max_position.saturating_mul(n_pairs)];

        // inv_freq[i] = freq_base ^ (-2i / rotary_dim)
        // Rotary dims are tiny (≪ 2^24), so f32 exponents are exact here.
        #[allow(clippy::cast_precision_loss)]
        let inv_freq: Vec<f32> = (0..n_pairs)
            .map(|i| {
                let exponent = (2 * i) as f32 / rotary_dim as f32;
                rope.freq_base.powf(-exponent)
            })
            .collect();

        for pos in 0..max_position {
            // Linear scale: angles use m' = m / scale (scale ≥ 1 stretches context).
            #[allow(clippy::cast_precision_loss)]
            let m = (pos as f32) / scale;
            for (pair, &freq) in inv_freq.iter().enumerate() {
                let angle = m * freq;
                let idx = pos * n_pairs + pair;
                cos[idx] = angle.cos();
                sin[idx] = angle.sin();
            }
        }

        debug!(
            rotary_dim,
            head_dim,
            max_position,
            freq_base = rope.freq_base,
            scale,
            "built RoPE cos/sin cache"
        );

        Ok(Self {
            rotary_dim,
            head_dim,
            freq_base: rope.freq_base,
            scale,
            max_position,
            cos,
            sin,
        })
    }

    /// Rotary feature count (`rope.dimension_count`).
    #[must_use]
    pub const fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Full head width this cache expects.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of cached absolute positions.
    #[must_use]
    pub const fn max_position(&self) -> usize {
        self.max_position
    }

    /// Base frequency θ.
    #[must_use]
    pub const fn freq_base(&self) -> f32 {
        self.freq_base
    }

    /// Linear position scale factor.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Apply `RoPE` to activations.
    ///
    /// Accepted shapes:
    /// - `[seq, head_dim]` — single head / packed vector
    /// - `[seq, n_heads, head_dim]` — multi-head Q or K
    ///
    /// `position_offset` is the absolute index of the first sequence step
    /// (0 for prefill start; `past_len` during decode).
    ///
    /// # Errors
    ///
    /// Shape mismatches or positions past the cache.
    pub fn forward(&self, input: &Tensor, position_offset: usize) -> Result<Tensor> {
        let shape = input.shape().as_slice();
        match shape.len() {
            2 => {
                let seq = shape[0];
                let dim = shape[1];
                self.validate_head_dim(dim)?;
                self.validate_positions(position_offset, seq)?;
                let mut out = input.as_slice().to_vec();
                for s in 0..seq {
                    let pos = position_offset + s;
                    let row = &mut out[s * dim..s * dim + dim];
                    self.rotate_inplace(row, pos);
                }
                Tensor::from_vec(out, Shape::new([seq, dim])?)
            }
            3 => {
                let seq = shape[0];
                let n_heads = shape[1];
                let dim = shape[2];
                self.validate_head_dim(dim)?;
                self.validate_positions(position_offset, seq)?;
                let mut out = input.as_slice().to_vec();
                let stride = n_heads * dim;
                for s in 0..seq {
                    let pos = position_offset + s;
                    for h in 0..n_heads {
                        let base = s * stride + h * dim;
                        let row = &mut out[base..base + dim];
                        self.rotate_inplace(row, pos);
                    }
                }
                Tensor::from_vec(out, Shape::new([seq, n_heads, dim])?)
            }
            rank => Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!(
                    "expected rank 2 [seq, head_dim] or rank 3 [seq, heads, head_dim], got rank {rank} ({shape:?})"
                ),
            }
            .into()),
        }
    }

    fn validate_head_dim(&self, dim: usize) -> Result<()> {
        if dim != self.head_dim {
            return Err(LayersError::InvalidActivationShape {
                op: "rope",
                reason: format!("last dim {dim} != configured head_dim {}", self.head_dim),
            }
            .into());
        }
        Ok(())
    }

    fn validate_positions(&self, offset: usize, seq: usize) -> Result<()> {
        if seq == 0 {
            return Ok(());
        }
        let last =
            offset
                .checked_add(seq - 1)
                .ok_or_else(|| LayersError::InvalidActivationShape {
                    op: "rope",
                    reason: "position offset + seq overflowed usize".into(),
                })?;
        if last >= self.max_position {
            return Err(LayersError::RopePositionOutOfRange {
                position: last,
                max_position: self.max_position,
            }
            .into());
        }
        Ok(())
    }

    fn rotate_inplace(&self, row: &mut [f32], position: usize) {
        debug_assert_eq!(row.len(), self.head_dim);
        debug_assert!(position < self.max_position);
        let n_pairs = self.rotary_dim / 2;
        let base = position * n_pairs;
        for pair in 0..n_pairs {
            let i = pair * 2;
            let cos = self.cos[base + pair];
            let sin = self.sin[base + pair];
            let x0 = row[i];
            let x1 = row[i + 1];
            row[i] = x0 * cos - x1 * sin;
            row[i + 1] = x0 * sin + x1 * cos;
        }
        // Features beyond rotary_dim are left unchanged (partial RoPE).
    }
}

fn linear_scale(rope: &RopeConfig) -> Result<f32> {
    let Some(scaling) = &rope.scaling else {
        return Ok(1.0);
    };
    let kind = scaling.scaling_type.as_str();
    // Accept common aliases used by converters.
    if !matches!(kind, "linear" | "Linear" | "") {
        // Capture metadata but refuse silent wrong math for yarn/ntk.
        return Err(LayersError::InvalidActivationShape {
            op: "rope",
            reason: format!(
                "unsupported rope.scaling.type '{kind}' (Phase 8 supports linear only)"
            ),
        }
        .into());
    }
    let factor = scaling.factor.unwrap_or(1.0);
    if !factor.is_finite() || factor <= 0.0 {
        return Err(LayersError::InvalidActivationShape {
            op: "rope",
            reason: "rope.scaling.factor must be finite and > 0".into(),
        }
        .into());
    }
    Ok(factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AttentionConfig, ModelConfig, RopeScaling};

    fn toy_config(ctx: u32, head_dim: u32, rope_dim: u32) -> ModelConfig {
        ModelConfig::from_parts(ModelConfig {
            architecture: crate::model::Architecture::Llama,
            name: None,
            vocab_size: None,
            context_length: ctx,
            embedding_length: head_dim * 2,
            feed_forward_length: head_dim * 4,
            block_count: 1,
            attention: AttentionConfig {
                head_count: 2,
                head_count_kv: 2,
                key_length: head_dim,
                value_length: head_dim,
            },
            rope: RopeConfig {
                dimension_count: rope_dim,
                freq_base: 10_000.0,
                scaling: None,
            },
            rms_norm_eps: 1e-5,
        })
        .unwrap()
    }

    #[test]
    fn position_zero_is_near_identity() {
        let rope = Rope::from_config(&toy_config(16, 4, 4)).unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new([1, 4]).unwrap()).unwrap();
        let y = rope.forward(&x, 0).unwrap();
        for (a, b) in x.as_slice().iter().zip(y.as_slice()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rotation_preserves_l2_norm() {
        let rope = Rope::from_config(&toy_config(32, 8, 8)).unwrap();
        let x = Tensor::from_vec(
            (0..24)
                .map(|i| f32::from(u8::try_from(i).unwrap()) * 0.1)
                .collect(),
            Shape::new([3, 8]).unwrap(),
        )
        .unwrap();
        let y = rope.forward(&x, 5).unwrap();
        for s in 0..3 {
            let mut nx = 0.0f32;
            let mut ny = 0.0f32;
            for d in 0..8 {
                let a = x.get(&[s, d]).unwrap();
                let b = y.get(&[s, d]).unwrap();
                nx += a * a;
                ny += b * b;
            }
            assert!(
                (nx.sqrt() - ny.sqrt()).abs() < 1e-5,
                "norm {nx} vs {ny} at step {s}"
            );
        }
    }

    #[test]
    fn multi_head_shape_round_trips_dims() {
        let rope = Rope::from_config(&toy_config(8, 4, 4)).unwrap();
        let x = Tensor::ones([2, 3, 4]).unwrap();
        let y = rope.forward(&x, 0).unwrap();
        assert_eq!(y.shape().as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn different_positions_change_output() {
        let rope = Rope::from_config(&toy_config(64, 4, 4)).unwrap();
        let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new([1, 4]).unwrap()).unwrap();
        let y0 = rope.forward(&x, 0).unwrap();
        let y7 = rope.forward(&x, 7).unwrap();
        let diff: f32 = y0
            .as_slice()
            .iter()
            .zip(y7.as_slice())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3, "expected positions to differ, diff={diff}");
    }

    #[test]
    fn partial_rope_leaves_tail_dims() {
        // Rotate first 2 dims only; last 2 must be unchanged.
        let rope = Rope::from_rope_config(
            &RopeConfig {
                dimension_count: 2,
                freq_base: 10_000.0,
                scaling: None,
            },
            4,
            16,
        )
        .unwrap();
        let x = Tensor::from_vec(vec![1.0, 2.0, 7.0, 8.0], Shape::new([1, 4]).unwrap()).unwrap();
        let y = rope.forward(&x, 3).unwrap();
        assert!((y.as_slice()[2] - 7.0).abs() < 1e-6);
        assert!((y.as_slice()[3] - 8.0).abs() < 1e-6);
        // First pair should move at position 3.
        assert!((y.as_slice()[0] - 1.0).abs() > 1e-4 || (y.as_slice()[1] - 2.0).abs() > 1e-4);
    }

    #[test]
    fn linear_scale_stretches_positions() {
        let base = Rope::from_rope_config(
            &RopeConfig {
                dimension_count: 4,
                freq_base: 10_000.0,
                scaling: None,
            },
            4,
            32,
        )
        .unwrap();
        let scaled = Rope::from_rope_config(
            &RopeConfig {
                dimension_count: 4,
                freq_base: 10_000.0,
                scaling: Some(RopeScaling {
                    scaling_type: "linear".into(),
                    factor: Some(2.0),
                }),
            },
            4,
            32,
        )
        .unwrap();
        let x = Tensor::from_vec(vec![1.0, 0.5, -0.5, 0.25], Shape::new([1, 4]).unwrap()).unwrap();
        // Scaled pos 10 uses angle of unscaled pos 5.
        let a = base.forward(&x, 5).unwrap();
        let b = scaled.forward(&x, 10).unwrap();
        for (u, v) in a.as_slice().iter().zip(b.as_slice()) {
            assert!((u - v).abs() < 1e-5, "{u} vs {v}");
        }
    }

    #[test]
    fn rejects_position_past_cache() {
        let rope = Rope::from_config(&toy_config(4, 4, 4)).unwrap();
        let x = Tensor::ones([1, 4]).unwrap();
        let err = rope.forward(&x, 4).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::RopePositionOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_yarn_scaling() {
        let err = Rope::from_rope_config(
            &RopeConfig {
                dimension_count: 4,
                freq_base: 10_000.0,
                scaling: Some(RopeScaling {
                    scaling_type: "yarn".into(),
                    factor: Some(2.0),
                }),
            },
            4,
            8,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::InvalidActivationShape { .. })
        ));
    }
}
