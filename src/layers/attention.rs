//! Causal multi-head / grouped-query attention.
//!
//! # Formula (Odyssey Spec v1.0.0)
//!
//! ```text
//! Q = x W_Qᵀ,  K = x W_Kᵀ,  V = x W_Vᵀ
//! Attn(Q,K,V) = softmax(Q Kᵀ / √d + M) V
//! output = Attn · W_Oᵀ
//! ```
//!
//! Causal mask `M`: `0` when `t ≤ s`, else `-∞`.
//!
//! GQA: `H / H_kv` query heads share one KV head (LLaMA-style).
//! Optional [`Rope`] rotates Q/K after the head reshape (Spec requirement for
//! the production decoder path).
//!
//! # References
//!
//! - Attention Is All You Need
//! - GQA (Ainslie et al.)
//! - `LLaMA` / Llama 2 architecture

use tracing::debug;

use super::error::LayersError;
use super::rope::Rope;
use crate::errors::Result;
use crate::model::ModelConfig;
use crate::tensor::{Shape, Tensor};
use crate::weights::WeightSet;

/// GGUF suffix for query projection.
pub const ATTN_Q_WEIGHT_SUFFIX: &str = ".attn_q.weight";
/// GGUF suffix for key projection.
pub const ATTN_K_WEIGHT_SUFFIX: &str = ".attn_k.weight";
/// GGUF suffix for value projection.
pub const ATTN_V_WEIGHT_SUFFIX: &str = ".attn_v.weight";
/// GGUF suffix for output projection.
pub const ATTN_OUTPUT_WEIGHT_SUFFIX: &str = ".attn_output.weight";
/// Prefix shared with other block tensors.
pub const ATTN_WEIGHT_PREFIX: &str = "blk.";

/// Build GGUF name for layer `i` query weight.
#[must_use]
pub fn attn_q_weight_name(layer: usize) -> String {
    format!("{ATTN_WEIGHT_PREFIX}{layer}{ATTN_Q_WEIGHT_SUFFIX}")
}

/// Build GGUF name for layer `i` key weight.
#[must_use]
pub fn attn_k_weight_name(layer: usize) -> String {
    format!("{ATTN_WEIGHT_PREFIX}{layer}{ATTN_K_WEIGHT_SUFFIX}")
}

/// Build GGUF name for layer `i` value weight.
#[must_use]
pub fn attn_v_weight_name(layer: usize) -> String {
    format!("{ATTN_WEIGHT_PREFIX}{layer}{ATTN_V_WEIGHT_SUFFIX}")
}

/// Build GGUF name for layer `i` output weight.
#[must_use]
pub fn attn_output_weight_name(layer: usize) -> String {
    format!("{ATTN_WEIGHT_PREFIX}{layer}{ATTN_OUTPUT_WEIGHT_SUFFIX}")
}

/// Causal self-attention with optional GQA.
#[derive(Debug, Clone, PartialEq)]
pub struct Attention {
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    /// Load `attn_q/k/v/output` for one block from a [`WeightSet`].
    ///
    /// # Errors
    ///
    /// Missing tensors, bad shapes, or config mismatch.
    pub fn from_weights(weights: &WeightSet, layer: usize, config: &ModelConfig) -> Result<Self> {
        let hidden = usize_from_u32(config.embedding_length, "embedding_length")?;
        let num_heads = usize_from_u32(config.attention.head_count, "head_count")?;
        let num_kv = usize_from_u32(config.attention.head_count_kv, "head_count_kv")?;
        let head_dim = usize_from_u32(config.attention.key_length, "key_length")?;
        let q_out = num_heads * head_dim;
        let kv_out = num_kv * head_dim;

        let w_q = load_matrix(weights, &attn_q_weight_name(layer), q_out, hidden)?;
        let w_k = load_matrix(weights, &attn_k_weight_name(layer), kv_out, hidden)?;
        let w_v = load_matrix(weights, &attn_v_weight_name(layer), kv_out, hidden)?;
        let w_o = load_matrix(weights, &attn_output_weight_name(layer), hidden, q_out)?;
        Self::from_tensors(w_q, w_k, w_v, w_o, num_heads, num_kv, head_dim)
    }

    /// Build from materialised Spec-shaped weights (validators / tests).
    ///
    /// Weight shapes: `w_q/w_o` `[H·d, D]` / `[D, H·d]`; `w_k/w_v` `[H_kv·d, D]`.
    ///
    /// # Errors
    ///
    /// Rank/dim mismatches.
    pub fn from_tensors(
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        w_o: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_q.weight",
                reason: "num_heads, num_kv_heads, and head_dim must be > 0".into(),
            }
            .into());
        }
        if num_heads % num_kv_heads != 0 {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_q.weight",
                reason: format!(
                    "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                ),
            }
            .into());
        }

        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        let q = w_q.shape().as_slice();
        let k = w_k.shape().as_slice();
        let v = w_v.shape().as_slice();
        let o = w_o.shape().as_slice();
        if q.len() != 2 || k.len() != 2 || v.len() != 2 || o.len() != 2 {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_q.weight",
                reason: format!("expected rank-2 weights, got q={q:?} k={k:?} v={v:?} o={o:?}"),
            }
            .into());
        }
        let hidden = q[1];
        if q[0] != q_out {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_q.weight",
                reason: format!("expected [{q_out}, {hidden}], got {q:?}"),
            }
            .into());
        }
        if k[0] != kv_out || k[1] != hidden {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_k.weight",
                reason: format!("expected [{kv_out}, {hidden}], got {k:?}"),
            }
            .into());
        }
        if v[0] != kv_out || v[1] != hidden {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_v.weight",
                reason: format!("expected [{kv_out}, {hidden}], got {v:?}"),
            }
            .into());
        }
        if o[0] != hidden || o[1] != q_out {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_output.weight",
                reason: format!("expected [{hidden}, {q_out}], got {o:?}"),
            }
            .into());
        }

        debug!(hidden, num_heads, num_kv_heads, head_dim, "built Attention");
        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            hidden_size: hidden,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Hidden / embedding length `D`.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Query head count `H`.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// KV head count `H_kv`.
    #[must_use]
    pub const fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Per-head dimension `d`.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Query projection weight `[H·d, D]`.
    #[must_use]
    pub fn w_q(&self) -> &Tensor {
        &self.w_q
    }

    /// Key projection weight `[H_kv·d, D]`.
    #[must_use]
    pub fn w_k(&self) -> &Tensor {
        &self.w_k
    }

    /// Value projection weight `[H_kv·d, D]`.
    #[must_use]
    pub fn w_v(&self) -> &Tensor {
        &self.w_v
    }

    /// Output projection weight `[D, H·d]`.
    #[must_use]
    pub fn w_o(&self) -> &Tensor {
        &self.w_o
    }

    /// Apply causal attention.
    ///
    /// Accepted shapes: `[seq, D]` or `[batch, seq, D]`.
    /// When `rope` is `Some`, Q/K are rotated at `position_offset`.
    ///
    /// # Errors
    ///
    /// Shape mismatches or rope / matmul failures.
    pub fn forward(
        &self,
        input: &Tensor,
        rope: Option<&Rope>,
        position_offset: usize,
    ) -> Result<Tensor> {
        let shape = input.shape().as_slice();
        if shape.len() < 2 {
            return Err(LayersError::InvalidActivationShape {
                op: "attention",
                reason: format!("expected rank >= 2 with last dim D, got {shape:?}"),
            }
            .into());
        }
        let dim = shape[shape.len() - 1];
        if dim != self.hidden_size {
            return Err(LayersError::InvalidActivationShape {
                op: "attention",
                reason: format!(
                    "last dim {dim} != configured hidden_size {}",
                    self.hidden_size
                ),
            }
            .into());
        }

        let (batch, seq) = if shape.len() == 2 {
            (1usize, shape[0])
        } else if shape.len() == 3 {
            (shape[0], shape[1])
        } else {
            return Err(LayersError::InvalidActivationShape {
                op: "attention",
                reason: format!("expected rank 2 or 3, got rank {} ({shape:?})", shape.len()),
            }
            .into());
        };

        let flat = input.reshape([batch * seq, self.hidden_size])?;
        let q = linear(&flat, &self.w_q)?; // [B*S, H*d]
        let k = linear(&flat, &self.w_k)?;
        let v = linear(&flat, &self.w_v)?;

        let mut q_heads = reshape_heads(q.as_slice(), batch, seq, self.num_heads, self.head_dim);
        let mut k_heads = reshape_heads(k.as_slice(), batch, seq, self.num_kv_heads, self.head_dim);
        let v_heads = reshape_heads(v.as_slice(), batch, seq, self.num_kv_heads, self.head_dim);

        if let Some(rope) = rope {
            apply_rope_batched(
                &mut q_heads,
                rope,
                batch,
                seq,
                self.num_heads,
                self.head_dim,
                position_offset,
            )?;
            apply_rope_batched(
                &mut k_heads,
                rope,
                batch,
                seq,
                self.num_kv_heads,
                self.head_dim,
                position_offset,
            )?;
        }

        let k_exp = expand_kv(
            &k_heads,
            batch,
            seq,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        );
        let v_exp = expand_kv(
            &v_heads,
            batch,
            seq,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        );

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let ctx = attention_sdpa(
            &q_heads,
            &k_exp,
            &v_exp,
            batch,
            self.num_heads,
            seq,
            self.head_dim,
            scale,
        );

        // ctx: [B, H, S, d] → merge → [B*S, H*d]
        let merged = merge_heads(&ctx, batch, self.num_heads, seq, self.head_dim);
        let merged_t = Tensor::from_vec(
            merged,
            Shape::new([batch * seq, self.num_heads * self.head_dim])?,
        )?;
        let out_flat = linear(&merged_t, &self.w_o)?;
        out_flat.into_shape(shape.to_vec())
    }
}

/// `nn.Linear`-style product: `x @ Wᵀ` for `W` shaped `[out, in]`.
fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let weight_t = weight.transpose()?;
    x.matmul(&weight_t)
}

/// Row-major `[B*S, H*d]` → `[B, H, S, d]` flat buffer.
fn reshape_heads(
    data: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * seq * head_dim];
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                for d in 0..head_dim {
                    let src = ((b * seq + s) * heads + h) * head_dim + d;
                    let dst = ((b * heads + h) * seq + s) * head_dim + d;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

fn merge_heads(data: &[f32], batch: usize, heads: usize, seq: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq * heads * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq {
                for d in 0..head_dim {
                    let src = ((b * heads + h) * seq + s) * head_dim + d;
                    let dst = ((b * seq + s) * heads + h) * head_dim + d;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

fn expand_kv(
    kv: &[f32],
    batch: usize,
    seq: usize,
    num_heads: usize,
    num_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    if num_heads == num_kv {
        return kv.to_vec();
    }
    let groups = num_heads / num_kv;
    let mut out = vec![0.0f32; batch * num_heads * seq * head_dim];
    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / groups;
            for s in 0..seq {
                for d in 0..head_dim {
                    let src = ((b * num_kv + kv_h) * seq + s) * head_dim + d;
                    let dst = ((b * num_heads + h) * seq + s) * head_dim + d;
                    out[dst] = kv[src];
                }
            }
        }
    }
    out
}

fn apply_rope_batched(
    heads: &mut [f32],
    rope: &Rope,
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: usize,
) -> Result<()> {
    // Phalanx Rope expects [seq, heads, head_dim] per batch item.
    for b in 0..batch {
        let mut seq_major = vec![0.0f32; seq * num_heads * head_dim];
        for h in 0..num_heads {
            for s in 0..seq {
                for d in 0..head_dim {
                    let src = ((b * num_heads + h) * seq + s) * head_dim + d;
                    let dst = (s * num_heads + h) * head_dim + d;
                    seq_major[dst] = heads[src];
                }
            }
        }
        let t = Tensor::from_vec(seq_major, Shape::new([seq, num_heads, head_dim])?)?;
        let rotated = rope.forward(&t, position_offset)?;
        let r = rotated.as_slice();
        for h in 0..num_heads {
            for s in 0..seq {
                for d in 0..head_dim {
                    let src = (s * num_heads + h) * head_dim + d;
                    let dst = ((b * num_heads + h) * seq + s) * head_dim + d;
                    heads[dst] = r[src];
                }
            }
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]
fn attention_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    scale: f64,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * seq * head_dim];
    let mut scores = vec![0.0f32; seq * seq];
    let mut weights = vec![0.0f32; seq * seq];

    for b in 0..batch {
        for h in 0..heads {
            let q_base = (b * heads + h) * seq * head_dim;
            let k_base = q_base;
            let v_base = q_base;

            // scores[s, t] = (q[s] · k[t]) * scale
            for s in 0..seq {
                for t in 0..seq {
                    let mut acc = 0.0f64;
                    let qb = q_base + s * head_dim;
                    let kb = k_base + t * head_dim;
                    for d in 0..head_dim {
                        acc += f64::from(q[qb + d]) * f64::from(k[kb + d]);
                    }
                    scores[s * seq + t] = (acc * scale) as f32;
                }
            }

            // Causal mask + stable softmax over keys for each query row.
            for s in 0..seq {
                let row = &mut scores[s * seq..(s + 1) * seq];
                for t in (s + 1)..seq {
                    row[t] = f32::NEG_INFINITY;
                }
                let mut max_v = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > max_v {
                        max_v = v;
                    }
                }
                if !max_v.is_finite() {
                    // All masked — should not happen for causal s>=0; zero row.
                    for t in 0..seq {
                        weights[s * seq + t] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f64;
                for t in 0..seq {
                    let e = if row[t].is_finite() {
                        (f64::from(row[t] - max_v)).exp()
                    } else {
                        0.0
                    };
                    weights[s * seq + t] = e as f32;
                    sum += e;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for t in 0..seq {
                    weights[s * seq + t] = (f64::from(weights[s * seq + t]) * inv) as f32;
                }
            }

            // context[s] = Σ_t weights[s,t] * v[t]
            for s in 0..seq {
                for d in 0..head_dim {
                    let mut acc = 0.0f64;
                    for t in 0..seq {
                        acc += f64::from(weights[s * seq + t])
                            * f64::from(v[v_base + t * head_dim + d]);
                    }
                    out[q_base + s * head_dim + d] = acc as f32;
                }
            }
        }
    }
    out
}

fn load_matrix(
    weights: &WeightSet,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Tensor> {
    let view = match weights.tensor(name) {
        Ok(v) => v,
        Err(crate::PhalanxError::Weights(crate::weights::WeightsError::TensorNotFound {
            ..
        })) => {
            return Err(LayersError::InvalidWeightShape {
                name: "attn_q.weight",
                reason: format!("tensor '{name}' not found in weight set"),
            }
            .into());
        }
        Err(other) => return Err(other),
    };

    let dims = squeeze_trailing_ones(&view.info.dimensions);
    if dims.len() != 2 {
        return Err(LayersError::InvalidWeightShape {
            name: "attn_q.weight",
            reason: format!(
                "expected rank-2 for '{name}', got {:?}",
                view.info.dimensions
            ),
        }
        .into());
    }

    let ne0 = usize_dim(dims[0], name)?;
    let ne1 = usize_dim(dims[1], name)?;
    let rows = expected_rows;
    let cols = expected_cols;

    if ne0 == cols && ne1 == rows {
        return view.to_f32_tensor()?.into_shape([rows, cols]);
    }
    if ne0 == rows && ne1 == cols {
        return view.to_f32_tensor()?.into_shape([rows, cols]);
    }
    Err(LayersError::ConfigMismatch {
        name: "attn_q.weight",
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
            name: "attn_q.weight",
            reason: format!("{name}: dimension {value} does not fit usize"),
        }
        .into()
    })
}

fn usize_from_u32(value: u32, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        LayersError::ConfigMismatch {
            name: "attention",
            reason: format!("{name} {value} does not fit usize"),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_attn(heads: usize, kv: usize, head_dim: usize) -> Attention {
        let hidden = heads * head_dim;
        let q_out = hidden;
        let kv_out = kv * head_dim;
        #[allow(clippy::cast_precision_loss)]
        let w_q = Tensor::from_vec(
            (0..q_out * hidden)
                .map(|i| (i % 7) as f32 * 0.01 - 0.03)
                .collect(),
            Shape::new([q_out, hidden]).unwrap(),
        )
        .unwrap();
        #[allow(clippy::cast_precision_loss)]
        let w_k = Tensor::from_vec(
            (0..kv_out * hidden)
                .map(|i| (i % 5) as f32 * 0.02 - 0.04)
                .collect(),
            Shape::new([kv_out, hidden]).unwrap(),
        )
        .unwrap();
        #[allow(clippy::cast_precision_loss)]
        let w_v = Tensor::from_vec(
            (0..kv_out * hidden)
                .map(|i| (i % 3) as f32 * 0.01 - 0.01)
                .collect(),
            Shape::new([kv_out, hidden]).unwrap(),
        )
        .unwrap();
        #[allow(clippy::cast_precision_loss)]
        let w_o = Tensor::from_vec(
            (0..hidden * q_out)
                .map(|i| (i % 11) as f32 * 0.005 - 0.02)
                .collect(),
            Shape::new([hidden, q_out]).unwrap(),
        )
        .unwrap();
        Attention::from_tensors(w_q, w_k, w_v, w_o, heads, kv, head_dim).unwrap()
    }

    #[test]
    fn preserves_batch_seq_shape_gqa() {
        let attn = toy_attn(4, 2, 8);
        let input = Tensor::ones([2, 5, 32]).unwrap();
        let output = attn.forward(&input, None, 0).unwrap();
        assert_eq!(output.shape().as_slice(), &[2, 5, 32]);
    }

    #[test]
    fn preserves_seq_shape_mha() {
        let attn = toy_attn(4, 4, 8);
        let input = Tensor::ones([6, 32]).unwrap();
        let output = attn.forward(&input, None, 0).unwrap();
        assert_eq!(output.shape().as_slice(), &[6, 32]);
    }

    #[test]
    fn rejects_wrong_last_dim() {
        let attn = toy_attn(4, 2, 8);
        let err = attn
            .forward(&Tensor::ones([2, 4]).unwrap(), None, 0)
            .unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Layers(LayersError::InvalidActivationShape { .. })
        ));
    }

    #[test]
    fn weight_name_helpers() {
        assert_eq!(attn_q_weight_name(0), "blk.0.attn_q.weight");
        assert_eq!(attn_k_weight_name(1), "blk.1.attn_k.weight");
        assert_eq!(attn_v_weight_name(2), "blk.2.attn_v.weight");
        assert_eq!(attn_output_weight_name(3), "blk.3.attn_output.weight");
    }

    #[test]
    fn causal_softmax_row_sums() {
        // Smoke: forward is finite.
        let attn = toy_attn(2, 1, 4);
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32) * 0.01).collect();
        let input = Tensor::from_vec(data, Shape::new([2, 3, 8]).unwrap()).unwrap();
        let out = attn.forward(&input, None, 0).unwrap();
        assert!(out.as_slice().iter().all(|v| v.is_finite()));
    }
}
