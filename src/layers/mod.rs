//! Decoder layer kernels and weight-bound tables.
//!
//! Phase 7–10: embedding, `RoPE`, `RMSNorm`, `SwiGlu`. Phase 11 adds [`Attention`].
//!
//! # Module map
//!
//! - [`embedding`] — [`EmbeddingTable`] gather from `token_embd.weight`
//! - [`rope`] — rotary positional embeddings for Q/K
//! - [`rmsnorm`] — Llama-style `RMSNorm` (γ ⊙ x / RMS(x))
//! - [`swiglu`] — Llama-style `SwiGLU` feed-forward
//! - [`attention`] — causal multi-head / grouped-query attention
//! - [`error`] — [`LayersError`]

mod attention;
mod embedding;
mod error;
mod rmsnorm;
mod rope;
mod swiglu;

pub use attention::{
    ATTN_K_WEIGHT_SUFFIX, ATTN_OUTPUT_WEIGHT_SUFFIX, ATTN_Q_WEIGHT_SUFFIX, ATTN_V_WEIGHT_SUFFIX,
    ATTN_WEIGHT_PREFIX, Attention, attn_k_weight_name, attn_output_weight_name, attn_q_weight_name,
    attn_v_weight_name,
};
pub use embedding::{EmbeddingTable, TOKEN_EMBD_WEIGHT};
pub use error::LayersError;
pub use rmsnorm::{
    ATTN_NORM_WEIGHT_PREFIX, ATTN_NORM_WEIGHT_SUFFIX, FFN_NORM_WEIGHT_SUFFIX, OUTPUT_NORM_WEIGHT,
    RmsNorm, attn_norm_weight_name, ffn_norm_weight_name,
};
pub use rope::Rope;
pub use swiglu::{
    FFN_DOWN_WEIGHT_SUFFIX, FFN_GATE_WEIGHT_SUFFIX, FFN_UP_WEIGHT_SUFFIX, FFN_WEIGHT_PREFIX,
    SwiGlu, ffn_down_weight_name, ffn_gate_weight_name, ffn_up_weight_name,
};
