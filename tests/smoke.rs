//! Integration smoke tests for the public API.
//!
//! These exercise the crate boundary (not `#[cfg(test)]` internals) so
//! refactors that break re-exports fail loudly.

use phalanx::{
    Architecture, Attention, EmbeddingTable, EncodeOptions, GgmlType, GgufError, GgufFile,
    LayersError, ModelConfig, ModelError, PhalanxError, QuantMeta, RUNTIME_NAME, RmsNorm, Rope,
    Shape, SpecialTokens, SwiGlu, Tensor, TensorError, Tokenizer, TokenizerModel, VERSION,
    Vocabulary,
};

#[test]
fn public_version_constant_is_exported() {
    assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn public_runtime_name_is_exported() {
    assert_eq!(RUNTIME_NAME, "Phalanx Runtime");
}

#[test]
fn config_errors_are_matchable_across_crate_boundary() {
    let err = PhalanxError::config("bad hyperparameter");
    match err {
        PhalanxError::Config(message) => assert!(message.contains("hyperparameter")),
        other => panic!("expected Config variant, got {other:?}"),
    }
}

#[test]
fn tensor_matmul_is_usable_from_integration_tests() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]).unwrap()).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]).unwrap()).unwrap();
    let c = a.matmul(&b).unwrap();
    // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn tensor_errors_nest_under_phalanx_error() {
    let a = Tensor::zeros([2, 2]).unwrap();
    let b = Tensor::zeros([3, 3]).unwrap();
    let err = a.add(&b).unwrap_err();
    assert!(matches!(
        err,
        PhalanxError::Tensor(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn gguf_rejects_bad_magic_across_crate_boundary() {
    let err = GgufFile::from_bytes(b"NOTG........").unwrap_err();
    assert!(matches!(
        err,
        PhalanxError::Gguf(GgufError::InvalidMagic { .. })
    ));
}

#[test]
fn gguf_type_names_are_exported() {
    assert_eq!(GgmlType::Q4K.name(), "q4_k");
}

#[test]
fn quant_meta_q4k_is_exported() {
    let meta = QuantMeta::for_type(GgmlType::Q4K).unwrap();
    assert_eq!(meta.block_size, 256);
    assert_eq!(meta.type_size, 144);
    assert!(meta.is_quantized);
}

#[test]
fn model_errors_nest_under_phalanx_error() {
    let err = PhalanxError::Model(ModelError::UnsupportedArchitecture {
        architecture: "qwen2".into(),
    });
    assert!(matches!(
        err,
        PhalanxError::Model(ModelError::UnsupportedArchitecture { .. })
    ));
    // Public re-exports stay usable at the crate boundary.
    assert_eq!(Architecture::Llama.as_str(), "llama");
    let _ = ModelConfig::from_parts;
}

#[test]
fn embedding_gather_works_across_crate_boundary() {
    let table = EmbeddingTable::from_tensor(
        Tensor::from_vec(
            vec![
                1.0, 2.0, // token 0
                3.0, 4.0, // token 1
            ],
            Shape::new([2, 2]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let out = table.forward(&[1, 0]).unwrap();
    assert_eq!(out.as_slice(), &[3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn rope_preserves_norm_across_crate_boundary() {
    use phalanx::{AttentionConfig, RopeConfig};
    let config = ModelConfig::from_parts(ModelConfig {
        architecture: Architecture::Llama,
        name: None,
        vocab_size: None,
        context_length: 16,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 1,
        attention: AttentionConfig {
            head_count: 2,
            head_count_kv: 2,
            key_length: 4,
            value_length: 4,
        },
        rope: RopeConfig {
            dimension_count: 4,
            freq_base: 10_000.0,
            scaling: None,
        },
        rms_norm_eps: 1e-5,
    })
    .unwrap();
    let rope = Rope::from_config(&config).unwrap();
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new([1, 4]).unwrap()).unwrap();
    let y = rope.forward(&x, 3).unwrap();
    let nx: f32 = x.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
    let ny: f32 = y.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((nx - ny).abs() < 1e-5);
}

#[test]
fn rmsnorm_unit_rms_across_crate_boundary() {
    let norm = RmsNorm::ones(4, 1e-6).unwrap();
    let x = Tensor::from_vec(vec![2.0, -2.0, 2.0, -2.0], Shape::new([1, 4]).unwrap()).unwrap();
    let y = norm.forward(&x).unwrap();
    let mean_sq: f32 = y.as_slice().iter().map(|v| v * v).sum::<f32>() / 4.0;
    assert!((mean_sq.sqrt() - 1.0).abs() < 1e-4);
}

#[test]
fn swiglu_preserves_shape_across_crate_boundary() {
    let gate = Tensor::ones([8, 4]).unwrap();
    let up = Tensor::ones([8, 4]).unwrap();
    let down = Tensor::ones([4, 8]).unwrap();
    let ffn = SwiGlu::from_tensors(gate, up, down).unwrap();
    let x = Tensor::ones([2, 3, 4]).unwrap();
    let y = ffn.forward(&x).unwrap();
    assert_eq!(y.shape().as_slice(), &[2, 3, 4]);
}

#[test]
fn layers_errors_nest_under_phalanx_error() {
    let err = PhalanxError::Layers(LayersError::TokenOutOfRange {
        id: 9,
        vocab_size: 2,
    });
    assert!(matches!(
        err,
        PhalanxError::Layers(LayersError::TokenOutOfRange { .. })
    ));
}

#[test]
fn tokenizer_encode_decode_round_trip_across_crate_boundary() {
    let vocab = Vocabulary::new(vec!["▁hi".into(), "!".into()], None, None).unwrap();
    let tok = Tokenizer::from_parts(
        TokenizerModel::Llama,
        vocab,
        SpecialTokens::default(),
        Vec::new(),
    );
    let ids = tok
        .encode(
            "hi!",
            EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        )
        .unwrap();
    assert_eq!(ids, vec![0, 1]);
    assert_eq!(tok.decode(&ids).unwrap(), " hi!");
}

#[test]
fn attention_gqa_forward_is_usable() {
    let hidden = 16usize;
    let heads = 4usize;
    let kv = 2usize;
    let head_dim = 4usize;
    let q_out = heads * head_dim;
    let kv_out = kv * head_dim;
    let w_q = Tensor::ones([q_out, hidden]).unwrap();
    let w_k = Tensor::ones([kv_out, hidden]).unwrap();
    let w_v = Tensor::ones([kv_out, hidden]).unwrap();
    let w_o = Tensor::ones([hidden, q_out]).unwrap();
    let attn = Attention::from_tensors(w_q, w_k, w_v, w_o, heads, kv, head_dim).unwrap();
    let y = attn
        .forward(&Tensor::ones([1, 3, hidden]).unwrap(), None, 0)
        .unwrap();
    assert_eq!(y.shape().as_slice(), &[1, 3, hidden]);
}
