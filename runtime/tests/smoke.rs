//! Integration smoke tests for the public API.
//!
//! These exercise the crate boundary (not `#[cfg(test)]` internals) so
//! refactors that break re-exports fail loudly.

use phalanx::{
    Architecture, EmbeddingTable, EncodeOptions, GgmlType, GgufError, GgufFile, LayersError,
    ModelConfig, ModelError, PhalanxError, QuantMeta, RUNTIME_NAME, Shape, SpecialTokens, Tensor,
    TensorError, Tokenizer, TokenizerModel, VERSION, Vocabulary,
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
