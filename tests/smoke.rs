//! Integration smoke tests for the public API.
//!
//! These exercise the crate boundary (not `#[cfg(test)]` internals) so
//! refactors that break re-exports fail loudly.

use phalanx::{PhalanxError, RUNTIME_NAME, Shape, Tensor, TensorError, VERSION};

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
