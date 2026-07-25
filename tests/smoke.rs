//! Integration smoke tests for the Phase 1 public API.
//!
//! These exercise the crate boundary (not `#[cfg(test)]` internals) so
//! refactors that break re-exports fail loudly.

use phalanx::{PhalanxError, RUNTIME_NAME, VERSION};

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
