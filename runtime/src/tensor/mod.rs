//! Tensor abstraction: shapes, contiguous storage, and elementary ops.
//!
//! This module is the math foundation for Phalanx. Later subsystems (GGUF
//! loaders, attention, FFN) should depend on [`Tensor`] rather than raw
//! `Vec<f32>` so layout conventions stay centralized.
//!
//! # Memory model (Phase 2)
//!
//! ```text
//! shape [2, 3]  dtype f32  row-major contiguous
//!
//! logical:        [[a00, a01, a02],
//!                  [a10, a11, a12]]
//!
//! memory:         [a00, a01, a02, a10, a11, a12]
//! strides:        [3, 1]
//! ```
//!
//! # Module map
//!
//! - [`dtype`] — element type tags
//! - [`shape`] — dimensions + stride / offset helpers
//! - [`dense`] — owned buffer + constructors
//! - [`ops`] — element-wise, matmul, transpose
//! - [`error`] — [`TensorError`]

mod dense;
mod dtype;
mod error;
mod ops;
mod shape;

pub use dense::Tensor;
pub use dtype::DType;
pub use error::TensorError;
pub use shape::Shape;
