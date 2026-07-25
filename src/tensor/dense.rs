//! Owned, contiguous, row-major tensors.
//!
//! # Design tradeoff: owned `Vec<f32>` vs views / ndarray
//!
//! | Approach | Pros | Cons |
//! |---|---|---|
//! | Owned contiguous `Vec<f32>` (chosen) | Simple aliasing model, easy to test, matches activation buffers | Copies on transpose / some reshapes of non-contig data |
//! | `ndarray` | Rich views & broadcasting | Heavy API surface; hides the memory model we want to teach |
//! | Byte buffer + dtype tag | Unified quantized storage | Awkward for Phase 2 float kernels |
//!
//! Phase 5 can introduce a `Storage` enum (owned f32 / mmap bytes / quant
//! blocks) behind this same `Tensor` façade without rewriting call sites that
//! only need `shape()` + typed accessors.

use super::dtype::DType;
use super::error::TensorError;
use super::shape::Shape;
use crate::errors::Result;

/// Dense tensor with contiguous row-major [`f32`] storage.
///
/// Contiguity is an intentional invariant: kernels can iterate `as_slice()`
/// without stride math. Non-contiguous views (as-strided KV cache windows)
/// will be a separate type when Phase 12 needs them.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    /// Wrap an existing buffer. Length must equal `shape.numel()`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DataLengthMismatch`] when lengths disagree.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Result<Self> {
        let expected = shape.numel();
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                shape: shape.to_string(),
                expected,
                got: data.len(),
            }
            .into());
        }
        Ok(Self::new_contiguous(data, shape))
    }

    /// Construct after the caller has already proven `data.len() == shape.numel()`.
    ///
    /// Kept private so the length invariant cannot be bypassed from outside
    /// the tensor module (e.g. by `scale`, which preserves element count).
    pub(super) fn new_contiguous(data: Vec<f32>, shape: Shape) -> Self {
        debug_assert_eq!(data.len(), shape.numel());
        Self {
            data,
            shape,
            dtype: DType::F32,
        }
    }

    /// Allocate a zero-filled tensor.
    ///
    /// # Errors
    ///
    /// Propagates shape construction errors.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        let data = vec![0.0; shape.numel()];
        Self::from_vec(data, shape)
    }

    /// Allocate a one-filled tensor.
    ///
    /// # Errors
    ///
    /// Propagates shape construction errors.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(shape)?;
        let data = vec![1.0; shape.numel()];
        Self::from_vec(data, shape)
    }

    /// Allocate a tensor filled with `value`.
    ///
    /// # Errors
    ///
    /// Propagates shape construction errors.
    pub fn full(shape: impl Into<Vec<usize>>, value: f32) -> Result<Self> {
        let shape = Shape::new(shape)?;
        let data = vec![value; shape.numel()];
        Self::from_vec(data, shape)
    }

    /// Borrowed constructor; copies into an owned buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DataLengthMismatch`] when lengths disagree.
    pub fn from_slice(data: &[f32], shape: Shape) -> Result<Self> {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Logical element type (always [`DType::F32`] in Phase 2).
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shape metadata.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Rank shortcut.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Element count shortcut.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Contiguous row-major strides for the current shape.
    ///
    /// # Errors
    ///
    /// Propagates stride overflow errors from [`Shape::row_major_strides`].
    pub fn strides(&self) -> Result<Vec<usize>> {
        self.shape.row_major_strides()
    }

    /// Immutable view of the backing buffer.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable view of the backing buffer.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Read one element by multi-index.
    ///
    /// # Errors
    ///
    /// Propagates rank / bounds errors from [`Shape::offset`].
    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        let offset = self.shape.offset(indices)?;
        Ok(self.data[offset])
    }

    /// Write one element by multi-index.
    ///
    /// # Errors
    ///
    /// Propagates rank / bounds errors from [`Shape::offset`].
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let offset = self.shape.offset(indices)?;
        self.data[offset] = value;
        Ok(())
    }

    /// Change shape without moving elements (metadata-only).
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DataLengthMismatch`] when `new_shape.numel()`
    /// differs — a silent reshape would reinterpret memory incorrectly.
    pub fn reshape(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(new_shape)?;
        Self::from_vec(self.data.clone(), shape)
    }

    /// Consume and reshape without cloning the buffer when lengths match.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DataLengthMismatch`] when element counts differ.
    pub fn into_shape(self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(new_shape)?;
        Self::from_vec(self.data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_correct_numel_and_values() {
        let t = Tensor::zeros([2, 3]).unwrap();
        assert_eq!(t.numel(), 6);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn from_vec_rejects_length_mismatch() {
        let shape = Shape::new([2, 2]).unwrap();
        let err = Tensor::from_vec(vec![1.0, 2.0], shape).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Tensor(TensorError::DataLengthMismatch { .. })
        ));
    }

    #[test]
    fn get_set_round_trip() {
        let mut t = Tensor::zeros([2, 2]).unwrap();
        t.set(&[1, 0], 3.5).unwrap();
        assert!((t.get(&[1, 0]).unwrap() - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn reshape_preserves_elements() {
        let t = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new([2, 3]).unwrap(),
        )
        .unwrap();
        let r = t.reshape([3, 2]).unwrap();
        assert_eq!(r.shape().as_slice(), &[3, 2]);
        assert_eq!(r.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn into_shape_avoids_needless_semantics_break() {
        let t = Tensor::ones([4]).unwrap();
        let t = t.into_shape([2, 2]).unwrap();
        assert_eq!(t.shape().as_slice(), &[2, 2]);
        assert_eq!(t.numel(), 4);
    }
}
