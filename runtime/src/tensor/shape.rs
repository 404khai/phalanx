//! Tensor shape metadata and row-major stride computation.
//!
//! Shapes are owned `Vec<usize>` rather than a fixed-rank array so GGUF tensors
//! (often rank 1–4, occasionally higher for packed quants) share one type.

use std::fmt;
use std::ops::Deref;

use super::TensorError;
use crate::errors::Result;

/// Ordered list of dimension sizes (`[rows, cols]` for a matrix).
///
/// # Invariants
///
/// - Rank is at least 1 (scalars are represented as `[1]`, not rank-0).
/// - Individual dimensions may be zero (empty batch / masked axis).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Construct a shape from dimension sizes.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidShape`] when `dims` is empty. Rank-0
    /// scalars are deferred — LLM kernels almost always want an explicit axis.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(TensorError::InvalidShape {
                reason: "shape must have rank >= 1 (use [1] for a scalar)".into(),
            }
            .into());
        }
        Ok(Self { dims })
    }

    /// Rank (number of dimensions).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements (`product` of dims; `0` if any dim is zero).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().copied().product()
    }

    /// Borrow the raw dimension slice.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Row-major (C-order) strides for a dense contiguous layout.
    ///
    /// For shape `[2, 3, 4]` strides are `[12, 4, 1]`: the leftmost axis varies
    /// slowest in memory. This matches `NumPy` defaults and typical BLAS packing,
    /// so GGUF-derived buffers map without a transpose surprise.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidShape`] if a stride product overflows `usize`
    /// (pathological shapes that cannot address a real buffer).
    pub fn row_major_strides(&self) -> Result<Vec<usize>> {
        let rank = self.dims.len();
        let mut strides = vec![0usize; rank];
        // Last axis is contiguous in memory — stride 1 element.
        strides[rank - 1] = 1usize;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1]
                .checked_mul(self.dims[i + 1])
                .ok_or_else(|| TensorError::InvalidShape {
                    reason: format!("stride overflow at axis {i} for shape {self}"),
                })?;
        }
        Ok(strides)
    }

    /// Linear offset of a multi-index into a contiguous row-major buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::RankMismatch`], [`TensorError::IndexOutOfBounds`],
    /// or [`TensorError::InvalidShape`] on arithmetic overflow.
    pub fn offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.rank() {
            return Err(TensorError::RankMismatch {
                expected: self.rank(),
                got: indices.len(),
            }
            .into());
        }

        let strides = self.row_major_strides()?;
        let mut offset = 0usize;
        for (axis, (&index, (&dim, &stride))) in indices
            .iter()
            .zip(self.dims.iter().zip(strides.iter()))
            .enumerate()
        {
            if index >= dim {
                return Err(TensorError::IndexOutOfBounds { axis, index, dim }.into());
            }
            let delta = index
                .checked_mul(stride)
                .ok_or_else(|| TensorError::InvalidShape {
                    reason: format!("offset overflow at axis {axis} for shape {self}"),
                })?;
            offset = offset
                .checked_add(delta)
                .ok_or_else(|| TensorError::InvalidShape {
                    reason: format!("offset overflow at axis {axis} for shape {self}"),
                })?;
        }
        Ok(offset)
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_dims() {
        let err = Shape::new([]).unwrap_err();
        assert!(matches!(err, crate::PhalanxError::Tensor(_)));
    }

    #[test]
    fn numel_and_strides_for_matrix() {
        let shape = Shape::new([2, 3]).unwrap();
        assert_eq!(shape.numel(), 6);
        assert_eq!(shape.row_major_strides().unwrap(), vec![3, 1]);
    }

    #[test]
    fn numel_and_strides_for_3d() {
        let shape = Shape::new([2, 3, 4]).unwrap();
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.row_major_strides().unwrap(), vec![12, 4, 1]);
    }

    #[test]
    fn offset_matches_row_major_formula() {
        let shape = Shape::new([2, 3]).unwrap();
        // index (1, 2) → 1*3 + 2 = 5
        assert_eq!(shape.offset(&[1, 2]).unwrap(), 5);
    }

    #[test]
    fn zero_dim_yields_zero_numel() {
        let shape = Shape::new([4, 0, 3]).unwrap();
        assert_eq!(shape.numel(), 0);
    }
}
