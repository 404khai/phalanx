//! Elementary tensor kernels used by later transformer layers.
//!
//! Phase 2 prioritizes **correctness and clarity** over throughput. Naïve
//! loops are intentional: they establish reference behaviour before SIMD,
//! multithreading, or BLAS backends arrive in the profiling phases.
//!
//! References for the linear-algebra building blocks:
//! - Golub & Van Loan, *Matrix Computations* (matmul structure)
//! - `NumPy` broadcasting rules (explicitly **not** implemented yet)

use super::dense::Tensor;
use super::error::TensorError;
use super::shape::Shape;
use crate::errors::Result;

impl Tensor {
    /// Element-wise addition; shapes must match exactly.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeMismatch`] when shapes differ.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary_elemwise(rhs, |a, b| a + b)
    }

    /// Element-wise subtraction; shapes must match exactly.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeMismatch`] when shapes differ.
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary_elemwise(rhs, |a, b| a - b)
    }

    /// Element-wise multiplication; shapes must match exactly.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeMismatch`] when shapes differ.
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary_elemwise(rhs, |a, b| a * b)
    }

    /// Element-wise division; shapes must match exactly.
    ///
    /// Division by zero follows IEEE-754 (`±inf` / `NaN`) rather than erroring,
    /// matching `NumPy` / GPU kernel defaults.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeMismatch`] when shapes differ.
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary_elemwise(rhs, |a, b| a / b)
    }

    /// Multiply every element by `scalar`.
    #[must_use]
    pub fn scale(&self, scalar: f32) -> Self {
        let data = self.as_slice().iter().map(|x| x * scalar).collect();
        // Length equals `shape.numel()` by construction — skip re-validation.
        Self::new_contiguous(data, self.shape().clone())
    }

    /// In-place scale — avoids an allocation on hot activation paths.
    pub fn scale_inplace(&mut self, scalar: f32) {
        for x in self.as_mut_slice() {
            *x *= scalar;
        }
    }

    /// Naive dense matrix product for rank-2 tensors: `[M, K] × [K, N] → [M, N]`.
    ///
    /// Complexity is \(O(M \cdot N \cdot K)\). No blocking / SIMD yet — this is
    /// the reference kernel later phases will compare against.
    ///
    /// # Errors
    ///
    /// - [`TensorError::RankMismatch`] if either operand is not rank 2
    /// - [`TensorError::MatMulIncompatible`] if inner dimensions differ
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                got: self.rank(),
            }
            .into());
        }
        if rhs.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                got: rhs.rank(),
            }
            .into());
        }

        let m = self.shape()[0];
        let k = self.shape()[1];
        let k_rhs = rhs.shape()[0];
        let n = rhs.shape()[1];

        if k != k_rhs {
            return Err(TensorError::MatMulIncompatible {
                lhs: self.shape().to_string(),
                rhs: rhs.shape().to_string(),
            }
            .into());
        }

        let left = self.as_slice();
        let right = rhs.as_slice();
        let mut out = vec![0.0f32; m * n];

        // ijk loop order: simple to audit; cache-friendlier ikj can wait for
        // Phase 17 profiling evidence.
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += left[i * k + t] * right[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }

        Self::from_vec(out, Shape::new([m, n])?)
    }

    /// Materialize the transpose of a rank-2 tensor (`[M, N] → [N, M]`).
    ///
    /// Copies into a new contiguous buffer so the contiguity invariant holds.
    /// A future strided view could avoid the copy for read-only paths.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::RankMismatch`] when rank ≠ 2.
    pub fn transpose(&self) -> Result<Self> {
        if self.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                got: self.rank(),
            }
            .into());
        }

        let rows = self.shape()[0];
        let cols = self.shape()[1];
        let src = self.as_slice();
        let mut out = vec![0.0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = src[i * cols + j];
            }
        }

        Self::from_vec(out, Shape::new([cols, rows])?)
    }

    /// Sum of all elements (useful for tests and reductions scaffolding).
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.as_slice().iter().copied().sum()
    }

    fn binary_elemwise(&self, rhs: &Self, op: impl Fn(f32, f32) -> f32) -> Result<Self> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_string(),
                got: rhs.shape().to_string(),
            }
            .into());
        }

        let data = self
            .as_slice()
            .iter()
            .zip(rhs.as_slice().iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Self::from_vec(data, self.shape().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} ≉ {b}");
    }

    #[test]
    fn elementwise_add_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]).unwrap()).unwrap();
        let b =
            Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new([2, 2]).unwrap()).unwrap();

        let sum = a.add(&b).unwrap();
        assert_eq!(sum.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

        let prod = a.mul(&b).unwrap();
        assert_eq!(prod.as_slice(), &[10.0, 40.0, 90.0, 160.0]);
    }

    #[test]
    fn shape_mismatch_is_typed() {
        let a = Tensor::zeros([2, 2]).unwrap();
        let b = Tensor::zeros([2, 3]).unwrap();
        let err = a.add(&b).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Tensor(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn matmul_2x3_times_3x2() {
        // [[1, 2, 3], [4, 5, 6]] × [[7, 8], [9, 10], [11, 12]]
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new([2, 3]).unwrap(),
        )
        .unwrap();
        let b = Tensor::from_vec(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::new([3, 2]).unwrap(),
        )
        .unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape().as_slice(), &[2, 2]);
        // Row0: 1*7+2*9+3*11 = 58, 1*8+2*10+3*12 = 64
        // Row1: 4*7+5*9+6*11 = 139, 4*8+5*10+6*12 = 154
        approx_eq(c.get(&[0, 0]).unwrap(), 58.0);
        approx_eq(c.get(&[0, 1]).unwrap(), 64.0);
        approx_eq(c.get(&[1, 0]).unwrap(), 139.0);
        approx_eq(c.get(&[1, 1]).unwrap(), 154.0);
    }

    #[test]
    fn matmul_rejects_inner_mismatch() {
        let a = Tensor::zeros([2, 3]).unwrap();
        let b = Tensor::zeros([4, 2]).unwrap();
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Tensor(TensorError::MatMulIncompatible { .. })
        ));
    }

    #[test]
    fn transpose_swaps_axes_and_values() {
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new([2, 3]).unwrap(),
        )
        .unwrap();
        let t = a.transpose().unwrap();
        assert_eq!(t.shape().as_slice(), &[3, 2]);
        assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn scale_and_sum() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], Shape::new([3]).unwrap()).unwrap();
        let scaled = a.scale(2.0);
        assert_eq!(scaled.as_slice(), &[2.0, 4.0, 6.0]);
        approx_eq(scaled.sum(), 12.0);
    }
}
