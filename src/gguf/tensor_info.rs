//! Tensor directory entries from the GGUF container.
//!
//! These describe *where* and *how* weights are stored. Phase 5 will use
//! [`TensorInfo::offset`] relative to [`crate::gguf::GgufFile::data_offset`]
//! to `mmap` or copy the payload — Phase 3 stops at the directory.

use super::types::GgmlType;

/// One tensor's metadata record (not the weight bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name, e.g. `token_embd.weight` (≤ 64 bytes per spec).
    pub name: String,
    /// Dimension sizes in GGUF order (typically `[n_embd, n_vocab]` etc.).
    pub dimensions: Vec<u64>,
    /// Element / quantization type.
    pub ggml_type: GgmlType,
    /// Byte offset **relative to the tensor data blob**, not the file start.
    ///
    /// Absolute file offset = [`crate::gguf::GgufFile::data_offset`] + `offset`.
    pub offset: u64,
}

impl TensorInfo {
    /// Number of dimensions.
    #[must_use]
    pub fn n_dims(&self) -> usize {
        self.dimensions.len()
    }

    /// Product of dimensions (element count for dense types; block count differs
    /// for quantized layouts and is resolved in Phase 5).
    #[must_use]
    pub fn numel(&self) -> Option<u64> {
        self.dimensions
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
    }
}
