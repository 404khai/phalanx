//! Quantization block metadata for `ggml_type` values.
//!
//! GGUF stores most weights as **blocks**: a fixed number of elements share a
//! scale (and optional min). Knowing `(block_size, type_size)` lets the loader
//! compute byte spans and validate the file **before** any dequant kernel runs.
//!
//! Sizes match ggml / llama.cpp (`sizeof(block_*)` and `QK*` constants).
//! Reference: <https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h>

use crate::gguf::GgmlType;
use crate::weights::error::WeightsError;

/// Layout description for one `ggml_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantMeta {
    /// Logical type this metadata describes.
    pub ggml_type: GgmlType,
    /// Elements covered by one on-disk block (`1` for dense floats/ints).
    pub block_size: u64,
    /// Bytes occupied by one on-disk block.
    pub type_size: u64,
    /// `true` when the type packs multiple values per block with a scale.
    pub is_quantized: bool,
}

impl QuantMeta {
    /// Look up layout metadata for a `ggml_type`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError::UnsupportedType`] for unknown / unimplemented tags.
    pub fn for_type(ggml_type: GgmlType) -> Result<Self, WeightsError> {
        let (block_size, type_size, is_quantized) = match ggml_type {
            GgmlType::F32 | GgmlType::I32 => (1, 4, false),
            GgmlType::F16 | GgmlType::Bf16 | GgmlType::I16 => (1, 2, false),
            GgmlType::F64 | GgmlType::I64 => (1, 8, false),
            GgmlType::I8 => (1, 1, false),
            // Legacy quants — block of 32
            GgmlType::Q4_0 => (32, 18, true),
            GgmlType::Q4_1 => (32, 20, true),
            GgmlType::Q5_0 => (32, 22, true),
            GgmlType::Q5_1 => (32, 24, true),
            GgmlType::Q8_0 => (32, 34, true),
            GgmlType::Q8_1 => (32, 36, true),
            // K-quants — super-block of 256
            GgmlType::Q2K => (256, 84, true),
            GgmlType::Q3K => (256, 110, true),
            GgmlType::Q4K => (256, 144, true),
            GgmlType::Q5K => (256, 176, true),
            GgmlType::Q6K => (256, 210, true),
            GgmlType::Q8K => (256, 292, true),
            // IQ / ternary / MX sizes vary across ggml revisions — enable with
            // golden tests when a Phase 7+ kernel needs a specific format.
            other => {
                return Err(WeightsError::UnsupportedType {
                    name: "<meta>".into(),
                    ggml_type: other,
                });
            }
        };

        Ok(Self {
            ggml_type,
            block_size,
            type_size,
            is_quantized,
        })
    }

    /// Byte length of a tensor with `numel` elements of this type.
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError::MisalignedElements`] when `numel` is not a
    /// multiple of [`QuantMeta::block_size`].
    pub fn nbytes(&self, numel: u64, tensor_name: &str) -> Result<u64, WeightsError> {
        if self.block_size == 0 {
            return Err(WeightsError::UnsupportedType {
                name: tensor_name.into(),
                ggml_type: self.ggml_type,
            });
        }
        if numel % self.block_size != 0 {
            return Err(WeightsError::MisalignedElements {
                name: tensor_name.into(),
                numel,
                block_size: self.block_size,
                ggml_type: self.ggml_type,
            });
        }
        let blocks = numel / self.block_size;
        blocks
            .checked_mul(self.type_size)
            .ok_or_else(|| WeightsError::InvalidElementCount {
                name: tensor_name.into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_and_q4_0_sizes() {
        let f32 = QuantMeta::for_type(GgmlType::F32).unwrap();
        assert_eq!(f32.nbytes(8, "t").unwrap(), 32);

        let q4 = QuantMeta::for_type(GgmlType::Q4_0).unwrap();
        assert_eq!(q4.block_size, 32);
        assert_eq!(q4.type_size, 18);
        assert_eq!(q4.nbytes(64, "t").unwrap(), 36);
        assert!(q4.nbytes(30, "t").is_err());
    }

    #[test]
    fn q4k_superblock() {
        let q = QuantMeta::for_type(GgmlType::Q4K).unwrap();
        assert_eq!(q.block_size, 256);
        assert_eq!(q.type_size, 144);
        assert_eq!(q.nbytes(256, "w").unwrap(), 144);
    }
}
