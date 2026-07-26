//! A single weight tensor view into mapped (or owned) GGUF bytes.

use half::f16;

use super::error::WeightsError;
use super::quant::QuantMeta;
use crate::errors::Result;
use crate::gguf::{GgmlType, TensorInfo};
use crate::tensor::{Shape, Tensor};

/// Borrowed view of one tensor's on-disk payload.
#[derive(Debug, Clone, Copy)]
pub struct WeightTensor<'a> {
    /// Directory record from the GGUF header.
    pub info: &'a TensorInfo,
    /// Quantization / dense layout metadata.
    pub quant: QuantMeta,
    /// Absolute file offset of the first payload byte.
    pub absolute_offset: u64,
    /// Raw payload bytes (block-packed for quantized types).
    pub data: &'a [u8],
}

impl WeightTensor<'_> {
    /// Tensor name shortcut.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Element type shortcut.
    #[must_use]
    pub fn ggml_type(&self) -> GgmlType {
        self.info.ggml_type
    }

    /// Materialize dense `f32` / `f16` weights into a runtime [`Tensor`].
    ///
    /// Quantized types return [`WeightsError::DequantNotImplemented`].
    /// Dense `f32`/`f16` embeddings materialize in Phase 7; block dequant
    /// arrives with the first quantized matmul path that needs it.
    ///
    /// # Errors
    ///
    /// Returns shape, length, or dequant errors.
    pub fn to_f32_tensor(&self) -> Result<Tensor> {
        let dims = self
            .info
            .dimensions
            .iter()
            .map(|&d| usize::try_from(d))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| WeightsError::InvalidElementCount {
                name: self.info.name.clone(),
            })?;
        let shape = Shape::new(dims)?;

        match self.info.ggml_type {
            GgmlType::F32 => self.decode_f32(shape),
            GgmlType::F16 => self.decode_f16(shape),
            other => Err(WeightsError::DequantNotImplemented {
                name: self.info.name.clone(),
                ggml_type: other,
            }
            .into()),
        }
    }

    fn decode_f32(&self, shape: Shape) -> Result<Tensor> {
        if self.data.len() % 4 != 0 {
            return Err(WeightsError::InvalidFloatPayload {
                name: self.info.name.clone(),
                got: self.data.len(),
            }
            .into());
        }
        let expected =
            shape
                .numel()
                .checked_mul(4)
                .ok_or_else(|| WeightsError::InvalidElementCount {
                    name: self.info.name.clone(),
                })?;
        if self.data.len() != expected {
            return Err(WeightsError::InvalidFloatPayload {
                name: self.info.name.clone(),
                got: self.data.len(),
            }
            .into());
        }

        let mut values = Vec::with_capacity(shape.numel());
        for chunk in self.data.chunks_exact(4) {
            let bytes: [u8; 4] =
                chunk
                    .try_into()
                    .map_err(|_| WeightsError::InvalidFloatPayload {
                        name: self.info.name.clone(),
                        got: self.data.len(),
                    })?;
            values.push(f32::from_le_bytes(bytes));
        }
        Tensor::from_vec(values, shape)
    }

    fn decode_f16(&self, shape: Shape) -> Result<Tensor> {
        if self.data.len() % 2 != 0 {
            return Err(WeightsError::InvalidFloatPayload {
                name: self.info.name.clone(),
                got: self.data.len(),
            }
            .into());
        }
        let expected =
            shape
                .numel()
                .checked_mul(2)
                .ok_or_else(|| WeightsError::InvalidElementCount {
                    name: self.info.name.clone(),
                })?;
        if self.data.len() != expected {
            return Err(WeightsError::InvalidFloatPayload {
                name: self.info.name.clone(),
                got: self.data.len(),
            }
            .into());
        }

        let mut values = Vec::with_capacity(shape.numel());
        for chunk in self.data.chunks_exact(2) {
            let bytes: [u8; 2] =
                chunk
                    .try_into()
                    .map_err(|_| WeightsError::InvalidFloatPayload {
                        name: self.info.name.clone(),
                        got: self.data.len(),
                    })?;
            values.push(f16::from_le_bytes(bytes).to_f32());
        }
        Tensor::from_vec(values, shape)
    }
}
