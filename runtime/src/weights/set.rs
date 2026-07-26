//! Loaded GGUF weights: directory + file-backed (or owned) byte store.
//!
//! # Load path
//!
//! ```text
//! path ──► mmap (or own bytes)
//!            │
//!            ├─► GgufFile::from_bytes(map)   // header + tensor infos
//!            └─► validate each tensor span
//! ```
//!
//! Parsing from the mapped bytes avoids a second full-file read.

use std::path::Path;

use tracing::debug;

use super::error::WeightsError;
use super::quant::QuantMeta;
use super::storage::WeightStorage;
use super::tensor::WeightTensor;
use crate::errors::Result;
use crate::gguf::{GgufFile, TensorInfo};

/// A GGUF model file with accessible weight payloads.
#[derive(Debug)]
pub struct WeightSet {
    gguf: GgufFile,
    storage: WeightStorage,
}

impl WeightSet {
    /// Memory-map `path` and parse its GGUF directory.
    ///
    /// # Errors
    ///
    /// Returns parse, mmap, or bounds-validation errors.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!(path = %path.display(), "mapping GGUF weights");
        let storage = WeightStorage::mmap_path(path)?;
        Self::from_storage(storage)
    }

    /// Parse an in-memory GGUF buffer (tests and tiny fixtures).
    ///
    /// # Errors
    ///
    /// Returns parse or bounds-validation errors.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        Self::from_storage(WeightStorage::owned(bytes.into()))
    }

    fn from_storage(storage: WeightStorage) -> Result<Self> {
        let gguf = GgufFile::from_bytes(storage.as_slice())?;
        let set = Self { gguf, storage };
        set.validate_all_tensors()?;
        debug!(
            tensors = set.gguf.header.tensor_count,
            mapped = set.storage.is_mapped(),
            file_len = set.storage.len(),
            "weight set ready"
        );
        Ok(set)
    }

    /// Borrow the parsed GGUF directory / metadata.
    #[must_use]
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// `true` when weights are OS-mapped rather than copied.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.storage.is_mapped()
    }

    /// Byte length of the backing file / buffer.
    #[must_use]
    pub fn file_len(&self) -> usize {
        self.storage.len()
    }

    /// Resolve a tensor by name to a borrowed payload view.
    ///
    /// # Errors
    ///
    /// Returns not-found, unsupported type, or out-of-bounds errors.
    pub fn tensor(&self, name: &str) -> Result<WeightTensor<'_>> {
        let info = self
            .gguf
            .tensor(name)
            .ok_or_else(|| WeightsError::TensorNotFound {
                name: name.to_owned(),
            })?;
        self.tensor_from_info(info)
    }

    /// Iterate all tensors as payload views.
    ///
    /// # Errors
    ///
    /// Fails on the first tensor that cannot be resolved.
    pub fn tensors(&self) -> Result<Vec<WeightTensor<'_>>> {
        self.gguf
            .tensors
            .iter()
            .map(|info| self.tensor_from_info(info))
            .collect()
    }

    fn tensor_from_info<'a>(&'a self, info: &'a TensorInfo) -> Result<WeightTensor<'a>> {
        let quant = quant_for_tensor(info)?;
        let numel = info
            .numel()
            .ok_or_else(|| WeightsError::InvalidElementCount {
                name: info.name.clone(),
            })?;
        if numel == 0 {
            return Err(WeightsError::InvalidElementCount {
                name: info.name.clone(),
            }
            .into());
        }

        let nbytes = quant.nbytes(numel, &info.name)?;
        let start = self.gguf.absolute_offset(info)?;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| WeightsError::OutOfBounds {
                name: info.name.clone(),
                start,
                end: u64::MAX,
                file_len: self.storage.len() as u64,
                ggml_type: info.ggml_type,
            })?;

        let file_len = self.storage.len() as u64;
        if end > file_len {
            return Err(WeightsError::OutOfBounds {
                name: info.name.clone(),
                start,
                end,
                file_len,
                ggml_type: info.ggml_type,
            }
            .into());
        }

        let start_usize = usize::try_from(start).map_err(|_| WeightsError::OutOfBounds {
            name: info.name.clone(),
            start,
            end,
            file_len,
            ggml_type: info.ggml_type,
        })?;
        let end_usize = usize::try_from(end).map_err(|_| WeightsError::OutOfBounds {
            name: info.name.clone(),
            start,
            end,
            file_len,
            ggml_type: info.ggml_type,
        })?;

        Ok(WeightTensor {
            info,
            quant,
            absolute_offset: start,
            data: &self.storage.as_slice()[start_usize..end_usize],
        })
    }

    fn validate_all_tensors(&self) -> Result<()> {
        for info in &self.gguf.tensors {
            let _ = self.tensor_from_info(info)?;
        }
        Ok(())
    }
}

fn quant_for_tensor(info: &TensorInfo) -> Result<QuantMeta> {
    QuantMeta::for_type(info.ggml_type).map_err(|err| match err {
        WeightsError::UnsupportedType { ggml_type, .. } => WeightsError::UnsupportedType {
            name: info.name.clone(),
            ggml_type,
        }
        .into(),
        other => other.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::test_support::GgufBuilder;
    use crate::gguf::{DEFAULT_ALIGNMENT, GgmlType, GgufFile, TensorInfo, align_offset};

    /// Build a minimal GGUF with one F32 tensor and its payload bytes.
    fn fixture_f32_matrix() -> Vec<u8> {
        let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut payload = Vec::with_capacity(16);
        for v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }

        let header = GgufBuilder::new()
            .architecture("llama")
            .tensor(TensorInfo {
                name: "weight".into(),
                dimensions: vec![2, 2],
                ggml_type: GgmlType::F32,
                offset: 0,
            })
            .build();

        let data_offset =
            usize::try_from(align_offset(header.len() as u64, DEFAULT_ALIGNMENT)).unwrap();
        let mut bytes = header;
        bytes.resize(data_offset, 0);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn loads_f32_tensor_from_bytes() {
        let bytes = fixture_f32_matrix();
        // Sanity: directory parse still works.
        let dir = GgufFile::from_bytes(&bytes).unwrap();
        assert_eq!(dir.tensor("weight").unwrap().ggml_type, GgmlType::F32);

        let set = WeightSet::from_bytes(bytes).unwrap();
        assert!(!set.is_mapped());
        let view = set.tensor("weight").unwrap();
        assert_eq!(view.data.len(), 16);
        assert!(!view.quant.is_quantized);

        let tensor = view.to_f32_tensor().unwrap();
        assert_eq!(tensor.shape().as_slice(), &[2, 2]);
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn missing_tensor_errors() {
        let set = WeightSet::from_bytes(fixture_f32_matrix()).unwrap();
        let err = set.tensor("nope").unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Weights(WeightsError::TensorNotFound { .. })
        ));
    }

    #[test]
    fn mmap_round_trip_tmpfile() {
        let bytes = fixture_f32_matrix();
        let dir = std::env::temp_dir();
        let path = dir.join("phalanx_phase5_weights.gguf");
        std::fs::write(&path, &bytes).unwrap();

        let set = WeightSet::open_mmap(&path).unwrap();
        assert!(set.is_mapped());
        let tensor = set.tensor("weight").unwrap().to_f32_tensor().unwrap();
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let header = GgufBuilder::new()
            .tensor(TensorInfo {
                name: "weight".into(),
                dimensions: vec![2, 2],
                ggml_type: GgmlType::F32,
                offset: 0,
            })
            .build();
        let data_offset =
            usize::try_from(align_offset(header.len() as u64, DEFAULT_ALIGNMENT)).unwrap();
        let mut bytes = header;
        bytes.resize(data_offset, 0);
        // Only 8 bytes instead of 16.
        bytes.extend_from_slice(&[0u8; 8]);

        let err = WeightSet::from_bytes(bytes).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Weights(WeightsError::OutOfBounds { .. })
        ));
    }
}
