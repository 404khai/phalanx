//! Top-level GGUF container parse (header + metadata + tensor directory).
//!
//! # Scope
//!
//! Phase 3 reads everything needed to *inspect* a model and locate tensors.
//! It deliberately stops before materializing weight bytes — that is Phase 5
//! (`mmap` / dequant), once vocabulary loading (Phase 4) has a home.
//!
//! # Tradeoff: streaming [`Read`] vs full-file buffer
//!
//! | Approach | Pros | Cons |
//! |---|---|---|
//! | Streaming `Read` + byte cursor (chosen) | Works on multi-GB models without loading weights | Slightly more bookkeeping |
//! | `mmap` whole file now | Simple slicing | Pulls `unsafe` / platform deps before needed |
//! | Read entire file to `Vec<u8>` | Easy tests | Wasteful for real checkpoints |
//!
//! Reference: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use tracing::debug;

use super::error::GgufError;
use super::reader::GgufReader;
use super::tensor_info::TensorInfo;
use super::types::GgmlType;
use super::types::{
    ALIGNMENT_KEY, ARCHITECTURE_KEY, DEFAULT_ALIGNMENT, GGUF_MAGIC, NAME_KEY, SUPPORTED_VERSIONS,
    align_offset, limits,
};
use super::value::{MetadataEntry, MetadataValue};
use crate::errors::Result;

/// Fixed fields from the start of a GGUF file (before KV pairs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// Format version (`2` or `3` accepted).
    pub version: u32,
    /// Number of tensor info records that follow metadata.
    pub tensor_count: u64,
    /// Number of metadata key-value pairs.
    pub metadata_kv_count: u64,
}

/// Parsed GGUF container **without** weight payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    /// Magic-validated header counts / version.
    pub header: GgufHeader,
    /// Metadata entries in file order.
    pub metadata: Vec<MetadataEntry>,
    /// Tensor directory in file order.
    pub tensors: Vec<TensorInfo>,
    /// Alignment used for the data section (from metadata or default 32).
    pub alignment: u64,
    /// Absolute file offset where the `tensor_data` blob begins.
    pub data_offset: u64,
}

impl GgufFile {
    /// Parse a GGUF file from a filesystem path.
    ///
    /// Only the header, metadata, and tensor infos are consumed; the weight
    /// blob is left unread on disk.
    ///
    /// # Errors
    ///
    /// Returns I/O or [`GgufError`] failures.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!(path = %path.display(), "opening GGUF file");
        let file = File::open(path)?;
        Self::parse(file)
    }

    /// Parse from an in-memory buffer (tests and small fixtures).
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(std::io::Cursor::new(bytes))
    }

    /// Parse from any [`Read`] stream positioned at the start of the file.
    ///
    /// # Errors
    ///
    /// Returns I/O or [`GgufError`] failures.
    pub fn parse<R: Read>(reader: R) -> Result<Self> {
        let mut reader = GgufReader::new(reader);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic, "magic")?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic {
                expected: GGUF_MAGIC,
                got: magic,
            }
            .into());
        }

        let version = reader.read_u32("version")?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(GgufError::UnsupportedVersion {
                version,
                supported: SUPPORTED_VERSIONS,
            }
            .into());
        }

        let tensor_count = reader.read_u64("tensor_count")?;
        if tensor_count > limits::MAX_TENSOR_COUNT {
            return Err(GgufError::LimitExceeded {
                context: "tensor_count",
                got: tensor_count,
                limit: limits::MAX_TENSOR_COUNT,
            }
            .into());
        }

        let metadata_kv_count = reader.read_u64("metadata_kv_count")?;
        if metadata_kv_count > limits::MAX_METADATA_COUNT {
            return Err(GgufError::LimitExceeded {
                context: "metadata_kv_count",
                got: metadata_kv_count,
                limit: limits::MAX_METADATA_COUNT,
            }
            .into());
        }

        let header = GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        let mut metadata = Vec::with_capacity(usize::try_from(metadata_kv_count).unwrap_or(0));
        for _ in 0..metadata_kv_count {
            metadata.push(read_metadata_entry(&mut reader)?);
        }

        let alignment = resolve_alignment(&metadata)?;

        let mut tensors = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
        for _ in 0..tensor_count {
            tensors.push(read_tensor_info(&mut reader, alignment)?);
        }

        let data_offset = align_offset(reader.position(), alignment);

        debug!(
            version,
            metadata = metadata.len(),
            tensors = tensors.len(),
            alignment,
            data_offset,
            "parsed GGUF container"
        );

        Ok(Self {
            header,
            metadata,
            tensors,
            alignment,
            data_offset,
        })
    }

    /// First metadata value for `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    /// `general.architecture` string, when present.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.get(ARCHITECTURE_KEY).and_then(MetadataValue::as_str)
    }

    /// `general.name` string, when present.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.get(NAME_KEY).and_then(MetadataValue::as_str)
    }

    /// Tensor info by exact name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Absolute file offset of a tensor's payload.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::Invalid`] when `relative + data_offset` overflows.
    pub fn absolute_offset(&self, tensor: &TensorInfo) -> Result<u64> {
        self.data_offset.checked_add(tensor.offset).ok_or_else(|| {
            GgufError::Invalid {
                context: "tensor absolute offset",
                reason: format!(
                    "data_offset {} + relative {} overflows u64",
                    self.data_offset, tensor.offset
                ),
            }
            .into()
        })
    }
}

fn read_metadata_entry<R: Read>(reader: &mut GgufReader<R>) -> Result<MetadataEntry> {
    let key = reader.read_string("metadata key", limits::MAX_KEY_LEN)?;
    validate_metadata_key(&key)?;
    let value_type = reader.read_value_type("metadata value type")?;
    let value = reader.read_value(value_type, 0)?;
    Ok(MetadataEntry { key, value })
}

fn validate_metadata_key(key: &str) -> Result<()> {
    // Spec: valid ASCII, hierarchical `lower_snake_case` segments separated by `.`.
    // We enforce ASCII + non-empty; snake_case is recommended but real files
    // occasionally bend the style rules — reject only clearly illegal keys.
    if key.is_empty() {
        return Err(GgufError::Invalid {
            context: "metadata key",
            reason: "key must not be empty".into(),
        }
        .into());
    }
    if !key.is_ascii() {
        return Err(GgufError::Invalid {
            context: "metadata key",
            reason: format!("key must be ASCII, got {key:?}"),
        }
        .into());
    }
    Ok(())
}

fn resolve_alignment(metadata: &[MetadataEntry]) -> Result<u64> {
    let Some(entry) = metadata.iter().find(|entry| entry.key == ALIGNMENT_KEY) else {
        return Ok(DEFAULT_ALIGNMENT);
    };

    let alignment = match &entry.value {
        MetadataValue::U32(v) => u64::from(*v),
        MetadataValue::U64(v) => *v,
        other => {
            return Err(GgufError::Invalid {
                context: ALIGNMENT_KEY,
                reason: format!("expected u32/u64, got {:?}", other.value_type()),
            }
            .into());
        }
    };

    if alignment == 0 || alignment % 8 != 0 {
        return Err(GgufError::Invalid {
            context: ALIGNMENT_KEY,
            reason: format!("alignment must be a non-zero multiple of 8, got {alignment}"),
        }
        .into());
    }

    Ok(alignment)
}

fn read_tensor_info<R: Read>(reader: &mut GgufReader<R>, alignment: u64) -> Result<TensorInfo> {
    let name = reader.read_string("tensor name", limits::MAX_TENSOR_NAME_LEN)?;
    if name.is_empty() {
        return Err(GgufError::Invalid {
            context: "tensor name",
            reason: "name must not be empty".into(),
        }
        .into());
    }

    let n_dimensions = reader.read_u32("tensor n_dimensions")?;
    if n_dimensions == 0 || n_dimensions > limits::MAX_DIMENSIONS {
        return Err(GgufError::Invalid {
            context: "tensor n_dimensions",
            reason: format!(
                "expected 1..={}, got {n_dimensions}",
                limits::MAX_DIMENSIONS
            ),
        }
        .into());
    }

    let mut dimensions = Vec::with_capacity(n_dimensions as usize);
    for _ in 0..n_dimensions {
        dimensions.push(reader.read_u64("tensor dimension")?);
    }

    let type_id = reader.read_u32("tensor ggml_type")?;
    let ggml_type = GgmlType::from_u32(type_id);

    let offset = reader.read_u64("tensor offset")?;
    if offset % alignment != 0 {
        return Err(GgufError::Invalid {
            context: "tensor offset",
            reason: format!("offset {offset} is not a multiple of alignment {alignment}"),
        }
        .into());
    }

    Ok(TensorInfo {
        name,
        dimensions,
        ggml_type,
        offset,
    })
}

/// Minimal writer used only by unit tests to synthesize valid containers.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unwrap_used,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::new_without_default,
    dead_code,
    missing_docs
)]
pub mod test_support {
    use super::*;
    use crate::gguf::types::MetadataValueType;

    /// Builder for tiny in-memory GGUF fixtures.
    pub struct GgufBuilder {
        version: u32,
        metadata: Vec<(String, MetadataValue)>,
        tensors: Vec<TensorInfo>,
        alignment: Option<u32>,
    }

    impl GgufBuilder {
        /// Start an empty version-3 container.
        pub fn new() -> Self {
            Self {
                version: 3,
                metadata: Vec::new(),
                tensors: Vec::new(),
                alignment: None,
            }
        }

        /// Set `general.architecture`.
        pub fn architecture(mut self, arch: &str) -> Self {
            self.metadata
                .push((ARCHITECTURE_KEY.into(), MetadataValue::String(arch.into())));
            self
        }

        /// Set `general.name`.
        pub fn name(mut self, name: &str) -> Self {
            self.metadata
                .push((NAME_KEY.into(), MetadataValue::String(name.into())));
            self
        }

        /// Set `general.alignment`.
        pub fn alignment(mut self, alignment: u32) -> Self {
            self.alignment = Some(alignment);
            self
        }

        /// Push a `u32` metadata value.
        pub fn meta_u32(mut self, key: &str, value: u32) -> Self {
            self.metadata.push((key.into(), MetadataValue::U32(value)));
            self
        }

        /// Push a `u32` array metadata value.
        pub fn meta_array_u32(mut self, key: &str, values: &[u32]) -> Self {
            self.metadata.push((
                key.into(),
                MetadataValue::Array(super::super::value::MetadataArray {
                    element_type: MetadataValueType::U32,
                    values: values.iter().copied().map(MetadataValue::U32).collect(),
                }),
            ));
            self
        }

        /// Push a string metadata value.
        pub fn meta_string(mut self, key: &str, value: &str) -> Self {
            self.metadata
                .push((key.into(), MetadataValue::String(value.into())));
            self
        }

        /// Push a string-array metadata value.
        pub fn meta_array_string(mut self, key: &str, values: &[&str]) -> Self {
            self.metadata.push((
                key.into(),
                MetadataValue::Array(super::super::value::MetadataArray {
                    element_type: MetadataValueType::String,
                    values: values
                        .iter()
                        .map(|s| MetadataValue::String((*s).into()))
                        .collect(),
                }),
            ));
            self
        }

        /// Push an `i32` array metadata value.
        pub fn meta_array_i32(mut self, key: &str, values: &[i32]) -> Self {
            self.metadata.push((
                key.into(),
                MetadataValue::Array(super::super::value::MetadataArray {
                    element_type: MetadataValueType::I32,
                    values: values.iter().copied().map(MetadataValue::I32).collect(),
                }),
            ));
            self
        }

        /// Push an `f32` array metadata value.
        pub fn meta_array_f32(mut self, key: &str, values: &[f32]) -> Self {
            self.metadata.push((
                key.into(),
                MetadataValue::Array(super::super::value::MetadataArray {
                    element_type: MetadataValueType::F32,
                    values: values.iter().copied().map(MetadataValue::F32).collect(),
                }),
            ));
            self
        }

        /// Append a tensor info record.
        pub fn tensor(mut self, info: TensorInfo) -> Self {
            self.tensors.push(info);
            self
        }

        /// Serialize to GGUF bytes (header + metadata + tensor infos only).
        pub fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&GGUF_MAGIC);
            out.extend_from_slice(&self.version.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());

            let mut metadata = self.metadata;
            if let Some(alignment) = self.alignment {
                metadata.push((ALIGNMENT_KEY.into(), MetadataValue::U32(alignment)));
            }

            out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
            for (key, value) in &metadata {
                write_string(&mut out, key);
                write_value(&mut out, value);
            }

            for tensor in &self.tensors {
                write_string(&mut out, &tensor.name);
                out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
                for dim in &tensor.dimensions {
                    out.extend_from_slice(&dim.to_le_bytes());
                }
                let type_id = ggml_type_id(tensor.ggml_type);
                out.extend_from_slice(&type_id.to_le_bytes());
                out.extend_from_slice(&tensor.offset.to_le_bytes());
            }

            out
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn write_value(out: &mut Vec<u8>, value: &MetadataValue) {
        let type_id = value.value_type() as u32;
        out.extend_from_slice(&type_id.to_le_bytes());
        match value {
            MetadataValue::U8(v) => out.push(*v),
            MetadataValue::I8(v) => out.push(*v as u8),
            MetadataValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::Bool(v) => out.push(u8::from(*v)),
            MetadataValue::String(v) => write_string(out, v),
            MetadataValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::Array(array) => {
                out.extend_from_slice(&(array.element_type as u32).to_le_bytes());
                out.extend_from_slice(&(array.values.len() as u64).to_le_bytes());
                for element in &array.values {
                    // Array elements omit a per-element type tag — only the payload.
                    write_value_payload(out, element);
                }
            }
        }
    }

    fn write_value_payload(out: &mut Vec<u8>, value: &MetadataValue) {
        match value {
            MetadataValue::U8(v) => out.push(*v),
            MetadataValue::I8(v) => out.push(*v as u8),
            MetadataValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::Bool(v) => out.push(u8::from(*v)),
            MetadataValue::String(v) => write_string(out, v),
            MetadataValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            MetadataValue::Array(array) => {
                out.extend_from_slice(&(array.element_type as u32).to_le_bytes());
                out.extend_from_slice(&(array.values.len() as u64).to_le_bytes());
                for element in &array.values {
                    write_value_payload(out, element);
                }
            }
        }
    }

    fn ggml_type_id(ggml_type: GgmlType) -> u32 {
        match ggml_type {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q8_1 => 9,
            GgmlType::Q2K => 10,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Q8K => 15,
            GgmlType::Unknown(id) => id,
            other => panic!("test builder missing ggml type id for {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::GgufBuilder;
    use super::*;

    #[test]
    fn parses_minimal_container() {
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .name("toy")
            .meta_u32("llama.block_count", 2)
            .meta_array_u32("llama.expert_used_count", &[1, 2, 3])
            .tensor(TensorInfo {
                name: "token_embd.weight".into(),
                dimensions: vec![4, 8],
                ggml_type: GgmlType::F32,
                offset: 0,
            })
            .build();

        let file = GgufFile::from_bytes(&bytes).unwrap();
        assert_eq!(file.header.version, 3);
        assert_eq!(file.header.tensor_count, 1);
        assert_eq!(file.architecture(), Some("llama"));
        assert_eq!(file.name(), Some("toy"));
        assert_eq!(file.alignment, DEFAULT_ALIGNMENT);
        assert_eq!(file.get("llama.block_count"), Some(&MetadataValue::U32(2)));

        let tensor = file.tensor("token_embd.weight").unwrap();
        assert_eq!(tensor.dimensions, vec![4, 8]);
        assert_eq!(tensor.ggml_type, GgmlType::F32);
        assert_eq!(tensor.offset, 0);
        assert!(file.data_offset >= 24); // at least the fixed header
        assert_eq!(file.data_offset % file.alignment, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let err = GgufFile::from_bytes(b"FFFF........").unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Gguf(GgufError::InvalidMagic { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = GgufBuilder::new().build();
        // version sits at bytes 4..8
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = GgufFile::from_bytes(&bytes).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Gguf(GgufError::UnsupportedVersion { version: 99, .. })
        ));
    }

    #[test]
    fn custom_alignment_is_honoured() {
        let bytes = GgufBuilder::new()
            .alignment(64)
            .tensor(TensorInfo {
                name: "w".into(),
                dimensions: vec![1],
                ggml_type: GgmlType::F16,
                offset: 64,
            })
            .build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        assert_eq!(file.alignment, 64);
        assert_eq!(file.data_offset % 64, 0);
        assert_eq!(
            file.absolute_offset(file.tensor("w").unwrap()).unwrap(),
            file.data_offset + 64
        );
    }

    #[test]
    fn rejects_unaligned_tensor_offset() {
        let bytes = GgufBuilder::new()
            .tensor(TensorInfo {
                name: "w".into(),
                dimensions: vec![1],
                ggml_type: GgmlType::F32,
                offset: 3, // not multiple of 32
            })
            .build();
        let err = GgufFile::from_bytes(&bytes).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Gguf(GgufError::Invalid {
                context: "tensor offset",
                ..
            })
        ));
    }
}
