//! Decoded GGUF metadata values.
//!
//! Values are owned so the parser can drop the input stream after reading
//! header + tensor info (weight bytes stay on disk until Phase 5).

use super::types::MetadataValueType;

/// One metadata value from a GGUF key-value pair.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Boolean (`0` / `1` on the wire).
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array (element type recorded for round-trip / inspection).
    Array(MetadataArray),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
}

impl MetadataValue {
    /// Wire type tag for this value.
    #[must_use]
    pub fn value_type(&self) -> MetadataValueType {
        match self {
            Self::U8(_) => MetadataValueType::U8,
            Self::I8(_) => MetadataValueType::I8,
            Self::U16(_) => MetadataValueType::U16,
            Self::I16(_) => MetadataValueType::I16,
            Self::U32(_) => MetadataValueType::U32,
            Self::I32(_) => MetadataValueType::I32,
            Self::F32(_) => MetadataValueType::F32,
            Self::Bool(_) => MetadataValueType::Bool,
            Self::String(_) => MetadataValueType::String,
            Self::Array(_) => MetadataValueType::Array,
            Self::U64(_) => MetadataValueType::U64,
            Self::I64(_) => MetadataValueType::I64,
            Self::F64(_) => MetadataValueType::F64,
        }
    }

    /// Borrow as `&str` when this is a string value.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Copy as `u32` when the value is an unsigned 32-bit integer.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            _ => None,
        }
    }

    /// Copy as `u64` when the value is an unsigned integer (32 or 64 bit).
    ///
    /// Many GGUF writers store counts as either width; readers are encouraged
    /// to accept both.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(u64::from(*v)),
            _ => None,
        }
    }
}

/// Array metadata payload.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataArray {
    /// Declared element type from the wire header of this array.
    pub element_type: MetadataValueType,
    /// Elements (`len` on the wire is `values.len()`, not byte length).
    pub values: Vec<MetadataValue>,
}

/// One metadata key-value entry.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    /// Hierarchical ASCII key (e.g. `general.architecture`).
    pub key: String,
    /// Typed value.
    pub value: MetadataValue,
}
