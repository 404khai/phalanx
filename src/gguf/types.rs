//! GGUF constants and wire-format enumerations.
//!
//! Numbers match the official ggml GGUF specification:
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

/// Magic bytes at offset 0 (`G G U F`).
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// Current structural format version described by the ggml docs.
pub const GGUF_VERSION_V3: u32 = 3;

/// Previous structural version still common in the wild.
pub const GGUF_VERSION_V2: u32 = 2;

/// Versions accepted by [`crate::gguf::GgufFile::parse`].
pub const SUPPORTED_VERSIONS: &[u32] = &[GGUF_VERSION_V2, GGUF_VERSION_V3];

/// Default tensor-data alignment when `general.alignment` is absent.
///
/// Chosen by the GGUF authors for efficient `mmap` / SIMD loads.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata key that overrides [`DEFAULT_ALIGNMENT`].
pub const ALIGNMENT_KEY: &str = "general.alignment";

/// Standardized architecture key used by almost every model card.
pub const ARCHITECTURE_KEY: &str = "general.architecture";

/// Standardized display-name key.
pub const NAME_KEY: &str = "general.name";

/// Safety caps so a malicious header cannot force multi-gigabyte allocations
/// before we have validated the rest of the container.
pub mod limits {
    /// Maximum metadata key length (spec: ≤ 65535).
    pub const MAX_KEY_LEN: u64 = 65_535;
    /// Maximum tensor name length (spec: ≤ 64).
    pub const MAX_TENSOR_NAME_LEN: u64 = 64;
    /// Soft cap for other UTF-8 strings (tokenizer pieces can be large).
    pub const MAX_STRING_LEN: u64 = 64 * 1024 * 1024;
    /// Maximum dimensions on a single tensor info record.
    pub const MAX_DIMENSIONS: u32 = 8;
    /// Maximum metadata KV count (far above real models).
    pub const MAX_METADATA_COUNT: u64 = 1_000_000;
    /// Maximum tensor count (far above real models).
    pub const MAX_TENSOR_COUNT: u64 = 1_000_000;
    /// Maximum array element count (vocab / merges can be huge).
    pub const MAX_ARRAY_LEN: u64 = 100_000_000;
    /// Nesting depth for metadata arrays.
    pub const MAX_ARRAY_DEPTH: u32 = 8;
}

/// `gguf_metadata_value_type` discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MetadataValueType {
    /// `u8`
    U8 = 0,
    /// `i8`
    I8 = 1,
    /// little-endian `u16`
    U16 = 2,
    /// little-endian `i16`
    I16 = 3,
    /// little-endian `u32`
    U32 = 4,
    /// little-endian `i32`
    I32 = 5,
    /// IEEE-754 `f32`
    F32 = 6,
    /// 1-byte bool (`0` / `1`)
    Bool = 7,
    /// length-prefixed UTF-8
    String = 8,
    /// typed array (element type + length + values)
    Array = 9,
    /// little-endian `u64`
    U64 = 10,
    /// little-endian `i64`
    I64 = 11,
    /// IEEE-754 `f64`
    F64 = 12,
}

impl MetadataValueType {
    /// Decode a raw type tag.
    ///
    /// # Errors
    ///
    /// Returns [`crate::gguf::GgufError::UnknownValueType`] for unknown ids.
    pub fn from_u32(value: u32) -> Result<Self, crate::gguf::GgufError> {
        Ok(match value {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => {
                return Err(crate::gguf::GgufError::UnknownValueType { type_id: other });
            }
        })
    }
}

/// `ggml_type` used by tensor info records.
///
/// Unknown future tags are preserved as [`GgmlType::Unknown`] so inspection
/// still works when ggml adds formats Phalanx does not yet dequantize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// 4-bit quantization, block variant 0
    Q4_0,
    /// 4-bit quantization, block variant 1
    Q4_1,
    /// 5-bit quantization, block variant 0
    Q5_0,
    /// 5-bit quantization, block variant 1
    Q5_1,
    /// 8-bit quantization, block variant 0
    Q8_0,
    /// 8-bit quantization, block variant 1
    Q8_1,
    /// k-quant Q2
    Q2K,
    /// k-quant Q3
    Q3K,
    /// k-quant Q4
    Q4K,
    /// k-quant Q5
    Q5K,
    /// k-quant Q6
    Q6K,
    /// k-quant Q8
    Q8K,
    /// Importance-matrix `IQ2_XXS`
    Iq2Xxs,
    /// Importance-matrix `IQ2_XS`
    Iq2Xs,
    /// Importance-matrix `IQ3_XXS`
    Iq3Xxs,
    /// Importance-matrix `IQ1_S`
    Iq1S,
    /// Importance-matrix `IQ4_NL`
    Iq4Nl,
    /// Importance-matrix `IQ3_S`
    Iq3S,
    /// Importance-matrix `IQ2_S`
    Iq2S,
    /// Importance-matrix `IQ4_XS`
    Iq4Xs,
    /// signed 8-bit integer
    I8,
    /// signed 16-bit integer
    I16,
    /// signed 32-bit integer
    I32,
    /// signed 64-bit integer
    I64,
    /// 64-bit float
    F64,
    /// Importance-matrix `IQ1_M`
    Iq1M,
    /// bfloat16
    Bf16,
    /// Ternary quant `TQ1_0`
    Tq1_0,
    /// Ternary quant `TQ2_0`
    Tq2_0,
    /// Microscaling FP4 (`MXFP4`)
    Mxfp4,
    /// Forward-compatible catch-all for tags ggml adds later.
    Unknown(u32),
}

impl GgmlType {
    /// Decode a raw `ggml_type` tag.
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::Mxfp4,
            other => Self::Unknown(other),
        }
    }

    /// Stable lowercase name for logs and inspection.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
            Self::Q8K => "q8_k",
            Self::Iq2Xxs => "iq2_xxs",
            Self::Iq2Xs => "iq2_xs",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq1S => "iq1_s",
            Self::Iq4Nl => "iq4_nl",
            Self::Iq3S => "iq3_s",
            Self::Iq2S => "iq2_s",
            Self::Iq4Xs => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::Iq1M => "iq1_m",
            Self::Bf16 => "bf16",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
            Self::Mxfp4 => "mxfp4",
            Self::Unknown(_) => "unknown",
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "unknown({id})"),
            other => f.write_str(other.name()),
        }
    }
}

/// Round `offset` up to the next multiple of `alignment`.
///
/// Mirrors the reference `align_offset` helper from the GGUF spec so tensor
/// data begins on an `mmap`-friendly boundary.
#[must_use]
pub fn align_offset(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    let rem = offset % alignment;
    if rem == 0 {
        offset
    } else {
        offset + (alignment - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_offset_rounds_up() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }

    #[test]
    fn ggml_type_round_trips_common_tags() {
        assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(12), GgmlType::Q4K);
        assert!(matches!(GgmlType::from_u32(999), GgmlType::Unknown(999)));
    }
}
