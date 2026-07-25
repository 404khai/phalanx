//! Little-endian binary reader with a running byte offset.
//!
//! Tracking `position` is required to compute the aligned start of
//! `tensor_data` after the variable-length header + tensor-info sections —
//! without ever loading the weight blob into RAM.

use std::io::Read;

use super::error::GgufError;
use super::types::{MetadataValueType, limits};
use super::value::{MetadataArray, MetadataValue};
use crate::errors::Result;

/// Streaming GGUF decoder over any [`Read`] source.
pub(super) struct GgufReader<R> {
    inner: R,
    position: u64,
}

impl<R: Read> GgufReader<R> {
    /// Wrap a reader; position starts at 0 (caller must not have consumed bytes).
    pub(super) fn new(inner: R) -> Self {
        Self { inner, position: 0 }
    }

    /// Bytes consumed so far.
    pub(super) fn position(&self) -> u64 {
        self.position
    }

    pub(super) fn read_exact(&mut self, buf: &mut [u8], context: &'static str) -> Result<()> {
        self.inner.read_exact(buf).map_err(|err| {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof { context }.into()
            } else {
                crate::PhalanxError::Io(err)
            }
        })?;
        self.position = self.position.saturating_add(buf.len() as u64);
        Ok(())
    }

    pub(super) fn read_u8(&mut self, context: &'static str) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf, context)?;
        Ok(buf[0])
    }

    pub(super) fn read_u16(&mut self, context: &'static str) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf, context)?;
        Ok(u16::from_le_bytes(buf))
    }

    pub(super) fn read_u32(&mut self, context: &'static str) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf, context)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub(super) fn read_u64(&mut self, context: &'static str) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf, context)?;
        Ok(u64::from_le_bytes(buf))
    }

    pub(super) fn read_i8(&mut self, context: &'static str) -> Result<i8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf, context)?;
        Ok(i8::from_le_bytes(buf))
    }

    pub(super) fn read_i16(&mut self, context: &'static str) -> Result<i16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf, context)?;
        Ok(i16::from_le_bytes(buf))
    }

    pub(super) fn read_i32(&mut self, context: &'static str) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf, context)?;
        Ok(i32::from_le_bytes(buf))
    }

    pub(super) fn read_i64(&mut self, context: &'static str) -> Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf, context)?;
        Ok(i64::from_le_bytes(buf))
    }

    pub(super) fn read_f32(&mut self, context: &'static str) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf, context)?;
        Ok(f32::from_le_bytes(buf))
    }

    pub(super) fn read_f64(&mut self, context: &'static str) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf, context)?;
        Ok(f64::from_le_bytes(buf))
    }

    pub(super) fn read_bool(&mut self, context: &'static str) -> Result<bool> {
        match self.read_u8(context)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(GgufError::InvalidBool { value }.into()),
        }
    }

    /// Length-prefixed UTF-8 string (`uint64` length + bytes).
    pub(super) fn read_string(&mut self, context: &'static str, max_len: u64) -> Result<String> {
        let len = self.read_u64(context)?;
        if len > max_len {
            return Err(GgufError::LimitExceeded {
                context,
                got: len,
                limit: max_len,
            }
            .into());
        }
        let len_usize = usize::try_from(len).map_err(|_| GgufError::LimitExceeded {
            context,
            got: len,
            limit: max_len,
        })?;
        let mut buf = vec![0u8; len_usize];
        self.read_exact(&mut buf, context)?;
        String::from_utf8(buf).map_err(|err| {
            GgufError::InvalidUtf8 {
                context,
                reason: err.to_string(),
            }
            .into()
        })
    }

    pub(super) fn read_value_type(&mut self, context: &'static str) -> Result<MetadataValueType> {
        let raw = self.read_u32(context)?;
        MetadataValueType::from_u32(raw).map_err(Into::into)
    }

    pub(super) fn read_value(
        &mut self,
        value_type: MetadataValueType,
        depth: u32,
    ) -> Result<MetadataValue> {
        Ok(match value_type {
            MetadataValueType::U8 => MetadataValue::U8(self.read_u8("metadata u8")?),
            MetadataValueType::I8 => MetadataValue::I8(self.read_i8("metadata i8")?),
            MetadataValueType::U16 => MetadataValue::U16(self.read_u16("metadata u16")?),
            MetadataValueType::I16 => MetadataValue::I16(self.read_i16("metadata i16")?),
            MetadataValueType::U32 => MetadataValue::U32(self.read_u32("metadata u32")?),
            MetadataValueType::I32 => MetadataValue::I32(self.read_i32("metadata i32")?),
            MetadataValueType::F32 => MetadataValue::F32(self.read_f32("metadata f32")?),
            MetadataValueType::Bool => MetadataValue::Bool(self.read_bool("metadata bool")?),
            MetadataValueType::String => {
                MetadataValue::String(self.read_string("metadata string", limits::MAX_STRING_LEN)?)
            }
            MetadataValueType::U64 => MetadataValue::U64(self.read_u64("metadata u64")?),
            MetadataValueType::I64 => MetadataValue::I64(self.read_i64("metadata i64")?),
            MetadataValueType::F64 => MetadataValue::F64(self.read_f64("metadata f64")?),
            MetadataValueType::Array => MetadataValue::Array(self.read_array(depth)?),
        })
    }

    fn read_array(&mut self, depth: u32) -> Result<MetadataArray> {
        if depth >= limits::MAX_ARRAY_DEPTH {
            return Err(GgufError::LimitExceeded {
                context: "metadata array depth",
                got: u64::from(depth) + 1,
                limit: u64::from(limits::MAX_ARRAY_DEPTH),
            }
            .into());
        }

        let element_type = self.read_value_type("metadata array element type")?;
        let len = self.read_u64("metadata array length")?;
        if len > limits::MAX_ARRAY_LEN {
            return Err(GgufError::LimitExceeded {
                context: "metadata array length",
                got: len,
                limit: limits::MAX_ARRAY_LEN,
            }
            .into());
        }

        let len_usize = usize::try_from(len).map_err(|_| GgufError::LimitExceeded {
            context: "metadata array length",
            got: len,
            limit: limits::MAX_ARRAY_LEN,
        })?;

        let mut values = Vec::with_capacity(len_usize.min(1024));
        for _ in 0..len_usize {
            values.push(self.read_value(element_type, depth + 1)?);
        }

        Ok(MetadataArray {
            element_type,
            values,
        })
    }
}
