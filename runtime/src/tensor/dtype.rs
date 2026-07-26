//! Element type tags for tensor storage.
//!
//! Phase 2 materializes only [`DType::F32`]. Later phases add half-precision
//! activations and GGUF quantized weight types without changing call sites that
//! already thread a [`DType`] through loaders and kernels.

/// Logical element type of a tensor buffer.
///
/// This is intentionally a tag, not a Rust generic parameter. Generics would
/// force every op to monomorphize across dtypes; a tag lets dispatch evolve
/// toward function tables / SIMD kernels as quantized formats land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE-754 float — default for activations and Phase 2 math.
    F32,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    pub const fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
        }
    }

    /// Human-readable name used in errors and model inspection.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_size_matches_rust() {
        assert_eq!(DType::F32.size_of(), std::mem::size_of::<f32>());
    }
}
