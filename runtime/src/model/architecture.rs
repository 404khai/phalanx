//! Supported decoder-only architectures.
//!
//! Phase 6 focuses on the Llama family. Additional GGUF architectures that
//! share the same hyperparameter schema can be added here without changing
//! the [`super::ModelConfig`] field layout.

/// Model family declared by `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// Meta Llama / Llama-2 / Llama-3 style decoder (GGUF arch string `llama`).
    ///
    /// Reference: [LLaMA](https://arxiv.org/abs/2302.13971).
    Llama,
}

impl Architecture {
    /// Parse a GGUF `general.architecture` string.
    ///
    /// Returns `None` for architectures this runtime does not yet configure.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "llama" => Some(Self::Llama),
            _ => None,
        }
    }

    /// Canonical GGUF architecture string for this family.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama() {
        assert_eq!(Architecture::parse("llama"), Some(Architecture::Llama));
        assert_eq!(Architecture::parse("qwen2"), None);
    }
}
