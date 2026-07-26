//! Special token ids extracted from GGUF metadata.
//!
//! These markers bound prompts and generations (`bos` / `eos`) and fill
//! batch pads. Ids are optional because not every GGUF file declares every
//! role — callers should treat `None` as “role unused by this model”.

/// Roles referenced by the GGUF tokenizer section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpecialTokens {
    /// Beginning of sequence.
    pub bos: Option<u32>,
    /// End of sequence / stop token for greedy decode loops.
    pub eos: Option<u32>,
    /// Unknown / OOV piece.
    pub unknown: Option<u32>,
    /// Separator (e.g. between segments).
    pub separator: Option<u32>,
    /// Padding for batched sequences.
    pub padding: Option<u32>,
}

impl SpecialTokens {
    /// `true` when `id` matches any declared special role.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        [
            self.bos,
            self.eos,
            self.unknown,
            self.separator,
            self.padding,
        ]
        .into_iter()
        .flatten()
        .any(|special| special == id)
    }
}
