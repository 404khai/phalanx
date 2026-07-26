//! Vocabulary tables loaded from `tokenizer.ggml.tokens` (+ optional scores/types).
//!
//! Token ids are [`u32`] to match GGUF vocabulary indices while keeping the
//! id space dense and cache-friendly during decode.

use std::collections::HashMap;

use super::error::TokenizerError;
use crate::errors::Result;

/// GGUF `tokenizer.ggml.token_type` discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TokenType {
    /// Ordinary vocabulary piece.
    Normal = 1,
    /// Unknown / OOV marker piece.
    Unknown = 2,
    /// Control / special structural token.
    Control = 3,
    /// User-defined added token.
    UserDefined = 4,
    /// Unused slot in a sparse vocab.
    Unused = 5,
    /// Single-byte fallback piece (`<0xXX>`).
    Byte = 6,
}

impl TokenType {
    /// Decode a raw type tag; unknown positives map to [`TokenType::Normal`].
    #[must_use]
    pub fn from_i32(value: i32) -> Self {
        match value {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            // `1` and any unknown tag: treat as ordinary text pieces.
            _ => Self::Normal,
        }
    }

    /// Control or unused pieces are usually omitted from detokenized text.
    #[must_use]
    pub const fn is_skippable_on_decode(self) -> bool {
        matches!(self, Self::Control | Self::Unused)
    }
}

/// Owned vocabulary: id → piece, with optional scores / types and a reverse map.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens: Vec<String>,
    scores: Option<Vec<f32>>,
    token_types: Option<Vec<TokenType>>,
    /// Reverse index for encode; built once at load time.
    piece_to_id: HashMap<String, u32>,
}

impl Vocabulary {
    /// Construct from parallel arrays already validated for length.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::LengthMismatch`] when optional arrays disagree
    /// with `tokens.len()`.
    pub fn new(
        tokens: Vec<String>,
        scores: Option<Vec<f32>>,
        token_types: Option<Vec<TokenType>>,
    ) -> Result<Self> {
        let n = tokens.len();
        if let Some(scores) = &scores
            && scores.len() != n
        {
            return Err(TokenizerError::LengthMismatch {
                key: "tokenizer.ggml.scores",
                expected: n,
                got: scores.len(),
            }
            .into());
        }
        if let Some(token_types) = &token_types
            && token_types.len() != n
        {
            return Err(TokenizerError::LengthMismatch {
                key: "tokenizer.ggml.token_type",
                expected: n,
                got: token_types.len(),
            }
            .into());
        }

        // Last duplicate piece wins — mirrors common HF export behaviour when
        // added tokens overwrite earlier ids.
        let mut piece_to_id = HashMap::with_capacity(n);
        for (id, piece) in tokens.iter().enumerate() {
            let id = u32::try_from(id).map_err(|_| TokenizerError::InvalidType {
                key: "tokenizer.ggml.tokens",
                reason: format!("vocabulary length {n} exceeds u32::MAX"),
            })?;
            piece_to_id.insert(piece.clone(), id);
        }

        Ok(Self {
            tokens,
            scores,
            token_types,
            piece_to_id,
        })
    }

    /// Number of vocabulary entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` when the vocabulary has no pieces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Borrow the piece string for `id`.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::UnknownTokenId`] when `id` is out of range.
    pub fn piece(&self, id: u32) -> Result<&str> {
        self.tokens
            .get(id as usize)
            .map(String::as_str)
            .ok_or_else(|| {
                TokenizerError::UnknownTokenId {
                    id,
                    vocab_size: self.tokens.len(),
                }
                .into()
            })
    }

    /// Look up a piece's id, if present.
    #[must_use]
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    /// Token type for `id`, defaulting to [`TokenType::Normal`].
    #[must_use]
    pub fn token_type(&self, id: u32) -> TokenType {
        self.token_types
            .as_ref()
            .and_then(|types| types.get(id as usize).copied())
            .unwrap_or(TokenType::Normal)
    }

    /// Optional SentencePiece-style score for `id`.
    #[must_use]
    pub fn score(&self, id: u32) -> Option<f32> {
        self.scores
            .as_ref()
            .and_then(|scores| scores.get(id as usize).copied())
    }

    /// Borrow all pieces in id order.
    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}
