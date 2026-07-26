//! Tokenizer facade: load from GGUF, encode text, decode token ids.
//!
//! # Pipeline
//!
//! ```text
//! GgufFile metadata ──► Vocabulary + SpecialTokens + merges
//!                              │
//!                     encode ◄─┴─► decode
//! ```

use tracing::debug;

use super::decode::{DecodeOptions, piece_to_bytes};
use super::encode::{MergeRule, encode_bpe, encode_greedy};
use super::error::TokenizerError;
use super::keys;
use super::special::SpecialTokens;
use super::vocab::{TokenType, Vocabulary};
use crate::errors::Result;
use crate::gguf::{GgufFile, MetadataValue};

/// Tokenizer family declared by `tokenizer.ggml.model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerModel {
    /// Llama-style `SentencePiece` pieces + scores.
    Llama,
    /// Replit `SentencePiece` variant.
    Replit,
    /// GPT-2 / GPT-NeoX BPE with merge rules.
    Gpt2,
    /// RWKV tokenizer.
    Rwkv,
    /// Unrecognised model string — still usable if tokens are present.
    Unknown,
}

impl TokenizerModel {
    /// Parse the GGUF model name string.
    #[must_use]
    pub fn parse(name: &str) -> Self {
        match name {
            "llama" => Self::Llama,
            "replit" => Self::Replit,
            "gpt2" => Self::Gpt2,
            "rwkv" => Self::Rwkv,
            _ => Self::Unknown,
        }
    }

    /// Prefer BPE merges when this family typically ships them.
    #[must_use]
    pub const fn prefers_bpe(self) -> bool {
        matches!(self, Self::Gpt2)
    }
}

/// Options for [`Tokenizer::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Prepend the BOS id when the model declares one.
    pub add_bos: bool,
    /// Append the EOS id when the model declares one.
    pub add_eos: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            // Prefill normally starts with BOS for decoder-only Llama-family models.
            add_bos: true,
            add_eos: false,
        }
    }
}

/// Loaded tokenizer ready for encode / decode.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    model: TokenizerModel,
    vocab: Vocabulary,
    special: SpecialTokens,
    merges: Vec<MergeRule>,
    added_tokens: Vec<String>,
}

impl Tokenizer {
    /// Load tokenizer tables from an already-parsed [`GgufFile`].
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError`] when required keys are missing or malformed.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let model_name = required_str(file, keys::MODEL)?;
        let model = TokenizerModel::parse(model_name);

        let tokens = required_string_array(file, keys::TOKENS)?;
        let scores = optional_f32_array(file, keys::SCORES, tokens.len())?;
        let raw_types = optional_i32_array(file, keys::TOKEN_TYPE, tokens.len())?;
        let token_types =
            raw_types.map(|values| values.into_iter().map(TokenType::from_i32).collect());

        let vocab = Vocabulary::new(tokens, scores, token_types)?;
        let special = load_special_tokens(file, vocab.len())?;

        let merges = match file.get(keys::MERGES) {
            None => Vec::new(),
            Some(value) => string_array(value, keys::MERGES)?
                .into_iter()
                .filter_map(|line| MergeRule::parse(&line))
                .collect(),
        };

        let added_tokens = match file.get(keys::ADDED_TOKENS) {
            None => Vec::new(),
            Some(value) => string_array(value, keys::ADDED_TOKENS)?,
        };

        debug!(
            model = ?model,
            vocab_size = vocab.len(),
            merges = merges.len(),
            "loaded tokenizer from GGUF metadata"
        );

        Ok(Self {
            model,
            vocab,
            special,
            merges,
            added_tokens,
        })
    }

    /// Construct from pre-built parts (unit tests and non-GGUF loaders).
    #[must_use]
    pub fn from_parts(
        model: TokenizerModel,
        vocab: Vocabulary,
        special: SpecialTokens,
        merges: Vec<MergeRule>,
    ) -> Self {
        Self {
            model,
            vocab,
            special,
            merges,
            added_tokens: Vec::new(),
        }
    }

    /// Tokenizer family.
    #[must_use]
    pub fn model(&self) -> TokenizerModel {
        self.model
    }

    /// Borrow the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Borrow special token ids.
    #[must_use]
    pub fn special(&self) -> &SpecialTokens {
        &self.special
    }

    /// BPE merge rules (empty for pure SentencePiece-style files).
    #[must_use]
    pub fn merges(&self) -> &[MergeRule] {
        &self.merges
    }

    /// Added-token strings from metadata, if any.
    #[must_use]
    pub fn added_tokens(&self) -> &[String] {
        &self.added_tokens
    }

    /// Encode `text` into token ids.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::EncodeFailure`] when a span cannot be mapped.
    pub fn encode(&self, text: &str, options: EncodeOptions) -> Result<Vec<u32>> {
        let mut ids = if !self.merges.is_empty() && self.model.prefers_bpe() {
            encode_bpe(&self.vocab, &self.merges, text)?
        } else if !self.merges.is_empty() {
            // Some Llama exports still ship merges; prefer BPE when present.
            encode_bpe(&self.vocab, &self.merges, text)?
        } else {
            encode_greedy(&self.vocab, text)?
        };

        if options.add_bos
            && let Some(bos) = self.special.bos
        {
            ids.insert(0, bos);
        }
        if options.add_eos
            && let Some(eos) = self.special.eos
        {
            ids.push(eos);
        }

        Ok(ids)
    }

    /// Decode token ids with default display options (skip specials/control).
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::UnknownTokenId`] for out-of-range ids.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.decode_with_options(ids, DecodeOptions::default())
    }

    /// Decode token ids with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::UnknownTokenId`] for out-of-range ids, or
    /// invalid UTF-8 if byte pieces concatenate into a partial sequence
    /// (lossy replacement is intentionally avoided — callers see the error).
    pub fn decode_with_options(&self, ids: &[u32], options: DecodeOptions) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if options.skip_special && self.special.contains(id) {
                continue;
            }
            let token_type = self.vocab.token_type(id);
            if options.skip_control && token_type.is_skippable_on_decode() {
                continue;
            }
            let piece = self.vocab.piece(id)?;
            bytes.extend_from_slice(&piece_to_bytes(piece));
        }

        String::from_utf8(bytes).map_err(|err| {
            TokenizerError::InvalidUtf8 {
                reason: err.to_string(),
            }
            .into()
        })
    }

    /// Decode a single id to its vocabulary piece (no ▁ rewriting).
    ///
    /// Useful for debugging and logits inspection UIs.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::UnknownTokenId`] when out of range.
    pub fn id_to_piece(&self, id: u32) -> Result<&str> {
        self.vocab.piece(id)
    }
}

fn load_special_tokens(file: &GgufFile, vocab_size: usize) -> Result<SpecialTokens> {
    Ok(SpecialTokens {
        bos: optional_token_id(file, keys::BOS_TOKEN_ID, "bos", vocab_size)?,
        eos: optional_token_id(file, keys::EOS_TOKEN_ID, "eos", vocab_size)?,
        unknown: optional_token_id(file, keys::UNKNOWN_TOKEN_ID, "unknown", vocab_size)?,
        separator: optional_token_id(file, keys::SEPARATOR_TOKEN_ID, "separator", vocab_size)?,
        padding: optional_token_id(file, keys::PADDING_TOKEN_ID, "padding", vocab_size)?,
    })
}

fn optional_token_id(
    file: &GgufFile,
    key: &'static str,
    name: &'static str,
    vocab_size: usize,
) -> Result<Option<u32>> {
    let Some(value) = file.get(key) else {
        return Ok(None);
    };
    let id = match value {
        MetadataValue::U32(v) => *v,
        MetadataValue::I32(v) => u32::try_from(*v).map_err(|_| TokenizerError::InvalidType {
            key,
            reason: format!("negative special token id {v}"),
        })?,
        other => {
            return Err(TokenizerError::InvalidType {
                key,
                reason: format!("expected u32/i32, got {:?}", other.value_type()),
            }
            .into());
        }
    };
    if usize::try_from(id).unwrap_or(usize::MAX) >= vocab_size {
        return Err(TokenizerError::InvalidSpecialToken {
            name,
            id,
            vocab_size,
        }
        .into());
    }
    Ok(Some(id))
}

fn required_str<'a>(file: &'a GgufFile, key: &'static str) -> Result<&'a str> {
    let value = file.get(key).ok_or(TokenizerError::MissingKey { key })?;
    value.as_str().ok_or_else(|| {
        TokenizerError::InvalidType {
            key,
            reason: format!("expected string, got {:?}", value.value_type()),
        }
        .into()
    })
}

fn required_string_array(file: &GgufFile, key: &'static str) -> Result<Vec<String>> {
    let value = file.get(key).ok_or(TokenizerError::MissingKey { key })?;
    string_array(value, key)
}

fn string_array(value: &MetadataValue, key: &'static str) -> Result<Vec<String>> {
    let array = value
        .as_array()
        .ok_or_else(|| TokenizerError::InvalidType {
            key,
            reason: format!("expected array, got {:?}", value.value_type()),
        })?;
    let mut out = Vec::with_capacity(array.values.len());
    for (i, element) in array.values.iter().enumerate() {
        let s = element
            .as_str()
            .ok_or_else(|| TokenizerError::InvalidType {
                key,
                reason: format!("element {i} is not a string ({:?})", element.value_type()),
            })?;
        out.push(s.to_owned());
    }
    Ok(out)
}

fn optional_f32_array(
    file: &GgufFile,
    key: &'static str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>> {
    let Some(value) = file.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| TokenizerError::InvalidType {
            key,
            reason: format!("expected array, got {:?}", value.value_type()),
        })?;
    if array.values.len() != expected_len {
        return Err(TokenizerError::LengthMismatch {
            key,
            expected: expected_len,
            got: array.values.len(),
        }
        .into());
    }
    let mut out = Vec::with_capacity(array.values.len());
    for (i, element) in array.values.iter().enumerate() {
        let v = element
            .as_f32()
            .ok_or_else(|| TokenizerError::InvalidType {
                key,
                reason: format!("element {i} is not f32 ({:?})", element.value_type()),
            })?;
        out.push(v);
    }
    Ok(Some(out))
}

fn optional_i32_array(
    file: &GgufFile,
    key: &'static str,
    expected_len: usize,
) -> Result<Option<Vec<i32>>> {
    let Some(value) = file.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| TokenizerError::InvalidType {
            key,
            reason: format!("expected array, got {:?}", value.value_type()),
        })?;
    if array.values.len() != expected_len {
        return Err(TokenizerError::LengthMismatch {
            key,
            expected: expected_len,
            got: array.values.len(),
        }
        .into());
    }
    let mut out = Vec::with_capacity(array.values.len());
    for (i, element) in array.values.iter().enumerate() {
        let v = element
            .as_i32()
            .ok_or_else(|| TokenizerError::InvalidType {
                key,
                reason: format!("element {i} is not i32 ({:?})", element.value_type()),
            })?;
        out.push(v);
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::gguf::test_support::GgufBuilder;

    #[test]
    fn from_gguf_loads_specials_and_round_trips() {
        let bytes = GgufBuilder::new()
            .architecture("llama")
            .meta_string(keys::MODEL, "llama")
            .meta_array_string(keys::TOKENS, &["<bos>", "<eos>", "▁Hello", "▁world", "!"])
            .meta_array_i32(keys::TOKEN_TYPE, &[3, 3, 1, 1, 1])
            .meta_u32(keys::BOS_TOKEN_ID, 0)
            .meta_u32(keys::EOS_TOKEN_ID, 1)
            .build();

        let file = GgufFile::from_bytes(&bytes).unwrap();
        let tok = Tokenizer::from_gguf(&file).unwrap();

        assert_eq!(tok.model(), TokenizerModel::Llama);
        assert_eq!(tok.special().bos, Some(0));
        assert_eq!(tok.special().eos, Some(1));
        assert_eq!(tok.vocab().len(), 5);

        let ids = tok
            .encode(
                "Hello world!",
                EncodeOptions {
                    add_bos: true,
                    add_eos: false,
                },
            )
            .unwrap();
        assert_eq!(ids, vec![0, 2, 3, 4]);
        assert_eq!(tok.decode(&ids).unwrap(), " Hello world!");
    }

    #[test]
    fn greedy_round_trip_hello_world() {
        let vocab = Vocabulary::new(
            vec![
                "<bos>".into(),
                "<eos>".into(),
                "▁Hello".into(),
                "▁world".into(),
                "!".into(),
            ],
            None,
            Some(vec![
                TokenType::Control,
                TokenType::Control,
                TokenType::Normal,
                TokenType::Normal,
                TokenType::Normal,
            ]),
        )
        .unwrap();

        let tok = Tokenizer::from_parts(
            TokenizerModel::Llama,
            vocab,
            SpecialTokens {
                bos: Some(0),
                eos: Some(1),
                ..SpecialTokens::default()
            },
            Vec::new(),
        );

        let ids = tok
            .encode(
                "Hello world!",
                EncodeOptions {
                    add_bos: true,
                    add_eos: false,
                },
            )
            .unwrap();
        assert_eq!(ids, vec![0, 2, 3, 4]);
        assert_eq!(tok.decode(&ids).unwrap(), " Hello world!");
    }

    #[test]
    fn bpe_merges_characters() {
        let vocab = Vocabulary::new(
            vec![
                "t".into(),
                "h".into(),
                "th".into(),
                "e".into(),
                "the".into(),
            ],
            None,
            None,
        )
        .unwrap();
        let merges = vec![
            MergeRule {
                left: "t".into(),
                right: "h".into(),
            },
            MergeRule {
                left: "th".into(),
                right: "e".into(),
            },
        ];
        let tok = Tokenizer::from_parts(
            TokenizerModel::Gpt2,
            vocab,
            SpecialTokens::default(),
            merges,
        );

        let ids = tok
            .encode(
                "the",
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .unwrap();
        assert_eq!(ids, vec![4]);
        assert_eq!(tok.decode(&ids).unwrap(), "the");
    }

    #[test]
    fn missing_tokens_key_errors() {
        let bytes = GgufBuilder::new().meta_string(keys::MODEL, "llama").build();
        let file = GgufFile::from_bytes(&bytes).unwrap();
        let err = Tokenizer::from_gguf(&file).unwrap_err();
        assert!(matches!(
            err,
            crate::PhalanxError::Tokenizer(TokenizerError::MissingKey { key: keys::TOKENS })
        ));
    }
}
