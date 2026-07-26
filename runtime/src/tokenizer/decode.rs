//! Piece → text conversion rules shared by detokenizers.
//!
//! Llama-style `SentencePiece` vocabs use U+2581 (`▁`) as a word-boundary mark
//! that becomes a space in surface text. Byte pieces (`<0x0A>`) round-trip
//! UTF-8 bytes that are not standalone vocabulary entries.
//!
//! References:
//! - `SentencePiece` (Kudo & Richardson, 2018)
//! - llama.cpp `llama_token_to_piece` / detokenize behaviour

/// `SentencePiece` word-boundary marker stored inside many GGUF vocabs.
pub const SENTENCEPIECE_SPACE: char = '\u{2581}';

/// Options controlling how token ids become surface text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Drop BOS/EOS/PAD/UNK/separator ids declared in [`super::SpecialTokens`].
    pub skip_special: bool,
    /// Drop pieces whose [`super::TokenType`] is control or unused.
    pub skip_control: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            // Generation UIs almost always want specials stripped from display text.
            skip_special: true,
            skip_control: true,
        }
    }
}

/// Convert one vocabulary piece into UTF-8 bytes for concatenation.
///
/// Returns an empty buffer for pieces that should contribute no surface text
/// under the active options (handled by the caller before invoking this).
pub(super) fn piece_to_bytes(piece: &str) -> Vec<u8> {
    if let Some(byte) = parse_byte_piece(piece) {
        return vec![byte];
    }

    // Replace SentencePiece ▁ with ASCII space so "▁Hello▁world" → " Hello world".
    piece
        .chars()
        .map(|ch| if ch == SENTENCEPIECE_SPACE { ' ' } else { ch })
        .collect::<String>()
        .into_bytes()
}

/// Parse `<0xNN>` byte pieces used by Llama byte-fallback vocabs.
fn parse_byte_piece(piece: &str) -> Option<u8> {
    let rest = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if rest.len() != 2 {
        return None;
    }
    u8::from_str_radix(rest, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentencepiece_marker_becomes_space() {
        let bytes = piece_to_bytes("▁Hi");
        assert_eq!(String::from_utf8(bytes).unwrap(), " Hi");
    }

    #[test]
    fn byte_piece_decodes() {
        assert_eq!(piece_to_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(piece_to_bytes("<0xFF>"), vec![0xFF]);
    }
}
