//! Text → token id encoders.
//!
//! # Design tradeoff
//!
//! | Approach | Pros | Cons |
//! |---|---|---|
//! | Full `SentencePiece` / `tokenizers` crate | Exact HF parity | Heavy deps; hides the algorithm |
//! | Greedy longest-match + optional BPE merges (chosen) | Small, auditable, matches GGUF data we already parse | Edge cases vs production SP may differ |
//!
//! Phase 4 prioritises a correct *data path* (load vocab → encode prompt →
//! decode ids) over perfect parity with every HF tokenizer quirk. Refine with
//! golden tests against llama.cpp when real models land in Phase 5+.

use std::collections::HashMap;

use super::error::TokenizerError;
use super::vocab::Vocabulary;
use crate::errors::Result;

/// One BPE merge rule: replace adjacent `(left, right)` with `left + right`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    /// Left piece.
    pub left: String,
    /// Right piece.
    pub right: String,
}

impl MergeRule {
    /// Parse a GGUF merge line `"left right"`.
    pub(super) fn parse(line: &str) -> Option<Self> {
        let (left, right) = line.split_once(' ')?;
        if left.is_empty() || right.is_empty() {
            return None;
        }
        Some(Self {
            left: left.to_owned(),
            right: right.to_owned(),
        })
    }

    /// Concatenated piece produced by applying this merge.
    #[must_use]
    pub fn merged(&self) -> String {
        format!("{}{}", self.left, self.right)
    }
}

/// Greedy longest-match encode used when merges are absent (Llama-style).
///
/// Spaces become the `SentencePiece` marker `▁`, and a leading `▁` is prepended
/// so the first word matches pieces like `▁Hello` (SP word-start convention).
pub(super) fn encode_greedy(vocab: &Vocabulary, text: &str) -> Result<Vec<u32>> {
    let prepared = prepare_sentencepiece_text(text);
    let chars: Vec<char> = prepared.chars().collect();
    let mut ids = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let mut matched = None;
        // Longest-first: try the remainder, then shrink until a piece hits.
        for end in (i + 1..=chars.len()).rev() {
            let candidate: String = chars[i..end].iter().collect();
            if let Some(id) = vocab.id(&candidate) {
                matched = Some((id, end));
                break;
            }
        }

        let Some((id, end)) = matched else {
            // Single-char / byte fallback via `<0xXX>` pieces when present.
            let ch = chars[i];
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            if encoded.len() == 1
                && let Some(id) = vocab.id(&format!("<0x{:02X}>", encoded.as_bytes()[0]))
            {
                ids.push(id);
                i += 1;
                continue;
            }

            return Err(TokenizerError::EncodeFailure {
                context: chars[i..].iter().take(16).collect(),
            }
            .into());
        };

        ids.push(id);
        i = end;
    }

    Ok(ids)
}

/// Map surface text into the `SentencePiece` piece alphabet.
fn prepare_sentencepiece_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let replaced = text.replace(' ', "\u{2581}");
    if replaced.starts_with('\u{2581}') {
        replaced
    } else {
        format!("\u{2581}{replaced}")
    }
}

/// GPT-2 style BPE using ordered merge rules from `tokenizer.ggml.merges`.
pub(super) fn encode_bpe(vocab: &Vocabulary, merges: &[MergeRule], text: &str) -> Result<Vec<u32>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    // Rank: lower index in the merges list = higher priority (applied first).
    let mut rank: HashMap<(String, String), usize> = HashMap::with_capacity(merges.len());
    for (i, rule) in merges.iter().enumerate() {
        rank.insert((rule.left.clone(), rule.right.clone()), i);
    }

    // Start from individual characters (GGUF gpt2 tokens are usually unicode chars / merges).
    let mut symbols: Vec<String> = text.chars().map(|ch| ch.to_string()).collect();

    loop {
        let mut best: Option<(usize, usize)> = None; // (rank, index)
        for i in 0..symbols.len().saturating_sub(1) {
            let key = (symbols[i].clone(), symbols[i + 1].clone());
            if let Some(&r) = rank.get(&key)
                && best.is_none_or(|(best_rank, _)| r < best_rank)
            {
                best = Some((r, i));
            }
        }

        let Some((_, idx)) = best else {
            break;
        };

        let joined = format!("{}{}", symbols[idx], symbols[idx + 1]);
        symbols[idx] = joined;
        symbols.remove(idx + 1);
    }

    let mut ids = Vec::with_capacity(symbols.len());
    for symbol in &symbols {
        let id = vocab
            .id(symbol)
            .ok_or_else(|| TokenizerError::EncodeFailure {
                context: symbol.clone(),
            })?;
        ids.push(id);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_merge_line() {
        let rule = MergeRule::parse("t h").unwrap();
        assert_eq!(rule.left, "t");
        assert_eq!(rule.right, "h");
        assert_eq!(rule.merged(), "th");
    }
}
