//! Backing bytes for a loaded GGUF file.
//!
//! # Why `mmap`
//!
//! Multi-GB checkpoints should not be copied into anonymous RAM on open.
//! Mapping the file lets the OS page weights in on demand and share pages
//! across processes. Phalanx keeps the map **read-only**.
//!
//! # Safety island
//!
//! `memmap2::MmapOptions::map` is `unsafe` because the OS can change file
//! contents under a mapping if another writer truncates or mutates the file.
//! This module is the **only** place Phalanx opts into `unsafe_code`. Callers
//! must treat model paths as immutable while a [`WeightStorage`] lives.

#![allow(unsafe_code)]

use std::fs::File;
use std::path::Path;

use memmap2::MmapOptions;

use super::error::WeightsError;
use crate::errors::Result;

/// Owned or memory-mapped file bytes.
#[derive(Debug)]
pub enum WeightStorage {
    /// Read-only file mapping (preferred for real checkpoints).
    Mapped(memmap2::Mmap),
    /// Heap copy (tests and tiny fixtures).
    Owned(Vec<u8>),
}

impl WeightStorage {
    /// Memory-map `path` read-only.
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError::Mmap`] or I/O errors.
    ///
    /// # Safety considerations
    ///
    /// The returned mapping assumes the file is not truncated or rewritten
    /// while in use. That is the normal contract for model weight files.
    pub fn mmap_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: We request a read-only map. Phalanx never writes through it.
        // External mutation/truncation of `path` while mapped is undefined at
        // the OS level; callers must not modify model files during load/infer.
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|err| WeightsError::Mmap {
                    reason: format!("{} ({})", err, path.display()),
                })?
        };
        Ok(Self::Mapped(mmap))
    }

    /// Wrap an owned buffer (full GGUF file bytes).
    #[must_use]
    pub fn owned(bytes: Vec<u8>) -> Self {
        Self::Owned(bytes)
    }

    /// Borrow the full file bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(mmap) => mmap,
            Self::Owned(bytes) => bytes,
        }
    }

    /// Byte length of the backing store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// `true` when the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` when this store is an OS file mapping.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }
}
