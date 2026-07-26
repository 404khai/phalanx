//! Phalanx CLI entrypoint.
//!
//! Phase 4 still ships a thin binary: logging init + version banner. A real
//! `inspect` / `generate` CLI lands in Phase 16 — keeping `main` small avoids
//! baking a framework choice before command requirements are clear.

use anyhow::{Context, Result};
use tracing::info;

use phalanx::{LogConfig, RUNTIME_NAME, VERSION, init_logging};

fn main() -> Result<()> {
    init_logging(&LogConfig::default()).context("failed to initialize logging")?;

    info!(version = VERSION, "{RUNTIME_NAME} starting");

    println!("{RUNTIME_NAME} v{VERSION}");
    println!("Phase 4 — tokenizer ready; weight loading (mmap/dequant) comes next.");
    println!("Set RUST_LOG=phalanx=debug for verbose diagnostics.");
    println!("Load a tokenizer with: Tokenizer::from_gguf(&GgufFile::from_path(...)? )?");

    Ok(())
}
