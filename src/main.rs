//! Phalanx CLI entrypoint.
//!
//! Phase 2 still ships a thin binary: logging init + version banner. Argument
//! parsing, model loading, and generation arrive in later phases — keeping
//! `main` small avoids baking a CLI framework choice before requirements are
//! clear.

use anyhow::{Context, Result};
use tracing::info;

use phalanx::{LogConfig, RUNTIME_NAME, VERSION, init_logging};

fn main() -> Result<()> {
    init_logging(&LogConfig::default()).context("failed to initialize logging")?;

    info!(version = VERSION, "{RUNTIME_NAME} starting");

    println!("{RUNTIME_NAME} v{VERSION}");
    println!("Phase 2 — tensor math ready; GGUF loading comes next.");
    println!("Set RUST_LOG=phalanx=debug for verbose diagnostics.");
    println!("Run tensor microbenchmarks with: cargo bench --bench tensor_ops");

    Ok(())
}
