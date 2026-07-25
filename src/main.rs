//! Phalanx CLI entrypoint.
//!
//! Phase 1 ships a thin binary that initializes logging and prints a
//! version banner. Argument parsing, model loading, and generation arrive
//! in later phases — keeping `main` small avoids baking a CLI framework
//! choice before requirements are clear.

use anyhow::{Context, Result};
use tracing::info;

use phalanx::{LogConfig, RUNTIME_NAME, VERSION, init_logging};

fn main() -> Result<()> {
    init_logging(&LogConfig::default()).context("failed to initialize logging")?;

    info!(version = VERSION, "{RUNTIME_NAME} starting");

    println!("{RUNTIME_NAME} v{VERSION}");
    println!("Phase 1 foundation — tensor math and GGUF loading come next.");
    println!("Set RUST_LOG=phalanx=debug for verbose diagnostics.");

    Ok(())
}
