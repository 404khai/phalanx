//! Structured logging bootstrap built on `tracing`.
//!
//! # Why `tracing` instead of `log` + `env_logger`
//!
//! | Approach | Pros | Cons |
//! |---|---|---|
//! | `tracing` + `tracing-subscriber` (chosen) | Structured spans, async-ready, ecosystem default for systems crates | Slightly heavier than bare `log` |
//! | `log` + `env_logger` | Minimal | Weak span/context support; harder to evolve into profiling |
//! | Custom stderr writer | Zero deps | Reinvents filtering, levels, and formatting |
//!
//! Inference runtimes benefit from spans around load / prefill / decode, so
//! we pay the small dependency cost up front rather than migrating later.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;

use crate::errors::{PhalanxError, Result};

/// Default filter when `RUST_LOG` is unset.
///
/// `info` is quiet enough for CLI use while still surfacing lifecycle events
/// (model load start/finish, generation begin). Developers can raise verbosity
/// with `RUST_LOG=phalanx=debug` or `RUST_LOG=trace`.
const DEFAULT_ENV_FILTER: &str = "info";

/// Options controlling how the global subscriber is installed.
///
/// `Default` yields CLI-friendly settings: inherit `RUST_LOG` (else `info`),
/// and hide module targets for cleaner stdout.
#[derive(Debug, Clone, Default)]
pub struct LogConfig {
    /// Filter directive string, e.g. `"info"` or `"phalanx=debug,warn"`.
    ///
    /// When `None`, the subscriber reads `RUST_LOG`, falling back to
    /// [`DEFAULT_ENV_FILTER`].
    pub filter: Option<String>,

    /// Include the log target (module path) in each line.
    ///
    /// Disabled by default for cleaner CLI output; enable when debugging
    /// which subsystem emitted an event.
    pub show_target: bool,
}

/// Install the global `tracing` subscriber.
///
/// Safe to call once from `main` (or tests that need logging). Calling twice
/// returns [`PhalanxError::Internal`] because `tracing` allows only one
/// global default subscriber.
///
/// # Errors
///
/// Returns [`PhalanxError::Config`] if the filter directive cannot be parsed,
/// or [`PhalanxError::Internal`] if a subscriber is already installed.
pub fn init_logging(config: &LogConfig) -> Result<()> {
    let filter = match &config.filter {
        Some(directive) => EnvFilter::try_new(directive).map_err(|err| {
            PhalanxError::config(format!("invalid log filter '{directive}': {err}"))
        })?,
        None => {
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_ENV_FILTER))
        }
    };

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(config.show_target)
        .finish();

    tracing::subscriber::set_global_default(subscriber).map_err(|err| {
        PhalanxError::internal(format!("logging subscriber already installed: {err}"))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_cli_friendly() {
        let config = LogConfig::default();
        assert!(config.filter.is_none());
        assert!(!config.show_target);
    }

    #[test]
    fn invalid_filter_returns_config_error() {
        // Use a unique subscriber attempt that fails at filter parse time
        // before touching the global default, so this test stays isolated.
        let config = LogConfig {
            filter: Some("this is not[[[valid".to_owned()),
            show_target: false,
        };

        let result = init_logging(&config);
        assert!(matches!(result, Err(PhalanxError::Config(_))));
    }
}
