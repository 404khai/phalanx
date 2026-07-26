//! Shared utilities that do not belong to a domain subsystem.
//!
//! Phase 1 only exposes logging initialization. Future phases may add
//! alignment helpers, timing primitives, or path utilities here — keep this
//! module free of model/math logic so domain crates stay cohesive.

mod logging;

pub use logging::{LogConfig, init_logging};
