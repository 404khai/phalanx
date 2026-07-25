# Changelog

All notable changes to Phalanx Runtime are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Phase 1 repository foundation: Cargo library + binary crate.
- Typed [`PhalanxError`](src/errors/mod.rs) surface via `thiserror`.
- Structured logging bootstrap via `tracing` / `tracing-subscriber`.
- Project documentation: README, architecture notes, implementation notes.
- Formatting (`rustfmt.toml`) and Clippy lint configuration.
- Smoke tests covering the public crate API.

## [0.1.0] - TBD

Initial development version; not yet published.
