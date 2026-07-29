#!/usr/bin/env python3
"""Shared suite: Attention parity (Odyssey ↔ Phalanx)."""

from __future__ import annotations

from _common import run_odyssey_validator


def main() -> int:
    return run_odyssey_validator("validate_attention.py")


if __name__ == "__main__":
    raise SystemExit(main())
