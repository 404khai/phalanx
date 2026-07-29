#!/usr/bin/env python3
"""Shared suite: SwiGLU Odyssey ↔ Phalanx numerical parity."""

from __future__ import annotations

import sys

from _common import run_odyssey_validator


def main() -> int:
    return run_odyssey_validator("validate_swiglu.py", sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
