#!/usr/bin/env python3
"""Shared suite: Embedding parity (placeholder).

Both sides have embedding layers, but a dedicated ``validate_embeddings``
binary/script pair is not wired yet. This entry exists so the suite layout
is complete; enable once the I/O contract lands.
"""

from __future__ import annotations

from _common import stub_message


def main() -> int:
    return stub_message("Embedding")


if __name__ == "__main__":
    raise SystemExit(main())
