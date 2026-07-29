"""Shared helpers for the Odyssey ↔ Phalanx validation suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Suite may live at monorepo `validation/` or inside `runtime/validation/`.
_SUITE_ROOT = Path(__file__).resolve().parent
_PARENT = _SUITE_ROOT.parent
if (_PARENT / "odyssey").is_dir() and (_PARENT / "runtime").is_dir():
    MONOREPO_ROOT = _PARENT
    ODYSSEY_ROOT = MONOREPO_ROOT / "odyssey"
    PHALANX_ROOT = MONOREPO_ROOT / "runtime"
else:
    # Installed under runtime/validation — Odyssey is the sibling checkout.
    PHALANX_ROOT = _PARENT
    MONOREPO_ROOT = _PARENT.parent
    ODYSSEY_ROOT = MONOREPO_ROOT / "odyssey"

DEFAULT_TOLERANCE = 1e-6


def run_odyssey_validator(script_name: str, extra_args: list[str] | None = None) -> int:
    """Execute an Odyssey ``scripts/validate_*.py`` driver and return its exit code."""
    script = ODYSSEY_ROOT / "scripts" / script_name
    if not script.is_file():
        print(f"missing Odyssey validator: {script}", file=sys.stderr)
        return 2
    if not PHALANX_ROOT.is_dir():
        print(f"missing Phalanx runtime at {PHALANX_ROOT}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script), "--phalanx-root", str(PHALANX_ROOT)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=ODYSSEY_ROOT)
    return int(completed.returncode)


def stub_message(component: str) -> int:
    """Exit 0 with an explicit skip for components not yet on both sides."""
    print(f"{component}: SKIP — not yet implemented on both Odyssey and Phalanx.")
    print("Add validate_<component>.py / validate_<component>.rs before enabling.")
    return 0
