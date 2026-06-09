#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C03 source guard — ensures ``scripts/run_boll_cvd_live.py`` does NOT
call ``SymbolWorkerApp`` yet (that is C04's job).

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text()


# ============================================================================
# 1. test_live_entry_does_not_call_symbol_worker_app_yet
# ============================================================================


def test_live_entry_does_not_call_symbol_worker_app_yet() -> None:
    """C03 must NOT wire SymbolWorkerApp into run_boll_cvd_live.py.
    That is C04's responsibility."""
    source = _live_source()

    assert "SymbolWorkerApp" not in source, (
        "C03 must not reference SymbolWorkerApp in run_boll_cvd_live.py"
    )


# ============================================================================
# 2. test_live_entry_still_has_current_main_for_c03
# ============================================================================


def test_live_entry_still_has_current_main_for_c03() -> None:
    """The live entry must still contain the full main() body — C03 does
    not remove it."""
    source = _live_source()

    required = [
        "async def main()",
        "factory.create_email_sender(",
        "factory.create_trader(",
        "await trader.start()",
        "await trader.initialize()",
        "asyncio.gather(",
        "monitor.run_forever()",
    ]
    for token in required:
        assert token in source, (
            f"C03 live entry must still contain {token!r}"
        )


# ============================================================================
# 3. test_symbol_worker_app_module_exists_but_not_wired
# ============================================================================


def test_symbol_worker_app_module_exists_but_not_wired() -> None:
    """SymbolWorkerApp module exists on disk but is NOT imported or
    referenced by run_boll_cvd_live.py."""
    assert _APP_MODULE.exists(), (
        "src/live/symbol_worker_app.py must exist"
    )
    live_source = _live_source()
    assert "SymbolWorkerApp" not in live_source, (
        "run_boll_cvd_live.py must not reference SymbolWorkerApp in C03"
    )
