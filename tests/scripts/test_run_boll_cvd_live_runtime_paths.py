#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests ensuring ``scripts/run_boll_cvd_live.py`` wires
RuntimePaths, legacy handoff, and symbol-scoped LiveTradeJournal /
LiveStateStore correctly (B05 + B06).

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"


def _source() -> str:
    return _LIVE_SCRIPT.read_text()


# ============================================================================
# 1. test_live_entry_uses_runtime_paths_for_state_and_journal
# ============================================================================


def test_live_entry_uses_runtime_paths_for_state_and_journal() -> None:
    """The live entry must construct RuntimePaths and wire it into
    LiveTradeJournal and LiveStateStore via from_runtime_paths."""
    source = _source()

    assert "RuntimePaths(" in source, (
        "live entry must construct RuntimePaths"
    )
    assert "runtime_configs.env_runtime.runtime_dir" in source, (
        "live entry must use runtime_configs.env_runtime.runtime_dir "
        "for RuntimePaths"
    )
    assert "handoff_legacy_runtime_files(" in source, (
        "live entry must call handoff_legacy_runtime_files"
    )
    assert "LiveTradeJournal.from_runtime_paths(" in source, (
        "live entry must use LiveTradeJournal.from_runtime_paths"
    )
    assert "LiveStateStore.from_runtime_paths(" in source, (
        "live entry must use LiveStateStore.from_runtime_paths"
    )
    assert "DailyTradeReporter(journal, email_sender)" in source, (
        "live entry must still construct DailyTradeReporter with journal"
    )


# ============================================================================
# 2. test_live_entry_no_longer_uses_bare_state_or_journal_constructors
# ============================================================================


def test_live_entry_no_longer_uses_bare_state_or_journal_constructors() -> None:
    """The live entry must NOT use the old bare constructors
    ``LiveTradeJournal()`` or ``LiveStateStore()``."""
    source = _source()

    assert "journal = LiveTradeJournal()" not in source, (
        "live entry must not use bare LiveTradeJournal() — "
        "use LiveTradeJournal.from_runtime_paths() instead"
    )
    assert "state_store = LiveStateStore()" not in source, (
        "live entry must not use bare LiveStateStore() — "
        "use LiveStateStore.from_runtime_paths() instead"
    )


# ============================================================================
# 3. test_live_entry_handoff_before_state_load
# ============================================================================


def test_live_entry_handoff_before_state_load() -> None:
    """handoff must happen before state_store.load() so the legacy state
    is seeded before the strategy tries to restore from it."""
    source = _source()

    handoff_idx = source.index("handoff_legacy_runtime_files(")
    state_load_idx = source.index("state_store.load()")
    from_runtime_paths_state_idx = source.index(
        "LiveStateStore.from_runtime_paths("
    )
    from_runtime_paths_journal_idx = source.index(
        "LiveTradeJournal.from_runtime_paths("
    )
    cash_baseline_idx = source.index("journal.record_cash_baseline(")

    assert handoff_idx < state_load_idx, (
        f"handoff_legacy_runtime_files must be called before state_store.load() — "
        f"handoff at {handoff_idx}, state_store.load at {state_load_idx}"
    )
    assert from_runtime_paths_state_idx < state_load_idx, (
        f"LiveStateStore.from_runtime_paths must be called before "
        f"state_store.load() — from_runtime_paths at {from_runtime_paths_state_idx}, "
        f"load at {state_load_idx}"
    )
    assert from_runtime_paths_journal_idx < cash_baseline_idx, (
        f"LiveTradeJournal.from_runtime_paths must be called before "
        f"journal.record_cash_baseline — from_runtime_paths at "
        f"{from_runtime_paths_journal_idx}, cash_baseline at {cash_baseline_idx}"
    )


# ============================================================================
# 4. test_live_entry_runtime_paths_after_runtime_configs
# ============================================================================


def test_live_entry_runtime_paths_after_runtime_configs() -> None:
    """RuntimePaths must be constructed AFTER build_live_symbol_runtime_configs
    because it depends on runtime_configs.env_runtime.runtime_dir."""
    source = _source()

    bootstrap_idx = source.index("build_live_symbol_runtime_configs(")
    runtime_paths_idx = source.index("RuntimePaths(")

    assert bootstrap_idx < runtime_paths_idx, (
        f"build_live_symbol_runtime_configs must be called before RuntimePaths — "
        f"bootstrap at {bootstrap_idx}, RuntimePaths at {runtime_paths_idx}"
    )


# ============================================================================
# 5. test_no_runtime_handoff_in_tick_path
# ============================================================================


# ============================================================================
# 5. B07 – RollingLossGuard uses from_runtime_paths, not from_env
# ============================================================================


def test_live_entry_uses_rolling_loss_guard_from_runtime_paths() -> None:
    """B07 wires RollingLossGuard.from_runtime_paths into the live entry."""
    source = _source()
    assert "RollingLossGuard.from_runtime_paths(" in source, (
        "B07 must wire RollingLossGuard.from_runtime_paths into run_boll_cvd_live.py"
    )


def test_live_entry_no_longer_uses_rolling_loss_guard_from_env() -> None:
    """B07 removes RollingLossGuard.from_env() from the live entry."""
    source = _source()
    assert "RollingLossGuard.from_env()" not in source, (
        "B07 must remove RollingLossGuard.from_env() from run_boll_cvd_live.py"
    )


def test_live_entry_rolling_loss_guard_ordering() -> None:
    """B07 ordering: handoff → from_runtime_paths → load_or_initialize,
    and RuntimePaths before from_runtime_paths."""
    source = _source()

    handoff_idx = source.index("handoff_legacy_runtime_files(")
    rlg_from_rp_idx = source.index("RollingLossGuard.from_runtime_paths(")
    rlg_load_idx = source.index("rolling_loss_guard.load_or_initialize(")
    runtime_paths_idx = source.index("RuntimePaths(")

    assert handoff_idx < rlg_load_idx, (
        f"handoff_legacy_runtime_files must be before "
        f"rolling_loss_guard.load_or_initialize — "
        f"handoff at {handoff_idx}, load_or_initialize at {rlg_load_idx}"
    )
    assert rlg_from_rp_idx < rlg_load_idx, (
        f"RollingLossGuard.from_runtime_paths must be before "
        f"rolling_loss_guard.load_or_initialize — "
        f"from_runtime_paths at {rlg_from_rp_idx}, load_or_initialize at {rlg_load_idx}"
    )
    assert runtime_paths_idx < rlg_from_rp_idx, (
        f"RuntimePaths must be constructed before "
        f"RollingLossGuard.from_runtime_paths — "
        f"RuntimePaths at {runtime_paths_idx}, from_runtime_paths at {rlg_from_rp_idx}"
    )


# ============================================================================
# 6. No runtime handoff in tick path
# ============================================================================


def test_no_runtime_handoff_in_tick_path() -> None:
    """RuntimePaths and handoff_legacy_runtime_files must NOT appear in
    tick / execution / sync worker files — they are startup-only."""
    worker_files = [
        "src/live/workers/strategy_tick_worker.py",
        "src/live/workers/execution_worker.py",
        "src/live/workers/execution_command_processor.py",
        "src/live/workers/account_position_sync_worker.py",
    ]

    for rel_path in worker_files:
        full_path = _PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue
        source = full_path.read_text(encoding="utf-8")
        assert "handoff_legacy_runtime_files" not in source, (
            f"{rel_path} must not reference handoff_legacy_runtime_files"
        )
        assert "RuntimePaths" not in source, (
            f"{rel_path} must not reference RuntimePaths"
        )
