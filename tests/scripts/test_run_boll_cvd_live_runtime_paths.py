#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests ensuring ``scripts/run_boll_cvd_live.py`` wires
RuntimePaths, legacy handoff, and symbol-scoped LiveTradeJournal /
LiveStateStore correctly (B05 + B06), and that
``src/live/symbol_worker_factory.py`` provides the construction
primitives (C02).

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_FACTORY_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_factory.py"


def _source() -> str:
    return _LIVE_SCRIPT.read_text()


def _factory_source() -> str:
    return _FACTORY_MODULE.read_text()


# ============================================================================
# 1. test_live_entry_uses_runtime_paths_for_state_and_journal (updated C02)
# ============================================================================


def test_live_entry_uses_runtime_paths_for_state_and_journal() -> None:
    """The live entry must construct RuntimePaths via the factory and
    wire it into persistence via factory.create_persistence."""
    source = _source()
    factory_source = _factory_source()

    assert "SymbolWorkerFactory" in source, (
        "C02 live entry must import SymbolWorkerFactory"
    )
    assert "factory.create_runtime_paths(" in source, (
        "C02 live entry must use factory.create_runtime_paths"
    )
    assert "handoff_legacy_runtime_files(" in source, (
        "live entry must call handoff_legacy_runtime_files"
    )
    assert "factory.create_persistence(" in source, (
        "C02 live entry must use factory.create_persistence"
    )

    # These from_runtime_paths calls now live in the factory, not the
    # live entry directly.
    assert "LiveTradeJournal.from_runtime_paths(" in factory_source, (
        "C02 factory must use LiveTradeJournal.from_runtime_paths"
    )
    assert "LiveStateStore.from_runtime_paths(" in factory_source, (
        "C02 factory must use LiveStateStore.from_runtime_paths"
    )
    assert "DailyTradeReporter(journal, email_sender)" in factory_source, (
        "C02 factory must construct DailyTradeReporter with journal"
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
    create_persistence_idx = source.index("factory.create_persistence(")
    cash_baseline_idx = source.index("journal.record_cash_baseline(")

    assert handoff_idx < state_load_idx, (
        f"handoff_legacy_runtime_files must be called before state_store.load() — "
        f"handoff at {handoff_idx}, state_store.load at {state_load_idx}"
    )
    assert create_persistence_idx < state_load_idx, (
        f"factory.create_persistence must be called before "
        f"state_store.load() — create_persistence at {create_persistence_idx}, "
        f"load at {state_load_idx}"
    )
    assert create_persistence_idx < cash_baseline_idx, (
        f"factory.create_persistence must be called before "
        f"journal.record_cash_baseline — create_persistence at "
        f"{create_persistence_idx}, cash_baseline at {cash_baseline_idx}"
    )


# ============================================================================
# 4. test_live_entry_runtime_paths_after_runtime_configs
# ============================================================================


def test_live_entry_runtime_paths_after_runtime_configs() -> None:
    """RuntimePaths must be constructed AFTER build_live_symbol_runtime_configs
    because it depends on runtime_configs.env_runtime.runtime_dir."""
    source = _source()

    bootstrap_idx = source.index("build_live_symbol_runtime_configs(")
    runtime_paths_idx = source.index("factory.create_runtime_paths(")

    assert bootstrap_idx < runtime_paths_idx, (
        f"build_live_symbol_runtime_configs must be called before "
        f"factory.create_runtime_paths — "
        f"bootstrap at {bootstrap_idx}, create_runtime_paths at {runtime_paths_idx}"
    )


# ============================================================================
# 5. B07 – RollingLossGuard uses from_runtime_paths, not from_env
# ============================================================================


def test_live_entry_uses_rolling_loss_guard_from_runtime_paths() -> None:
    """B07 wires RollingLossGuard.from_runtime_paths into the factory."""
    factory_source = _factory_source()
    assert "RollingLossGuard.from_runtime_paths(" in factory_source, (
        "B07 must wire RollingLossGuard.from_runtime_paths into "
        "symbol_worker_factory.py"
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
    rlg_from_rp_idx = _factory_source().index("RollingLossGuard.from_runtime_paths(")
    rlg_load_idx = source.index("rolling_loss_guard.load_or_initialize(")
    create_runtime_paths_idx = source.index("factory.create_runtime_paths(")

    assert handoff_idx < rlg_load_idx, (
        f"handoff_legacy_runtime_files must be before "
        f"rolling_loss_guard.load_or_initialize — "
        f"handoff at {handoff_idx}, load_or_initialize at {rlg_load_idx}"
    )
    # RollingLossGuard.from_runtime_paths is now in the factory, which is
    # called after handoff — the ordering is preserved structurally.
    assert create_runtime_paths_idx < rlg_load_idx, (
        f"factory.create_runtime_paths must be constructed before "
        f"rolling_loss_guard.load_or_initialize — "
        f"create_runtime_paths at {create_runtime_paths_idx}, "
        f"load_or_initialize at {rlg_load_idx}"
    )
    # Verify factory's from_runtime_paths is after handoff in
    # architectural sense: factory.create_persistence is called after
    # handoff in the live entry.
    create_persistence_idx = source.index("factory.create_persistence(")
    assert handoff_idx < create_persistence_idx, (
        f"handoff_legacy_runtime_files must be before "
        f"factory.create_persistence — "
        f"handoff at {handoff_idx}, create_persistence at {create_persistence_idx}"
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
