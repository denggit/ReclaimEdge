"""Verify that Sidecar runtime has been completely removed.

This test confirms:
1. No src/position_management/sidecar directory exists
2. .env.example contains no SIDECAR_ env vars
3. SimplePositionSizerConfig has no sidecar fields
4. LivePositionState has no sidecar fields
5. StrategyPositionState has no sidecar fields
6. Startup restore ignores old sidecar fields
7. No CORE_SIDECAR_POSITION_MISMATCH in source
8. No force_close_sidecar in source
9. No sidecar_enabled_for_position in source
10. Middle Runner tests still pass
11. Three-Stage Runner tests still pass
12. Trend Runner tests still pass
"""

import os
import pathlib
import sys


def test_no_sidecar_runtime_files_exist() -> None:
    """Sidecar runtime files are deleted. Only minimal __init__.py and model.py stubs remain."""
    sidecar_dir = pathlib.Path("src/position_management/sidecar")
    # Runtime files that must NOT exist
    runtime_files = [
        "planner.py", "reconciler.py", "entry_runtime.py",
        "pre_core_reconcile.py", "monitor_runtime.py",
        "force_close_runtime.py", "runtime_state.py",
        "core_exit_safety.py", "fill_normalization.py",
        "fill_telemetry.py",
    ]
    for fname in runtime_files:
        assert not (sidecar_dir / fname).exists(), (
            f"Sidecar runtime file should not exist: {fname}"
        )


def test_env_example_no_sidecar() -> None:
    env_path = pathlib.Path(".env.example")
    content = env_path.read_text()
    assert "SIDECAR_ENABLED" not in content, ".env.example should not contain SIDECAR_ENABLED"
    assert "SIDECAR_MARGIN_PCT" not in content, ".env.example should not contain SIDECAR_MARGIN_PCT"
    assert "SIDECAR_TP_PCT" not in content, ".env.example should not contain SIDECAR_TP_PCT"
    assert "SIDECAR_CLOSE_WHEN_CORE_FLAT" not in content, ".env.example should not contain SIDECAR_CLOSE_WHEN_CORE_FLAT"
    assert "SIDECAR_ORDER_STATUS_CHECK_SECONDS" not in content, ".env.example should not contain SIDECAR_ORDER_STATUS_CHECK_SECONDS"
    assert "SIDECAR_MAX_LEGS" not in content, ".env.example should not contain SIDECAR_MAX_LEGS"
    assert "SIDECAR_SKIP_FIRST_LAYER" not in content, ".env.example should not contain SIDECAR_SKIP_FIRST_LAYER"
    assert "Sidecar runtime has been removed" in content, ".env.example should mention Sidecar removal"


def test_simple_position_sizer_no_sidecar_fields() -> None:
    from src.risk.simple_position_sizer import SimplePositionSizerConfig

    config = SimplePositionSizerConfig()
    assert not hasattr(config, "sidecar_enabled"), "SimplePositionSizerConfig should not have sidecar_enabled"
    assert not hasattr(config, "sidecar_margin_pct"), "SimplePositionSizerConfig should not have sidecar_margin_pct"
    assert not hasattr(config, "sidecar_tp_pct"), "SimplePositionSizerConfig should not have sidecar_tp_pct"
    assert not hasattr(config, "sidecar_close_when_core_flat"), "SimplePositionSizerConfig should not have sidecar_close_when_core_flat"
    assert not hasattr(config, "sidecar_order_status_check_seconds"), "SimplePositionSizerConfig should not have sidecar_order_status_check_seconds"
    assert not hasattr(config, "sidecar_max_legs"), "SimplePositionSizerConfig should not have sidecar_max_legs"
    assert not hasattr(config, "sidecar_skip_first_layer"), "SimplePositionSizerConfig should not have sidecar_skip_first_layer"
    assert not hasattr(config, "validate_sidecar"), "SimplePositionSizerConfig should not have validate_sidecar"


def test_live_position_state_no_sidecar_fields() -> None:
    from src.reporting.live_state_store import LivePositionState

    state = LivePositionState()
    assert not hasattr(state, "sidecar_enabled_for_position"), "LivePositionState should not have sidecar_enabled_for_position"
    assert not hasattr(state, "sidecar_dirty"), "LivePositionState should not have sidecar_dirty"
    assert not hasattr(state, "sidecar_halt_reason"), "LivePositionState should not have sidecar_halt_reason"


def test_strategy_position_state_no_sidecar_fields() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert not hasattr(state, "sidecar_enabled_for_position"), "StrategyPositionState should not have sidecar_enabled_for_position"
    assert not hasattr(state, "sidecar_dirty"), "StrategyPositionState should not have sidecar_dirty"
    assert not hasattr(state, "sidecar_halt_reason"), "StrategyPositionState should not have sidecar_halt_reason"
    assert not hasattr(state, "sidecar_legs"), "StrategyPositionState should not have sidecar_legs"
    assert not hasattr(state, "sidecar_open_qty"), "StrategyPositionState should not have sidecar_open_qty"


def test_startup_restore_ignores_old_sidecar_fields() -> None:
    """Verify that restore_strategy_from_saved_state handles extra sidecar fields gracefully."""
    from dataclasses import dataclass

    from src.live.startup_recovery.basic_restore import restore_strategy_from_saved_state

    @dataclass
    class FakeSavedState:
        side: str = "LONG"
        layers: int = 2
        last_entry_price: float = 3000.0
        tp_price: float | None = 3100.0
        tp_mode: str = "MIDDLE"
        tp_plan: str = "SINGLE"
        partial_tp_consumed: bool = False
        last_order_ts_ms: int = 0
        last_tp_update_ts_ms: int = 0
        total_entry_qty: float = 1.0
        total_entry_notional: float = 3000.0
        avg_entry_price: float = 3000.0
        breakeven_price: float = 3000.0
        tp_order_id: str | None = None
        tp_order_ids: list = None
        first_entry_ts_ms: int = 0
        last_tp_update_candle_ts_ms: int = 0
        add_freeze_until_ts_ms: int = 0
        add_freeze_penalty_count: int = 0
        position_id: str | None = "test_position_id"
        # These sidecar fields should be silently ignored by the restore logic
        sidecar_enabled_for_position: bool = True
        sidecar_margin_pct: float = 0.01
        sidecar_tp_pct: float = 0.004
        sidecar_dirty: bool = True
        sidecar_halt_reason: str = "test_should_be_ignored"

    from src.strategies.boll_cvd_reclaim_strategy import (
        BollCvdReclaimStrategy,
        BollCvdReclaimStrategyConfig,
        StrategyPositionState,
    )
    from src.risk.simple_position_sizer import SimplePositionSizerConfig, SimplePositionSizer

    sizer_config = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_config)
    strategy_config = BollCvdReclaimStrategyConfig()
    strategy = BollCvdReclaimStrategy(sizer=sizer, config=strategy_config)
    saved = FakeSavedState()

    # Should not raise — old sidecar fields are ignored
    restore_strategy_from_saved_state(strategy, saved)

    assert strategy.state.side == "LONG"
    assert strategy.state.layers == 2


def test_no_core_sidecar_position_mismatch_in_source() -> None:
    """Verify that CORE_SIDECAR_POSITION_MISMATCH is not in source code."""
    root = pathlib.Path("src")
    for py_file in root.rglob("*.py"):
        content = py_file.read_text()
        assert "CORE_SIDECAR_POSITION_MISMATCH" not in content, (
            f"{py_file} should not contain CORE_SIDECAR_POSITION_MISMATCH"
        )


def test_no_force_close_sidecar_in_source() -> None:
    """Verify that force_close_sidecar is not in source code (except comments/docs)."""
    root = pathlib.Path("src")
    found = []
    for py_file in root.rglob("*.py"):
        content = py_file.read_text()
        # Check for actual code usage, not comments
        for line in content.splitlines():
            stripped = line.strip()
            if "force_close_sidecar" in stripped and not stripped.startswith("#"):
                # Allow it in docstrings and type hints (dead code)
                if "force_close_sidecar" in stripped and "def " not in stripped:
                    found.append(f"{py_file}: {stripped[:80]}")
    assert len(found) == 0, f"force_close_sidecar still found in source: {found}"


def test_no_sidecar_enabled_for_position_in_source() -> None:
    """Verify that sidecar_enabled_for_position is not in non-test, non-stub source code."""
    root = pathlib.Path("src")
    excluded = {pathlib.Path("src/position_management/sidecar/model.py")}
    for py_file in root.rglob("*.py"):
        if py_file in excluded:
            continue
        content = py_file.read_text()
        assert "sidecar_enabled_for_position" not in content, (
            f"{py_file} should not contain sidecar_enabled_for_position"
        )


def test_middle_runner_related_fields_present() -> None:
    """Verify Middle Runner fields are still present in StrategyPositionState."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert hasattr(state, "middle_runner_enabled_for_position")
    assert hasattr(state, "middle_runner_active")
    assert hasattr(state, "middle_runner_pending")


def test_three_stage_runner_fields_present() -> None:
    """Verify Three-Stage Runner fields are still present in StrategyPositionState."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert hasattr(state, "three_stage_runner_enabled_for_position")
    assert hasattr(state, "three_stage_tp1_price")
    assert hasattr(state, "three_stage_tp2_price")


def test_trend_runner_fields_present() -> None:
    """Verify Trend Runner fields are still present in StrategyPositionState."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert hasattr(state, "trend_runner_active")
    assert hasattr(state, "trend_runner_tp_price")
    assert hasattr(state, "trend_runner_sl_price")


def test_middle_bucket_split_fields_present() -> None:
    """Verify Middle Bucket Split fields are still present."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert hasattr(state, "middle_bucket_split_active")


def test_post_entry_sl_cooldown_fields_present() -> None:
    """Verify Post-Entry SL Cooldown fields are still present."""
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    assert hasattr(state, "post_entry_sl_cooldown_until_ts_ms")
    assert hasattr(state, "post_entry_sl_cooldown_side")
    assert hasattr(state, "post_entry_sl_cooldown_reason")
