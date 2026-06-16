"""Verify that Sidecar runtime has been completely removed.

This test confirms:
1. No src/position_management/sidecar directory exists
2. No src/execution/tp_sl_sidecar_manager.py exists
3. No sidecar naming in src/ Python files (case-insensitive)
4. No SIDECAR_ env vars in .env.example
5. SimplePositionSizerConfig has no sidecar fields
6. StrategyPositionState has no sidecar fields
7. LivePositionState has no sidecar fields
8. Startup restore ignores old sidecar fields
9. Middle Runner, Three-Stage Runner, Trend Runner are preserved
"""

import pathlib


def test_no_sidecar_package_exists() -> None:
    sidecar_dir = pathlib.Path("src/position_management/sidecar")
    assert not sidecar_dir.exists(), f"Sidecar directory must not exist: {sidecar_dir}"


def test_no_sidecar_manager_exists() -> None:
    manager_path = pathlib.Path("src/execution/tp_sl_sidecar_manager.py")
    assert not manager_path.exists(), f"Sidecar manager must not exist: {manager_path}"


def test_no_sidecar_naming_in_src() -> None:
    """Verify that sidecar/Sidecar/SIDECAR does not appear in src/ Python files.

    Allowed exceptions:
    - Exchange adapter files (exchanges/) — dead legacy code, not Sidecar runtime
    - Comments referencing historical Sidecar removal (e.g. "Sidecar removed")
    """
    allowed_patterns = {
        "sidecar has been removed",
        "sidecar runtime has been removed",
        "sidecar removed",
        "sidecar runtime removed",
    }
    # Exchange adapter files are excluded (not Sidecar runtime, but adapter layer)
    excluded_prefixes = (
        str(pathlib.Path("src/exchanges/")),
    )
    root = pathlib.Path("src")
    violations = []
    for py_file in sorted(root.rglob("*.py")):
        py_str = str(py_file)
        if py_str.startswith(excluded_prefixes):
            continue
        content = py_file.read_text()
        lower = content.lower()
        if "sidecar" not in lower and "side_car" not in lower and "core_sidecar" not in lower:
            continue
        # Check each line for sidecar references
        for i, line in enumerate(content.splitlines(), 1):
            line_lower = line.strip().lower()
            if "sidecar" not in line_lower and "side_car" not in line_lower:
                continue
            # Allow lines that explicitly state Sidecar has been removed
            if any(p in line_lower for p in allowed_patterns):
                continue
            # Allow pure comments that are removal notes
            if line.strip().startswith("#") and any(p in line_lower for p in allowed_patterns):
                continue
            violations.append(f"{py_file}:{i}: {line.strip()[:100]}")

    if violations:
        msg = "Sidecar references found in src/:\n" + "\n".join(violations[:20])
        if len(violations) > 20:
            msg += f"\n... and {len(violations) - 20} more"
        raise AssertionError(msg)


def test_env_example_no_sidecar() -> None:
    env_path = pathlib.Path(".env.example")
    content = env_path.read_text()
    assert "SIDECAR_ENABLED" not in content
    assert "SIDECAR_MARGIN_PCT" not in content
    assert "SIDECAR_TP_PCT" not in content
    assert "SIDECAR_CLOSE_WHEN_CORE_FLAT" not in content
    assert "SIDECAR_ORDER_STATUS_CHECK_SECONDS" not in content
    assert "SIDECAR_MAX_LEGS" not in content
    assert "SIDECAR_SKIP_FIRST_LAYER" not in content
    assert "Sidecar runtime has been removed" in content


def test_simple_position_sizer_no_sidecar_fields() -> None:
    from src.risk.simple_position_sizer import SimplePositionSizerConfig

    config = SimplePositionSizerConfig()
    for field in ("sidecar_enabled", "sidecar_margin_pct", "sidecar_tp_pct",
                  "sidecar_close_when_core_flat", "sidecar_order_status_check_seconds",
                  "sidecar_max_legs", "sidecar_skip_first_layer", "validate_sidecar"):
        assert not hasattr(config, field), f"SimplePositionSizerConfig should not have {field}"


def test_strategy_position_state_no_sidecar_fields() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

    state = StrategyPositionState()
    for field in ("sidecar_enabled_for_position", "sidecar_dirty", "sidecar_halt_reason",
                  "sidecar_legs", "sidecar_open_qty", "sidecar_margin_pct",
                  "sidecar_tp_pct", "sidecar_total_qty", "sidecar_total_notional",
                  "sidecar_realized_qty"):
        assert not hasattr(state, field), f"StrategyPositionState should not have {field}"


def test_live_position_state_no_sidecar_fields() -> None:
    from src.reporting.live_state_store import LivePositionState

    state = LivePositionState()
    for field in ("sidecar_enabled_for_position", "sidecar_dirty", "sidecar_halt_reason",
                  "sidecar_margin_pct", "sidecar_tp_pct", "sidecar_legs",
                  "sidecar_open_qty", "sidecar_total_qty", "sidecar_total_notional",
                  "sidecar_realized_qty"):
        assert not hasattr(state, field), f"LivePositionState should not have {field}"


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


def test_middle_runner_related_fields_present() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
    state = StrategyPositionState()
    assert hasattr(state, "middle_runner_enabled_for_position")
    assert hasattr(state, "middle_runner_active")
    assert hasattr(state, "middle_runner_pending")


def test_three_stage_runner_fields_present() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
    state = StrategyPositionState()
    assert hasattr(state, "three_stage_runner_enabled_for_position")
    assert hasattr(state, "three_stage_tp1_price")
    assert hasattr(state, "three_stage_tp2_price")


def test_trend_runner_fields_present() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
    state = StrategyPositionState()
    assert hasattr(state, "trend_runner_active")
    assert hasattr(state, "trend_runner_tp_price")
    assert hasattr(state, "trend_runner_sl_price")


def test_middle_bucket_split_fields_present() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
    state = StrategyPositionState()
    assert hasattr(state, "middle_bucket_split_active")


def test_post_entry_sl_cooldown_fields_present() -> None:
    from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
    state = StrategyPositionState()
    assert hasattr(state, "post_entry_sl_cooldown_until_ts_ms")
    assert hasattr(state, "post_entry_sl_cooldown_side")
    assert hasattr(state, "post_entry_sl_cooldown_reason")


def test_run_boll_cvd_live_has_no_sidecar_startup_recovery_call() -> None:
    source = pathlib.Path("scripts/run_boll_cvd_live.py").read_text()
    assert "apply_sidecar_startup_recovery" not in source


def test_order_recovery_has_no_sidecar_public_api_dependency() -> None:
    source = pathlib.Path("src/live/startup_recovery/order_recovery.py").read_text()
    assert "sidecar" not in source.lower()
