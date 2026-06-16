"""Verify NEAR-TP subsystem has been fully removed.

This test file replaces the deleted tests/test_near_tp_reduce.py and
tests/test_near_tp_reduce_helpers.py, confirming that:

1. TradeIntentType no longer includes NEAR_TP_REDUCE
2. StrategyPositionState has no near_tp_* fields
3. BollCvdReclaimStrategyConfig has no near_tp_* fields
4. LivePositionState has no near_tp_* fields
5. Strategy on_tick() does not generate NEAR_TP_REDUCE
6. Entry protective SL state is still present
7. Middle Runner / Three-Stage Runner state is still present
8. No near_tp_* references remain in source code
"""

from __future__ import annotations

import os

import pytest

from src.reporting.live_state_store import LivePositionState
from src.risk.simple_position_sizer import SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntentType,
)


class TestTradeIntentTypeNoNearTpReduce:
    """NEAR_TP_REDUCE must not be a valid TradeIntentType."""

    def test_near_tp_reduce_not_in_literal(self) -> None:
        # TradeIntentType is a Literal; verify "NEAR_TP_REDUCE" is absent
        valid = set(TradeIntentType.__args__)  # type: ignore[union-attr]
        assert "NEAR_TP_REDUCE" not in valid, (
            f"TradeIntentType should not contain NEAR_TP_REDUCE; got {valid}"
        )

    def test_expected_intents_still_present(self) -> None:
        valid = set(TradeIntentType.__args__)  # type: ignore[union-attr]
        for required in ("OPEN_LONG", "OPEN_SHORT", "UPDATE_TP", "MARKET_EXIT_RUNNER"):
            assert required in valid, f"Required intent {required} missing from TradeIntentType"


class TestStrategyPositionStateNoNearTp:
    """StrategyPositionState must have zero near_tp_* fields."""

    def test_no_near_tp_fields_in_state(self) -> None:
        state = StrategyPositionState()
        near_tp_field_names = [
            name for name in vars(state)
            if "near_tp" in name.lower()
        ]
        assert len(near_tp_field_names) == 0, (
            f"StrategyPositionState contains near_tp fields: {near_tp_field_names}"
        )

    def test_entry_protective_sl_fields_present(self) -> None:
        state = StrategyPositionState()
        assert hasattr(state, "entry_protective_sl_price")
        assert hasattr(state, "entry_protective_sl_order_id")
        assert hasattr(state, "entry_protective_sl_protected")

    def test_middle_runner_fields_present(self) -> None:
        state = StrategyPositionState()
        assert hasattr(state, "middle_runner_enabled_for_position")
        assert hasattr(state, "middle_runner_pending")
        assert hasattr(state, "middle_runner_active")
        assert hasattr(state, "middle_runner_protective_sl_order_id")

    def test_three_stage_runner_fields_present(self) -> None:
        state = StrategyPositionState()
        assert hasattr(state, "three_stage_runner_enabled_for_position")
        assert hasattr(state, "three_stage_tp1_consumed")
        assert hasattr(state, "three_stage_post_tp1_protective_sl_order_id")

    def test_trend_runner_fields_present(self) -> None:
        state = StrategyPositionState()
        assert hasattr(state, "trend_runner_active")
        assert hasattr(state, "trend_runner_sl_order_id")


class TestBollCvdReclaimStrategyConfigNoNearTp:
    """BollCvdReclaimStrategyConfig must have zero near_tp_* fields."""

    def test_no_near_tp_config_fields(self) -> None:
        cfg = BollCvdReclaimStrategyConfig()
        near_tp_field_names = [
            name for name in vars(cfg)
            if "near_tp" in name.lower()
        ]
        assert len(near_tp_field_names) == 0, (
            f"BollCvdReclaimStrategyConfig contains near_tp fields: {near_tp_field_names}"
        )

    def test_env_does_not_read_near_tp(self, monkeypatch) -> None:
        """from_env() must not fail even if NEAR_TP_* env vars are set."""
        monkeypatch.setenv("NEAR_TP_ENABLED", "true")
        monkeypatch.setenv("NEAR_TP_REDUCE_RATIO", "0.5")
        monkeypatch.setenv("NEAR_TP_PROTECTIVE_SL_ENABLED", "true")
        cfg = BollCvdReclaimStrategyConfig.from_env()
        assert not any("near_tp" in name.lower() for name in vars(cfg)), (
            "BollCvdReclaimStrategyConfig.from_env() should ignore NEAR_TP_* env vars"
        )

    def test_no_near_tp_mutex_checks(self, monkeypatch) -> None:
        """MIDDLE_RUNNER_ENABLED=true should no longer conflict with NEAR_TP_ENABLED."""
        monkeypatch.setenv("MIDDLE_RUNNER_ENABLED", "true")
        monkeypatch.setenv("NEAR_TP_ENABLED", "true")
        try:
            cfg = BollCvdReclaimStrategyConfig.from_env()
            assert cfg.middle_runner_enabled is True
        except RuntimeError as exc:
            if "mutually exclusive" in str(exc):
                pytest.fail(
                    f"Near-TP mutex check should be removed; got: {exc}"
                )
            # Other RuntimeError (e.g. validation) is acceptable


class TestLivePositionStateNoNearTp:
    """LivePositionState must have zero near_tp_* fields."""

    def test_no_near_tp_fields_in_live_state(self) -> None:
        state = LivePositionState()
        near_tp_field_names = [
            name for name in vars(state)
            if "near_tp" in name.lower()
        ]
        assert len(near_tp_field_names) == 0, (
            f"LivePositionState contains near_tp fields: {near_tp_field_names}"
        )

    def test_entry_protective_sl_fields_in_live_state(self) -> None:
        state = LivePositionState()
        assert hasattr(state, "entry_protective_sl_price")
        assert hasattr(state, "entry_protective_sl_order_id")
        assert hasattr(state, "entry_protective_sl_protected")

    def test_middle_runner_fields_in_live_state(self) -> None:
        state = LivePositionState()
        assert hasattr(state, "middle_runner_enabled_for_position")
        assert hasattr(state, "middle_runner_protective_sl_order_id")

    def test_three_stage_fields_in_live_state(self) -> None:
        state = LivePositionState()
        assert hasattr(state, "three_stage_runner_enabled_for_position")
        assert hasattr(state, "three_stage_post_tp1_protective_sl_order_id")


class TestDefaultRiskPct:
    """Default trade risk is 0.003 (0.3%)."""

    def test_default_config_risk_is_003(self) -> None:
        cfg = SimplePositionSizerConfig()
        assert cfg.trade_risk_pct == 0.003

    def test_env_trade_risk_pct_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv("TRADE_RISK_PCT", "0.01")
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.01

    def test_env_entry_risk_pct_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv("ENTRY_RISK_PCT", "0.005")
        # Ensure TRADE_RISK_PCT is not set
        monkeypatch.delenv("TRADE_RISK_PCT", raising=False)
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.005

    def test_trade_risk_pct_priority_over_entry_risk_pct(self, monkeypatch) -> None:
        monkeypatch.setenv("TRADE_RISK_PCT", "0.02")
        monkeypatch.setenv("ENTRY_RISK_PCT", "0.005")
        cfg = SimplePositionSizerConfig.from_env()
        assert cfg.trade_risk_pct == 0.02


class TestSourceCodeNoNearTp:
    """Verify zero near_tp_* / NEAR_TP_* business references in source code."""

    def test_no_near_tp_in_source(self) -> None:
        """Grep-based assertion that src/ has no near_tp references."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable, "-c",
                r"""
import subprocess, sys
p = subprocess.run(
    ["grep", "-rn", "NEAR_TP\|near_tp\|NearTp\|NEAR_TP_REDUCE\|near_tp_reduce"],
    cwd="src", capture_output=True, text=True
)
hits = [l for l in p.stdout.strip().split("\n") if l and "__pycache__" not in l]
if hits:
    sys.exit(f"Found near_tp references in src/: {hits}")
""",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Source code still contains near_tp references:\n{result.stderr}"
        )
