#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_small_live_launch_check.py
@Description: Unit tests for scripts/binance_small_live_launch_check.py

Covers:
  1. Happy path — all env ready, no state file → exit 0
  2. Wrong exchange (EXCHANGE=okx) → exit 3
  3. Preflight blocked (EXCHANGE=binance only) → exit 2
  4. Order notional too high → exit 2
  5. Position notional too high → exit 2
  6. Sidecar enabled default block → exit 2
  7. Sidecar allow via CLI → exit 0 with warning
  8. Local state has old position default block → exit 2
  9. Allow existing local position without startup_force_tp_reconcile → exit 2
  10. Allow existing local position with startup_force_tp_reconcile=true → exit 0
  11. JSON output for ready and blocked paths
  12. Source-level no side effects
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.binance_small_live_launch_check import main


# ======================================================================
# Helpers
# ======================================================================

READY_ENV: dict[str, str] = {
    "EXCHANGE": "binance",
    "EXCHANGE_API_KEY": "test-key",
    "EXCHANGE_API_SECRET": "test-secret",
    "LIVE_ENABLED": "true",
    "LIVE_ALLOW_ORDERS": "true",
    "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
    "LIVE_MAX_ORDER_NOTIONAL_USDT": "10",
    "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
    "LIVE_LEVERAGE": "20",
    "SIDECAR_ENABLED": "false",
}


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Set environment variables, clearing any that are not in *env*."""
    # Clear all relevant vars first
    for key in (
        "EXCHANGE",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "EXCHANGE_API_PASSPHRASE",
        "SIGNAL_ONLY",
        "BINANCE_SIGNAL_ONLY",
        "LIVE_ENABLED",
        "BINANCE_LIVE_ENABLED",
        "LIVE_ALLOW_ORDERS",
        "BINANCE_LIVE_ALLOW_ORDERS",
        "LIVE_CONFIRMATION",
        "BINANCE_LIVE_CONFIRMATION",
        "LIVE_MAX_ORDER_NOTIONAL_USDT",
        "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT",
        "LIVE_MAX_POSITION_NOTIONAL_USDT",
        "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT",
        "LIVE_LEVERAGE",
        "BINANCE_LIVE_LEVERAGE",
        "TRADE_ASSET",
        "QUOTE_ASSET",
        "MARKET_TYPE",
        "MARGIN_MODE",
        "POSITION_MODE",
        "KLINE_INTERVAL",
        "LEVERAGE",
        "SIDECAR_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _make_state_file(overrides: dict | None = None) -> Path:
    """Create a temporary live_state.json and return its path."""
    state: dict = {
        "symbol": "ETHUSDT",
        "side": None,
        "layers": 0,
        "core_eth_qty": 0.0,
        "position_cost_remaining_qty": 0.0,
        "startup_force_tp_reconcile": False,
    }
    if overrides:
        state.update(overrides)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="test_live_state_"
    )
    json.dump(state, tmp)
    tmp.close()
    return Path(tmp.name)


# ======================================================================
# 1. Happy path
# ======================================================================


class TestHappyPath:
    """All env ready, no state file → exit 0, BINANCE_SMALL_LIVE_LAUNCH_READY."""

    def test_ready_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "BINANCE_SMALL_LIVE_LAUNCH_READY" in captured.out

    def test_ready_output_includes_symbol(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "symbol=ETHUSDT" in captured.out

    def test_ready_output_includes_exchange(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "exchange=binance" in captured.out

    def test_ready_output_includes_config_fields(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "trade_asset=ETH" in captured.out
        assert "quote_asset=USDT" in captured.out
        assert "market_type=PERPETUAL" in captured.out
        assert "margin_mode=isolated" in captured.out
        assert "position_mode=net" in captured.out
        assert "leverage=20" in captured.out
        assert "live_enabled=true" in captured.out
        assert "live_allow_orders=true" in captured.out

    def test_ready_output_includes_qty_check(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "qty_check_0_05_eth=0.05" in captured.out

    def test_ready_output_side_effects_false(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "orders_executed=false" in captured.out
        assert "websocket_started=false" in captured.out

    def test_ready_output_includes_notional_values(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "max_order_notional_usdt=10" in captured.out
        assert "max_position_notional_usdt=20" in captured.out

    def test_ready_output_includes_sizing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "contract_multiplier=1" in captured.out
        assert "contract_precision=0.001" in captured.out
        assert "min_contracts=0.001" in captured.out

    def test_ready_output_sidecar_disabled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "sidecar_enabled=false" in captured.out

    def test_ready_no_state_file_status_absent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "local_state_status=flat_or_absent" in captured.out


# ======================================================================
# 2. Wrong exchange
# ======================================================================


class TestWrongExchange:
    """When EXCHANGE is not binance, exit 3."""

    def test_okx_exchange_returns_3(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "okx"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3, f"Expected exit 3, got {rc}. stdout={captured.out}"
        assert "BINANCE_SMALL_LIVE_WRONG_EXCHANGE" in captured.out

    def test_okx_exchange_shows_exchange_name(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "okx"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "okx" in captured.out.lower()

    def test_empty_exchange_returns_3(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "BINANCE_SMALL_LIVE_WRONG_EXCHANGE" in captured.out


# ======================================================================
# 3. Preflight blocked
# ======================================================================


class TestPreflightBlocked:
    """When only EXCHANGE=binance, preflight is blocked → exit 2."""

    def test_preflight_blocked_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
        assert "BINANCE_SMALL_LIVE_PREFLIGHT_BLOCKED" in captured.out

    def test_preflight_blocked_includes_reasons(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 2
        assert "blocking_reasons=" in captured.out
        assert "binance_live_enabled_not_true" in captured.out


# ======================================================================
# 4. Order notional too high
# ======================================================================


class TestOrderNotionalTooHigh:
    """LIVE_MAX_ORDER_NOTIONAL_USDT exceeds small live cap → exit 2.

    Uses a value (22) that passes the preflight hard cap (25) but exceeds
    the default small-live cap (20).
    """

    def test_order_notional_too_high_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_ORDER_NOTIONAL_USDT"] = "22"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
        assert "LIVE_MAX_ORDER_NOTIONAL_TOO_HIGH" in captured.out

    def test_order_notional_too_high_blocked_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_ORDER_NOTIONAL_USDT"] = "22"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 2
        assert "BINANCE_SMALL_LIVE_LAUNCH_BLOCKED" in captured.out

    def test_order_notional_within_limit_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_ORDER_NOTIONAL_USDT"] = "15"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "BINANCE_SMALL_LIVE_LAUNCH_READY" in captured.out

    def test_order_notional_cli_override_allows_higher(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """22 > default cap (20) but passes with --max-allowed-order-notional 25."""
        env = dict(READY_ENV)
        env["LIVE_MAX_ORDER_NOTIONAL_USDT"] = "22"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(
                argv=[
                    "--state-path", tf.name,
                    "--max-allowed-order-notional", "25",
                ]
            )
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "BINANCE_SMALL_LIVE_LAUNCH_READY" in captured.out


# ======================================================================
# 5. Position notional too high
# ======================================================================


class TestPositionNotionalTooHigh:
    """LIVE_MAX_POSITION_NOTIONAL_USDT exceeds small live cap → exit 2.

    Uses CLI override to make the small-live cap stricter than the preflight
    cap, so the small-live check triggers before the preflight would.
    """

    def test_position_notional_too_high_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_POSITION_NOTIONAL_USDT"] = "30"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(
                argv=[
                    "--state-path", tf.name,
                    "--max-allowed-position-notional", "25",
                ]
            )
        captured = capsys.readouterr()
        assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
        assert "LIVE_MAX_POSITION_NOTIONAL_TOO_HIGH" in captured.out

    def test_position_notional_too_high_blocked_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_POSITION_NOTIONAL_USDT"] = "30"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(
                argv=[
                    "--state-path", tf.name,
                    "--max-allowed-position-notional", "25",
                ]
            )
        captured = capsys.readouterr()
        assert rc == 2
        assert "BINANCE_SMALL_LIVE_LAUNCH_BLOCKED" in captured.out

    def test_position_notional_cli_override_allows_higher(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_POSITION_NOTIONAL_USDT"] = "30"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(
                argv=[
                    "--state-path", tf.name,
                    "--max-allowed-position-notional", "35",
                ]
            )
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "BINANCE_SMALL_LIVE_LAUNCH_READY" in captured.out


# ======================================================================
# 6. Sidecar default block
# ======================================================================


class TestSidecarDefaultBlock:
    """SIDECAR_ENABLED=true blocks by default → exit 2."""

    def test_sidecar_enabled_default_block_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "true"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
        assert "SIDECAR_ENABLED_FOR_FIRST_BINANCE_LIVE" in captured.out

    def test_sidecar_enabled_default_block_token(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "true"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 2
        assert "BINANCE_SMALL_LIVE_LAUNCH_BLOCKED" in captured.out


# ======================================================================
# 7. Sidecar allow via CLI → exit 0 with warning
# ======================================================================


class TestSidecarAllow:
    """--allow-sidecar permits SIDECAR_ENABLED=true with warning."""

    def test_sidecar_allow_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "true"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--allow-sidecar"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "WARNING_SIDECAR_ENABLED" in captured.out

    def test_sidecar_allow_still_shows_ready(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "true"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--allow-sidecar"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "BINANCE_SMALL_LIVE_LAUNCH_READY" in captured.out

    def test_sidecar_disabled_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "false"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name])
        captured = capsys.readouterr()
        assert rc == 0
        assert "WARNING_SIDECAR_ENABLED" not in captured.out


# ======================================================================
# 8. Local state has old position — default block
# ======================================================================


class TestLocalStateHasOpenPosition:
    """Local state with open position blocks by default → exit 2."""

    def test_open_position_blocks_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        state_path = _make_state_file(
            {
                "side": "LONG",
                "layers": 1,
                "core_eth_qty": 0.05,
                "position_cost_remaining_qty": 0.05,
                "startup_force_tp_reconcile": False,
            }
        )
        try:
            rc = main(argv=["--state-path", str(state_path)])
            captured = capsys.readouterr()
            assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
            assert "LOCAL_STATE_HAS_OPEN_POSITION" in captured.out
        finally:
            state_path.unlink(missing_ok=True)

    def test_open_position_blocked_token(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        state_path = _make_state_file(
            {
                "side": "LONG",
                "layers": 1,
                "core_eth_qty": 0.05,
                "position_cost_remaining_qty": 0.05,
            }
        )
        try:
            rc = main(argv=["--state-path", str(state_path)])
            captured = capsys.readouterr()
            assert rc == 2
            assert "BINANCE_SMALL_LIVE_LAUNCH_BLOCKED" in captured.out
        finally:
            state_path.unlink(missing_ok=True)

    def test_flat_state_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        state_path = _make_state_file(
            {
                "side": None,
                "layers": 0,
                "core_eth_qty": 0.0,
                "position_cost_remaining_qty": 0.0,
            }
        )
        try:
            rc = main(argv=["--state-path", str(state_path)])
            captured = capsys.readouterr()
            assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
            assert "local_state_status=flat_or_absent" in captured.out
        finally:
            state_path.unlink(missing_ok=True)

    def test_absent_state_passes(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        # Use a path that definitely doesn't exist
        rc = main(argv=["--state-path", "/tmp/does_not_exist_test_state.json"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "local_state_status=flat_or_absent" in captured.out


# ======================================================================
# 9. Allow existing local position but no startup_force_tp_reconcile
# ======================================================================


class TestAllowExistingPositionNoReconcile:
    """--allow-existing-local-position but startup_force_tp_reconcile=false → block."""

    def test_no_reconcile_blocks_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        state_path = _make_state_file(
            {
                "side": "LONG",
                "layers": 1,
                "core_eth_qty": 0.05,
                "position_cost_remaining_qty": 0.05,
                "startup_force_tp_reconcile": False,
            }
        )
        try:
            rc = main(
                argv=[
                    "--state-path", str(state_path),
                    "--allow-existing-local-position",
                ]
            )
            captured = capsys.readouterr()
            assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
            assert "EXISTING_POSITION_REQUIRES_STARTUP_FORCE_TP_RECONCILE" in captured.out
        finally:
            state_path.unlink(missing_ok=True)


# ======================================================================
# 10. Allow existing local position with startup_force_tp_reconcile=true
# ======================================================================


class TestAllowExistingPositionWithReconcile:
    """--allow-existing-local-position with startup_force_tp_reconcile=true → pass."""

    def test_with_reconcile_passes_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        state_path = _make_state_file(
            {
                "side": "LONG",
                "layers": 1,
                "core_eth_qty": 0.05,
                "position_cost_remaining_qty": 0.05,
                "startup_force_tp_reconcile": True,
            }
        )
        try:
            rc = main(
                argv=[
                    "--state-path", str(state_path),
                    "--allow-existing-local-position",
                ]
            )
            captured = capsys.readouterr()
            assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
            assert "BINANCE_SMALL_LIVE_LAUNCH_READY" in captured.out
        finally:
            state_path.unlink(missing_ok=True)


# ======================================================================
# 11. JSON output
# ======================================================================


class TestJsonOutput:
    """--json flag produces valid JSON with all required fields."""

    def test_json_ready_has_required_fields(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["status"] == "ready"
        assert data["exchange"] == "binance"
        assert data["symbol"] == "ETHUSDT"
        assert data["preflight_ok"] is True
        assert data["checks"]["small_live_caps_ok"] is True
        assert data["checks"]["sidecar_ok"] is True
        assert data["checks"]["local_state_ok"] is True
        assert data["checks"]["trader_sizing_ok"] is True

    def test_json_ready_has_runtime(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        runtime = data["runtime"]
        assert runtime["contract_multiplier"] == "1"
        assert runtime["contract_precision"] == "0.001"
        assert runtime["min_contracts"] == "0.001"
        assert runtime["qty_check_0_05_eth"] == "0.05"

    def test_json_ready_has_side_effects(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(READY_ENV))
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["side_effects"]["orders_executed"] is False
        assert data["side_effects"]["websocket_started"] is False

    def test_json_blocked_has_required_fields(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["LIVE_MAX_ORDER_NOTIONAL_USDT"] = "22"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--json"])
        captured = capsys.readouterr()
        assert rc == 2
        data = json.loads(captured.out)
        assert data["status"] == "blocked"
        assert data["exchange"] == "binance"
        assert data["symbol"] == "ETHUSDT"
        assert "LIVE_MAX_ORDER_NOTIONAL_TOO_HIGH" in data["blocking_reasons"]

    def test_json_wrong_exchange(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "okx"})
        rc = main(argv=["--json"])
        captured = capsys.readouterr()
        assert rc == 3
        data = json.loads(captured.out)
        assert data["status"] == "wrong_exchange"
        assert data["exchange"] == "okx"
        assert data["error"] == "BINANCE_SMALL_LIVE_WRONG_EXCHANGE"

    def test_json_config_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance", "TRADE_ASSET": "BTC"})
        rc = main(argv=["--json"])
        captured = capsys.readouterr()
        assert rc == 1
        data = json.loads(captured.out)
        assert data["status"] == "config_error"
        assert data["exchange"] == "binance"
        assert "BINANCE_SMALL_LIVE_CONFIG_ERROR" in data["error"]

    def test_json_blocked_includes_warnings(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "true"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(argv=["--state-path", tf.name, "--json"])
        captured = capsys.readouterr()
        assert rc == 2
        data = json.loads(captured.out)
        assert data["status"] == "blocked"
        assert "SIDECAR_ENABLED_FOR_FIRST_BINANCE_LIVE" in data["blocking_reasons"]

    def test_json_sidecar_allow_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = dict(READY_ENV)
        env["SIDECAR_ENABLED"] = "true"
        _set_env(monkeypatch, env)
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="no_state_") as tf:
            rc = main(
                argv=[
                    "--state-path", tf.name,
                    "--allow-sidecar",
                    "--json",
                ]
            )
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["status"] == "ready"
        assert "WARNING_SIDECAR_ENABLED" in data["warnings"]


# ======================================================================
# 12. Source-level no side effects
# ======================================================================


class TestSourceLevelNoSideEffects:
    """The script source must NOT contain any forbidden method calls."""

    SCRIPT_SOURCE: str = Path(
        "scripts/binance_small_live_launch_check.py"
    ).read_text(encoding="utf-8")

    FORBIDDEN_TOKENS: tuple[str, ...] = (
        "place_market_order",
        "place_limit_order",
        "place_stop_market_order",
        "cancel_order",
        "cancel_algo_order",
        "stream_market_events(",
        "fetch_balance(",
        "fetch_position(",
        "configure_instrument(",
        "initialize_instrument(",
    )

    def test_no_forbidden_method_calls(self) -> None:
        """Verify the script source does not contain any forbidden tokens."""
        violations: list[str] = []
        for i, line in enumerate(self.SCRIPT_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in self.FORBIDDEN_TOKENS:
                if token in stripped:
                    violations.append(
                        f"scripts/binance_small_live_launch_check.py:{i}: {stripped}"
                    )
        assert not violations, (
            "Script must not contain forbidden method calls:\n"
            + "\n".join(violations)
        )

    def test_no_websocket_imports(self) -> None:
        """The script must not import websocket modules."""
        for i, line in enumerate(self.SCRIPT_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "import" in stripped and "websocket" in stripped.lower():
                pytest.fail(
                    f"scripts/binance_small_live_launch_check.py:{i}: "
                    f"must not import websocket: {stripped}"
                )

    def test_no_aiohttp_imports(self) -> None:
        """The script must not import aiohttp."""
        for i, line in enumerate(self.SCRIPT_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "import" in stripped and "aiohttp" in stripped.lower():
                pytest.fail(
                    f"scripts/binance_small_live_launch_check.py:{i}: "
                    f"must not import aiohttp: {stripped}"
                )

    def test_no_run_boll_cvd_live_import(self) -> None:
        """The script must not call or import run_boll_cvd_live."""
        for i, line in enumerate(self.SCRIPT_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "run_boll_cvd_live" in stripped:
                pytest.fail(
                    f"scripts/binance_small_live_launch_check.py:{i}: "
                    f"must not reference run_boll_cvd_live: {stripped}"
                )
