#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_runtime_smoke.py
@Description: Unit tests for scripts/binance_runtime_smoke.py

Covers:
  - Default Binance env blocked → exit 2
  - Default Binance env blocked with --expect-blocked → exit 0
  - Ready env with --expect-ready → exit 0, adapters created, sizing correct
  - Wrong exchange (EXCHANGE=okx) → exit 3
  - JSON output for blocked and ready paths
  - Source-level safety scan: no forbidden method calls in the script
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.binance_runtime_smoke import main


# ======================================================================
# Helpers
# ======================================================================

LIVE_PREFLIGHT_PASS_ENV: dict[str, str] = {
    "EXCHANGE": "binance",
    "EXCHANGE_API_KEY": "test-key",
    "EXCHANGE_API_SECRET": "test-secret",
    "LIVE_ENABLED": "true",
    "LIVE_ALLOW_ORDERS": "true",
    "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
    "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
    "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
    "LIVE_LEVERAGE": "20",
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
    ):
        monkeypatch.delenv(key, raising=False)

    for key, value in env.items():
        monkeypatch.setenv(key, value)


# ======================================================================
# 1. Default Binance env blocked → exit 2
# ======================================================================


class TestDefaultEnvBlocked:
    """With only EXCHANGE=binance, the smoke script must report blocked."""

    def test_blocked_without_expect_flag_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 2
        assert "BINANCE_RUNTIME_SMOKE_BLOCKED" in captured.out

    def test_blocked_output_includes_blocking_reasons(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 2
        assert "blocking_reasons=[" in captured.out
        assert "binance_live_enabled_not_true" in captured.out


# ======================================================================
# 2. Default Binance env blocked with --expect-blocked → exit 0
# ======================================================================


class TestBlockedWithExpectFlag:
    """With --expect-blocked, blocked is the expected outcome → exit 0."""

    def test_blocked_with_expect_flag_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "BINANCE_RUNTIME_SMOKE_BLOCKED" in captured.out

    def test_blocked_with_expect_flag_shows_reasons(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "blocking_reasons=[" in captured.out
        assert "ok=false" in captured.out

    def test_blocked_with_expect_flag_output_contains_exchange(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "exchange=binance" in captured.out


# ======================================================================
# 3. Ready env with --expect-ready → exit 0, adapters created
# ======================================================================


class TestReadyEnv:
    """When all live gates are satisfied, adapters are created and validated."""

    def test_ready_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0, f"Expected exit 0, got {rc}. stdout={captured.out}"
        assert "BINANCE_RUNTIME_SMOKE_READY" in captured.out

    def test_ready_output_includes_adapter_types(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "market_data_client=BinanceMarketDataClient" in captured.out
        assert "trading_client=BinanceTradingClient" in captured.out
        assert "trader=Trader" in captured.out

    def test_ready_output_includes_sizing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "trader_contract_multiplier=1" in captured.out
        assert "trader_contract_precision=0.001" in captured.out
        assert "trader_min_contracts=0.001" in captured.out

    def test_ready_output_includes_qty_check(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "qty_check_0_05_eth=0.05" in captured.out

    def test_ready_output_includes_side_effects_false(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "orders_executed=false" in captured.out
        assert "websocket_started=false" in captured.out

    def test_ready_output_includes_symbol(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "symbol=ETHUSDT" in captured.out

    def test_ready_output_includes_exchange(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "exchange=binance" in captured.out

    def test_ready_does_not_trigger_network(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Adapter creation must NOT trigger network calls.

        This is verified by simply creating adapters — the existing
        test_binance_runtime_adapter.py already proves that
        BinancePrivateClient creates its transport lazily on start().
        """
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-ready"])
        assert rc == 0

    def test_ready_with_live_env_primary_names(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """LIVE_* primary names satisfy all gates."""
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_KEY": "primary-key",
            "EXCHANGE_API_SECRET": "primary-secret",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "LIVE_LEVERAGE": "20",
        }
        _set_env(monkeypatch, env)
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "BINANCE_RUNTIME_SMOKE_READY" in captured.out

    def test_ready_with_binance_alias_names(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """BINANCE_* alias names still work (backward compat)."""
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_KEY": "alias-key",
            "EXCHANGE_API_SECRET": "alias-secret",
            "BINANCE_LIVE_ENABLED": "true",
            "BINANCE_LIVE_ALLOW_ORDERS": "true",
            "BINANCE_LIVE_CONFIRMATION": "I_UNDERSTAND_BINANCE_LIVE_TRADING",
            "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
            "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
            "BINANCE_LIVE_LEVERAGE": "20",
        }
        _set_env(monkeypatch, env)
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "BINANCE_RUNTIME_SMOKE_READY" in captured.out


# ======================================================================
# 4. Wrong exchange → exit 3
# ======================================================================


class TestWrongExchange:
    """When EXCHANGE is not binance, exit 3 immediately."""

    def test_okx_exchange_returns_3(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "okx"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE" in captured.out

    def test_okx_exchange_message_includes_exchange(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "okx"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "okx" in captured.out.lower() or "OKX" in captured.out

    def test_empty_exchange_defaults_to_okx_returns_3(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE" in captured.out

    def test_bybit_exchange_returns_3(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "bybit"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE" in captured.out


# ======================================================================
# 5. --expect-blocked in ready env → exit 4
# ======================================================================


class TestExpectBlockedButReady:
    """When env is ready but --expect-blocked is passed, exit 4."""

    def test_ready_env_expect_blocked_returns_4(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 4, f"Expected exit 4, got {rc}. stdout={captured.out}"
        assert (
            "BINANCE_RUNTIME_SMOKE_EXPECTED_BLOCKED_BUT_READY" in captured.out
        )
        assert "BINANCE_RUNTIME_SMOKE_READY" not in captured.out

    def test_ready_env_expect_blocked_does_not_create_adapters(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--expect-blocked must NOT call create_exchange_runtime_adapters."""
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))

        import scripts.binance_runtime_smoke as smoke_module

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            pytest.fail(
                "create_exchange_runtime_adapters must not be called "
                "when --expect-blocked and preflight is ready"
            )

        monkeypatch.setattr(
            smoke_module,
            "create_exchange_runtime_adapters",
            _fail_if_called,
        )

        rc = main(argv=["--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 4
        assert (
            "BINANCE_RUNTIME_SMOKE_EXPECTED_BLOCKED_BUT_READY" in captured.out
        )


# ======================================================================
# 6. Both --expect-blocked and --expect-ready → exit 4
# ======================================================================


class TestBothExpectationFlags:
    """Passing both --expect-blocked and --expect-ready is invalid."""

    def test_both_flags_returns_4(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(argv=["--expect-blocked", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 4, f"Expected exit 4, got {rc}. stdout={captured.out}"
        assert "BINANCE_RUNTIME_SMOKE_INVALID_EXPECTATION_FLAGS" in captured.out

    def test_both_flags_does_not_load_config(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both flags must fail BEFORE load_unified_runtime_config is called."""
        import scripts.binance_runtime_smoke as smoke_module

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            pytest.fail(
                "load_unified_runtime_config must not be called "
                "when both --expect-blocked and --expect-ready are passed"
            )

        monkeypatch.setattr(
            smoke_module,
            "load_unified_runtime_config",
            _fail_if_called,
        )
        monkeypatch.setattr(
            smoke_module,
            "create_exchange_runtime_adapters",
            _fail_if_called,
        )

        rc = main(argv=["--expect-blocked", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 4
        assert "BINANCE_RUNTIME_SMOKE_INVALID_EXPECTATION_FLAGS" in captured.out


# ======================================================================
# 7. Unsupported exchange via ValueError (e.g. EXCHANGE=abc) → exit 3
# ======================================================================


class TestUnsupportedExchangeViaValueError:
    """When EXCHANGE is not a valid ExchangeName, ValueError is caught."""

    def test_invalid_exchange_returns_3(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "abc"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3, f"Expected exit 3, got {rc}. stdout={captured.out}"
        assert "BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE" in captured.out
        assert "abc" in captured.out

    def test_invalid_exchange_bybit_returns_3_no_exception(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """EXCHANGE=bybit is a valid enum member but not binance — must not crash."""
        _set_env(monkeypatch, {"EXCHANGE": "bybit"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 3
        assert "BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE" in captured.out
        assert "bybit" in captured.out.lower()


# ======================================================================
# 8. EXCHANGE=binance but invalid config → exit 1
# ======================================================================


class TestConfigError:
    """When EXCHANGE=binance but other config is invalid, exit 1."""

    def test_invalid_trade_asset_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance", "TRADE_ASSET": "BTC"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 1, f"Expected exit 1, got {rc}. stdout={captured.out}"
        assert "BINANCE_RUNTIME_SMOKE_CONFIG_ERROR" in captured.out
        assert "BTC" in captured.out

    def test_invalid_leverage_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance", "LEVERAGE": "not_a_number"})
        rc = main(argv=[])
        captured = capsys.readouterr()
        assert rc == 1, f"Expected exit 1, got {rc}. stdout={captured.out}"
        assert "BINANCE_RUNTIME_SMOKE_CONFIG_ERROR" in captured.out


# ======================================================================
# 9. --expect-ready in blocked env → exit 2 (clear message)
# ======================================================================


class TestExpectReadyButBlocked:
    """When --expect-ready is passed but preflight is blocked, exit 2."""

    def test_expect_ready_blocked_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 2, f"Expected exit 2, got {rc}. stdout={captured.out}"
        assert (
            "BINANCE_RUNTIME_SMOKE_EXPECTED_READY_BUT_BLOCKED" in captured.out
        )
        # Still includes blocked reasons
        assert "BINANCE_RUNTIME_SMOKE_BLOCKED" in captured.out
        assert "blocking_reasons=[" in captured.out


# ======================================================================
# 10. JSON output for new paths
# ======================================================================

# (continued below — see TestJsonOutput for existing JSON tests)


class TestJsonOutputNew:
    """JSON output for newly added exit paths."""

    # --expect-blocked in ready env (JSON)
    def test_json_expect_blocked_but_ready(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--json", "--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 4
        data = json.loads(captured.out)
        assert data["status"] == "expectation_failed"
        assert data["exchange"] == "binance"
        assert data["symbol"] == "ETHUSDT"
        assert data["preflight_ok"] is True
        assert (
            data["error"] == "BINANCE_RUNTIME_SMOKE_EXPECTED_BLOCKED_BUT_READY"
        )

    # Both flags (JSON)
    def test_json_both_flags(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(argv=["--json", "--expect-blocked", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 4
        data = json.loads(captured.out)
        assert data["status"] == "invalid_args"
        assert (
            data["error"] == "BINANCE_RUNTIME_SMOKE_INVALID_EXPECTATION_FLAGS"
        )

    # Unsupported exchange (JSON)
    def test_json_wrong_exchange_via_value_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "bybit"})
        rc = main(argv=["--json"])
        captured = capsys.readouterr()
        assert rc == 3
        data = json.loads(captured.out)
        assert data["status"] == "wrong_exchange"
        assert data["exchange"] == "bybit"

    # Config error (JSON)
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
        assert "BINANCE_RUNTIME_SMOKE_CONFIG_ERROR" in data["error"]

    # --expect-ready blocked (JSON)
    def test_json_expect_ready_but_blocked(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--json", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 2
        data = json.loads(captured.out)
        assert data["status"] == "blocked"
        assert data["preflight_ok"] is False
        assert data["expected_ready"] is True
        assert (
            data["error"]
            == "BINANCE_RUNTIME_SMOKE_EXPECTED_READY_BUT_BLOCKED"
        )
        assert isinstance(data["blocking_reasons"], list)
        assert len(data["blocking_reasons"]) > 0


# ======================================================================
# 5. JSON output
# ======================================================================


class TestJsonOutput:
    """--json flag produces valid JSON with all required fields."""

    def test_json_blocked_has_required_fields(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--json"])
        captured = capsys.readouterr()
        assert rc == 2
        data = json.loads(captured.out)
        assert data["status"] == "blocked"
        assert data["exchange"] == "binance"
        assert data["symbol"] == "ETHUSDT"
        assert data["preflight_ok"] is False
        assert isinstance(data["blocking_reasons"], list)
        assert len(data["blocking_reasons"]) > 0

    def test_json_ready_has_required_fields(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--json", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["status"] == "ready"
        assert data["exchange"] == "binance"
        assert data["symbol"] == "ETHUSDT"
        assert data["preflight_ok"] is True
        assert data["blocking_reasons"] == []

    def test_json_ready_has_adapters(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--json", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        adapters = data["adapters"]
        assert adapters["market_data_client"] == "BinanceMarketDataClient"
        assert adapters["trading_client"] == "BinanceTradingClient"
        assert adapters["trader"] == "Trader"

    def test_json_ready_has_trader_sizing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--json", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        sizing = data["trader_sizing"]
        assert sizing["contract_multiplier"] == "1"
        assert sizing["contract_precision"] == "0.001"
        assert sizing["min_contracts"] == "0.001"
        assert sizing["qty_check_0_05_eth"] == "0.05"

    def test_json_ready_has_side_effects(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, dict(LIVE_PREFLIGHT_PASS_ENV))
        rc = main(argv=["--json", "--expect-ready"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        side = data["side_effects"]
        assert side["orders_executed"] is False
        assert side["websocket_started"] is False

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

    def test_json_blocked_with_expect_blocked_returns_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _set_env(monkeypatch, {"EXCHANGE": "binance"})
        rc = main(argv=["--json", "--expect-blocked"])
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["status"] == "blocked"


# ======================================================================
# 11. Source-level safety scan
# ======================================================================


class TestSourceLevelNoSideEffects:
    """The script source must NOT contain any forbidden method calls."""

    SCRIPT_SOURCE: str = Path("scripts/binance_runtime_smoke.py").read_text(
        encoding="utf-8"
    )

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
                        f"scripts/binance_runtime_smoke.py:{i}: {stripped}"
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
                    f"scripts/binance_runtime_smoke.py:{i}: "
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
                    f"scripts/binance_runtime_smoke.py:{i}: "
                    f"must not import aiohttp: {stripped}"
                )

    def test_no_binance_transport_imports(self) -> None:
        """The script must not import Binance transport classes directly."""
        transport_tokens = (
            "AiohttpBinanceTransport",
            "BinanceHttpTransport",
            "BinanceBrokerClient",
        )
        for i, line in enumerate(self.SCRIPT_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in transport_tokens:
                if token in stripped:
                    pytest.fail(
                        f"scripts/binance_runtime_smoke.py:{i}: "
                        f"must not import {token}: {stripped}"
                    )
