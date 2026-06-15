#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_live_preflight.py
@Description: Unit tests for the Binance live preflight guard.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.exchanges.binance.live_preflight import (
    BINANCE_LIVE_CONFIRMATION_PHRASE,
    LIVE_CONFIRMATION_PHRASE,
    BinanceLivePreflightConfig,
    BinanceLivePreflightReport,
    build_binance_live_preflight_report,
    format_binance_live_blocked_message,
    load_binance_live_preflight_config,
)

# ======================================================================
# Config loading
# ======================================================================


class TestConfigLoadingDefaults:
    """Default env returns sensible defaults."""

    def test_default_exchange_is_okx(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.exchange == "okx"

    def test_default_signal_only_is_false(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.signal_only is False

    def test_default_live_enabled_is_false(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.live_enabled is False

    def test_default_allow_orders_is_false(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.allow_orders is False

    def test_default_confirmation_is_empty(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.confirmation == ""

    def test_default_max_order_none(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.max_order_notional_usdt is None

    def test_default_max_position_none(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.max_position_notional_usdt is None

    def test_default_leverage_none(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.leverage is None


class TestConfigExchangeBinance:
    """EXCHANGE=binance parses correctly."""

    def test_exchange_binance(self) -> None:
        cfg = load_binance_live_preflight_config({"EXCHANGE": "binance"})
        assert cfg.exchange == "binance"

    def test_exchange_binance_uppercase(self) -> None:
        cfg = load_binance_live_preflight_config({"EXCHANGE": "BINANCE"})
        assert cfg.exchange == "binance"


class TestConfigSignalOnly:
    """SIGNAL_ONLY (primary) truthy values."""

    def test_signal_only_true(self) -> None:
        cfg = load_binance_live_preflight_config({"SIGNAL_ONLY": "true"})
        assert cfg.signal_only is True

    def test_signal_only_1(self) -> None:
        cfg = load_binance_live_preflight_config({"SIGNAL_ONLY": "1"})
        assert cfg.signal_only is True

    def test_signal_only_false_explicit(self) -> None:
        cfg = load_binance_live_preflight_config({"SIGNAL_ONLY": "false"})
        assert cfg.signal_only is False

    # --- Alias (backward compat) ---

    def test_alias_true(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "true"})
        assert cfg.signal_only is True

    def test_alias_false(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "false"})
        assert cfg.signal_only is False

    # --- Conflict detection ---

    def test_conflict_primary_vs_alias_raises(self) -> None:
        """SIGNAL_ONLY=true vs BINANCE_SIGNAL_ONLY=false → blocked."""
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "SIGNAL_ONLY": "false", "BINANCE_SIGNAL_ONLY": "true"}
        )
        assert report.ok is False
        assert "live_env_var_conflict" in report.blocking_reasons


class TestConfigLiveEnabled:
    """BINANCE_LIVE_ENABLED truthy values."""

    @pytest.mark.parametrize("value", ["true", "1", "yes", "y", "on"])
    def test_live_enabled_truthy(self, value: str) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_LIVE_ENABLED": value})
        assert cfg.live_enabled is True

    def test_live_enabled_missing_is_false(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.live_enabled is False

    def test_live_enabled_false(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_LIVE_ENABLED": "false"})
        assert cfg.live_enabled is False


class TestConfigAllowOrders:
    """BINANCE_LIVE_ALLOW_ORDERS truthy values."""

    @pytest.mark.parametrize("value", ["true", "1", "yes", "y", "on"])
    def test_allow_orders_truthy(self, value: str) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_LIVE_ALLOW_ORDERS": value})
        assert cfg.allow_orders is True

    def test_allow_orders_missing_is_false(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.allow_orders is False


class TestConfigConfirmation:
    """LIVE_CONFIRMATION field — accepts both new and legacy phrases."""

    def test_confirmation_new_phrase(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"LIVE_CONFIRMATION": LIVE_CONFIRMATION_PHRASE}
        )
        assert cfg.confirmation == LIVE_CONFIRMATION_PHRASE

    def test_confirmation_legacy_phrase(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE}
        )
        assert cfg.confirmation == BINANCE_LIVE_CONFIRMATION_PHRASE

    def test_confirmation_wrong_value(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"LIVE_CONFIRMATION": "I_AGREE"}
        )
        assert cfg.confirmation == "I_AGREE"

    def test_confirmation_missing_is_empty(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.confirmation == ""


class TestConfigMaxOrderNotional:
    """BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT Decimal parsing."""

    def test_parse_decimal(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "5"}
        )
        assert cfg.max_order_notional_usdt == Decimal("5")

    def test_parse_decimal_with_decimals(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "7.5"}
        )
        assert cfg.max_order_notional_usdt == Decimal("7.5")

    def test_missing_is_none(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.max_order_notional_usdt is None

    def test_invalid_returns_none(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "not-a-number"}
        )
        assert cfg.max_order_notional_usdt is None


class TestConfigMaxPositionNotional:
    """BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT Decimal parsing."""

    def test_parse_decimal(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "20"}
        )
        assert cfg.max_position_notional_usdt == Decimal("20")

    def test_missing_is_none(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.max_position_notional_usdt is None


class TestConfigLeverage:
    """BINANCE_LIVE_LEVERAGE int parsing."""

    def test_parse_int(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_LIVE_LEVERAGE": "10"})
        assert cfg.leverage == 10

    def test_missing_is_none(self) -> None:
        cfg = load_binance_live_preflight_config({})
        assert cfg.leverage is None

    def test_invalid_returns_none(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_LIVE_LEVERAGE": "abc"})
        assert cfg.leverage is None


class TestConfigFrozen:
    """BinanceLivePreflightConfig is frozen."""

    def test_config_is_frozen(self) -> None:
        cfg = load_binance_live_preflight_config({})
        with pytest.raises(FrozenInstanceError):
            cfg.exchange = "binance"  # type: ignore[misc]

    def test_report_is_frozen(self) -> None:
        cfg = load_binance_live_preflight_config({})
        report = BinanceLivePreflightReport(ok=False, config=cfg, blocking_reasons=())
        with pytest.raises(FrozenInstanceError):
            report.ok = True  # type: ignore[misc]


# ======================================================================
# Blocking reasons
# ======================================================================


class TestBlockingExchangeNotBinance:
    """Exchange not binance → exchange_is_not_binance."""

    def test_okx_blocked(self) -> None:
        report = build_binance_live_preflight_report({"EXCHANGE": "okx"})
        assert report.ok is False
        assert "exchange_is_not_binance" in report.blocking_reasons

    def test_default_exchange_blocked(self) -> None:
        report = build_binance_live_preflight_report({})
        assert report.ok is False
        assert "exchange_is_not_binance" in report.blocking_reasons


class TestBlockingSignalOnly:
    """Signal-only → binance_signal_only_enabled."""

    def test_signal_only_true_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "SIGNAL_ONLY": "true"}
        )
        assert report.ok is False
        assert "binance_signal_only_enabled" in report.blocking_reasons

    def test_signal_only_alias_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "yes"}
        )
        assert report.ok is False
        assert "binance_signal_only_enabled" in report.blocking_reasons


class TestBlockingLiveEnabled:
    """live_enabled not truthy → binance_live_enabled_not_true."""

    def test_live_enabled_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "SIGNAL_ONLY": "false"}
        )
        assert report.ok is False
        assert "binance_live_enabled_not_true" in report.blocking_reasons

    def test_live_enabled_false_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "false",
            }
        )
        assert "binance_live_enabled_not_true" in report.blocking_reasons


class TestBlockingAllowOrders:
    """allow_orders not truthy → binance_live_allow_orders_not_true."""

    def test_allow_orders_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
            }
        )
        assert "binance_live_allow_orders_not_true" in report.blocking_reasons

    def test_allow_orders_false_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "false",
            }
        )
        assert "binance_live_allow_orders_not_true" in report.blocking_reasons


class TestBlockingConfirmation:
    """Confirmation missing/wrong → binance_live_confirmation_missing_or_invalid."""

    def test_confirmation_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
            }
        )
        assert "binance_live_confirmation_missing_or_invalid" in report.blocking_reasons

    def test_confirmation_wrong_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": "WRONG_PHRASE",
            }
        )
        assert "binance_live_confirmation_missing_or_invalid" in report.blocking_reasons

    def test_new_confirmation_phrase_passes(self) -> None:
        """LIVE_CONFIRMATION=I_UNDERSTAND_EXCHANGE_LIVE_TRADING passes."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True

    def test_legacy_confirmation_phrase_passes(self) -> None:
        """BINANCE_LIVE_CONFIRMATION=old-phrase still passes."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True


class TestBlockingMaxOrderNotional:
    """max_order_notional invalid → binance_live_max_order_notional_invalid."""

    def test_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
            }
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_high_value_passes(self) -> None:
        """LIVE_MAX_ORDER_NOTIONAL_USDT=100 passes (no hard upper bound)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "100",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert "binance_live_max_order_notional_invalid" not in report.blocking_reasons

    def test_valid_order_notional_passes(self) -> None:
        """LIVE_MAX_ORDER_NOTIONAL_USDT=25 is valid (any positive value works)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert "binance_live_max_order_notional_invalid" not in report.blocking_reasons

    def test_zero_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "0",
            }
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_negative_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "-1",
            }
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons


class TestBlockingMaxPositionNotional:
    """max_position_notional invalid → binance_live_max_position_notional_invalid."""

    def test_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
            }
        )
        assert "binance_live_max_position_notional_invalid" in report.blocking_reasons

    def test_high_value_passes(self) -> None:
        """LIVE_MAX_POSITION_NOTIONAL_USDT=500 passes (no hard upper bound)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "500",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert "binance_live_max_position_notional_invalid" not in report.blocking_reasons

    def test_valid_position_notional_passes(self) -> None:
        """LIVE_MAX_POSITION_NOTIONAL_USDT=30 is valid (any positive value works)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "30",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True


class TestBlockingLeverage:
    """Leverage invalid → binance_live_leverage_invalid."""

    def test_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
            }
        )
        assert "binance_live_leverage_invalid" in report.blocking_reasons

    def test_too_high_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
                "LIVE_LEVERAGE": "21",
            }
        )
        assert "binance_live_leverage_invalid" in report.blocking_reasons

    def test_at_hard_cap_valid(self) -> None:
        """LIVE_LEVERAGE=20 is valid (at hard cap)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "20",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True

    def test_zero_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
                "LIVE_LEVERAGE": "0",
            }
        )
        assert "binance_live_leverage_invalid" in report.blocking_reasons

    def test_negative_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
                "LIVE_LEVERAGE": "-1",
            }
        )
        assert "binance_live_leverage_invalid" in report.blocking_reasons


class TestBlockingOrdersDisabledByBuild:
    """orders_globally_enabled=False → binance_live_orders_disabled_by_build."""

    def test_all_env_satisfied_still_blocked(self) -> None:
        """Even with all env set correctly, orders_disabled_by_build blocks."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
                "BINANCE_LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=False,
        )
        assert report.ok is False
        assert "binance_live_orders_disabled_by_build" in report.blocking_reasons

    def test_all_env_satisfied_globally_enabled_ok(self) -> None:
        """All env correct + orders_globally_enabled=True → ok."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
                "BINANCE_LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True
        assert len(report.blocking_reasons) == 0


class TestBlockingReasonCount:
    """Report accurately reflects how many reasons are blocking."""

    def test_multiple_reasons_accumulate(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
            }
        )
        # live_enabled, allow_orders, confirmation, max_order, max_position,
        # leverage, orders_disabled_by_build → 7 reasons
        assert len(report.blocking_reasons) == 7
        assert report.ok is False


# ======================================================================
# Message formatting
# ======================================================================


class TestMessageFormatting:
    """format_binance_live_blocked_message produces correct output."""

    def test_contains_not_wired_yet(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"},
        )
        msg = format_binance_live_blocked_message(report)
        assert "Binance live trading runtime is not wired yet" in msg

    def test_contains_signal_only_hint(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"},
        )
        msg = format_binance_live_blocked_message(report)
        assert "SIGNAL_ONLY=true" in msg

    def test_contains_blocking_reasons(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"},
        )
        msg = format_binance_live_blocked_message(report)
        assert "blocking_reasons=" in msg
        assert "binance_live_enabled_not_true" in msg

    def test_contains_disabled_by_build(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"},
        )
        msg = format_binance_live_blocked_message(report)
        assert "binance_live_orders_disabled_by_build" in msg

    def test_no_secret_key_names(self) -> None:
        """Message must not contain secret key names."""
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"},
        )
        msg = format_binance_live_blocked_message(report)
        assert "EXCHANGE_API_KEY" not in msg
        assert "EXCHANGE_API_SECRET" not in msg
        assert "EXCHANGE_API_PASSPHRASE" not in msg
        assert "BINANCE_API_KEY" not in msg
        assert "BINANCE_SECRET_KEY" not in msg

    def test_no_secret_values(self) -> None:
        """Message must not expose confirmation phrase literally as a secret."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
            },
        )
        msg = format_binance_live_blocked_message(report)
        # The confirmation phrase itself should not appear verbatim in the
        # blocked message — that would leak the required secret phrase.
        assert BINANCE_LIVE_CONFIRMATION_PHRASE not in msg


# ======================================================================
# Exchange-neutral env var naming (20C-4C-MIN-NOTIONAL-FIX)
# ======================================================================


class TestEnvVarDualNames:
    """Dual-name env var reading: LIVE_* (primary) with BINANCE_* alias."""

    def test_only_generic_name_works(self) -> None:
        """Only LIVE_* env vars set → valid."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True

    def test_only_binance_alias_works(self) -> None:
        """Only BINANCE_* env vars set (no LIVE_*) → valid (backward compat)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "BINANCE_LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True

    def test_both_same_works(self) -> None:
        """Both LIVE_* and BINANCE_* set to the same values → valid."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "BINANCE_LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
                "BINANCE_LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is True

    def test_conflict_blocked(self) -> None:
        """LIVE_* and BINANCE_* set to different values → blocked."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "BINANCE_LIVE_ENABLED": "false",
                "LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT": "25",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT": "25",
                "LIVE_LEVERAGE": "10",
                "BINANCE_LIVE_LEVERAGE": "10",
            },
            orders_globally_enabled=True,
        )
        assert report.ok is False
        assert "live_env_var_conflict" in report.blocking_reasons


# ======================================================================
# Legacy OKX live config compatibility
# ======================================================================


LEGACY_OKX_ENV: dict[str, str] = {
    "EXCHANGE": "binance",
    "LIVE_TRADING": "true",
    "MAX_LIVE_EQUITY_USDT": "40000",
    "LAYER_MARGIN_PCT": "0.04",
    "LEVERAGE": "10",
    "MAX_LAYERS": "12",
}


class TestLegacyOkxConfig:
    """Legacy OKX live config (LIVE_TRADING=true, MAX_LIVE_EQUITY_USDT, etc.)
    passes Binance live preflight without requiring new Binance-only env vars.
    """

    def test_legacy_okx_config_passes(self) -> None:
        """LIVE_TRADING=true + legacy sizing → report.ok is True."""
        report = build_binance_live_preflight_report(
            LEGACY_OKX_ENV, orders_globally_enabled=True
        )
        assert report.ok is True
        assert report.config.live_enabled is True
        assert report.config.allow_orders is True
        assert report.config.max_order_notional_usdt == Decimal("16000")
        assert report.config.max_position_notional_usdt == Decimal("192000")
        assert report.config.leverage == 10

    def test_legacy_okx_config_sources(self) -> None:
        """Legacy config correctly annotates source for each field."""
        report = build_binance_live_preflight_report(
            LEGACY_OKX_ENV, orders_globally_enabled=True
        )
        cfg = report.config
        assert cfg.live_enabled_source == "LIVE_TRADING"
        assert cfg.allow_orders_source == "LIVE_TRADING"
        assert cfg.confirmation_source == "LEGACY_LIVE_TRADING"
        assert cfg.max_order_notional_source == "DERIVED_FROM_MAX_LIVE_EQUITY"
        assert (
            cfg.max_position_notional_source
            == "DERIVED_FROM_MAX_LIVE_EQUITY_AND_MAX_LAYERS"
        )
        assert cfg.leverage_source == "LEVERAGE"

    def test_legacy_okx_config_warns_on_confirmation(self) -> None:
        """Legacy LIVE_TRADING=true without explicit confirmation emits warning."""
        report = build_binance_live_preflight_report(
            LEGACY_OKX_ENV, orders_globally_enabled=True
        )
        assert "WARNING_LEGACY_LIVE_TRADING_CONFIRMATION_USED" in report.warnings

    def test_legacy_okx_config_live_trading_1(self) -> None:
        """LIVE_TRADING=1 also enables live mode."""
        env = dict(LEGACY_OKX_ENV)
        env["LIVE_TRADING"] = "1"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert report.ok is True
        assert report.config.live_enabled is True
        assert report.config.allow_orders is True

    def test_legacy_okx_config_live_trading_yes(self) -> None:
        """LIVE_TRADING=yes also enables live mode."""
        env = dict(LEGACY_OKX_ENV)
        env["LIVE_TRADING"] = "yes"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert report.ok is True
        assert report.config.live_enabled is True


class TestExplicitLiveOverridesLegacy:
    """Explicit LIVE_* env vars take priority over legacy OKX config."""

    def test_explicit_overrides_legacy(self) -> None:
        """LIVE_ENABLED=false overrides LIVE_TRADING=true."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
            "LIVE_ENABLED": "false",
            "LIVE_ALLOW_ORDERS": "false",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "123",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "456",
            "LIVE_LEVERAGE": "7",
            "MAX_LIVE_EQUITY_USDT": "40000",
            "LAYER_MARGIN_PCT": "0.04",
            "LEVERAGE": "10",
            "MAX_LAYERS": "12",
        }
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        cfg = report.config
        assert cfg.live_enabled is False
        assert cfg.allow_orders is False
        assert cfg.max_order_notional_usdt == Decimal("123")
        assert cfg.max_position_notional_usdt == Decimal("456")
        assert cfg.leverage == 7
        assert cfg.live_enabled_source == "LIVE_ENABLED"
        assert cfg.allow_orders_source == "LIVE_ALLOW_ORDERS"
        assert cfg.max_order_notional_source == "LIVE_MAX_ORDER_NOTIONAL_USDT"
        assert cfg.max_position_notional_source == "LIVE_MAX_POSITION_NOTIONAL_USDT"
        assert cfg.leverage_source == "LIVE_LEVERAGE"
        # live_enabled=false and allow_orders=false → blocked
        assert report.ok is False
        assert "binance_live_enabled_not_true" in report.blocking_reasons
        assert "binance_live_allow_orders_not_true" in report.blocking_reasons

    def test_explicit_order_notional_overrides_derived(self) -> None:
        """LIVE_MAX_ORDER_NOTIONAL_USDT=20000 overrides derived 16000."""
        env = dict(LEGACY_OKX_ENV)
        env["LIVE_MAX_ORDER_NOTIONAL_USDT"] = "20000"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert report.ok is True
        assert report.config.max_order_notional_usdt == Decimal("20000")
        assert report.config.max_order_notional_source == "LIVE_MAX_ORDER_NOTIONAL_USDT"
        # position still derived: 20000 * 12 = 240000
        assert report.config.max_position_notional_usdt == Decimal("240000")
        assert (
            report.config.max_position_notional_source
            == "DERIVED_FROM_MAX_LIVE_EQUITY_AND_MAX_LAYERS"
        )

    def test_explicit_position_notional_overrides_derived(self) -> None:
        """LIVE_MAX_POSITION_NOTIONAL_USDT=50000 overrides derived value."""
        env = dict(LEGACY_OKX_ENV)
        env["LIVE_MAX_POSITION_NOTIONAL_USDT"] = "50000"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert report.ok is True
        assert report.config.max_position_notional_usdt == Decimal("50000")
        assert (
            report.config.max_position_notional_source
            == "LIVE_MAX_POSITION_NOTIONAL_USDT"
        )

    def test_explicit_confirmation_overrides_legacy(self) -> None:
        """Explicit LIVE_CONFIRMATION removes legacy warning."""
        env = dict(LEGACY_OKX_ENV)
        env["LIVE_CONFIRMATION"] = LIVE_CONFIRMATION_PHRASE
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert report.ok is True
        assert (
            "WARNING_LEGACY_LIVE_TRADING_CONFIRMATION_USED"
            not in report.warnings
        )
        assert report.config.confirmation_source == "LIVE_CONFIRMATION"

    def test_explicit_allow_orders_false_respected(self) -> None:
        """LIVE_ALLOW_ORDERS=false must be respected even with LIVE_TRADING=true."""
        env = dict(LEGACY_OKX_ENV)
        env["LIVE_ALLOW_ORDERS"] = "false"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert report.config.allow_orders is False
        assert report.config.allow_orders_source == "LIVE_ALLOW_ORDERS"
        assert "binance_live_allow_orders_not_true" in report.blocking_reasons


class TestMissingMaxLiveEquityStillBlocks:
    """When neither explicit notional env vars nor MAX_LIVE_EQUITY_USDT
    are provided, the preflight must still block."""

    def test_missing_max_equity_blocks(self) -> None:
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
        }
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons
        assert "binance_live_max_position_notional_invalid" in report.blocking_reasons

    def test_max_equity_without_layer_margin_blocks(self) -> None:
        """MAX_LIVE_EQUITY_USDT without LAYER_MARGIN_PCT cannot derive."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
            "MAX_LIVE_EQUITY_USDT": "40000",
            "LEVERAGE": "10",
            "MAX_LAYERS": "12",
        }
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_max_equity_without_leverage_blocks(self) -> None:
        """MAX_LIVE_EQUITY_USDT without LEVERAGE cannot derive."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
            "MAX_LIVE_EQUITY_USDT": "40000",
            "LAYER_MARGIN_PCT": "0.04",
            "MAX_LAYERS": "12",
        }
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons


class TestInvalidLegacyDerivedValuesBlock:
    """Invalid legacy OKX sizing values must still block."""

    def test_invalid_max_live_equity_blocks(self) -> None:
        """MAX_LIVE_EQUITY_USDT=abc → cannot derive → blocked."""
        env = dict(LEGACY_OKX_ENV)
        env["MAX_LIVE_EQUITY_USDT"] = "abc"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_invalid_layer_margin_blocks(self) -> None:
        """LAYER_MARGIN_PCT=abc → cannot derive → blocked."""
        env = dict(LEGACY_OKX_ENV)
        env["LAYER_MARGIN_PCT"] = "abc"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_invalid_leverage_in_legacy_blocks(self) -> None:
        """LEVERAGE=abc → cannot derive notional and leverage is invalid."""
        env = dict(LEGACY_OKX_ENV)
        env["LEVERAGE"] = "abc"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        assert "binance_live_leverage_invalid" in report.blocking_reasons

    def test_zero_max_live_equity_derives_zero_order_notional(self) -> None:
        """MAX_LIVE_EQUITY_USDT=0 → derives 0 → blocked as invalid."""
        env = dict(LEGACY_OKX_ENV)
        env["MAX_LIVE_EQUITY_USDT"] = "0"
        report = build_binance_live_preflight_report(
            env, orders_globally_enabled=True
        )
        # order = 0 * 0.04 * 10 = 0 → not > 0 → invalid
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons


class TestLeverageFallback:
    """LEVERAGE env var acts as fallback for LIVE_LEVERAGE."""

    def test_leverage_fallback_works(self) -> None:
        """LEVERAGE=10 → live leverage=10."""
        env = dict(LEGACY_OKX_ENV)
        # LEGACY_OKX_ENV already has LEVERAGE=10 and no LIVE_LEVERAGE
        cfg = load_binance_live_preflight_config(env)
        assert cfg.leverage == 10
        assert cfg.leverage_source == "LEVERAGE"

    def test_leverage_fallback_5(self) -> None:
        """LEVERAGE=5 → live leverage=5."""
        env = {"EXCHANGE": "binance", "LEVERAGE": "5"}
        cfg = load_binance_live_preflight_config(env)
        assert cfg.leverage == 5
        assert cfg.leverage_source == "LEVERAGE"

    def test_no_leverage_anywhere_is_none(self) -> None:
        """Without LIVE_LEVERAGE, BINANCE_LIVE_LEVERAGE, or LEVERAGE → None."""
        cfg = load_binance_live_preflight_config({"EXCHANGE": "binance"})
        assert cfg.leverage is None
        assert cfg.leverage_source == ""


class TestLiveLeverageOverridesLeverage:
    """LIVE_LEVERAGE explicit takes priority over LEVERAGE fallback."""

    def test_live_leverage_overrides_leverage(self) -> None:
        """LIVE_LEVERAGE=5 overrides LEVERAGE=10."""
        env = {"EXCHANGE": "binance", "LIVE_LEVERAGE": "5", "LEVERAGE": "10"}
        cfg = load_binance_live_preflight_config(env)
        assert cfg.leverage == 5
        assert cfg.leverage_source == "LIVE_LEVERAGE"

    def test_binance_live_leverage_overrides_leverage(self) -> None:
        """BINANCE_LIVE_LEVERAGE=15 overrides LEVERAGE=10."""
        env = {
            "EXCHANGE": "binance",
            "BINANCE_LIVE_LEVERAGE": "15",
            "LEVERAGE": "10",
        }
        cfg = load_binance_live_preflight_config(env)
        assert cfg.leverage == 15
        assert cfg.leverage_source == "BINANCE_LIVE_LEVERAGE"


class TestLegacyConfigDerivationMath:
    """Verify the derivation math for notional values."""

    def test_derivation_formula(self) -> None:
        """order = equity × margin_pct × leverage; position = order × layers."""
        cfg = load_binance_live_preflight_config(LEGACY_OKX_ENV)
        # 40000 * 0.04 * 10 = 16000
        assert cfg.max_order_notional_usdt == Decimal("16000")
        # 16000 * 12 = 192000
        assert cfg.max_position_notional_usdt == Decimal("192000")

    def test_different_equity_values(self) -> None:
        """50000 * 0.04 * 10 = 20000; 20000 * 12 = 240000."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
            "MAX_LIVE_EQUITY_USDT": "50000",
            "LAYER_MARGIN_PCT": "0.04",
            "LEVERAGE": "10",
            "MAX_LAYERS": "12",
        }
        cfg = load_binance_live_preflight_config(env)
        assert cfg.max_order_notional_usdt == Decimal("20000")
        assert cfg.max_position_notional_usdt == Decimal("240000")

    def test_different_margin_pct(self) -> None:
        """40000 * 0.05 * 10 = 20000."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
            "MAX_LIVE_EQUITY_USDT": "40000",
            "LAYER_MARGIN_PCT": "0.05",
            "LEVERAGE": "10",
            "MAX_LAYERS": "12",
        }
        cfg = load_binance_live_preflight_config(env)
        assert cfg.max_order_notional_usdt == Decimal("20000")

    def test_different_leverage_affects_derivation(self) -> None:
        """40000 * 0.04 * 5 = 8000."""
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
            "MAX_LIVE_EQUITY_USDT": "40000",
            "LAYER_MARGIN_PCT": "0.04",
            "LEVERAGE": "5",
            "MAX_LAYERS": "12",
        }
        cfg = load_binance_live_preflight_config(env)
        assert cfg.max_order_notional_usdt == Decimal("8000")
        assert cfg.max_position_notional_usdt == Decimal("96000")
