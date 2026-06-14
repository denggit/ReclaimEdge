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

from src.live.binance_live_preflight import (
    BINANCE_LIVE_CONFIRMATION_PHRASE,
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
    """BINANCE_SIGNAL_ONLY truthy values."""

    def test_signal_only_true(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "true"})
        assert cfg.signal_only is True

    def test_signal_only_1(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "1"})
        assert cfg.signal_only is True

    def test_signal_only_yes(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "yes"})
        assert cfg.signal_only is True

    def test_signal_only_y(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "y"})
        assert cfg.signal_only is True

    def test_signal_only_on(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "on"})
        assert cfg.signal_only is True

    def test_signal_only_false_explicit(self) -> None:
        cfg = load_binance_live_preflight_config({"BINANCE_SIGNAL_ONLY": "false"})
        assert cfg.signal_only is False


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
    """BINANCE_LIVE_CONFIRMATION field."""

    def test_confirmation_correct_phrase(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE}
        )
        assert cfg.confirmation == BINANCE_LIVE_CONFIRMATION_PHRASE

    def test_confirmation_wrong_value(self) -> None:
        cfg = load_binance_live_preflight_config(
            {"BINANCE_LIVE_CONFIRMATION": "I_AGREE"}
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
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "true"}
        )
        assert report.ok is False
        assert "binance_signal_only_enabled" in report.blocking_reasons

    def test_signal_only_yes_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "yes"}
        )
        assert report.ok is False
        assert "binance_signal_only_enabled" in report.blocking_reasons


class TestBlockingLiveEnabled:
    """live_enabled not truthy → binance_live_enabled_not_true."""

    def test_live_enabled_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {"EXCHANGE": "binance", "BINANCE_SIGNAL_ONLY": "false"}
        )
        assert report.ok is False
        assert "binance_live_enabled_not_true" in report.blocking_reasons

    def test_live_enabled_false_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "false",
            }
        )
        assert "binance_live_enabled_not_true" in report.blocking_reasons


class TestBlockingAllowOrders:
    """allow_orders not truthy → binance_live_allow_orders_not_true."""

    def test_allow_orders_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
            }
        )
        assert "binance_live_allow_orders_not_true" in report.blocking_reasons

    def test_allow_orders_false_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "false",
            }
        )
        assert "binance_live_allow_orders_not_true" in report.blocking_reasons


class TestBlockingConfirmation:
    """Confirmation missing/wrong → binance_live_confirmation_missing_or_invalid."""

    def test_confirmation_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
            }
        )
        assert "binance_live_confirmation_missing_or_invalid" in report.blocking_reasons

    def test_confirmation_wrong_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "BINANCE_LIVE_ENABLED": "true",
                "BINANCE_LIVE_ALLOW_ORDERS": "true",
                "BINANCE_LIVE_CONFIRMATION": "WRONG_PHRASE",
            }
        )
        assert "binance_live_confirmation_missing_or_invalid" in report.blocking_reasons


class TestBlockingMaxOrderNotional:
    """max_order_notional invalid → binance_live_max_order_notional_invalid."""

    def test_missing_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
            }
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_too_high_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "26",
            }
        )
        assert "binance_live_max_order_notional_invalid" in report.blocking_reasons

    def test_at_hard_cap_valid(self) -> None:
        """LIVE_MAX_ORDER_NOTIONAL_USDT=25 is valid (at the hard cap)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
            }
        )
        assert "binance_live_max_position_notional_invalid" in report.blocking_reasons

    def test_too_high_blocked(self) -> None:
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
                "LIVE_ENABLED": "true",
                "LIVE_ALLOW_ORDERS": "true",
                "LIVE_CONFIRMATION": BINANCE_LIVE_CONFIRMATION_PHRASE,
                "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
                "LIVE_MAX_POSITION_NOTIONAL_USDT": "31",
            }
        )
        assert "binance_live_max_position_notional_invalid" in report.blocking_reasons

    def test_at_hard_cap_valid(self) -> None:
        """LIVE_MAX_POSITION_NOTIONAL_USDT=30 is valid (at hard cap)."""
        report = build_binance_live_preflight_report(
            {
                "EXCHANGE": "binance",
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
                "BINANCE_SIGNAL_ONLY": "false",
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
