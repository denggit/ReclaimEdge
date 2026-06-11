#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``config.symbol_config`` — per-symbol configuration schema."""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.symbol_config import (
    SymbolCapitalConfig,
    SymbolConfig,
    SymbolCvdConfig,
    SymbolEntryConfig,
    SymbolExecutionConfig,
    SymbolIdentityConfig,
    SymbolMarketConfig,
    SymbolMiddleBucketSplitConfig,
    SymbolRiskConfig,
    SymbolRuntimeConfig,
    SymbolSidecarConfig,
    SymbolTpConfig,
    decimal_from_any,
)


# ---------------------------------------------------------------------------
# decimal_from_any
# ---------------------------------------------------------------------------


class TestDecimalFromAny:
    """Smoke-tests for the conversion helper."""

    def test_from_str(self) -> None:
        assert decimal_from_any("0.1") == Decimal("0.1")

    def test_from_int(self) -> None:
        assert decimal_from_any(1) == Decimal("1")

    def test_from_decimal(self) -> None:
        assert decimal_from_any(Decimal("2.5")) == Decimal("2.5")

    def test_from_float_uses_str_roundtrip(self) -> None:
        result = decimal_from_any(0.6)
        assert result == Decimal("0.6")
        # The critical property: the value must NOT be the IEEE-754
        # approximation.
        assert result == Decimal(str(0.6))

    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="None"):
            decimal_from_any(None)

    def test_unknown_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            decimal_from_any([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SymbolConfig.default_eth() – identity
# ---------------------------------------------------------------------------


class TestDefaultEthIdentity:
    """Basic identity checks on the default ETH config."""

    @staticmethod
    def config() -> SymbolConfig:
        return SymbolConfig.default_eth()

    def test_inst_id_is_eth(self) -> None:
        assert self.config().inst_id == "ETH-USDT-SWAP"

    def test_is_enabled(self) -> None:
        assert self.config().is_enabled is True

    def test_is_live_trading_enabled_is_false(self) -> None:
        # live_trading defaults to False to prevent accidental live trading.
        assert self.config().is_live_trading_enabled is False

    def test_symbol_identity_defaults(self) -> None:
        identity = self.config().symbol
        assert identity.inst_id == "ETH-USDT-SWAP"
        assert identity.enabled is True
        assert identity.live_trading is False


# ---------------------------------------------------------------------------
# SymbolConfig.default_eth() – market
# ---------------------------------------------------------------------------


class TestDefaultEthMarketConfig:
    """Market-parameter defaults."""

    @staticmethod
    def config() -> SymbolConfig:
        return SymbolConfig.default_eth()

    def test_bar(self) -> None:
        assert self.config().market.bar == "15m"

    def test_contract_value(self) -> None:
        assert self.config().market.contract_value == Decimal("0.1")

    def test_min_contracts(self) -> None:
        assert self.config().market.min_contracts == Decimal("0.01")

    def test_boll_window(self) -> None:
        assert self.config().market.boll_window == 20

    def test_tp_boll_window(self) -> None:
        assert self.config().market.tp_boll_window == 15

    def test_min_outside_pct(self) -> None:
        assert self.config().market.min_outside_pct == Decimal("0.0005")


# ---------------------------------------------------------------------------
# SymbolConfig.default_eth() – entry
# ---------------------------------------------------------------------------


class TestDefaultEthEntryConfig:
    """Entry-parameter defaults."""

    @staticmethod
    def config() -> SymbolConfig:
        return SymbolConfig.default_eth()

    def test_first_add_block_seconds(self) -> None:
        assert self.config().entry.first_add_block_seconds == 3600

    def test_add_min_interval_seconds(self) -> None:
        assert self.config().entry.add_min_interval_seconds == 1800


# ---------------------------------------------------------------------------
# SymbolConfig.default_eth() – TP safety defaults
# ---------------------------------------------------------------------------


class TestDefaultEthTpSafetyDefaults:
    """Three-Stage TP defaults that must not be changed accidentally."""

    @staticmethod
    def config() -> SymbolConfig:
        return SymbolConfig.default_eth()

    def test_three_stage_runner_enabled(self) -> None:
        assert self.config().tp.three_stage_runner_enabled is True

    def test_tp1_ratio(self) -> None:
        assert self.config().tp.three_stage_tp1_ratio == Decimal("0.70")

    def test_tp2_ratio(self) -> None:
        assert self.config().tp.three_stage_tp2_ratio == Decimal("0.20")

    def test_runner_ratio(self) -> None:
        assert self.config().tp.three_stage_runner_ratio == Decimal("0.10")

    def test_tp2_use_structure_boll_is_true(self) -> None:
        assert self.config().tp.three_stage_tp2_use_structure_boll is True

    def test_split_tp_enabled_is_false(self) -> None:
        assert self.config().tp.split_tp_enabled is False


# ---------------------------------------------------------------------------
# SymbolConfig.default_eth() – DME delay
# ---------------------------------------------------------------------------


class TestDefaultEthDmeDelay:
    """The market-exit delay after an order failure must be 1800 seconds."""

    def test_order_failure_market_exit_delay(self) -> None:
        config = SymbolConfig.default_eth()
        assert config.risk.order_failure_market_exit_delay_seconds == 1800


# ---------------------------------------------------------------------------
# SymbolConfig.default_eth() – Decimal fields are Decimal (not float)
# ---------------------------------------------------------------------------


class TestDecimalFieldsAreDecimalNotFloat:
    """Every field declared as ``Decimal`` must actually *be* a ``Decimal``
    instance, never a plain ``float``."""

    @staticmethod
    def config() -> SymbolConfig:
        return SymbolConfig.default_eth()

    # market
    def test_market_contract_value(self) -> None:
        assert isinstance(self.config().market.contract_value, Decimal)

    # capital
    def test_capital_layer_margin_pct(self) -> None:
        assert isinstance(self.config().capital.layer_margin_pct, Decimal)

    def test_capital_leverage(self) -> None:
        assert isinstance(self.config().capital.leverage, Decimal)

    # entry
    def test_entry_add_gap_base_pct(self) -> None:
        assert isinstance(self.config().entry.add_gap_base_pct, Decimal)

    # cvd
    def test_cvd_fast_window_seconds(self) -> None:
        assert isinstance(self.config().cvd.fast_window_seconds, Decimal)

    # tp
    def test_tp_three_stage_tp1_ratio(self) -> None:
        assert isinstance(self.config().tp.three_stage_tp1_ratio, Decimal)

    # middle bucket split
    def test_middle_bucket_fast_ratio(self) -> None:
        assert isinstance(self.config().middle_bucket_split.fast_ratio, Decimal)

    # sidecar
    def test_sidecar_margin_pct(self) -> None:
        assert isinstance(self.config().sidecar.margin_pct, Decimal)

    # execution
    def test_execution_private_write_min_interval(self) -> None:
        assert isinstance(
            self.config().execution.private_write_min_interval_seconds, Decimal
        )

    # runtime
    def test_runtime_account_sync_seconds(self) -> None:
        assert isinstance(self.config().runtime.account_sync_seconds, Decimal)
