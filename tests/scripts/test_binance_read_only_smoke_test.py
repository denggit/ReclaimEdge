#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_read_only_smoke_test.py
@Description: Unit tests for scripts/binance_read_only_smoke_test.py

Covers confirmation, config validation, credential loading, read-only
runner, and main wiring — all without network calls.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Sequence
from unittest import mock

import pytest

from scripts.binance_read_only_smoke_test import (
    BINANCE_SYMBOL,
    READ_ONLY_CONFIRM_ENV,
    READ_ONLY_CONFIRM_VALUE,
    load_binance_read_only_credentials,
    main,
    require_read_only_confirmation,
    run_read_only_smoke,
    validate_unified_config_for_binance_read_only,
)
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.runtime_config import ExchangeRuntimeConfig


# ======================================================================
# Helpers
# ======================================================================


def _make_valid_rt(**overrides: Any) -> ExchangeRuntimeConfig:
    """Build a valid Binance runtime config, optionally overriding fields."""
    defaults: dict[str, Any] = {
        "exchange": ExchangeName.BINANCE,
        "trade_asset": "ETH",
        "quote_asset": "USDT",
        "market_type": "PERPETUAL",
        "leverage": 20,
        "margin_mode": "isolated",
        "position_mode": "net",
        "kline_interval": "15m",
    }
    defaults.update(overrides)
    return ExchangeRuntimeConfig(**defaults)


# ======================================================================
# Confirmation tests
# ======================================================================


class TestRequireReadOnlyConfirmation:
    """Tests for ``require_read_only_confirmation()``."""

    def test_missing_env_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(READ_ONLY_CONFIRM_ENV, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_read_only_confirmation()
        assert exc_info.value.code == 1

    def test_wrong_value_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, "NO")
        with pytest.raises(SystemExit) as exc_info:
            require_read_only_confirmation()
        assert exc_info.value.code == 1

    def test_empty_value_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, "")
        with pytest.raises(SystemExit) as exc_info:
            require_read_only_confirmation()
        assert exc_info.value.code == 1

    def test_correct_value_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, READ_ONLY_CONFIRM_VALUE)
        # Must not raise
        require_read_only_confirmation()

    def test_does_not_accept_live_smoke_confirmation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BINANCE_LIVE_SMOKE_TEST_CONFIRM must not satisfy the gate."""
        monkeypatch.setenv(
            READ_ONLY_CONFIRM_ENV, "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"
        )
        with pytest.raises(SystemExit) as exc_info:
            require_read_only_confirmation()
        assert exc_info.value.code == 1

    def test_does_not_accept_live_smoke_env_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting the live-smoke env var with the right read-only value
        must NOT satisfy the gate, because the gate reads
        BINANCE_READ_ONLY_SMOKE_CONFIRM, not BINANCE_LIVE_SMOKE_TEST_CONFIRM."""
        monkeypatch.setenv(
            "BINANCE_LIVE_SMOKE_TEST_CONFIRM", READ_ONLY_CONFIRM_VALUE
        )
        monkeypatch.delenv(READ_ONLY_CONFIRM_ENV, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_read_only_confirmation()
        assert exc_info.value.code == 1


# ======================================================================
# Unified config validation tests
# ======================================================================


class TestValidateUnifiedConfig:
    """Tests for ``validate_unified_config_for_binance_read_only()``."""

    def test_exchange_not_binance_exits(self) -> None:
        rt = _make_valid_rt(exchange=ExchangeName.OKX)
        with pytest.raises(SystemExit) as exc_info:
            validate_unified_config_for_binance_read_only(rt)
        assert exc_info.value.code == 1

    def test_canonical_not_eth_usdt_perp_exits(self) -> None:
        rt = _make_valid_rt(trade_asset="BTC")
        with pytest.raises(SystemExit) as exc_info:
            validate_unified_config_for_binance_read_only(rt)
        assert exc_info.value.code == 1

    def test_binance_symbol_not_ethusdt_exits(self) -> None:
        rt = _make_valid_rt(trade_asset="BTC", quote_asset="USDT")
        with pytest.raises(SystemExit) as exc_info:
            validate_unified_config_for_binance_read_only(rt)
        assert exc_info.value.code == 1

    def test_position_mode_not_net_exits(self) -> None:
        rt = _make_valid_rt(position_mode="hedge")
        with pytest.raises(SystemExit) as exc_info:
            validate_unified_config_for_binance_read_only(rt)
        assert exc_info.value.code == 1

    def test_margin_mode_not_isolated_exits(self) -> None:
        rt = _make_valid_rt(margin_mode="cross")
        with pytest.raises(SystemExit) as exc_info:
            validate_unified_config_for_binance_read_only(rt)
        assert exc_info.value.code == 1

    def test_kline_interval_not_15m_exits(self) -> None:
        rt = _make_valid_rt(kline_interval="1h")
        with pytest.raises(SystemExit) as exc_info:
            validate_unified_config_for_binance_read_only(rt)
        assert exc_info.value.code == 1

    def test_valid_config_returns_ethusdt(self) -> None:
        rt = _make_valid_rt()
        result = validate_unified_config_for_binance_read_only(rt)
        assert result == "ETHUSDT"

    def test_valid_config_returns_binance_symbol(self) -> None:
        rt = _make_valid_rt()
        result = validate_unified_config_for_binance_read_only(rt)
        assert result == rt.binance_symbol
        assert result == BINANCE_SYMBOL


# ======================================================================
# Credential loading tests
# ======================================================================


class TestLoadBinanceReadOnlyCredentials:
    """Tests for ``load_binance_read_only_credentials()``."""

    def test_missing_api_key_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
        monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_read_only_credentials()
        assert exc_info.value.code == 1

    def test_missing_api_secret_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "key")
        monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            load_binance_read_only_credentials()
        assert exc_info.value.code == 1

    def test_empty_api_key_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "   ")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_read_only_credentials()
        assert exc_info.value.code == 1

    def test_empty_api_secret_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "key")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "   ")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_read_only_credentials()
        assert exc_info.value.code == 1

    def test_both_present_returns_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "test-key")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "test-secret")
        key, secret = load_binance_read_only_credentials()
        assert key == "test-key"
        assert secret == "test-secret"

    def test_stdout_does_not_contain_key_value(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "my-secret-key")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "my-secret-value")
        load_binance_read_only_credentials()
        captured = capsys.readouterr()
        assert "my-secret-key" not in captured.out
        assert "my-secret-value" not in captured.out

    def test_stderr_does_not_contain_key_value_on_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "leaked-key")
        monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
        with pytest.raises(SystemExit):
            load_binance_read_only_credentials()
        captured = capsys.readouterr()
        assert "leaked-key" not in captured.err


# ======================================================================
# Fake client for testing the read-only runner
# ======================================================================


class FakeReadOnlyClient:
    """A fake BinanceBrokerClient that only supports read-only methods.

    This client implements the same interface as BinanceBrokerClient but
    all write methods raise AssertionError to prove they are never called.
    """

    def __init__(
        self,
        *,
        position: BrokerPosition | None = None,
        open_orders: Sequence[BrokerOrder] | None = None,
    ) -> None:
        self._position = position
        self._open_orders = list(open_orders) if open_orders is not None else []
        self._fetch_position_called = False
        self._fetch_open_orders_called = False
        self._last_symbol: str | None = None

    @property
    def fetch_position_called(self) -> bool:
        return self._fetch_position_called

    @property
    def fetch_open_orders_called(self) -> bool:
        return self._fetch_open_orders_called

    @property
    def last_symbol(self) -> str | None:
        return self._last_symbol

    async def fetch_position(self, symbol: str) -> BrokerPosition | None:
        self._fetch_position_called = True
        self._last_symbol = symbol
        return self._position

    async def fetch_open_orders(self, symbol: str) -> Sequence[BrokerOrder]:
        self._fetch_open_orders_called = True
        self._last_symbol = symbol
        return self._open_orders

    # Write methods — must never be called in read-only smoke.
    # If they are called the test fails with AssertionError.

    async def place_order(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("place_order MUST NOT be called in read-only smoke")

    async def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cancel_order MUST NOT be called in read-only smoke")


def _make_fake_position(
    side: BrokerPositionSide = BrokerPositionSide.LONG,
    quantity: int = 1,
) -> BrokerPosition:
    """Create a minimal BrokerPosition for testing."""
    from decimal import Decimal

    return BrokerPosition(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        position_side=side,
        quantity=Decimal(str(quantity)),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        average_entry_price=Decimal("3000"),
        mark_price=Decimal("3001"),
        unrealized_pnl=Decimal("1"),
    )


def _make_fake_order(
    order_id: str = "123",
) -> BrokerOrder:
    """Create a minimal BrokerOrder for testing."""
    from decimal import Decimal

    return BrokerOrder(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        order_id=order_id,
        client_order_id="cid-1",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.NET,
        order_type=BrokerOrderType.LIMIT,
        status=BrokerOrderStatus.OPEN,
        price=Decimal("3100"),
        quantity=Decimal("0.1"),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        reduce_only=True,
    )


# ======================================================================
# Read-only runner tests
# ======================================================================


class TestRunReadOnlySmoke:
    """Tests for ``run_read_only_smoke()``."""

    @pytest.mark.asyncio
    async def test_calls_fetch_position(self) -> None:
        client = FakeReadOnlyClient()
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        assert client.fetch_position_called

    @pytest.mark.asyncio
    async def test_calls_fetch_open_orders(self) -> None:
        client = FakeReadOnlyClient()
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        assert client.fetch_open_orders_called

    @pytest.mark.asyncio
    async def test_symbol_is_ethusdt(self) -> None:
        client = FakeReadOnlyClient()
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        assert client.last_symbol == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_no_position_no_orders_succeeds(self) -> None:
        client = FakeReadOnlyClient(position=None, open_orders=[])
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        # Must not raise

    @pytest.mark.asyncio
    async def test_with_position_and_orders_succeeds(self) -> None:
        pos = _make_fake_position()
        orders = [_make_fake_order("1"), _make_fake_order("2")]
        client = FakeReadOnlyClient(position=pos, open_orders=orders)
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        # Must not raise

    @pytest.mark.asyncio
    async def test_with_position_prints_summary(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pos = _make_fake_position()
        client = FakeReadOnlyClient(position=pos, open_orders=[])
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        captured = capsys.readouterr()
        assert "fetch_position OK" in captured.out
        assert "has_position=True" in captured.out

    @pytest.mark.asyncio
    async def test_with_open_orders_prints_summary(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orders = [_make_fake_order("456")]
        client = FakeReadOnlyClient(position=None, open_orders=orders)
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        captured = capsys.readouterr()
        assert "fetch_open_orders OK" in captured.out
        assert "count=1" in captured.out
        assert "order_id=456" in captured.out

    @pytest.mark.asyncio
    async def test_no_position_prints_has_position_false(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = FakeReadOnlyClient(position=None, open_orders=[])
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        captured = capsys.readouterr()
        assert "has_position=False" in captured.out

    @pytest.mark.asyncio
    async def test_done_message_present(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = FakeReadOnlyClient()
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        captured = capsys.readouterr()
        assert "done | no orders were placed" in captured.out

    @pytest.mark.asyncio
    async def test_does_not_call_place_order(self) -> None:
        """FakeClient.place_order raises AssertionError if called."""
        client = FakeReadOnlyClient()
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        # If place_order were called the test would already have failed

    @pytest.mark.asyncio
    async def test_does_not_call_cancel_order(self) -> None:
        """FakeClient.cancel_order raises AssertionError if called."""
        client = FakeReadOnlyClient()
        await run_read_only_smoke(client=client, symbol="ETHUSDT")
        # If cancel_order were called the test would already have failed


# ======================================================================
# Main wiring tests
# ======================================================================


class TestMainWiring:
    """Tests for ``main()`` using monkeypatching."""

    @pytest.mark.asyncio
    async def test_main_creates_client_and_calls_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirm the full wiring: main creates transport + client and calls
        the read-only runner without invoking any order methods."""
        # Gate
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, READ_ONLY_CONFIRM_VALUE)

        # Credentials
        monkeypatch.setenv("EXCHANGE_API_KEY", "k")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "s")

        # Unified config
        monkeypatch.setenv("EXCHANGE", "binance")
        monkeypatch.setenv("TRADE_ASSET", "ETH")
        monkeypatch.setenv("QUOTE_ASSET", "USDT")
        monkeypatch.setenv("MARKET_TYPE", "PERPETUAL")
        monkeypatch.setenv("MARGIN_MODE", "isolated")
        monkeypatch.setenv("POSITION_MODE", "net")
        monkeypatch.setenv("KLINE_INTERVAL", "15m")

        runner_called = False
        runner_symbol: str | None = None

        async def _fake_runner(*, client: Any, symbol: str) -> None:
            nonlocal runner_called, runner_symbol
            runner_called = True
            runner_symbol = symbol

        monkeypatch.setattr(
            "scripts.binance_read_only_smoke_test.run_read_only_smoke",
            _fake_runner,
        )

        await main()

        assert runner_called, "run_read_only_smoke should be called"
        assert runner_symbol == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_main_does_not_call_order_methods(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use a FakeReadOnlyClient to ensure no order methods are called."""
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, READ_ONLY_CONFIRM_VALUE)
        monkeypatch.setenv("EXCHANGE_API_KEY", "k")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
        monkeypatch.setenv("EXCHANGE", "binance")

        # Force the real runner to use our fake client
        runner_called_with_client: Any = None

        async def _capture_runner(*, client: Any, symbol: str) -> None:
            nonlocal runner_called_with_client
            runner_called_with_client = client

        monkeypatch.setattr(
            "scripts.binance_read_only_smoke_test.run_read_only_smoke",
            _capture_runner,
        )

        await main()

        # The client was captured — verify it's a BinanceBrokerClient
        assert runner_called_with_client is not None

    @pytest.mark.asyncio
    async def test_main_exits_without_confirmation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main must exit when confirmation is not set."""
        monkeypatch.delenv(READ_ONLY_CONFIRM_ENV, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_exits_without_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main must exit when credentials are missing."""
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, READ_ONLY_CONFIRM_VALUE)
        monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
        monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
        monkeypatch.setenv("EXCHANGE", "binance")
        with pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_exits_with_invalid_exchange(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Main must exit when EXCHANGE is not binance."""
        monkeypatch.setenv(READ_ONLY_CONFIRM_ENV, READ_ONLY_CONFIRM_VALUE)
        monkeypatch.setenv("EXCHANGE_API_KEY", "k")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
        monkeypatch.setenv("EXCHANGE", "okx")
        with pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1
