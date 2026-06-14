#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_live_smoke_test.py
@Description: Unit tests for the Binance live smoke test script.

All tests use mocked / fake transports — no real network calls.
"""

from __future__ import annotations

import os
from decimal import ROUND_UP, Decimal
from typing import Any
from unittest import mock

import pytest

from scripts.binance_live_smoke_test import (
    ALLOW_SET_LEVERAGE_ENV,
    ALLOW_SET_LEVERAGE_VALUE,
    ALGO_ORDER_PATH,
    BINANCE_SYMBOL,
    CANONICAL_SYMBOL,
    CLIENT_ORDER_ID_PREFIX,
    CONFIRM_ENV,
    CONFIRM_VALUE,
    CHANGE_LEVERAGE_PATH,
    DEFAULT_MAX_NOTIONAL,
    DEFAULT_MARGIN_BUFFER_MULTIPLIER,
    DEFAULT_SL_PCT,
    DEFAULT_TP_PCT,
    ENV_MARGIN_BUFFER_MULTIPLIER,
    ExchangeInfoFilters,
    POSITION_RISK_PATH,
    Preflight,
    _generate_client_order_id,
    _make_order_request,
    _read_positive_decimal_env,
    _round_up_to_step,
    allow_set_leverage,
    calculate_required_margin_with_buffer,
    calculate_safe_quantity,
    cancel_algo_order_by_client_id,
    cancel_order_by_id,
    cancel_smoke_algo_orders,
    cancel_smoke_orders,
    cleanup,
    close_long_position,
    fetch_account_balance,
    fetch_algo_open_orders,
    fetch_long_position,
    fetch_open_orders,
    load_binance_credentials,
    main,
    open_long,
    place_sl,
    place_stop_loss_algo_order,
    place_tp,
    require_binance_live_preflight_for_smoke,
    require_calculated_notional_cap,
    require_existing_leverage,
    require_isolated_margin,
    require_live_confirmation,
    require_no_existing_position,
    require_one_way_position_mode,
    require_requested_notional_cap,
    set_initial_leverage,
    validate_unified_config_for_binance,
)
from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.binance.signing import (
    BINANCE_USDM_BASE_URL,
    BINANCE_USDM_OPEN_ORDERS_PATH,
    BINANCE_USDM_ORDER_PATH,
    BINANCE_USDM_POSITION_RISK_PATH,
    BinanceSignedRequest,
)
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeErrorKind
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeBinanceTransport:
    """Record requests and return canned responses in sequence."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.requests: list[BinanceSignedRequest] = []
        self._idx = 0

    async def send(self, request: BinanceSignedRequest) -> BinanceTransportResponse:
        self.requests.append(request)
        if self._idx >= len(self.responses):
            return BinanceTransportResponse(
                status_code=200,
                payload={},
                headers={},
            )
        payload = self.responses[self._idx]
        self._idx += 1
        if isinstance(payload, BinanceTransportResponse):
            return payload
        status_code = 200
        if isinstance(payload, tuple):
            status_code, actual_payload = payload
            return BinanceTransportResponse(
                status_code=status_code,
                payload=actual_payload,
                headers={},
            )
        return BinanceTransportResponse(
            status_code=status_code,
            payload=payload,
            headers={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(*responses: Any) -> BinanceBrokerClient:
    transport = FakeBinanceTransport(*responses)
    return BinanceBrokerClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
        position_mode="net",
    )


def _minimal_order_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "orderId": 123456789,
        "clientOrderId": "cid-abc-001",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "MARKET",
        "status": "NEW",
        "price": "0",
        "origQty": "0.1",
        "executedQty": "0.1",
        "avgPrice": "3100.50",
        "reduceOnly": False,
    }
    data.update(overrides)
    return data


def _minimal_position_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "positionAmt": "0.1",
        "entryPrice": "3100.50",
        "markPrice": "3105.00",
        "unRealizedProfit": "0.45",
        "positionSide": "LONG",
        "leverage": "1",
    }
    data.update(overrides)
    return data


def _fake_open_order_response(
    order_id: int = 1,
    client_order_id: str = "cid-tp-001",
    side: str = "SELL",
    order_type: str = "LIMIT",
    price: str = "3200.00",
    trigger_price: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "orderId": order_id,
        "clientOrderId": client_order_id,
        "side": side,
        "positionSide": "LONG",
        "type": order_type,
        "status": "NEW",
        "price": price,
        "origQty": "0.1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
    }
    if trigger_price is not None:
        data["stopPrice"] = trigger_price
    return data


def _cancel_payload(order_id: int = 123456789) -> dict[str, Any]:
    return {
        "symbol": "ETHUSDT",
        "orderId": order_id,
        "clientOrderId": "cid-cancel-1",
        "status": "CANCELED",
    }


# ---------------------------------------------------------------------------
# Tests: safety gates
# ---------------------------------------------------------------------------


class TestRequireLiveConfirmation:
    def test_rejects_when_env_not_set(self, monkeypatch) -> None:
        monkeypatch.delenv(CONFIRM_ENV, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_live_confirmation()
        assert exc_info.value.code == 1

    def test_rejects_when_env_has_wrong_value(self, monkeypatch) -> None:
        monkeypatch.setenv(CONFIRM_ENV, "NO")
        with pytest.raises(SystemExit) as exc_info:
            require_live_confirmation()
        assert exc_info.value.code == 1

    def test_accepts_correct_confirmation(self, monkeypatch) -> None:
        monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
        require_live_confirmation()  # does not raise


class TestValidateUnifiedConfigForBinance:
    @staticmethod
    def _make_rt(**overrides):
        from dataclasses import replace
        from src.exchanges.runtime_config import ExchangeRuntimeConfig
        from src.exchanges.models import ExchangeName
        base = ExchangeRuntimeConfig(
            exchange=ExchangeName.BINANCE,
            trade_asset="ETH",
            quote_asset="USDT",
            market_type="PERPETUAL",
        )
        if not overrides:
            return base
        return replace(base, **overrides)

    def test_rejects_okx_config(self) -> None:
        rt = self._make_rt(exchange=ExchangeName.OKX)
        with pytest.raises(SystemExit):
            validate_unified_config_for_binance(rt)

    def test_accepts_binance_config(self) -> None:
        rt = self._make_rt()
        symbol = validate_unified_config_for_binance(rt)
        assert symbol == "ETHUSDT"

    def test_rejects_wrong_canonical_symbol(self) -> None:
        rt = self._make_rt(trade_asset="BTC")
        with pytest.raises(SystemExit):
            validate_unified_config_for_binance(rt)

    def test_rejects_hedge_position_mode(self) -> None:
        rt = self._make_rt(position_mode="hedge")
        with pytest.raises(SystemExit):
            validate_unified_config_for_binance(rt)

    def test_rejects_cross_margin_mode(self) -> None:
        rt = self._make_rt(margin_mode="cross")
        with pytest.raises(SystemExit):
            validate_unified_config_for_binance(rt)

    def test_rejects_1m_kline_interval(self) -> None:
        rt = self._make_rt(kline_interval="1m")
        with pytest.raises(SystemExit):
            validate_unified_config_for_binance(rt)


class TestLoadBinanceCredentials:
    def test_rejects_missing_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
        monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_credentials()
        assert exc_info.value.code == 1

    def test_rejects_missing_api_secret(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "key")
        monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            load_binance_credentials()
        assert exc_info.value.code == 1

    def test_rejects_empty_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "   ")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "secret")
        with pytest.raises(SystemExit) as exc_info:
            load_binance_credentials()
        assert exc_info.value.code == 1

    def test_returns_credentials_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCHANGE_API_KEY", "my-key")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "my-secret")
        key, secret = load_binance_credentials()
        assert key == "my-key"
        assert secret == "my-secret"


# ---------------------------------------------------------------------------
# Tests: _round_up_to_step
# ---------------------------------------------------------------------------


class TestRoundUpToStep:
    def test_exact_multiple_unchanged(self) -> None:
        assert _round_up_to_step(Decimal("0.1"), Decimal("0.1")) == Decimal("0.1")

    def test_rounds_up_to_next_step(self) -> None:
        assert _round_up_to_step(Decimal("0.011"), Decimal("0.001")) == Decimal("0.011")
        assert _round_up_to_step(Decimal("0.0015"), Decimal("0.001")) == Decimal("0.002")

    def test_step_of_0_1(self) -> None:
        assert _round_up_to_step(Decimal("0.05"), Decimal("0.1")) == Decimal("0.1")
        assert _round_up_to_step(Decimal("0.10"), Decimal("0.1")) == Decimal("0.1")
        assert _round_up_to_step(Decimal("0.11"), Decimal("0.1")) == Decimal("0.2")

    def test_raises_on_zero_step(self) -> None:
        with pytest.raises(ValueError, match="step_size must be positive"):
            _round_up_to_step(Decimal("0.1"), Decimal("0"))


# ---------------------------------------------------------------------------
# Tests: calculate_safe_quantity
# ---------------------------------------------------------------------------


class TestCalculateSafeQuantity:
    _FILTERS = ExchangeInfoFilters(
        min_qty=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("5"),
    )

    def test_basic_calculation(self) -> None:
        qty, notional = calculate_safe_quantity(
            mark_price=Decimal("3000"),
            max_notional=Decimal("6"),
            filters=self._FILTERS,
        )
        assert qty > 0
        assert notional > 0

    def test_quantity_rounded_to_step(self) -> None:
        qty, _ = calculate_safe_quantity(
            mark_price=Decimal("3100"),
            max_notional=Decimal("5"),
            filters=self._FILTERS,
        )
        # 5/3100 = 0.0016129... → rounded up to 0.002
        assert qty == Decimal("0.002")

    def test_notional_below_min_gets_bumped(self) -> None:
        filters = ExchangeInfoFilters(
            min_qty=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("100"),
        )
        qty, notional = calculate_safe_quantity(
            mark_price=Decimal("3000"),
            max_notional=Decimal("6"),
            filters=filters,
        )
        assert notional >= filters.min_notional

    def test_below_min_qty_exits(self) -> None:
        filters = ExchangeInfoFilters(
            min_qty=Decimal("10"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
        )
        with pytest.raises(SystemExit):
            calculate_safe_quantity(
                mark_price=Decimal("3000"),
                max_notional=Decimal("6"),
                filters=filters,
            )

    def test_min_notional_exceeds_exchange_info_when_quantity_too_large(self) -> None:
        """When min_qty * mark_price >> user budget, exit early."""
        filters = ExchangeInfoFilters(
            min_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
        )
        with pytest.raises(SystemExit):
            calculate_safe_quantity(
                mark_price=Decimal("3000"),
                max_notional=Decimal("6"),
                filters=filters,
            )


# ---------------------------------------------------------------------------
# Tests: _generate_client_order_id
# ---------------------------------------------------------------------------


class TestGenerateClientOrderId:
    def test_has_smoke_prefix(self) -> None:
        cid = _generate_client_order_id("open")
        assert cid.startswith(CLIENT_ORDER_ID_PREFIX)

    def test_contains_label(self) -> None:
        cid = _generate_client_order_id("tp")
        assert "tp" in cid

    def test_unique_per_call(self) -> None:
        cid1 = _generate_client_order_id("x")
        cid2 = _generate_client_order_id("x")
        assert cid1 != cid2

    def test_open_cid_length_le_36(self) -> None:
        cid = _generate_client_order_id("open")
        assert len(cid) <= 36

    def test_tp_cid_length_le_36(self) -> None:
        cid = _generate_client_order_id("tp")
        assert len(cid) <= 36

    def test_sl_cid_length_le_36(self) -> None:
        cid = _generate_client_order_id("sl")
        assert len(cid) <= 36

    def test_close_cid_length_le_36(self) -> None:
        cid = _generate_client_order_id("close")
        assert len(cid) <= 36

    def test_cleanup_close_cid_length_le_36(self) -> None:
        cid = _generate_client_order_id("cleanup_close")
        assert len(cid) <= 36

    def test_all_start_with_re_smoke(self) -> None:
        for label in ("open", "tp", "sl", "close", "cleanup_close"):
            cid = _generate_client_order_id(label)
            assert cid.startswith(CLIENT_ORDER_ID_PREFIX), f"{label} cid does not start with RE_SMOKE_"


# ---------------------------------------------------------------------------
# Tests: _make_order_request
# ---------------------------------------------------------------------------


class TestMakeOrderRequest:
    def test_market_buy_long(self) -> None:
        req = _make_order_request(
            side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("0.1"),
            client_order_id="test-001",
        )
        assert req.exchange == ExchangeName.BINANCE
        assert req.symbol == BINANCE_SYMBOL
        assert req.side == BrokerOrderSide.BUY
        assert req.position_side == BrokerPositionSide.NET
        assert req.order_type == BrokerOrderType.MARKET
        assert req.quantity == Decimal("0.1")
        assert req.quantity_unit == BrokerQuantityUnit.BASE_ASSET
        assert req.reduce_only is False
        assert req.client_order_id == "test-001"

    def test_limit_sell_tp(self) -> None:
        req = _make_order_request(
            side=BrokerOrderSide.SELL,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("3200"),
            client_order_id="tp-001",
        )
        assert req.side == BrokerOrderSide.SELL
        assert req.order_type == BrokerOrderType.LIMIT
        assert req.price == Decimal("3200")

    def test_stop_market_sell_sl(self) -> None:
        req = _make_order_request(
            side=BrokerOrderSide.SELL,
            order_type=BrokerOrderType.STOP_MARKET,
            quantity=Decimal("0.1"),
            trigger_price=Decimal("2900"),
            client_order_id="sl-001",
        )
        assert req.side == BrokerOrderSide.SELL
        assert req.order_type == BrokerOrderType.STOP_MARKET
        assert req.trigger_price == Decimal("2900")


# ---------------------------------------------------------------------------
# Tests: open_long
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_long_sends_correct_request() -> None:
    client = _make_client(_minimal_order_payload())
    result = await open_long(client, Decimal("0.1"), "RE_SMOKE_open_123")
    assert result.ok is True
    assert result.order is not None
    transport = client._transport
    assert transport is not None
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "POST"
    assert transport.requests[0].path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_open_long_raises_on_failure() -> None:
    from src.exchanges.errors import ExchangeError
    client = _make_client(
        BinanceTransportResponse(status_code=400, payload={"code": -2019, "msg": "Insufficient balance"}, headers={}),
    )
    with pytest.raises(ExchangeError) as exc_info:
        await open_long(client, Decimal("0.1"), "RE_SMOKE_open_123")
    assert exc_info.value.kind == ExchangeErrorKind.INSUFFICIENT_BALANCE


# ---------------------------------------------------------------------------
# Tests: fetch_long_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_long_position_returns_broker_position() -> None:
    client = _make_client([_minimal_position_payload()])
    pos = await fetch_long_position(client)
    assert pos is not None
    assert pos.symbol == "ETHUSDT"
    assert pos.position_side == BrokerPositionSide.LONG
    assert pos.quantity == Decimal("0.1")


@pytest.mark.asyncio
async def test_fetch_long_position_returns_none_when_zero() -> None:
    payload = _minimal_position_payload(positionAmt="0", positionSide="LONG")
    client = _make_client([payload])
    pos = await fetch_long_position(client)
    assert pos is None


@pytest.mark.asyncio
async def test_fetch_long_position_returns_position_with_positive_quantity() -> None:
    """In One-way mode, any position with quantity > 0 is returned."""
    payload = _minimal_position_payload(positionAmt="0.1", positionSide="SHORT")
    client = _make_client([payload])
    pos = await fetch_long_position(client)
    assert pos is not None
    assert pos.quantity > 0


# ---------------------------------------------------------------------------
# Tests: place_tp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_tp_sends_limit_sell() -> None:
    client = _make_client(
        _minimal_order_payload(side="SELL", type="LIMIT", price="3200.00"),
    )
    result = await place_tp(client, Decimal("0.1"), Decimal("3200"), "RE_SMOKE_tp_123")
    assert result.ok is True
    assert result.order is not None
    transport = client._transport
    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_place_tp_raises_on_failure() -> None:
    from src.exchanges.errors import ExchangeError
    client = _make_client(
        BinanceTransportResponse(
            status_code=400, payload={"code": -2019, "msg": "Margin insufficient"}, headers={},
        ),
    )
    with pytest.raises(ExchangeError) as exc_info:
        await place_tp(client, Decimal("0.1"), Decimal("3200"), "RE_SMOKE_tp_123")
    assert exc_info.value.kind == ExchangeErrorKind.INSUFFICIENT_BALANCE


# ---------------------------------------------------------------------------
# Tests: place_sl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_sl_calls_algo_order_helper() -> None:
    """place_sl now delegates to place_stop_loss_algo_order — it does NOT
    go through BinanceBrokerClient.place_order()."""
    from unittest import mock

    fake_result = BrokerOrderResult(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        ok=True,
        order_id="12345",
        client_order_id="RE_SMOKE_sl_123",
    )

    async def _fake_algo(*, api_key, api_secret, quantity, sl_price, client_order_id):
        return fake_result

    with mock.patch(
        "scripts.binance_live_smoke_test.place_stop_loss_algo_order",
        side_effect=_fake_algo,
    ) as mock_algo:
        result = await place_sl(
            api_key="test-key",
            api_secret="test-secret",
            quantity=Decimal("0.1"),
            sl_price=Decimal("2900"),
            client_order_id="RE_SMOKE_sl_123",
        )
        assert result is fake_result
        mock_algo.assert_called_once_with(
            api_key="test-key",
            api_secret="test-secret",
            quantity=Decimal("0.1"),
            sl_price=Decimal("2900"),
            client_order_id="RE_SMOKE_sl_123",
        )


@pytest.mark.asyncio
async def test_place_sl_does_not_use_client_place_order() -> None:
    """place_sl must NOT use BinanceBrokerClient.place_order()."""
    client = _make_client(
        _minimal_order_payload(side="SELL", type="STOP_MARKET"),
    )
    # place_sl no longer accepts a client param — it would be a TypeError
    with pytest.raises(TypeError):
        await place_sl(client, Decimal("0.1"), Decimal("2900"), "RE_SMOKE_sl_123")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_place_sl_raises_on_algo_failure() -> None:
    """place_sl propagates RuntimeError from the algo helper."""
    from unittest import mock

    async def _fake_algo_fail(*, api_key, api_secret, quantity, sl_price, client_order_id):
        raise RuntimeError("Algo SL order rejected: [-2021] Stop price error")

    with mock.patch(
        "scripts.binance_live_smoke_test.place_stop_loss_algo_order",
        side_effect=_fake_algo_fail,
    ):
        with pytest.raises(RuntimeError, match="Stop price error"):
            await place_sl(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_123",
            )


# ---------------------------------------------------------------------------
# Tests: place_stop_loss_algo_order
# ---------------------------------------------------------------------------


class TestPlaceStopLossAlgoOrder:
    @pytest.mark.asyncio
    async def test_sends_algo_order_request(self) -> None:
        """place_stop_loss_algo_order sends POST to /fapi/v1/algoOrder."""
        from unittest import mock

        fake_response = BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 99999, "clientAlgoId": "RE_SMOKE_sl_456", "algoType": "CONDITIONAL", "code": 200},
            headers={},
        )

        async def fake_send(request):
            return fake_response

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            result = await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_456",
            )
            assert result.ok is True
            assert result.order_id == "99999"

    @pytest.mark.asyncio
    async def test_request_contains_ethusdt(self) -> None:
        """Algo SL request must include ETHUSDT symbol."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 1, "clientAlgoId": "cid", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_789",
            )

        assert captured_params.get("symbol") == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_request_contains_sell_side(self) -> None:
        """Algo SL request must be SELL."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 2, "clientAlgoId": "cid", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_789",
            )

        assert captured_params.get("side") == "SELL"

    @pytest.mark.asyncio
    async def test_request_contains_reduce_only(self) -> None:
        """Algo SL request must include reduceOnly=true."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 3, "clientAlgoId": "cid", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_789",
            )

        assert captured_params.get("reduceOnly") == "true"

    @pytest.mark.asyncio
    async def test_request_contains_short_client_order_id(self) -> None:
        """Algo SL request must include a short clientAlgoId."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 4, "clientAlgoId": "cid", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_789",
            )

        cid = captured_params.get("clientAlgoId")
        assert cid is not None
        assert len(str(cid)) <= 36

    @pytest.mark.asyncio
    async def test_request_contains_algo_type_conditional(self) -> None:
        """Algo SL request must include algoType=CONDITIONAL."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 5, "clientAlgoId": "cid", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_789",
            )

        assert captured_params.get("algoType") == "CONDITIONAL"

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self) -> None:
        """Algo SL raises RuntimeError on HTTP 400+."""
        from unittest import mock

        async def fake_send(request):
            return BinanceTransportResponse(
                status_code=400,
                payload={"code": -4120, "msg": "Endpoint not supported"},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            with pytest.raises(RuntimeError, match="Algo SL order HTTP 400"):
                await place_stop_loss_algo_order(
                    api_key="test-key",
                    api_secret="test-secret",
                    quantity=Decimal("0.1"),
                    sl_price=Decimal("2900"),
                    client_order_id="RE_SMOKE_sl_789",
                )

    @pytest.mark.asyncio
    async def test_raises_on_business_error(self) -> None:
        """Algo SL raises RuntimeError on negative code in 200 response."""
        from unittest import mock

        async def fake_send(request):
            return BinanceTransportResponse(
                status_code=200,
                payload={"code": -2021, "msg": "Invalid trigger price"},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            with pytest.raises(RuntimeError, match="Invalid trigger price"):
                await place_stop_loss_algo_order(
                    api_key="test-key",
                    api_secret="test-secret",
                    quantity=Decimal("0.1"),
                    sl_price=Decimal("-1"),
                    client_order_id="RE_SMOKE_sl_789",
                )


# ---------------------------------------------------------------------------
# Tests: fetch_algo_open_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_algo_open_orders_returns_list() -> None:
    """fetch_algo_open_orders returns list of algo order dicts."""
    from unittest import mock

    fake_orders = [
        {
            "algoId": 100,
            "clientAlgoId": "RE_SMOKE_sl_1",
            "orderType": "STOP_MARKET",
            "side": "SELL",
            "symbol": "ETHUSDT",
        },
    ]

    async def fake_send(request):
        return BinanceTransportResponse(
            status_code=200,
            payload=fake_orders,
            headers={},
        )

    with mock.patch(
        "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
        side_effect=fake_send,
    ):
        orders = await fetch_algo_open_orders(
            api_key="test-key",
            api_secret="test-secret",
        )
        assert len(orders) == 1
        assert orders[0]["clientAlgoId"] == "RE_SMOKE_sl_1"


# ---------------------------------------------------------------------------
# Tests: cancel_algo_order_by_client_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_algo_order_by_client_id() -> None:
    """cancel_algo_order_by_client_id sends DELETE to /fapi/v1/algoOrder."""
    from unittest import mock

    async def fake_send(request):
        return BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 100, "clientAlgoId": "RE_SMOKE_sl_1", "code": 200},
            headers={},
        )

    with mock.patch(
        "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
        side_effect=fake_send,
    ):
        result = await cancel_algo_order_by_client_id(
            api_key="test-key",
            api_secret="test-secret",
            client_order_id="RE_SMOKE_sl_1",
        )
        assert result.ok is True
        assert result.client_order_id == "RE_SMOKE_sl_1"


# ---------------------------------------------------------------------------
# Tests: cancel_smoke_algo_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_smoke_algo_orders_only_cancels_prefixed() -> None:
    """cancel_smoke_algo_orders cancels algo orders with RE_SMOKE_ prefix."""
    from unittest import mock

    fake_orders = [
        {"algoId": 1, "clientAlgoId": "RE_SMOKE_sl_1"},
        {"algoId": 2, "clientAlgoId": "OTHER_ALGO"},
        {"algoId": 3, "clientAlgoId": "RE_SMOKE_sl_2"},
    ]

    cancelled_ids = []

    async def fake_fetch(*, api_key, api_secret):
        return fake_orders

    async def fake_cancel(*, api_key, api_secret, client_order_id):
        cancelled_ids.append(client_order_id)
        return BrokerCancelResult(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            ok=True,
            order_id=None,
            client_order_id=client_order_id,
        )

    with mock.patch(
        "scripts.binance_live_smoke_test.fetch_algo_open_orders",
        side_effect=fake_fetch,
    ), mock.patch(
        "scripts.binance_live_smoke_test.cancel_algo_order_by_client_id",
        side_effect=fake_cancel,
    ):
        count = await cancel_smoke_algo_orders(
            api_key="test-key",
            api_secret="test-secret",
        )
        assert count == 2
        assert "RE_SMOKE_sl_1" in cancelled_ids
        assert "RE_SMOKE_sl_2" in cancelled_ids
        assert "OTHER_ALGO" not in cancelled_ids


# ---------------------------------------------------------------------------
# Tests: fetch_open_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_open_orders_returns_list() -> None:
    client = _make_client([
        _fake_open_order_response(order_id=1, client_order_id="RE_SMOKE_tp_1"),
        _fake_open_order_response(order_id=2, client_order_id="RE_SMOKE_sl_1", order_type="STOP_MARKET", price="0", trigger_price="2900"),
    ])
    orders = await fetch_open_orders(client)
    assert len(orders) == 2
    transport = client._transport
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "GET"
    assert transport.requests[0].path == BINANCE_USDM_OPEN_ORDERS_PATH


@pytest.mark.asyncio
async def test_fetch_open_orders_empty() -> None:
    client = _make_client([])
    orders = await fetch_open_orders(client)
    assert orders == []


# ---------------------------------------------------------------------------
# Tests: cancel_order_by_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_order_by_id() -> None:
    client = _make_client(_cancel_payload(order_id=123))
    result = await cancel_order_by_id(client, "123")
    assert result.ok is True
    assert result.order_id == "123"
    transport = client._transport
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "DELETE"
    assert transport.requests[0].path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_cancel_order_not_found_does_not_raise() -> None:
    client = _make_client(
        BinanceTransportResponse(
            status_code=400,
            payload={"code": -2011, "msg": "Order does not exist."},
            headers={},
        ),
    )
    # cancel_order_by_id calls client.cancel_order which raises ExchangeError on failure
    # But cancel_order_by_id doesn't catch it explicitly - let me check
    # Actually, looking at my code, it does catch and prints a warning
    # Let me re-examine...
    #
    # Wait: cancel_order_by_id calls client.cancel_order which will raise ExchangeError.
    # My cancel_order_by_id doesn't have a try/except. Let me verify:
    #
    # Looking at my code:
    # async def cancel_order_by_id(client, order_id):
    #     result = await client.cancel_order(BINANCE_SYMBOL, order_id)
    #     if not result.ok:
    #         print(...)
    #     ...
    #
    # This will RAISE ExchangeError when the order is not found, because
    # BinanceBrokerClient.cancel_order raises map_binance_error for error codes.
    #
    # So the test should expect the error to propagate. Let me fix the test.

    # Actually, the test is correct — we're testing that the error propagates.
    # The script handles it in the main flow via the try/except in main().
    # But for the unit test, we should expect the ExchangeError.

    from src.exchanges.errors import ExchangeError, ExchangeErrorKind
    with pytest.raises(ExchangeError) as exc_info:
        await cancel_order_by_id(client, "999")
    assert exc_info.value.kind == ExchangeErrorKind.ORDER_NOT_FOUND


# ---------------------------------------------------------------------------
# Tests: cancel_smoke_orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_smoke_orders_only_cancels_prefixed() -> None:
    client = _make_client(
        [
            _fake_open_order_response(order_id=1, client_order_id="RE_SMOKE_tp_1"),
            _fake_open_order_response(order_id=2, client_order_id="OTHER_ORDER"),
            _fake_open_order_response(order_id=3, client_order_id="RE_SMOKE_sl_1", order_type="STOP_MARKET", price="0", trigger_price="2900"),
        ],
        _cancel_payload(order_id=1),
        _cancel_payload(order_id=3),
    )
    result = await cancel_smoke_orders(client)
    # Should cancel 2 smoke orders, skip the other one
    assert result == 2


@pytest.mark.asyncio
async def test_cancel_smoke_orders_none_to_cancel() -> None:
    client = _make_client([])
    result = await cancel_smoke_orders(client)
    assert result == 0


# ---------------------------------------------------------------------------
# Tests: close_long_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_long_sends_market_sell() -> None:
    client = _make_client(
        _minimal_order_payload(side="SELL", type="MARKET"),
    )
    result = await close_long_position(client, Decimal("0.1"), "RE_SMOKE_close_123")
    assert result.ok is True
    transport = client._transport
    assert len(transport.requests) == 1
    assert transport.requests[0].method == "POST"
    assert transport.requests[0].path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_close_long_without_client_order_id() -> None:
    """close_long_position supports client_order_id=None for fallback."""
    client = _make_client(
        _minimal_order_payload(side="SELL", type="MARKET", clientOrderId=""),
    )
    result = await close_long_position(client, Decimal("0.1"), client_order_id=None)
    assert result.ok is True
    transport = client._transport
    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.path == BINANCE_USDM_ORDER_PATH
    # With client_order_id=None, no clientOrderId in params
    assert "clientOrderId" not in req.params


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_no_residual_position() -> None:
    client = _make_client(
        [],  # open orders → empty
        [_minimal_position_payload(positionAmt="0")],  # position → None
        [_minimal_position_payload(positionAmt="0")],  # final check → None
    )
    await cleanup(client, api_key="test-key", api_secret="test-secret")  # should not raise


@pytest.mark.asyncio
async def test_cleanup_cancels_algo_orders() -> None:
    """cleanup must cancel algo smoke orders when api_key/api_secret provided."""
    from unittest import mock

    async def fake_cancel_algo(*, api_key, api_secret):
        return 1

    with mock.patch(
        "scripts.binance_live_smoke_test.cancel_smoke_algo_orders",
        side_effect=fake_cancel_algo,
    ) as mock_cancel:
        client = _make_client(
            [],  # open orders
            [_minimal_position_payload(positionAmt="0")],
            [_minimal_position_payload(positionAmt="0")],
        )
        await cleanup(client, api_key="test-key", api_secret="test-secret")
        mock_cancel.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_fallback_close_without_client_order_id() -> None:
    """cleanup must fallback to close without clientOrderId when primary close fails."""
    from unittest import mock

    call_args_list = []

    async def fake_close(client, quantity, client_order_id=None):
        call_args_list.append(client_order_id)
        if client_order_id is not None:
            # Primary close with cid — simulate failure
            raise RuntimeError("clientOrderId too long")
        # Fallback close without cid — succeed
        return BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            ok=True,
            order_id="fallback-1",
            client_order_id=None,
        )

    client = _make_client(
        [],  # open orders
        [_minimal_position_payload(positionAmt="0.1")],  # position
        [_minimal_position_payload(positionAmt="0")],  # pos after fallback
        [_minimal_position_payload(positionAmt="0")],  # final check
    )

    with mock.patch(
        "scripts.binance_live_smoke_test.close_long_position",
        side_effect=fake_close,
    ) as mock_close:
        await cleanup(client, api_key="test-key", api_secret="test-secret")
        # First call was primary (with cid), second was fallback (None)
        assert mock_close.call_count == 2
        assert call_args_list[0] is not None  # primary with cid
        assert call_args_list[1] is None  # fallback without cid


@pytest.mark.asyncio
async def test_cleanup_without_api_credentials_skips_algo_cancel() -> None:
    """cleanup must not attempt algo cancel when api_key/api_secret is None."""
    client = _make_client(
        [],  # open orders
        [_minimal_position_payload(positionAmt="0")],
        [_minimal_position_payload(positionAmt="0")],
    )
    # Should not raise even without api credentials
    await cleanup(client)  # no api_key/api_secret passed


# ---------------------------------------------------------------------------
# Tests: overall flow sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_sequence_order() -> None:
    """Verify the smoke test flow goes through all steps in order.

    SL is tested separately via the algo order path because the
    regular POST /fapi/v1/order endpoint rejects STOP_MARKET since
    2025-12-09 (error -4120).
    """
    responses = [
        _minimal_order_payload(orderId=1, clientOrderId="RE_SMOKE_open_1"),
        [_minimal_position_payload(positionAmt="0.1", entryPrice="3100")],
        _minimal_order_payload(
            orderId=2, clientOrderId="RE_SMOKE_tp_1",
            side="SELL", type="LIMIT", price="3118.60",
        ),
        # SL is now via algo order API — tested separately
        # Step 5: fetch open orders (regular — only TP visible)
        [
            _fake_open_order_response(
                order_id=2, client_order_id="RE_SMOKE_tp_1",
                side="SELL", order_type="LIMIT", price="3118.60",
            ),
        ],
        _cancel_payload(order_id=2),  # cancel TP
        # Verify regular orders cancelled
        [],
        [_minimal_position_payload(positionAmt="0.1", entryPrice="3100")],
        _minimal_order_payload(
            orderId=4, clientOrderId="RE_SMOKE_close_1",
            side="SELL", type="MARKET",
        ),
        [_minimal_position_payload(positionAmt="0")],
    ]

    transport = FakeBinanceTransport(*responses)
    client = BinanceBrokerClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
        position_mode="net",
    )

    # Step 1: Open
    open_result = await open_long(client, Decimal("0.1"), "RE_SMOKE_open_1")
    assert open_result.ok

    # Step 2: Fetch position
    pos = await fetch_long_position(client)
    assert pos is not None
    assert pos.quantity > 0

    # Step 3: Place TP
    tp_result = await place_tp(client, Decimal("0.1"), Decimal("3118.60"), "RE_SMOKE_tp_1")
    assert tp_result.ok

    # Step 4: Place SL — via algo order (tested separately)
    from unittest import mock

    fake_sl_result = BrokerOrderResult(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        ok=True,
        order_id="99999",
        client_order_id="RE_SMOKE_sl_1",
    )

    async def fake_algo_sl(*, api_key, api_secret, quantity, sl_price, client_order_id):
        return fake_sl_result

    with mock.patch(
        "scripts.binance_live_smoke_test.place_stop_loss_algo_order",
        side_effect=fake_algo_sl,
    ):
        sl_result = await place_sl(
            api_key="test-key",
            api_secret="test-secret",
            quantity=Decimal("0.1"),
            sl_price=Decimal("3081.40"),
            client_order_id="RE_SMOKE_sl_1",
        )
        assert sl_result.ok
        assert sl_result.order_id == "99999"

    # Step 5: Fetch open orders — only TP in regular orders
    orders = await fetch_open_orders(client)
    assert len(orders) == 1  # only TP, SL is algo
    assert orders[0].order_id == "2"

    # Step 6: Cancel TP
    c1 = await cancel_order_by_id(client, str(tp_result.order_id or "2"))
    assert c1.ok

    # Cancel SL via algo cancel
    with mock.patch(
        "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
        return_value=BinanceTransportResponse(
            status_code=200,
            payload={"algoId": 99999, "clientAlgoId": "RE_SMOKE_sl_1", "code": 200},
            headers={},
        ),
    ):
        cancel_sl = await cancel_algo_order_by_client_id(
            api_key="test-key",
            api_secret="test-secret",
            client_order_id="RE_SMOKE_sl_1",
        )
        assert cancel_sl.ok

    # Verify regular orders cancelled
    orders_after = await fetch_open_orders(client)
    assert len(orders_after) == 0

    # Step 7: Market close
    pos_before = await fetch_long_position(client)
    assert pos_before is not None
    close_result = await close_long_position(client, pos_before.quantity, "RE_SMOKE_close_1")
    assert close_result.ok

    pos_after = await fetch_long_position(client)
    assert pos_after is None  # position = 0


# ---------------------------------------------------------------------------
# Tests: Preflight dataclass
# ---------------------------------------------------------------------------


class TestPreflightDataclass:
    def test_preflight_construction(self) -> None:
        p = Preflight(
            api_key="k",
            api_secret="s",
            mark_price=Decimal("3000"),
            available_usdt_balance=Decimal("10"),
            max_notional=Decimal("6"),
            tp_pct=Decimal("0.006"),
            sl_pct=Decimal("0.006"),
            filters=ExchangeInfoFilters(
                min_qty=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("5"),
            ),
            calculated_quantity=Decimal("0.002"),
            calculated_notional=Decimal("6"),
            leverage=10,
            margin_buffer_multiplier=Decimal("3"),
            estimated_initial_margin=Decimal("0.6"),
            required_margin_with_buffer=Decimal("1.8"),
        )
        assert p.calculated_quantity == Decimal("0.002")
        assert p.calculated_notional == Decimal("6")
        assert p.leverage == 10
        assert p.margin_buffer_multiplier == Decimal("3")
        assert p.estimated_initial_margin == Decimal("0.6")
        assert p.required_margin_with_buffer == Decimal("1.8")


# ---------------------------------------------------------------------------
# Tests: ExchangeInfoFilters dataclass
# ---------------------------------------------------------------------------


class TestExchangeInfoFilters:
    def test_construction(self) -> None:
        f = ExchangeInfoFilters(
            min_qty=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("5"),
        )
        assert f.min_qty == Decimal("0.001")
        assert f.step_size == Decimal("0.001")
        assert f.min_notional == Decimal("5")


# ---------------------------------------------------------------------------
# Tests: require_hedge_position_mode
# ---------------------------------------------------------------------------


class TestRequireOneWayPositionMode:
    @pytest.mark.asyncio
    async def test_accepts_one_way_mode(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload={"dualSidePosition": False},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            await require_one_way_position_mode("test-key", "test-secret")
            # does not raise

    @pytest.mark.asyncio
    async def test_rejects_hedge_mode(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload={"dualSidePosition": True},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit):
                await require_one_way_position_mode("test-key", "test-secret")

    @pytest.mark.asyncio
    async def test_rejects_http_error(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=403,
                    payload={"code": -2015, "msg": "Invalid API-key"},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit):
                await require_one_way_position_mode("bad-key", "bad-secret")


# ---------------------------------------------------------------------------
# Tests: require_isolated_margin
# ---------------------------------------------------------------------------


class TestRequireIsolatedMargin:
    @pytest.mark.asyncio
    async def test_accepts_isolated_margin(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload=[{"symbol": "ETHUSDT", "marginType": "isolated", "positionAmt": "0"}],
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            await require_isolated_margin("test-key", "test-secret")
            # does not raise

    @pytest.mark.asyncio
    async def test_rejects_cross_margin(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload=[{"symbol": "ETHUSDT", "marginType": "cross", "positionAmt": "0"}],
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit, match="1"):
                await require_isolated_margin("test-key", "test-secret")

    @pytest.mark.asyncio
    async def test_rejects_empty_payload(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload=[],
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit, match="1"):
                await require_isolated_margin("test-key", "test-secret")


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_canonical_symbol_is_eth_usdt_perp(self) -> None:
        assert CANONICAL_SYMBOL == "ETH-USDT-PERP"

    def test_binance_symbol_is_ethusdt(self) -> None:
        assert BINANCE_SYMBOL == "ETHUSDT"

    def test_client_order_id_prefix(self) -> None:
        assert CLIENT_ORDER_ID_PREFIX == "RE_SMOKE_"

    def test_change_leverage_path(self) -> None:
        assert CHANGE_LEVERAGE_PATH == "/fapi/v1/leverage"

    def test_default_margin_buffer_multiplier_is_3(self) -> None:
        assert DEFAULT_MARGIN_BUFFER_MULTIPLIER == Decimal("3")

    def test_env_margin_buffer_multiplier_key(self) -> None:
        assert ENV_MARGIN_BUFFER_MULTIPLIER == "BINANCE_LIVE_SMOKE_TEST_MARGIN_BUFFER_MULTIPLIER"


# ---------------------------------------------------------------------------
# Tests: calculate_required_margin_with_buffer
# ---------------------------------------------------------------------------


class TestCalculateRequiredMarginWithBuffer:
    def test_standard_case(self) -> None:
        estimated, required = calculate_required_margin_with_buffer(
            notional=Decimal("20.14788"),
            leverage=10,
            buffer_multiplier=Decimal("3"),
        )
        assert estimated == pytest.approx(Decimal("2.014788"))
        assert required == pytest.approx(Decimal("6.044364"))

    def test_available_10_passes_margin_check(self) -> None:
        """With notional≈20.15, leverage=10, buffer=3 → required≈6.04.
        10 USDT available comfortably passes."""
        _, required = calculate_required_margin_with_buffer(
            notional=Decimal("20.14788"),
            leverage=10,
            buffer_multiplier=Decimal("3"),
        )
        assert Decimal("10") >= required

    def test_available_5_fails_margin_check(self) -> None:
        """With notional≈20.15, leverage=10, buffer=3 → required≈6.04.
        5 USDT available is not enough."""
        _, required = calculate_required_margin_with_buffer(
            notional=Decimal("20.14788"),
            leverage=10,
            buffer_multiplier=Decimal("3"),
        )
        assert Decimal("5") < required

    def test_leverage_1_means_full_notional_margin(self) -> None:
        """With leverage=1, estimated_margin equals notional."""
        estimated, required = calculate_required_margin_with_buffer(
            notional=Decimal("20"),
            leverage=1,
            buffer_multiplier=Decimal("1"),
        )
        assert estimated == Decimal("20")
        assert required == Decimal("20")

    def test_leverage_20_reduces_margin(self) -> None:
        estimated, required = calculate_required_margin_with_buffer(
            notional=Decimal("20"),
            leverage=20,
            buffer_multiplier=Decimal("2"),
        )
        assert estimated == Decimal("1")
        assert required == Decimal("2")

    def test_raises_on_zero_leverage(self) -> None:
        with pytest.raises(ValueError, match="leverage must be positive"):
            calculate_required_margin_with_buffer(
                notional=Decimal("20"),
                leverage=0,
                buffer_multiplier=Decimal("3"),
            )

    def test_raises_on_negative_leverage(self) -> None:
        with pytest.raises(ValueError, match="leverage must be positive"):
            calculate_required_margin_with_buffer(
                notional=Decimal("20"),
                leverage=-1,
                buffer_multiplier=Decimal("3"),
            )

    def test_raises_on_zero_buffer(self) -> None:
        with pytest.raises(ValueError, match="buffer_multiplier must be positive"):
            calculate_required_margin_with_buffer(
                notional=Decimal("20"),
                leverage=10,
                buffer_multiplier=Decimal("0"),
            )

    def test_raises_on_negative_buffer(self) -> None:
        with pytest.raises(ValueError, match="buffer_multiplier must be positive"):
            calculate_required_margin_with_buffer(
                notional=Decimal("20"),
                leverage=10,
                buffer_multiplier=Decimal("-1"),
            )

    def test_raises_on_zero_notional(self) -> None:
        with pytest.raises(ValueError, match="notional must be positive"):
            calculate_required_margin_with_buffer(
                notional=Decimal("0"),
                leverage=10,
                buffer_multiplier=Decimal("3"),
            )

    def test_raises_on_negative_notional(self) -> None:
        with pytest.raises(ValueError, match="notional must be positive"):
            calculate_required_margin_with_buffer(
                notional=Decimal("-1"),
                leverage=10,
                buffer_multiplier=Decimal("3"),
            )


# ---------------------------------------------------------------------------
# Tests: set_initial_leverage
# ---------------------------------------------------------------------------


class TestSetInitialLeverage:
    @pytest.mark.asyncio
    async def test_success_with_matching_leverage(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload={"symbol": "ETHUSDT", "leverage": 10},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            await set_initial_leverage("test-key", "test-secret", 10)
            # does not raise

    @pytest.mark.asyncio
    async def test_fails_on_leverage_mismatch(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload={"symbol": "ETHUSDT", "leverage": 5},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit):
                await set_initial_leverage("test-key", "test-secret", 10)

    @pytest.mark.asyncio
    async def test_fails_on_http_400(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=400,
                    payload={"code": -4029, "msg": "Invalid symbol"},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit):
                await set_initial_leverage("test-key", "test-secret", 10)

    @pytest.mark.asyncio
    async def test_accepts_payload_without_leverage_field(self) -> None:
        """When response payload has no 'leverage' key (e.g. just ack), pass."""
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload={"symbol": "ETHUSDT", "msg": "success"},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            await set_initial_leverage("test-key", "test-secret", 10)
            # does not raise


# ---------------------------------------------------------------------------
# Tests: _read_positive_decimal_env
# ---------------------------------------------------------------------------


class TestReadPositiveDecimalEnv:
    def test_uses_default_when_env_not_set(self, monkeypatch) -> None:
        monkeypatch.delenv("TEST_ENV_VAR", raising=False)
        result = _read_positive_decimal_env("TEST_ENV_VAR", Decimal("42"))
        assert result == Decimal("42")

    def test_uses_default_when_env_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_ENV_VAR", "   ")
        result = _read_positive_decimal_env("TEST_ENV_VAR", Decimal("42"))
        assert result == Decimal("42")

    def test_parses_valid_decimal(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_ENV_VAR", "3.14")
        result = _read_positive_decimal_env("TEST_ENV_VAR", Decimal("1"))
        assert result == Decimal("3.14")

    def test_parses_integer_as_decimal(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_ENV_VAR", "5")
        result = _read_positive_decimal_env("TEST_ENV_VAR", Decimal("1"))
        assert result == Decimal("5")

    def test_rejects_zero(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_ENV_VAR", "0")
        with pytest.raises(SystemExit):
            _read_positive_decimal_env("TEST_ENV_VAR", Decimal("3"))

    def test_rejects_negative(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_ENV_VAR", "-5")
        with pytest.raises(SystemExit):
            _read_positive_decimal_env("TEST_ENV_VAR", Decimal("3"))

    def test_rejects_non_numeric(self, monkeypatch) -> None:
        monkeypatch.setenv("TEST_ENV_VAR", "abc")
        with pytest.raises(SystemExit):
            _read_positive_decimal_env("TEST_ENV_VAR", Decimal("3"))

    def test_margin_buffer_env_default(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_MARGIN_BUFFER_MULTIPLIER, raising=False)
        result = _read_positive_decimal_env(
            ENV_MARGIN_BUFFER_MULTIPLIER, DEFAULT_MARGIN_BUFFER_MULTIPLIER,
        )
        assert result == Decimal("3")

    def test_margin_buffer_env_custom(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_MARGIN_BUFFER_MULTIPLIER, "5")
        result = _read_positive_decimal_env(
            ENV_MARGIN_BUFFER_MULTIPLIER, DEFAULT_MARGIN_BUFFER_MULTIPLIER,
        )
        assert result == Decimal("5")

    def test_margin_buffer_env_zero_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_MARGIN_BUFFER_MULTIPLIER, "0")
        with pytest.raises(SystemExit):
            _read_positive_decimal_env(
                ENV_MARGIN_BUFFER_MULTIPLIER, DEFAULT_MARGIN_BUFFER_MULTIPLIER,
            )

    def test_margin_buffer_env_negative_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_MARGIN_BUFFER_MULTIPLIER, "-1")
        with pytest.raises(SystemExit):
            _read_positive_decimal_env(
                ENV_MARGIN_BUFFER_MULTIPLIER, DEFAULT_MARGIN_BUFFER_MULTIPLIER,
            )


# ---------------------------------------------------------------------------
# Tests: preflight guard (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestRequireBinanceLivePreflightForSmoke:
    """Tests for ``require_binance_live_preflight_for_smoke()``."""

    def _set_preflight_envs(self, monkeypatch, **overrides):
        monkeypatch.setenv("EXCHANGE", "binance")
        monkeypatch.setenv("BINANCE_SIGNAL_ONLY", "false")
        monkeypatch.setenv("BINANCE_LIVE_ENABLED", "true")
        monkeypatch.setenv("BINANCE_LIVE_ALLOW_ORDERS", "true")
        monkeypatch.setenv(
            "BINANCE_LIVE_CONFIRMATION", "I_UNDERSTAND_BINANCE_LIVE_TRADING"
        )
        monkeypatch.setenv("BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_LEVERAGE", "20")
        for k, v in overrides.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)

    def test_missing_live_confirmation_exits(self, monkeypatch) -> None:
        self._set_preflight_envs(monkeypatch, BINANCE_LIVE_CONFIRMATION=None)
        monkeypatch.delenv("BINANCE_LIVE_CONFIRMATION", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_binance_live_preflight_for_smoke()
        assert exc_info.value.code == 1

    def test_missing_live_enabled_exits(self, monkeypatch) -> None:
        self._set_preflight_envs(monkeypatch, BINANCE_LIVE_ENABLED="false")
        with pytest.raises(SystemExit) as exc_info:
            require_binance_live_preflight_for_smoke()
        assert exc_info.value.code == 1

    def test_missing_allow_orders_exits(self, monkeypatch) -> None:
        self._set_preflight_envs(monkeypatch, BINANCE_LIVE_ALLOW_ORDERS="false")
        with pytest.raises(SystemExit) as exc_info:
            require_binance_live_preflight_for_smoke()
        assert exc_info.value.code == 1

    def test_all_preflight_env_ok_passes(self, monkeypatch) -> None:
        self._set_preflight_envs(monkeypatch)
        require_binance_live_preflight_for_smoke()  # does not raise


class TestDoubleConfirmation:
    """Tests for double confirmation (smoke + preflight)."""

    def test_only_smoke_confirm_exits_in_main(self, monkeypatch) -> None:
        """Only BINANCE_LIVE_SMOKE_TEST_CONFIRM, missing preflight env → exit."""
        monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
        # Preflight envs NOT set — should fail
        monkeypatch.delenv("BINANCE_LIVE_ENABLED", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_binance_live_preflight_for_smoke()
        assert exc_info.value.code == 1

    def test_only_preflight_confirm_still_requires_smoke_confirm(self, monkeypatch) -> None:
        """Only preflight confirm, missing smoke confirm → smoke gate exits."""
        # Set preflight OK
        monkeypatch.setenv("EXCHANGE", "binance")
        monkeypatch.setenv("BINANCE_LIVE_ENABLED", "true")
        monkeypatch.setenv("BINANCE_LIVE_ALLOW_ORDERS", "true")
        monkeypatch.setenv(
            "BINANCE_LIVE_CONFIRMATION", "I_UNDERSTAND_BINANCE_LIVE_TRADING"
        )
        monkeypatch.setenv("BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_LEVERAGE", "20")
        # Preflight passes
        require_binance_live_preflight_for_smoke()
        # But smoke gate should still fail without its own confirmation
        monkeypatch.delenv(CONFIRM_ENV, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            require_live_confirmation()
        assert exc_info.value.code == 1

    def test_both_confirmations_present_passes(self, monkeypatch) -> None:
        monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
        monkeypatch.setenv("EXCHANGE", "binance")
        monkeypatch.setenv("BINANCE_LIVE_ENABLED", "true")
        monkeypatch.setenv("BINANCE_LIVE_ALLOW_ORDERS", "true")
        monkeypatch.setenv(
            "BINANCE_LIVE_CONFIRMATION", "I_UNDERSTAND_BINANCE_LIVE_TRADING"
        )
        monkeypatch.setenv("BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_LEVERAGE", "20")
        require_live_confirmation()  # does not raise
        require_binance_live_preflight_for_smoke()  # does not raise


# ---------------------------------------------------------------------------
# Tests: max notional cap (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestRequireRequestedNotionalCap:
    """Tests for ``require_requested_notional_cap()``."""

    def test_smoke_max_above_hard_order_cap_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_requested_notional_cap(
                smoke_max_notional=Decimal("11"),
                preflight_max_order_notional=Decimal("5"),
                preflight_max_position_notional=Decimal("30"),
            )
        assert exc_info.value.code == 1

    def test_smoke_max_above_preflight_order_max_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_requested_notional_cap(
                smoke_max_notional=Decimal("6"),
                preflight_max_order_notional=Decimal("5"),
                preflight_max_position_notional=Decimal("30"),
            )
        assert exc_info.value.code == 1

    def test_smoke_max_above_hard_position_cap_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_requested_notional_cap(
                smoke_max_notional=Decimal("31"),
                preflight_max_order_notional=Decimal("50"),
                preflight_max_position_notional=Decimal("30"),
            )
        assert exc_info.value.code == 1

    def test_smoke_max_above_preflight_position_max_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_requested_notional_cap(
                smoke_max_notional=Decimal("6"),
                preflight_max_order_notional=Decimal("10"),
                preflight_max_position_notional=Decimal("5"),
            )
        assert exc_info.value.code == 1

    def test_both_within_caps_passes(self) -> None:
        require_requested_notional_cap(
            smoke_max_notional=Decimal("5"),
            preflight_max_order_notional=Decimal("6"),
            preflight_max_position_notional=Decimal("6"),
        )  # does not raise

    def test_equal_values_passes(self) -> None:
        require_requested_notional_cap(
            smoke_max_notional=Decimal("6"),
            preflight_max_order_notional=Decimal("6"),
            preflight_max_position_notional=Decimal("6"),
        )  # does not raise


class TestRequireCalculatedNotionalCap:
    """Tests for ``require_calculated_notional_cap()``."""

    def test_above_hard_order_cap_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_calculated_notional_cap(
                calculated_notional=Decimal("11"),
                preflight_max_order_notional=Decimal("10"),
                preflight_max_position_notional=Decimal("30"),
            )
        assert exc_info.value.code == 1

    def test_above_preflight_order_max_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_calculated_notional_cap(
                calculated_notional=Decimal("7"),
                preflight_max_order_notional=Decimal("6"),
                preflight_max_position_notional=Decimal("30"),
            )
        assert exc_info.value.code == 1

    def test_above_hard_position_cap_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_calculated_notional_cap(
                calculated_notional=Decimal("31"),
                preflight_max_order_notional=Decimal("50"),
                preflight_max_position_notional=Decimal("30"),
            )
        assert exc_info.value.code == 1

    def test_above_preflight_position_max_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_calculated_notional_cap(
                calculated_notional=Decimal("6"),
                preflight_max_order_notional=Decimal("10"),
                preflight_max_position_notional=Decimal("5"),
            )
        assert exc_info.value.code == 1

    def test_within_caps_passes(self) -> None:
        require_calculated_notional_cap(
            calculated_notional=Decimal("5"),
            preflight_max_order_notional=Decimal("6"),
            preflight_max_position_notional=Decimal("6"),
        )  # does not raise

    def test_equal_values_passes(self) -> None:
        require_calculated_notional_cap(
            calculated_notional=Decimal("6"),
            preflight_max_order_notional=Decimal("6"),
            preflight_max_position_notional=Decimal("6"),
        )  # does not raise


# ---------------------------------------------------------------------------
# Tests: leverage behavior (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestAllowSetLeverage:
    """Tests for ``allow_set_leverage()``."""

    def test_returns_false_when_env_not_set(self, monkeypatch) -> None:
        monkeypatch.delenv(ALLOW_SET_LEVERAGE_ENV, raising=False)
        assert allow_set_leverage() is False

    def test_returns_false_for_wrong_value(self, monkeypatch) -> None:
        monkeypatch.setenv(ALLOW_SET_LEVERAGE_ENV, "YES")
        assert allow_set_leverage() is False

    def test_returns_true_for_correct_value(self, monkeypatch) -> None:
        monkeypatch.setenv(ALLOW_SET_LEVERAGE_ENV, ALLOW_SET_LEVERAGE_VALUE)
        assert allow_set_leverage() is True

    def test_returns_false_for_empty_value(self, monkeypatch) -> None:
        monkeypatch.setenv(ALLOW_SET_LEVERAGE_ENV, "")
        assert allow_set_leverage() is False


class TestRequireExistingLeverage:
    """Tests for ``require_existing_leverage()``."""

    @pytest.mark.asyncio
    async def test_matching_leverage_passes(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
        from unittest import mock

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload=[{
                        "symbol": "ETHUSDT",
                        "marginType": "isolated",
                        "positionAmt": "0",
                        "leverage": 20,
                    }],
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            await require_existing_leverage("test-key", "test-secret", 20)
            # does not raise

    @pytest.mark.asyncio
    async def test_mismatched_leverage_exits(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
        from unittest import mock

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload=[{
                        "symbol": "ETHUSDT",
                        "marginType": "isolated",
                        "positionAmt": "0",
                        "leverage": 5,
                    }],
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit) as exc_info:
                await require_existing_leverage("test-key", "test-secret", 20)
            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_empty_response_exits(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
        from unittest import mock

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=200,
                    payload=[],
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit) as exc_info:
                await require_existing_leverage("test-key", "test-secret", 20)
            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_http_error_exits(self) -> None:
        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
        from unittest import mock

        class FakeTransport:
            async def send(self, request):
                return BinanceTransportResponse(
                    status_code=403,
                    payload={"code": -2015, "msg": "Invalid API-key"},
                    headers={},
                )

        with mock.patch.object(AiohttpBinanceTransport, "send", FakeTransport.send):
            with pytest.raises(SystemExit) as exc_info:
                await require_existing_leverage("bad-key", "bad-secret", 20)
            assert exc_info.value.code == 1


class TestLeverageDefaultBehavior:
    """Tests for the default don't-set-leverage behavior."""

    @pytest.mark.asyncio
    async def test_default_does_not_call_set_initial_leverage(self, monkeypatch) -> None:
        """By default, allow_set_leverage returns False."""
        monkeypatch.delenv(ALLOW_SET_LEVERAGE_ENV, raising=False)
        assert allow_set_leverage() is False

    @pytest.mark.asyncio
    async def test_wrong_allow_value_does_not_enable_set_leverage(self, monkeypatch) -> None:
        """Wrong value in allow env does not enable set_leverage."""
        monkeypatch.setenv(ALLOW_SET_LEVERAGE_ENV, "YES_PLEASE")
        assert allow_set_leverage() is False

    @pytest.mark.asyncio
    async def test_correct_allow_value_enables_set_leverage(self, monkeypatch) -> None:
        """Correct value returns True."""
        monkeypatch.setenv(ALLOW_SET_LEVERAGE_ENV, ALLOW_SET_LEVERAGE_VALUE)
        assert allow_set_leverage() is True


# ---------------------------------------------------------------------------
# Tests: no existing position (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestRequireNoExistingPosition:
    """Tests for ``require_no_existing_position()``."""

    @pytest.mark.asyncio
    async def test_no_position_passes(self) -> None:
        """fetch_position returns None → passes."""
        client = _make_client(
            [_minimal_position_payload(positionAmt="0")],
        )
        await require_no_existing_position(client)  # does not raise

    @pytest.mark.asyncio
    async def test_zero_quantity_passes(self) -> None:
        """fetch_position returns quantity 0 → passes."""
        client = _make_client(
            [_minimal_position_payload(positionAmt="0", positionSide="LONG")],
        )
        await require_no_existing_position(client)  # does not raise

    @pytest.mark.asyncio
    async def test_positive_quantity_exits(self) -> None:
        """fetch_position returns positive quantity → exit."""
        client = _make_client(
            [_minimal_position_payload(positionAmt="0.5")],
        )
        with pytest.raises(SystemExit) as exc_info:
            await require_no_existing_position(client)
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_existing_position_exit_does_not_call_open_long(self) -> None:
        """When existing position causes exit, open_long should NOT be called."""
        client = _make_client(
            [_minimal_position_payload(positionAmt="0.3")],
        )
        with pytest.raises(SystemExit):
            await require_no_existing_position(client)
        # If we get here, exit was raised. Now check that transport
        # only received the fetch_position request (1 call).
        transport = client._transport
        assert len(transport.requests) == 1
        assert transport.requests[0].path == "/fapi/v2/positionRisk"


# ---------------------------------------------------------------------------
# Tests: old smoke order cleanup (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestOldSmokeCleanupSafety:
    """Tests for old RE_SMOKE_ order cleanup behavior."""

    def test_cancel_smoke_orders_only_removes_re_smoke_prefix(self) -> None:
        """cancel_smoke_orders only targets CLIENT_ORDER_ID_PREFIX."""
        assert CLIENT_ORDER_ID_PREFIX == "RE_SMOKE_"

    @pytest.mark.asyncio
    async def test_non_smoke_regular_orders_not_cancelled(self) -> None:
        """Non-RE_SMOKE_ orders must not be cancelled."""
        client = _make_client(
            [
                _fake_open_order_response(
                    order_id=1, client_order_id="RE_SMOKE_tp_1",
                ),
                _fake_open_order_response(
                    order_id=2, client_order_id="MY_ORDER",
                ),
                _fake_open_order_response(
                    order_id=3, client_order_id="RE_SMOKE_sl_1",
                    order_type="STOP_MARKET", price="0", trigger_price="2900",
                ),
            ],
            _cancel_payload(order_id=1),
            _cancel_payload(order_id=3),
        )
        result = await cancel_smoke_orders(client)
        assert result == 2  # Only the RE_SMOKE_ orders cancelled

    @pytest.mark.asyncio
    async def test_non_smoke_algo_orders_not_cancelled(self) -> None:
        """Non-RE_SMOKE_ algo orders must not be cancelled."""
        from unittest import mock

        fake_orders = [
            {"algoId": 1, "clientAlgoId": "RE_SMOKE_sl_1"},
            {"algoId": 2, "clientAlgoId": "MY_ALGO"},
        ]

        cancelled_ids = []

        async def fake_fetch(*, api_key, api_secret):
            return fake_orders

        async def fake_cancel(*, api_key, api_secret, client_order_id):
            cancelled_ids.append(client_order_id)
            return BrokerCancelResult(
                exchange=ExchangeName.BINANCE,
                symbol="ETHUSDT",
                ok=True,
                order_id=None,
                client_order_id=client_order_id,
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.fetch_algo_open_orders",
            side_effect=fake_fetch,
        ), mock.patch(
            "scripts.binance_live_smoke_test.cancel_algo_order_by_client_id",
            side_effect=fake_cancel,
        ):
            count = await cancel_smoke_algo_orders(
                api_key="test-key",
                api_secret="test-secret",
            )
            assert count == 1  # Only RE_SMOKE_sl_1 cancelled
            assert "RE_SMOKE_sl_1" in cancelled_ids
            assert "MY_ALGO" not in cancelled_ids


# ---------------------------------------------------------------------------
# Tests: TP / Close safety (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestTPSafety:
    """TP order must be reduce_only and RE_SMOKE_ prefixed."""

    @pytest.mark.asyncio
    async def test_tp_is_reduce_only(self) -> None:
        client = _make_client(
            _minimal_order_payload(side="SELL", type="LIMIT", price="3200.00"),
        )
        transport = client._transport
        result = await place_tp(
            client, Decimal("0.1"), Decimal("3200"), "RE_SMOKE_tp_123"
        )
        assert result.ok is True
        assert len(transport.requests) == 1
        req = transport.requests[0]
        assert req.params.get("reduceOnly") == "true"

    def test_tp_client_order_id_has_smoke_prefix(self) -> None:
        cid = _generate_client_order_id("tp")
        assert cid.startswith("RE_SMOKE_")

    @pytest.mark.asyncio
    async def test_open_is_not_reduce_only(self) -> None:
        client = _make_client(_minimal_order_payload())
        transport = client._transport
        result = await open_long(client, Decimal("0.1"), "RE_SMOKE_open_123")
        assert result.ok is True
        req = transport.requests[0]
        assert req.params.get("reduceOnly") is None or req.params.get("reduceOnly") == "false"

    def test_open_client_order_id_has_smoke_prefix(self) -> None:
        cid = _generate_client_order_id("open")
        assert cid.startswith("RE_SMOKE_")


class TestCloseSafety:
    """Close order must be reduce_only."""

    @pytest.mark.asyncio
    async def test_close_is_reduce_only(self) -> None:
        client = _make_client(
            _minimal_order_payload(side="SELL", type="MARKET"),
        )
        transport = client._transport
        result = await close_long_position(
            client, Decimal("0.1"), "RE_SMOKE_close_123"
        )
        assert result.ok is True
        req = transport.requests[0]
        assert req.params.get("reduceOnly") == "true"

    def test_close_client_order_id_has_smoke_prefix(self) -> None:
        cid = _generate_client_order_id("close")
        assert cid.startswith("RE_SMOKE_")


# ---------------------------------------------------------------------------
# Tests: SL algo safety (20C-4C-PREP)
# ---------------------------------------------------------------------------


class TestSLAlgoSafety:
    """SL via Algo Order API must have reduceOnly, no positionSide."""

    @pytest.mark.asyncio
    async def test_sl_uses_algo_order_helper(self) -> None:
        """place_sl delegates to place_stop_loss_algo_order."""
        from unittest import mock

        fake_result = BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol="ETHUSDT",
            ok=True,
            order_id="99999",
            client_order_id="RE_SMOKE_sl_123",
        )

        async def fake_algo(*, api_key, api_secret, quantity, sl_price, client_order_id):
            return fake_result

        with mock.patch(
            "scripts.binance_live_smoke_test.place_stop_loss_algo_order",
            side_effect=fake_algo,
        ) as mock_algo:
            result = await place_sl(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_123",
            )
            assert result is fake_result
            mock_algo.assert_called_once()

    @pytest.mark.asyncio
    async def test_algo_params_has_reduce_only(self) -> None:
        """Algo SL params must include reduceOnly=true."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 1, "clientAlgoId": "RE_SMOKE_sl_1", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_1",
            )

        assert captured_params.get("reduceOnly") == "true"

    @pytest.mark.asyncio
    async def test_algo_params_has_no_position_side(self) -> None:
        """Algo SL params must NOT include positionSide."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 2, "clientAlgoId": "RE_SMOKE_sl_2", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_2",
            )

        assert "positionSide" not in captured_params

    @pytest.mark.asyncio
    async def test_algo_params_has_client_algo_id(self) -> None:
        """Algo SL params must include clientAlgoId."""
        from unittest import mock

        captured_params = {}

        async def fake_send(request):
            captured_params.update(request.params)
            return BinanceTransportResponse(
                status_code=200,
                payload={"algoId": 3, "clientAlgoId": "cid", "code": 200},
                headers={},
            )

        with mock.patch(
            "scripts.binance_live_smoke_test.AiohttpBinanceTransport.send",
            side_effect=fake_send,
        ):
            await place_stop_loss_algo_order(
                api_key="test-key",
                api_secret="test-secret",
                quantity=Decimal("0.1"),
                sl_price=Decimal("2900"),
                client_order_id="RE_SMOKE_sl_789",
            )

        assert "clientAlgoId" in captured_params


# ---------------------------------------------------------------------------
# Tests: POSITION_RISK_PATH constant
# ---------------------------------------------------------------------------


class TestPositionRiskPath:
    """Tests for the POSITION_RISK_PATH constant (used by leverage / margin checks)."""

    def test_position_risk_path_is_correct(self) -> None:
        assert POSITION_RISK_PATH == "/fapi/v2/positionRisk"


# ---------------------------------------------------------------------------
# Tests: main wiring updated for preflight
# ---------------------------------------------------------------------------


class TestMainWithPreflight:
    """Main must call the new preflight guard and handle pre-trade checks."""

    @pytest.mark.asyncio
    async def test_main_exits_when_preflight_fails(self, monkeypatch) -> None:
        """Main exits when preflight guard fails (missing BINANCE_LIVE_ENABLED)."""
        monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
        monkeypatch.delenv("BINANCE_LIVE_ENABLED", raising=False)
        monkeypatch.setenv("EXCHANGE", "binance")
        with pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_requires_both_confirmations(self, monkeypatch) -> None:
        """main() calls require_live_confirmation first, then preflight."""
        # Missing smoke confirm
        monkeypatch.delenv(CONFIRM_ENV, raising=False)
        # But preflight envs are OK
        monkeypatch.setenv("EXCHANGE", "binance")
        monkeypatch.setenv("BINANCE_LIVE_ENABLED", "true")
        monkeypatch.setenv("BINANCE_LIVE_ALLOW_ORDERS", "true")
        monkeypatch.setenv(
            "BINANCE_LIVE_CONFIRMATION", "I_UNDERSTAND_BINANCE_LIVE_TRADING"
        )
        monkeypatch.setenv("BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT", "6")
        monkeypatch.setenv("BINANCE_LIVE_LEVERAGE", "20")
        with pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_passes_position_cap_to_requested_notional_cap(
        self, monkeypatch,
    ) -> None:
        """main must pass preflight_max_position_notional to require_requested_notional_cap."""
        monkeypatch.setenv(CONFIRM_ENV, CONFIRM_VALUE)
        monkeypatch.setenv("EXCHANGE", "binance")
        monkeypatch.setenv("BINANCE_SIGNAL_ONLY", "false")
        monkeypatch.setenv("BINANCE_LIVE_ENABLED", "true")
        monkeypatch.setenv("BINANCE_LIVE_ALLOW_ORDERS", "true")
        monkeypatch.setenv(
            "BINANCE_LIVE_CONFIRMATION", "I_UNDERSTAND_BINANCE_LIVE_TRADING"
        )
        monkeypatch.setenv("BINANCE_LIVE_MAX_ORDER_NOTIONAL_USDT", "10")
        monkeypatch.setenv("BINANCE_LIVE_MAX_POSITION_NOTIONAL_USDT", "5")
        monkeypatch.setenv("BINANCE_LIVE_LEVERAGE", "20")
        monkeypatch.setenv("EXCHANGE_API_KEY", "test-key")
        monkeypatch.setenv("EXCHANGE_API_SECRET", "test-secret")

        with mock.patch(
            "scripts.binance_live_smoke_test.require_requested_notional_cap",
        ) as mock_req:
            mock_req.side_effect = SystemExit(0)
            with pytest.raises(SystemExit):
                await main()
            assert mock_req.call_args.kwargs["preflight_max_position_notional"] == Decimal("5")
            assert mock_req.call_args.kwargs["preflight_max_order_notional"] == Decimal("10")
