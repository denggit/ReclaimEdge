#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_okx_semantic_order_body_parity.py
@Description: OKX semantic executor request body parity with legacy order_specs.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.execution import order_specs
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
from src.exchanges.okx.client import OkxBrokerClient
from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
from src.exchanges.semantic_models import BrokerSemanticOrderRole


SYMBOL = "ETH-USDT-SWAP"
TD_MODE = "isolated"
POS_SIDE_MODE = "long_short"


class CapturingTrader:
    symbol = SYMBOL
    td_mode = TD_MODE
    pos_side_mode = POS_SIDE_MODE

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any | None]] = []

    async def request(
        self,
        method: str,
        endpoint: str,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        self.requests.append((method, endpoint, payload))
        if endpoint == "/api/v5/trade/order":
            return {"data": [{"ordId": "order-1"}]}
        if endpoint == "/api/v5/trade/order-algo":
            return {"data": [{"algoId": "algo-1"}]}
        if endpoint == "/api/v5/trade/cancel-order":
            return {"data": [{"ordId": payload["ordId"], "sCode": "0"}]}
        if endpoint == "/api/v5/trade/cancel-algos":
            return {"data": [{"algoId": payload[0]["algoId"], "sCode": "0"}]}
        raise AssertionError(f"unexpected request {method} {endpoint} {payload}")

    @staticmethod
    def decimal_to_str(value: Any) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: Any) -> str:
        return f"{float(price):.2f}"

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        return str(res["data"][0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        item = res["data"][0]
        return str(item.get("algoId") or item.get("ordId"))


def _executor(trader: CapturingTrader) -> OkxBrokerSemanticExecutor:
    client = OkxBrokerClient(trader)
    return OkxBrokerSemanticExecutor(client)


def _last_request(trader: CapturingTrader) -> tuple[str, str, Any | None]:
    assert len(trader.requests) == 1
    return trader.requests[0]


@pytest.mark.asyncio
async def test_open_long_market_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.open_position(
        symbol=SYMBOL,
        side=BrokerPositionSide.LONG,
        quantity=Decimal("12"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    expected_body = order_specs.build_market_entry_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="LONG",
        contracts_text="12",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_open_short_market_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.open_position(
        symbol=SYMBOL,
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("12"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    expected_body = order_specs.build_market_entry_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="SHORT",
        contracts_text="12",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_reduce_only_tp_long_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.place_reduce_only_tp(
        symbol=SYMBOL,
        side=BrokerPositionSide.LONG,
        quantity=Decimal("10"),
        trigger_price=Decimal("3500"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        role=BrokerSemanticOrderRole.TP1,
        client_order_id="tp-client-1",
    )

    expected_body = order_specs.build_reduce_only_tp_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="LONG",
        contracts_text="10",
        price_text="3500.00",
        pos_side_mode=POS_SIDE_MODE,
        client_order_id="tp-client-1",
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_reduce_only_tp_short_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.place_reduce_only_tp(
        symbol=SYMBOL,
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("10"),
        trigger_price=Decimal("3200"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        role=BrokerSemanticOrderRole.TP1,
        client_order_id="tp-short-1",
    )

    expected_body = order_specs.build_reduce_only_tp_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="SHORT",
        contracts_text="10",
        price_text="3200.00",
        pos_side_mode=POS_SIDE_MODE,
        client_order_id="tp-short-1",
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_protective_sl_long_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.place_protective_stop(
        symbol=SYMBOL,
        side=BrokerPositionSide.LONG,
        quantity=Decimal("10"),
        trigger_price=Decimal("3400"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        role=BrokerSemanticOrderRole.PROTECTIVE_SL,
    )

    expected_body = order_specs.build_conditional_protective_sl_algo_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="LONG",
        contracts_text="10",
        stop_price_text="3400.00",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order-algo"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_protective_sl_short_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.place_protective_stop(
        symbol=SYMBOL,
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("10"),
        trigger_price=Decimal("3600"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        role=BrokerSemanticOrderRole.PROTECTIVE_SL,
    )

    expected_body = order_specs.build_conditional_protective_sl_algo_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="SHORT",
        contracts_text="10",
        stop_price_text="3600.00",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order-algo"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_market_exit_long_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.market_exit(
        symbol=SYMBOL,
        side=BrokerPositionSide.LONG,
        quantity=Decimal("5"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    expected_body = order_specs.build_reduce_only_market_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="LONG",
        contracts_text="5",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_market_exit_short_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.market_exit(
        symbol=SYMBOL,
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("5"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    expected_body = order_specs.build_reduce_only_market_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="SHORT",
        contracts_text="5",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_sidecar_entry_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.sidecar_entry(
        symbol=SYMBOL,
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    expected_body = order_specs.build_market_entry_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="LONG",
        contracts_text="2",
        pos_side_mode=POS_SIDE_MODE,
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


@pytest.mark.asyncio
async def test_sidecar_tp_body_parity() -> None:
    trader = CapturingTrader()
    executor = _executor(trader)

    await executor.sidecar_tp(
        symbol=SYMBOL,
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        trigger_price=Decimal("3550"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        client_order_id="sidecar-tp-1",
    )

    expected_body = order_specs.build_reduce_only_tp_order_body(
        inst_id=SYMBOL,
        td_mode=TD_MODE,
        side="LONG",
        contracts_text="2",
        price_text="3550.00",
        pos_side_mode=POS_SIDE_MODE,
        client_order_id="sidecar-tp-1",
    )
    method, endpoint, payload = _last_request(trader)
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload == expected_body


def test_semantic_parity_does_not_import_or_modify_live_modules() -> None:
    root = Path(__file__).resolve().parents[3]
    guarded_files = (
        root / "src/execution/tp_sl_execution_manager.py",
        root / "src/execution/tp_sl_core_tp_manager.py",
        root / "src/execution/tp_sl_protective_stop_manager.py",
        root / "src/execution/tp_sl_market_exit_manager.py",
        root / "src/execution/tp_sl_sidecar_manager.py",
        root / "scripts/run_boll_cvd_live.py",
    )
    forbidden_tokens = (
        "OkxBrokerSemanticExecutor",
        "broker_semantic_executor",
        "BROKER_SEMANTIC_EXECUTION",
    )

    for source_path in guarded_files:
        source = source_path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source, f"{token} unexpectedly found in {source_path}"
