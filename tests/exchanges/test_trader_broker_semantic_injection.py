#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_trader_broker_semantic_injection.py
@Description: Tests for Trader broker semantic executor lazy injection.
"""

from __future__ import annotations

from pathlib import Path

from src.execution.trader import Trader
from src.exchanges.models import ExchangeName


def test_trader_broker_exchange_name_defaults_to_okx() -> None:
    trader = object.__new__(Trader)

    assert trader.broker_exchange_name == "okx"


def test_trader_broker_semantic_executor_lazy_loads_okx_adapter() -> None:
    trader = object.__new__(Trader)
    trader._broker_client = None
    trader._broker_semantic_executor = None
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "long_short"

    executor = trader.broker_semantic_executor

    assert executor is trader._broker_semantic_executor
    assert trader._broker_client is not None
    assert executor.exchange == ExchangeName.OKX
    assert trader._broker_client.exchange == ExchangeName.OKX
    assert trader.broker_semantic_executor is executor


def test_trader_broker_semantic_executor_imports_are_lazy() -> None:
    text = Path("src/execution/trader.py").read_text()
    top_import_section = text.split("class Trader", 1)[0]

    assert "src.exchanges.okx.client" not in top_import_section
    assert "src.exchanges.okx.semantic_executor" not in top_import_section


def test_execute_intent_does_not_route_to_broker_semantic_executor() -> None:
    text = Path("src/execution/trader.py").read_text()
    execute_intent_block = text.split("async def execute_intent", 1)[1].split(
        "\n    async def ",
        1,
    )[0]

    assert "broker_semantic_executor" not in execute_intent_block
    assert "OkxBrokerClient" not in execute_intent_block
    assert "OkxBrokerSemanticExecutor" not in execute_intent_block
