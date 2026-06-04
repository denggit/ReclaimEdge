from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

import src.execution.trader as trader_module  # noqa: E402
from src.execution.trader import Trader  # noqa: E402
from src.risk.simple_position_sizer import PositionSize  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent  # noqa: E402


def make_intent(**overrides) -> TradeIntent:
    kwargs = dict(
        intent_type="UPDATE_TP",
        side="LONG",
        price=3000.0,
        layer_index=1,
        tp_price=3100.0,
        reason="test",
        size=PositionSize(30.0, 1500.0, 0.5, 1, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=1000,
        avg_entry_price=3000.0,
        breakeven_price=3003.0,
        tp_mode="MIDDLE",
    )
    kwargs.update(overrides)
    return TradeIntent(**kwargs)  # type: ignore[arg-type]


def make_trader(**overrides) -> Trader:
    t = Trader.__new__(Trader)
    t.base_url = "https://www.okx.test"
    t.api_key = "key"
    t.secret_key = "secret"
    t.passphrase = "pass"
    t._session = None
    t._timeout_seconds = 7.0
    t.symbol = "ETH-USDT-SWAP"
    t.td_mode = "isolated"
    t.leverage = "50"
    t.pos_side_mode = "net"
    t.live_trading = True
    t.max_live_equity_usdt = 30.0
    t.contract_multiplier = Decimal("0.1")
    t.contract_precision = Decimal("0.01")
    t.min_contracts = Decimal("0.01")
    t.tp_order_id = None
    t.near_tp_protective_sl_order_id = None
    t.middle_runner_protective_sl_order_id = None
    t.three_stage_post_tp1_protective_sl_order_id = None
    t.trend_runner_sl_order_id = None
    t.position_contracts = Decimal("0")
    t.account_equity_usdt = 0.0
    t._protected_reduce_only_order_ids = set()
    t._managed_reduce_only_order_ids = set()
    t._allow_cancel_unmanaged_reduce_only = True
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


class TraderManagedCoreContractsTest(unittest.TestCase):
    """Tests for Trader._managed_core_contracts_from_intent."""

    def test_managed_core_contracts_valid_no_attribute_error(self) -> None:
        """replace_take_profit(managed_core_contracts="10") does not raise AttributeError."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="10")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNotNone(result)
        self.assertEqual(result, Decimal("10"))

    def test_managed_core_contracts_none_returns_none(self) -> None:
        """managed_core_contracts=None returns None (old logic path)."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts=None)
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_empty_string_returns_none(self) -> None:
        """managed_core_contracts='' returns None."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_negative_returns_none(self) -> None:
        """managed_core_contracts <= 0 returns None."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="-5")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_zero_returns_none(self) -> None:
        """managed_core_contracts=0 returns None."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="0")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_invalid_string_raises_runtime_error(self) -> None:
        """managed_core_contracts invalid string raises RuntimeError."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="not_a_number")
        with self.assertRaises(RuntimeError) as ctx:
            trader._managed_core_contracts_from_intent(intent)
        self.assertIn("invalid managed_core_contracts", str(ctx.exception))

    def test_managed_core_contracts_below_min_contracts_raises_runtime_error(self) -> None:
        """managed_core_contracts < min_contracts raises RuntimeError."""
        trader = make_trader(min_contracts=Decimal("1"))
        # 0.01 < 1.0 min_contracts
        intent = make_intent(managed_core_contracts="0.001")
        with self.assertRaises(RuntimeError) as ctx:
            trader._managed_core_contracts_from_intent(intent)
        self.assertIn("managed_core_contracts below min_contracts", str(ctx.exception))

    def test_managed_core_contracts_rounded_down(self) -> None:
        """managed_core_contracts is rounded down to contract_precision."""
        trader = make_trader()
        # 10.005 should be rounded down to 10.00
        intent = make_intent(managed_core_contracts="10.005")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertEqual(result, Decimal("10.00"))

    def test_replace_take_profit_uses_managed_core_contracts(self) -> None:
        """When managed_core_contracts is set, replace_take_profit should not fetch OKX position.
        We verify that _managed_core_contracts_from_intent extracts the value correctly,
        which is then used by replace_take_profit to set position_contracts.
        """
        trader = make_trader()
        intent = make_intent(managed_core_contracts="10")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertEqual(result, Decimal("10"))


if __name__ == "__main__":
    unittest.main()
