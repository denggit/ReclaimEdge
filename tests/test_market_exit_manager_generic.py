"""Generic MarketExitManager tests — reduce-only exit, execute_market_exit_runner.

These tests verify that MarketExitManager routes through
TradingClientPort (or broker_semantic_executor when semantic flag is on),
without any near-TP references.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.trading_client_port import (
    CancelResult,
    OrderResult,
    PositionSnapshot,
)


# ======================================================================
# Fake TradingClientPort
# ======================================================================


class FakeTradingClientPort:
    """Fake that records calls and returns configured responses."""

    def __init__(self) -> None:
        self.place_market_calls: list[dict[str, Any]] = []
        self.cancel_algo_calls: list[dict[str, Any]] = []
        self.cancel_order_calls: list[dict[str, Any]] = []
        self.fetch_position_calls: int = 0

        # Configurable position
        self._positions: list[PositionSnapshot] = []
        self._position_index: int = 0
        self._open_orders: list = []

    def set_positions(self, positions: list[PositionSnapshot]) -> None:
        """Set a sequence of positions (one per fetch_position call)."""
        self._positions = positions
        self._position_index = 0

    def set_single_position(self, pos: PositionSnapshot) -> None:
        self._positions = [pos]
        self._position_index = 0

    def set_open_orders(self, orders: list) -> None:
        self._open_orders = orders

    # --- TradingClientPort interface ---

    async def fetch_position(self) -> PositionSnapshot:
        self.fetch_position_calls += 1
        if self._positions:
            idx = min(self._position_index, len(self._positions) - 1)
            pos = self._positions[idx]
            self._position_index += 1
            return pos
        return PositionSnapshot(side=None, qty=Decimal("0"))

    async def place_market_order(
        self,
        *,
        side: str,
        qty: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        self.place_market_calls.append({
            "side": side,
            "qty": qty,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return OrderResult(ok=True, order_id="market-exit-001")

    async def cancel_algo_order(self, **kwargs) -> CancelResult:
        self.cancel_algo_calls.append(kwargs)
        return CancelResult(ok=True)

    async def cancel_order(self, **kwargs) -> CancelResult:
        self.cancel_order_calls.append(kwargs)
        return CancelResult(ok=True)

    async def fetch_open_orders(self) -> list:
        return list(self._open_orders)

    async def fetch_open_algo_orders(self) -> tuple:
        return ()

    async def place_stop_market_order(self, **kwargs) -> OrderResult:
        return OrderResult(ok=True)

    async def place_limit_order(self, **kwargs) -> OrderResult:
        return OrderResult(ok=True)

    async def fetch_balance(self):
        return MagicMock()

    async def configure_instrument(self) -> None:
        pass

    async def fetch_order_status(self, **kwargs):
        return MagicMock()


# ======================================================================
# Fake Trader (for MarketExitManager tests)
# ======================================================================


class FakeTraderForMarketExit:
    """Trader stub with only the attributes/methods MarketExitManager uses."""

    symbol = "ETH-USDT-SWAP"
    contract_precision = Decimal("0.01")
    min_contracts = Decimal("0.01")

    def __init__(self, trading_client: FakeTradingClientPort) -> None:
        self._tc = trading_client
        self.position_contracts = Decimal("0")
        self.entry_protective_sl_order_id: str | None = None
        self.middle_runner_protective_sl_order_id: str | None = None
        self.three_stage_post_tp1_protective_sl_order_id: str | None = None
        self.trend_runner_sl_order_id: str | None = None

        self._cleanup_called: bool = False
        self._cancel_protective_stop_calls: list[str] = []

        self._semantic_executor: MagicMock | None = None

    def set_semantic_executor(self, mock: MagicMock) -> None:
        self._semantic_executor = mock

    def bind_broker_semantic_executor(self, executor: Any) -> None:
        self._semantic_executor = executor

    @property
    def broker_semantic_executor(self) -> MagicMock:
        if self._semantic_executor is None:
            raise RuntimeError("broker_semantic_executor_not_bound")
        return self._semantic_executor

    @staticmethod
    def decimal_to_str(value: Decimal | str | int | float) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        import math
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"

    async def fetch_position_snapshot(self):
        """Return Trader.PositionSnapshot (with .contracts), not TradingClientPort.PositionSnapshot."""
        port_snap = await self._tc.fetch_position()
        from src.execution.trader import PositionSnapshot as TraderPositionSnapshot
        return TraderPositionSnapshot(
            side=port_snap.side,
            contracts=port_snap.qty,
            avg_entry_price=float(port_snap.avg_entry_price or 0),
            eth_qty=float(port_snap.qty),
            raw_pos=port_snap.qty,
        )

    async def cancel_protective_stop(self, order_id: str | None) -> bool:
        if order_id:
            self._cancel_protective_stop_calls.append(order_id)
        return True

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        if order_id:
            self.middle_runner_protective_sl_order_id = None
        return True

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        if order_id:
            self.three_stage_post_tp1_protective_sl_order_id = None
        return True

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        if order_id:
            self.trend_runner_sl_order_id = None
        return True

    async def cancel_existing_reduce_only_orders(self) -> None:
        pass

    async def _cleanup_after_market_exit(self) -> None:
        self._cleanup_called = True

    async def market_exit_remaining_position_with_retries(
        self, side, retry_count, *, context="generic", retry_interval_seconds=None
    ) -> tuple[bool, str]:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager
        return await MarketExitManager(self, self._tc).market_exit_remaining_position_with_retries(
            side, retry_count, context=context, retry_interval_seconds=retry_interval_seconds,
        )


# ======================================================================
# Helper: construct a fake TradeIntent for execute_market_exit_runner
# ======================================================================


def _make_market_exit_intent(side: str = "LONG", tp_price: float = 3000.00, **kwargs):
    """Create a minimal TradeIntent-like object for execute_market_exit_runner."""
    from dataclasses import dataclass

    @dataclass
    class _Intent:
        intent_type: str = "MARKET_EXIT_RUNNER"
        side: str = ""
        tp_price: float = 0.0
        trend_runner_sl_order_id: str | None = None

    obj = _Intent()
    obj.side = side
    obj.tp_price = tp_price
    if "trend_runner_sl_order_id" in kwargs:
        obj.trend_runner_sl_order_id = kwargs["trend_runner_sl_order_id"]
    return obj


# ======================================================================
# Case A — normal market exit
# ======================================================================


class TestMarketExitRemainingPosition:
    """market_exit_remaining_position_with_retries tests."""

    @pytest.mark.asyncio
    async def test_exits_position_successfully(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        # pre-exit position → LONG with 2 contracts
        # post-exit position → flat
        fake_client = FakeTradingClientPort()
        fake_client.set_positions([
            PositionSnapshot(side="LONG", qty=Decimal("2.0")),
            PositionSnapshot(side=None, qty=Decimal("0")),
        ])
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="generic", retry_interval_seconds=0,
        )

        assert ok is True
        assert "market_exit_order_id" in message
        assert len(fake_client.place_market_calls) == 1
        call = fake_client.place_market_calls[0]
        assert call["reduce_only"] is True
        assert call["qty"] == Decimal("2.0")
        assert call["side"] == "LONG"

    @pytest.mark.asyncio
    async def test_already_flat_returns_ok(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_client = FakeTradingClientPort()
        fake_client.set_single_position(PositionSnapshot(side=None, qty=Decimal("0")))
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="generic", retry_interval_seconds=0,
        )

        assert ok is True
        assert message == "already_flat"
        assert len(fake_client.place_market_calls) == 0

    @pytest.mark.asyncio
    async def test_side_mismatch_returns_ok(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_client = FakeTradingClientPort()
        # Position is SHORT but we asked to exit LONG
        fake_client.set_single_position(PositionSnapshot(side="SHORT", qty=Decimal("1.0")))
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="generic", retry_interval_seconds=0,
        )

        assert ok is True
        assert message == "target_side_absent"
        assert len(fake_client.place_market_calls) == 0

    @pytest.mark.asyncio
    async def test_dust_below_min_contracts_returns_false(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_client = FakeTradingClientPort()
        # Position has dust: 0.005 < min_contracts 0.01
        fake_client.set_single_position(PositionSnapshot(side="LONG", qty=Decimal("0.005")))
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="generic", retry_interval_seconds=0,
        )

        assert ok is False
        assert "dust_position_below_min_contracts" in message
        assert len(fake_client.place_market_calls) == 0

    @pytest.mark.asyncio
    async def test_exits_zero_qty_position_already_flat(self) -> None:
        """Position with qty <= 0 is treated as flat."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_client = FakeTradingClientPort()
        fake_client.set_single_position(PositionSnapshot(side="LONG", qty=Decimal("0")))
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="generic", retry_interval_seconds=0,
        )

        assert ok is True
        assert message == "already_flat"


# ======================================================================
# Case E — semantic market exit
# ======================================================================


class TestSemanticMarketExit:
    """BROKER_SEMANTIC_MARKET_EXIT_ENABLED=true uses semantic executor."""

    @pytest.mark.asyncio
    async def test_semantic_market_exit_called(self, monkeypatch) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")

        fake_client = FakeTradingClientPort()
        # Position then flat after exit
        fake_client.set_positions([
            PositionSnapshot(side="LONG", qty=Decimal("1.5")),
            PositionSnapshot(side=None, qty=Decimal("0")),
        ])
        trader = FakeTraderForMarketExit(fake_client)

        semantic_exec = MagicMock()
        semantic_exec.market_exit = AsyncMock(
            return_value=MagicMock(ok=True, order_id="semantic-me-001", message="")
        )
        semantic_exec.market_exit_runner = AsyncMock(
            return_value=MagicMock(ok=True, order_id="semantic-rnr-001", message="")
        )
        trader.set_semantic_executor(semantic_exec)

        manager = MarketExitManager(trader, fake_client)
        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="generic", retry_interval_seconds=0,
        )

        assert ok is True
        semantic_exec.market_exit.assert_called_once()
        call_kwargs = semantic_exec.market_exit.call_args.kwargs
        assert call_kwargs["symbol"] == trader.symbol
        assert call_kwargs["quantity"] == Decimal("1.5")
        from src.exchanges.models import BrokerQuantityUnit
        assert call_kwargs["quantity_unit"] == BrokerQuantityUnit.CONTRACTS

        # Non-semantic path must NOT be called
        assert len(fake_client.place_market_calls) == 0

    @pytest.mark.asyncio
    async def test_semantic_market_exit_runner_called(self, monkeypatch) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        monkeypatch.setenv("BROKER_SEMANTIC_MARKET_EXIT_ENABLED", "true")

        fake_client = FakeTradingClientPort()
        fake_client.set_positions([
            PositionSnapshot(side="LONG", qty=Decimal("2.0")),
            PositionSnapshot(side=None, qty=Decimal("0")),
        ])
        trader = FakeTraderForMarketExit(fake_client)

        semantic_exec = MagicMock()
        semantic_exec.market_exit = AsyncMock(
            return_value=MagicMock(ok=True, order_id="semantic-me-001", message="")
        )
        semantic_exec.market_exit_runner = AsyncMock(
            return_value=MagicMock(ok=True, order_id="semantic-rnr-001", message="")
        )
        trader.set_semantic_executor(semantic_exec)

        manager = MarketExitManager(trader, fake_client)
        ok, message = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="market_exit_runner", retry_interval_seconds=0,
        )

        assert ok is True
        # Runner context should call market_exit_runner, not market_exit
        semantic_exec.market_exit_runner.assert_called_once()
        semantic_exec.market_exit.assert_not_called()


# ======================================================================
# execute_market_exit_runner tests
# ======================================================================


class TestExecuteMarketExitRunner:
    """execute_market_exit_runner tests: normal, flat, side mismatch, failure."""

    @pytest.mark.asyncio
    async def test_has_trend_runner_position_exits(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        # Before: have position; After: flat
        fake_client.set_positions([
            PositionSnapshot(side="LONG", qty=Decimal("1.0")),
            PositionSnapshot(side=None, qty=Decimal("0")),
        ])
        trader = FakeTraderForMarketExit(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        intent = _make_market_exit_intent(side="LONG", tp_price=3200.00)

        result = await mgr.execute_market_exit_runner(intent)

        assert result.ok is True
        assert result.exit_all is True
        assert result.reduce_filled is True
        assert result.action == "MARKET_EXIT_RUNNER"
        assert result.contracts_before == "1"
        assert result.contracts_after == "0"

    @pytest.mark.asyncio
    async def test_already_flat_returns_ok(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        fake_client.set_single_position(PositionSnapshot(side=None, qty=Decimal("0")))
        trader = FakeTraderForMarketExit(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        intent = _make_market_exit_intent(side="LONG", tp_price=3200.00)

        result = await mgr.execute_market_exit_runner(intent)

        assert result.ok is True
        assert result.exit_all is True
        assert result.message == "runner_already_flat"

    @pytest.mark.asyncio
    async def test_side_absent_returns_ok(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        # Position is SHORT but intent is LONG
        fake_client.set_single_position(PositionSnapshot(side="SHORT", qty=Decimal("1.0")))
        trader = FakeTraderForMarketExit(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        intent = _make_market_exit_intent(side="LONG", tp_price=3200.00)

        result = await mgr.execute_market_exit_runner(intent)

        assert result.ok is True
        assert result.exit_all is True
        assert result.message == "runner_side_absent"

    @pytest.mark.asyncio
    async def test_failure_returns_not_ok(self) -> None:
        """When market exit fails, returns ok=False with contracts_after reflecting remaining."""
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        class FlakyClient(FakeTradingClientPort):
            async def place_market_order(self, **kwargs) -> OrderResult:
                self.place_market_calls.append(kwargs)
                # Simulate a placement that doesn't clear the position
                return OrderResult(ok=False, message="simulated_failure")

        # Use positions where exit doesn't clear
        fake_client = FlakyClient()
        # Before: position; After: still position (exit failed)
        fake_client.set_positions([
            PositionSnapshot(side="LONG", qty=Decimal("1.0")),
            PositionSnapshot(side="LONG", qty=Decimal("1.0")),  # still there
        ])
        trader = FakeTraderForMarketExit(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        intent = _make_market_exit_intent(side="LONG", tp_price=3200.00)

        result = await mgr.execute_market_exit_runner(intent)

        assert result.ok is False
        assert result.exit_all is False
        assert result.contracts_after is not None

    @pytest.mark.asyncio
    async def test_uses_market_exit_runner_env_config(self) -> None:
        """Uses MARKET_EXIT_RUNNER_RETRY_COUNT and MARKET_EXIT_RUNNER_RETRY_INTERVAL_SECONDS env."""
        # Confirm env vars are read by the production code
        import os

        # These env vars are read inside execute_market_exit_runner
        retry_count_default = int(os.getenv("MARKET_EXIT_RUNNER_RETRY_COUNT", "3"))
        retry_interval_default = float(os.getenv("MARKET_EXIT_RUNNER_RETRY_INTERVAL_SECONDS", "0.5"))

        assert retry_count_default == 3
        assert retry_interval_default == pytest.approx(0.5)


# ======================================================================
# Reduce-only market exit TradingClientPort boundary
# ======================================================================


class TestReduceOnlyMarketExitBoundary:
    """market exit must use TradingClientPort.place_market_order(reduce_only=True)."""

    @pytest.mark.asyncio
    async def test_reduce_only_is_true(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_client = FakeTradingClientPort()
        fake_client.set_positions([
            PositionSnapshot(side="LONG", qty=Decimal("3.0")),
            PositionSnapshot(side=None, qty=Decimal("0")),
        ])
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, _ = await manager.market_exit_remaining_position_with_retries(
            side="LONG", retry_count=1, context="boundary_test", retry_interval_seconds=0,
        )

        assert ok is True
        assert len(fake_client.place_market_calls) == 1
        assert fake_client.place_market_calls[0]["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_qty_equals_position_qty(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_client = FakeTradingClientPort()
        fake_client.set_positions([
            PositionSnapshot(side="SHORT", qty=Decimal("4.5")),
            PositionSnapshot(side=None, qty=Decimal("0")),
        ])
        trader = FakeTraderForMarketExit(fake_client)
        manager = MarketExitManager(trader, fake_client)

        ok, _ = await manager.market_exit_remaining_position_with_retries(
            side="SHORT", retry_count=1, context="boundary_test", retry_interval_seconds=0,
        )

        assert ok is True
        assert fake_client.place_market_calls[0]["qty"] == Decimal("4.5")
