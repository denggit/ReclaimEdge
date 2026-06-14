from __future__ import annotations

from decimal import Decimal

import pytest
from _pytest.logging import LogCaptureFixture

from src.execution.tp_sl_market_exit_manager import MarketExitManager
from src.execution.trader import PositionSnapshot
from src.execution.trading_client_port import OrderResult


class FakeTradingClient:
    """A fake trading client that records market order calls."""

    def __init__(self):
        self.market_calls: list[dict] = []
        self.next_order_id: str | None = "fake-market-exit-1"
        self._raise_on_place: Exception | None = None

    async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
        if self._raise_on_place:
            raise self._raise_on_place
        self.market_calls.append({
            "side": side,
            "qty": qty,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return OrderResult(
            ok=True,
            order_id=self.next_order_id,
            client_order_id=None,
            raw={"fake": True},
        )


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    contract_multiplier = Decimal("0.1")
    contract_precision = Decimal("0.01")
    min_contracts = Decimal("0.01")
    position_contracts = Decimal("0")
    near_tp_protective_sl_order_id: str | None = None
    middle_runner_protective_sl_order_id: str | None = None
    three_stage_post_tp1_protective_sl_order_id: str | None = None
    trend_runner_sl_order_id: str | None = None
    _protected_reduce_only_order_ids: set = set()
    _managed_reduce_only_order_ids: set = set()
    _allow_cancel_unmanaged_reduce_only = True

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []
        self.cancelled_protective_stops: list[str] = []
        self.cancelled_reduce_only: list[str] = []
        self._position_was_flat = False
        self._position_side_wrong = False
        self._dust_after_order = False
        self._not_flat_after_order = False
        self._request_fails: Exception | None = None
        self._position_snapshot = PositionSnapshot("LONG", Decimal("1"), 3000.0, 0.1, Decimal("1"))
        self.trading_client = FakeTradingClient()

    def decimal_to_str(self, value):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        if self._position_was_flat:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))
        if self._position_side_wrong:
            return PositionSnapshot("SHORT", Decimal("1"), 3000.0, 0.1, Decimal("1"))
        return self._position_snapshot

    async def request(self, method, endpoint, payload=None):
        if self._request_fails:
            raise self._request_fails
        self.requests.append((method, endpoint, payload or {}))
        return {"code": "0", "data": [{"ordId": "test-order-123"}]}

    def extract_order_id(self, res):
        return str(res["data"][0]["ordId"])

    def _reduce_only_market_order_body(self, side, contracts):
        return {"side": side, "contracts": str(contracts)}

    async def cancel_existing_reduce_only_orders(self):
        self.cancelled_reduce_only.append("all")

    async def cancel_near_tp_protective_stop(self, order_id):
        self.cancelled_protective_stops.append(order_id)
        return True

    async def cancel_middle_runner_protective_stop(self, order_id):
        self.cancelled_protective_stops.append(order_id)
        return True

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id):
        self.cancelled_protective_stops.append(order_id)
        return True

    async def cancel_trend_runner_protective_stop(self, order_id):
        self.cancelled_protective_stops.append(order_id)
        return True

    async def _cleanup_after_market_exit(self):
        await self.cancel_existing_reduce_only_orders()

    async def _cleanup_after_near_tp_market_exit(self):
        await self._cleanup_after_market_exit()


@pytest.mark.asyncio
async def test_market_exit_context_in_logs(caplog: LogCaptureFixture) -> None:
    """market_exit logs should contain context and MARKET_EXIT_* prefix, not NEAR_TP_MARKET_EXIT_*."""
    trader = FakeTrader()
    trader._position_was_flat = True
    mgr = MarketExitManager(trader, trader.trading_client)

    with caplog.at_level("WARNING"):
        ok, msg = await mgr.market_exit_remaining_position_with_retries(
            "LONG", 3, context="sidecar_tp_place_failed",
        )

    assert ok is True
    assert msg == "already_flat"

    log_text = caplog.text
    assert "NEAR_TP_MARKET_EXIT" not in log_text, f"Found NEAR_TP_MARKET_EXIT in logs: {log_text}"
    assert "MARKET_EXIT_SUCCESS" in log_text
    assert "context=sidecar_tp_place_failed" in log_text
    assert "side=LONG" in log_text


@pytest.mark.asyncio
async def test_market_exit_with_retry_interval(caplog: LogCaptureFixture) -> None:
    """market_exit should accept retry_interval_seconds and delay between retries."""
    trader = FakeTrader()
    trader._not_flat_after_order = True
    mgr = MarketExitManager(trader, trader.trading_client)

    with caplog.at_level("ERROR"):
        ok, msg = await mgr.market_exit_remaining_position_with_retries(
            "LONG", 2, context="middle_bucket_fast_sl_failed", retry_interval_seconds=0.5,
        )

    assert ok is False
    log_text = caplog.text
    assert "NEAR_TP_MARKET_EXIT" not in log_text
    assert "MARKET_EXIT_FAILED" in log_text
    assert "context=middle_bucket_fast_sl_failed" in log_text
    assert "attempt=1/2" in log_text or "attempt=2/2" in log_text
    # After migration, market exit routes through trading_client
    assert len(trader.trading_client.market_calls) >= 1
    call = trader.trading_client.market_calls[0]
    assert call["reduce_only"] is True
    assert call["side"] == "LONG"


@pytest.mark.asyncio
async def test_market_exit_default_context() -> None:
    """market_exit with no context defaults to 'generic'."""
    trader = FakeTrader()
    trader._position_was_flat = True
    mgr = MarketExitManager(trader, trader.trading_client)
    ok, msg = await mgr.market_exit_remaining_position_with_retries("LONG", 1)
    assert ok is True


@pytest.mark.asyncio
async def test_cleanup_after_market_exit_renamed() -> None:
    """_cleanup_after_market_exit is the new name; old name still works as alias."""
    trader = FakeTrader()
    mgr = MarketExitManager(trader, trader.trading_client)

    # New name works
    await mgr._cleanup_after_market_exit()
    assert len(trader.cancelled_reduce_only) == 1

    # Old name is backward-compat alias
    await mgr._cleanup_after_near_tp_market_exit()
    assert len(trader.cancelled_reduce_only) == 2


@pytest.mark.asyncio
async def test_market_exit_request_exception_logs_context(caplog: LogCaptureFixture) -> None:
    """When trading_client raises, log should contain context."""
    trader = FakeTrader()
    trader.trading_client._raise_on_place = RuntimeError("50011: Rate limit reached")
    mgr = MarketExitManager(trader, trader.trading_client)

    with caplog.at_level("ERROR"):
        ok, msg = await mgr.market_exit_remaining_position_with_retries(
            "LONG", 2, context="sidecar_tp_place_rate_limited",
        )

    assert ok is False
    log_text = caplog.text
    assert "NEAR_TP_MARKET_EXIT" not in log_text
    assert "MARKET_EXIT_FAILED" in log_text
    assert "context=sidecar_tp_place_rate_limited" in log_text
