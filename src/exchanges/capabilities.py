from __future__ import annotations

from dataclasses import dataclass

from src.exchanges.models import ExchangeName


@dataclass(frozen=True)
class ExchangeCapabilities:
    exchange: ExchangeName
    supports_hedge_mode: bool
    supports_one_way_mode: bool
    supports_reduce_only: bool
    supports_reduce_only_in_hedge_mode: bool
    supports_algo_orders: bool
    supports_conditional_orders: bool
    supports_close_position_stop: bool
    supports_client_order_id: bool
    market_trade_stream: str
    market_trade_stream_interval_ms: int | None
    requires_position_side_for_hedge_mode: bool


def okx_capabilities() -> ExchangeCapabilities:
    return ExchangeCapabilities(
        exchange=ExchangeName.OKX,
        supports_hedge_mode=True,
        supports_one_way_mode=True,
        supports_reduce_only=True,
        supports_reduce_only_in_hedge_mode=True,
        supports_algo_orders=True,
        supports_conditional_orders=True,
        # Keep this conservative until an OKX adapter codifies close-position stop semantics.
        supports_close_position_stop=False,
        supports_client_order_id=True,
        market_trade_stream="trades",
        market_trade_stream_interval_ms=None,
        requires_position_side_for_hedge_mode=True,
    )


def binance_usdm_capabilities() -> ExchangeCapabilities:
    return ExchangeCapabilities(
        exchange=ExchangeName.BINANCE,
        supports_hedge_mode=True,
        supports_one_way_mode=True,
        supports_reduce_only=True,
        supports_reduce_only_in_hedge_mode=False,
        supports_algo_orders=True,
        supports_conditional_orders=True,
        supports_close_position_stop=True,
        supports_client_order_id=True,
        market_trade_stream="aggTrade",
        market_trade_stream_interval_ms=100,
        requires_position_side_for_hedge_mode=True,
    )
