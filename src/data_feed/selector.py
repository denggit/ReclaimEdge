from __future__ import annotations

from src.data_feed.base import MarketDataFeed
from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.okx.adapter import OkxMarketDataFeed
from src.exchanges.models import ExchangeName


def normalize_exchange_name(exchange: str | ExchangeName | None) -> ExchangeName:
    if exchange is None:
        return ExchangeName.OKX
    if isinstance(exchange, ExchangeName):
        return exchange

    normalized = str(exchange).strip().lower()
    if normalized == ExchangeName.OKX.value:
        return ExchangeName.OKX
    if normalized == ExchangeName.BINANCE.value:
        return ExchangeName.BINANCE

    raise ValueError(f"Unsupported data feed exchange: {exchange!r}")


def build_market_data_feed(
    *,
    exchange: str | ExchangeName | None = None,
    canonical_symbol: str = "ETH-USDT-PERP",
    raw_symbol: str | None = None,
    kline_interval: str = "15m",
) -> MarketDataFeed:
    exchange_name = normalize_exchange_name(exchange)

    if exchange_name == ExchangeName.OKX:
        return OkxMarketDataFeed(
            canonical_symbol=canonical_symbol,
            raw_symbol=raw_symbol or "ETH-USDT-SWAP",
        )

    if exchange_name == ExchangeName.BINANCE:
        return BinanceMarketDataFeed(
            canonical_symbol=canonical_symbol,
            raw_symbol=raw_symbol or "ETHUSDT",
            kline_interval=kline_interval,
        )

    raise ValueError(f"Unsupported data feed exchange: {exchange_name!r}")
