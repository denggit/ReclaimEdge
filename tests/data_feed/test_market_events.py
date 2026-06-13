from decimal import Decimal

import pytest

from src.data_feed import (
    MarketCandleEvent,
    MarketEventType,
    MarketTradeEvent,
    MarketTradeSide,
)
from src.data_feed.market_events import (
    require_non_negative_decimal,
    require_positive_decimal,
)
from src.exchanges.models import ExchangeName


class TestMarketTradeSide:
    def test_enum_values(self) -> None:
        assert MarketTradeSide.BUY == "BUY"
        assert MarketTradeSide.SELL == "SELL"
        assert MarketTradeSide.UNKNOWN == "UNKNOWN"


class TestMarketEventType:
    def test_enum_values(self) -> None:
        assert MarketEventType.TRADE == "TRADE"
        assert MarketEventType.CANDLE == "CANDLE"


class TestRequirePositiveDecimal:
    def test_passes_for_positive(self) -> None:
        result = require_positive_decimal(Decimal("1.0"), field_name="price")
        assert result == Decimal("1.0")

    def test_raises_for_zero(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            require_positive_decimal(Decimal("0"), field_name="price")

    def test_raises_for_negative(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            require_positive_decimal(Decimal("-1"), field_name="price")


class TestRequireNonNegativeDecimal:
    def test_passes_for_positive(self) -> None:
        result = require_non_negative_decimal(Decimal("1.0"), field_name="volume")
        assert result == Decimal("1.0")

    def test_passes_for_zero(self) -> None:
        result = require_non_negative_decimal(Decimal("0"), field_name="volume")
        assert result == Decimal("0")

    def test_raises_for_negative(self) -> None:
        with pytest.raises(ValueError, match="volume must be non-negative"):
            require_non_negative_decimal(Decimal("-1"), field_name="volume")


class TestMarketTradeEventNonAggregated:
    def test_can_be_created(self) -> None:
        event = MarketTradeEvent(
            exchange=ExchangeName.BINANCE,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETHUSDT",
            price=Decimal("3000.00"),
            quantity=Decimal("1.5"),
            taker_side=MarketTradeSide.BUY,
            event_time_ms=1718234567890,
        )
        assert event.exchange == ExchangeName.BINANCE
        assert event.canonical_symbol == "ETH-USDT"
        assert event.raw_symbol == "ETHUSDT"
        assert event.price == Decimal("3000.00")
        assert event.quantity == Decimal("1.5")
        assert event.taker_side == MarketTradeSide.BUY
        assert event.event_time_ms == 1718234567890
        assert event.trade_time_ms is None
        assert event.is_aggregated is False
        assert event.aggregation_window_ms is None
        assert event.trade_id is None
        assert event.first_trade_id is None
        assert event.last_trade_id is None
        assert event.raw == {}

    def test_keeps_raw_payload(self) -> None:
        raw = {"e": "trade", "p": "3000.00", "q": "1.5"}
        event = MarketTradeEvent(
            exchange=ExchangeName.OKX,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETH-USDT-SWAP",
            price=Decimal("3000.00"),
            quantity=Decimal("1.5"),
            taker_side=MarketTradeSide.SELL,
            event_time_ms=1718234567890,
            raw=raw,
        )
        assert event.raw is raw
        assert event.raw["e"] == "trade"


class TestMarketTradeEventAggregated:
    def test_requires_aggregation_window_ms(self) -> None:
        with pytest.raises(
            ValueError,
            match="aggregation_window_ms is required for aggregated trade events",
        ):
            MarketTradeEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                price=Decimal("3000.00"),
                quantity=Decimal("2.0"),
                taker_side=MarketTradeSide.BUY,
                event_time_ms=1718234567890,
                is_aggregated=True,
            )

    def test_supports_first_last_trade_id(self) -> None:
        event = MarketTradeEvent(
            exchange=ExchangeName.BINANCE,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETHUSDT",
            price=Decimal("3000.00"),
            quantity=Decimal("5.0"),
            taker_side=MarketTradeSide.BUY,
            event_time_ms=1718234567890,
            is_aggregated=True,
            aggregation_window_ms=100,
            first_trade_id="12345",
            last_trade_id="12350",
        )
        assert event.is_aggregated is True
        assert event.aggregation_window_ms == 100
        assert event.first_trade_id == "12345"
        assert event.last_trade_id == "12350"

    def test_non_aggregated_rejects_aggregation_window_ms(self) -> None:
        with pytest.raises(
            ValueError,
            match="aggregation_window_ms must be None for non-aggregated trade events",
        ):
            MarketTradeEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                price=Decimal("3000.00"),
                quantity=Decimal("1.0"),
                taker_side=MarketTradeSide.BUY,
                event_time_ms=1718234567890,
                is_aggregated=False,
                aggregation_window_ms=100,
            )


class TestMarketTradeEventValidation:
    def test_rejects_price_zero(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            MarketTradeEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                price=Decimal("0"),
                quantity=Decimal("1.0"),
                taker_side=MarketTradeSide.BUY,
                event_time_ms=1718234567890,
            )

    def test_rejects_price_negative(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            MarketTradeEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                price=Decimal("-1"),
                quantity=Decimal("1.0"),
                taker_side=MarketTradeSide.BUY,
                event_time_ms=1718234567890,
            )

    def test_rejects_quantity_zero(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            MarketTradeEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                price=Decimal("3000.00"),
                quantity=Decimal("0"),
                taker_side=MarketTradeSide.BUY,
                event_time_ms=1718234567890,
            )

    def test_rejects_quantity_negative(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            MarketTradeEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                price=Decimal("3000.00"),
                quantity=Decimal("-0.5"),
                taker_side=MarketTradeSide.BUY,
                event_time_ms=1718234567890,
            )


class TestMarketCandleEvent:
    def test_can_be_created(self) -> None:
        event = MarketCandleEvent(
            exchange=ExchangeName.BINANCE,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETHUSDT",
            timeframe="1m",
            open_time_ms=1718234520000,
            close_time_ms=1718234579999,
            open_price=Decimal("3000.00"),
            high_price=Decimal("3010.00"),
            low_price=Decimal("2990.00"),
            close_price=Decimal("3005.00"),
            volume=Decimal("100.5"),
            is_closed=True,
        )
        assert event.exchange == ExchangeName.BINANCE
        assert event.canonical_symbol == "ETH-USDT"
        assert event.raw_symbol == "ETHUSDT"
        assert event.timeframe == "1m"
        assert event.open_time_ms == 1718234520000
        assert event.close_time_ms == 1718234579999
        assert event.open_price == Decimal("3000.00")
        assert event.high_price == Decimal("3010.00")
        assert event.low_price == Decimal("2990.00")
        assert event.close_price == Decimal("3005.00")
        assert event.volume == Decimal("100.5")
        assert event.is_closed is True
        assert event.raw == {}

    def test_supports_is_closed_false(self) -> None:
        event = MarketCandleEvent(
            exchange=ExchangeName.OKX,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETH-USDT-SWAP",
            timeframe="5m",
            open_time_ms=1718234520000,
            close_time_ms=1718234819999,
            open_price=Decimal("3000.00"),
            high_price=Decimal("3010.00"),
            low_price=Decimal("2990.00"),
            close_price=Decimal("3005.00"),
            volume=Decimal("50.0"),
            is_closed=False,
        )
        assert event.is_closed is False

    def test_rejects_negative_volume(self) -> None:
        with pytest.raises(ValueError, match="volume must be non-negative"):
            MarketCandleEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                timeframe="1m",
                open_time_ms=1718234520000,
                close_time_ms=1718234579999,
                open_price=Decimal("3000.00"),
                high_price=Decimal("3010.00"),
                low_price=Decimal("2990.00"),
                close_price=Decimal("3005.00"),
                volume=Decimal("-1"),
                is_closed=True,
            )

    def test_allows_zero_volume(self) -> None:
        event = MarketCandleEvent(
            exchange=ExchangeName.BINANCE,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETHUSDT",
            timeframe="1m",
            open_time_ms=1718234520000,
            close_time_ms=1718234579999,
            open_price=Decimal("3000.00"),
            high_price=Decimal("3000.00"),
            low_price=Decimal("3000.00"),
            close_price=Decimal("3000.00"),
            volume=Decimal("0"),
            is_closed=True,
        )
        assert event.volume == Decimal("0")

    def test_rejects_close_before_open(self) -> None:
        with pytest.raises(
            ValueError,
            match="close_time_ms must be greater than or equal to open_time_ms",
        ):
            MarketCandleEvent(
                exchange=ExchangeName.BINANCE,
                canonical_symbol="ETH-USDT",
                raw_symbol="ETHUSDT",
                timeframe="1m",
                open_time_ms=1718234579999,
                close_time_ms=1718234520000,
                open_price=Decimal("3000.00"),
                high_price=Decimal("3010.00"),
                low_price=Decimal("2990.00"),
                close_price=Decimal("3005.00"),
                volume=Decimal("100.5"),
                is_closed=True,
            )

    def test_allows_equal_open_close_time(self) -> None:
        event = MarketCandleEvent(
            exchange=ExchangeName.BINANCE,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETHUSDT",
            timeframe="1m",
            open_time_ms=1718234520000,
            close_time_ms=1718234520000,
            open_price=Decimal("3000.00"),
            high_price=Decimal("3000.00"),
            low_price=Decimal("3000.00"),
            close_price=Decimal("3000.00"),
            volume=Decimal("0"),
            is_closed=True,
        )
        assert event.open_time_ms == event.close_time_ms

    def test_keeps_raw_payload(self) -> None:
        raw = {"e": "kline", "k": {"o": "3000.00"}}
        event = MarketCandleEvent(
            exchange=ExchangeName.BINANCE,
            canonical_symbol="ETH-USDT",
            raw_symbol="ETHUSDT",
            timeframe="1m",
            open_time_ms=1718234520000,
            close_time_ms=1718234579999,
            open_price=Decimal("3000.00"),
            high_price=Decimal("3010.00"),
            low_price=Decimal("2990.00"),
            close_price=Decimal("3005.00"),
            volume=Decimal("100.5"),
            is_closed=True,
            raw=raw,
        )
        assert event.raw is raw
        assert event.raw["e"] == "kline"


class TestExports:
    def test_all_exports_expected_names(self) -> None:
        from src.data_feed import __all__ as exports

        expected = {
            "MarketCandleEvent",
            "MarketEventType",
            "MarketTradeEvent",
            "MarketTradeSide",
            "require_non_negative_decimal",
            "require_positive_decimal",
        }
        assert set(exports) == expected
