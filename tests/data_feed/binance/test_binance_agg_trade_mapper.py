from __future__ import annotations

from decimal import Decimal

import pytest

from src.data_feed.binance.agg_trade_mapper import (
    BINANCE_AGG_TRADE_AGGREGATION_WINDOW_MS,
    DEFAULT_BINANCE_CANONICAL_SYMBOL,
    map_binance_agg_trade_event,
)
from src.data_feed.market_events import MarketTradeEvent, MarketTradeSide
from src.exchanges.models import ExchangeName


def _agg_trade_payload(**overrides):
    payload = {
        "e": "aggTrade",
        "E": 1710000000123,
        "s": "ETHUSDT",
        "a": 5933014,
        "p": "3100.50",
        "q": "1.25",
        "f": 100,
        "l": 105,
        "T": 1710000000111,
        "m": True,
    }
    payload.update(overrides)
    return payload


class TestBinanceAggTradeMapperValid:
    # 1. valid aggTrade payload maps to MarketTradeEvent
    def test_valid_payload_maps_to_market_trade_event(self) -> None:
        payload = _agg_trade_payload()
        event = map_binance_agg_trade_event(payload)
        assert isinstance(event, MarketTradeEvent)

    # 2. exchange == ExchangeName.BINANCE
    def test_exchange_is_binance(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload())
        assert event.exchange == ExchangeName.BINANCE

    # 3. canonical_symbol default == ETH-USDT-PERP
    def test_canonical_symbol_default(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload())
        assert event.canonical_symbol == "ETH-USDT-PERP"
        assert event.canonical_symbol == DEFAULT_BINANCE_CANONICAL_SYMBOL

    # 4. raw_symbol from s == ETHUSDT
    def test_raw_symbol_from_s(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload())
        assert event.raw_symbol == "ETHUSDT"

    # 5. price from p
    def test_price_from_p(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(p="3100.50"))
        assert event.price == Decimal("3100.50")

    # 6. quantity from q
    def test_quantity_from_q(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(q="1.25"))
        assert event.quantity == Decimal("1.25")

    # 7. event_time_ms from E
    def test_event_time_ms_from_E(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(E=1710000000123))
        assert event.event_time_ms == 1710000000123

    # 8. trade_time_ms from T
    def test_trade_time_ms_from_T(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(T=1710000000111))
        assert event.trade_time_ms == 1710000000111

    # 9. trade_id from a
    def test_trade_id_from_a(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(a=5933014))
        assert event.trade_id == "5933014"

    # 10. first_trade_id from f
    def test_first_trade_id_from_f(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(f=100))
        assert event.first_trade_id == "100"

    # 11. last_trade_id from l
    def test_last_trade_id_from_l(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(l=105))
        assert event.last_trade_id == "105"

    # 12. m=true -> MarketTradeSide.SELL
    def test_m_true_maps_to_sell(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(m=True))
        assert event.taker_side == MarketTradeSide.SELL

    # 13. m=false -> MarketTradeSide.BUY
    def test_m_false_maps_to_buy(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(m=False))
        assert event.taker_side == MarketTradeSide.BUY

    # 14. is_aggregated=True
    def test_is_aggregated_true(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload())
        assert event.is_aggregated is True

    # 15. aggregation_window_ms=100
    def test_aggregation_window_ms(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload())
        assert event.aggregation_window_ms == 100
        assert event.aggregation_window_ms == BINANCE_AGG_TRADE_AGGREGATION_WINDOW_MS

    # 16. raw payload preserved
    def test_raw_payload_preserved(self) -> None:
        payload = _agg_trade_payload()
        event = map_binance_agg_trade_event(payload)
        assert event.raw == payload
        assert event.raw["e"] == "aggTrade"
        assert event.raw["p"] == "3100.50"

    # 17. custom canonical_symbol supported
    def test_custom_canonical_symbol(self) -> None:
        event = map_binance_agg_trade_event(
            _agg_trade_payload(), canonical_symbol="BTC-USDT-PERP"
        )
        assert event.canonical_symbol == "BTC-USDT-PERP"


class TestBinanceAggTradeMapperMissingFields:
    # 18. missing p raises ValueError
    def test_missing_p_raises_value_error(self) -> None:
        payload = _agg_trade_payload()
        del payload["p"]
        with pytest.raises(ValueError, match="missing p"):
            map_binance_agg_trade_event(payload)

    # 19. missing q raises ValueError
    def test_missing_q_raises_value_error(self) -> None:
        payload = _agg_trade_payload()
        del payload["q"]
        with pytest.raises(ValueError, match="missing q"):
            map_binance_agg_trade_event(payload)

    # 20. invalid p raises ValueError
    def test_invalid_p_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid decimal p"):
            map_binance_agg_trade_event(_agg_trade_payload(p="not_a_number"))

    # 21. invalid q raises ValueError
    def test_invalid_q_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid decimal q"):
            map_binance_agg_trade_event(_agg_trade_payload(q="not_a_number"))

    # 22. missing E raises ValueError
    def test_missing_E_raises_value_error(self) -> None:
        payload = _agg_trade_payload()
        del payload["E"]
        with pytest.raises(ValueError, match="missing E"):
            map_binance_agg_trade_event(payload)

    # 23. missing T raises ValueError
    def test_missing_T_raises_value_error(self) -> None:
        payload = _agg_trade_payload()
        del payload["T"]
        with pytest.raises(ValueError, match="missing T"):
            map_binance_agg_trade_event(payload)

    # 24. invalid E raises ValueError
    def test_invalid_E_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid int E"):
            map_binance_agg_trade_event(_agg_trade_payload(E="not_a_number"))

    # 25. invalid T raises ValueError
    def test_invalid_T_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid int T"):
            map_binance_agg_trade_event(_agg_trade_payload(T="not_a_number"))

    # 26. invalid m raises ValueError
    def test_invalid_m_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid m="):
            map_binance_agg_trade_event(_agg_trade_payload(m="not_bool"))

    # 27. missing optional ids a/f/l are allowed and become None
    def test_missing_optional_ids_become_none(self) -> None:
        payload = _agg_trade_payload()
        del payload["a"]
        del payload["f"]
        del payload["l"]
        event = map_binance_agg_trade_event(payload)
        assert event.trade_id is None
        assert event.first_trade_id is None
        assert event.last_trade_id is None


class TestBinanceAggTradeMapperPriceInteger:
    """Binance often sends prices as integers in aggTrade (e.g. p=310050)."""

    def test_price_as_integer(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(p=310050))
        assert event.price == Decimal("310050")

    def test_quantity_as_integer(self) -> None:
        event = map_binance_agg_trade_event(_agg_trade_payload(q=125))
        assert event.quantity == Decimal("125")
