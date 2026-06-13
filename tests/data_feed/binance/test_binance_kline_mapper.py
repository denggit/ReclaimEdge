from __future__ import annotations

from decimal import Decimal

import pytest

from src.data_feed.binance.kline_mapper import (
    DEFAULT_BINANCE_CANONICAL_SYMBOL,
    map_binance_kline_event,
)
from src.data_feed.market_events import MarketCandleEvent
from src.exchanges.models import ExchangeName


def _kline_payload(**k_overrides):
    kline = {
        "t": 1710000000000,
        "T": 1710000899999,
        "s": "ETHUSDT",
        "i": "15m",
        "f": 100,
        "L": 200,
        "o": "3100.00",
        "c": "3120.00",
        "h": "3130.00",
        "l": "3090.00",
        "v": "123.45",
        "n": 100,
        "x": True,
        "q": "382000.00",
        "V": "60.00",
        "Q": "186000.00",
        "B": "0",
    }
    kline.update(k_overrides)
    return {
        "e": "kline",
        "E": 1710000900000,
        "s": "ETHUSDT",
        "k": kline,
    }


class TestBinanceKlineMapperValid:
    # 1. valid kline payload maps to MarketCandleEvent
    def test_valid_payload_maps_to_market_candle_event(self) -> None:
        payload = _kline_payload()
        event = map_binance_kline_event(payload)
        assert isinstance(event, MarketCandleEvent)

    # 2. exchange == ExchangeName.BINANCE
    def test_exchange_is_binance(self) -> None:
        event = map_binance_kline_event(_kline_payload())
        assert event.exchange == ExchangeName.BINANCE

    # 3. canonical_symbol default == ETH-USDT-PERP
    def test_canonical_symbol_default(self) -> None:
        event = map_binance_kline_event(_kline_payload())
        assert event.canonical_symbol == "ETH-USDT-PERP"
        assert event.canonical_symbol == DEFAULT_BINANCE_CANONICAL_SYMBOL

    # 4. custom canonical_symbol supported
    def test_custom_canonical_symbol(self) -> None:
        event = map_binance_kline_event(
            _kline_payload(), canonical_symbol="BTC-USDT-PERP"
        )
        assert event.canonical_symbol == "BTC-USDT-PERP"

    # 5. raw_symbol from top-level s
    def test_raw_symbol_from_top_level_s(self) -> None:
        payload = _kline_payload()
        payload["s"] = "ETHUSDT"
        payload["k"]["s"] = "BTCUSDT"
        event = map_binance_kline_event(payload)
        assert event.raw_symbol == "ETHUSDT"

    # 6. raw_symbol fallback from k.s
    def test_raw_symbol_fallback_from_k_s(self) -> None:
        payload = _kline_payload()
        del payload["s"]
        payload["k"]["s"] = "BTCUSDT"
        event = map_binance_kline_event(payload)
        assert event.raw_symbol == "BTCUSDT"

    # 7. raw_symbol fallback default ETHUSDT
    def test_raw_symbol_fallback_default(self) -> None:
        payload = _kline_payload()
        del payload["s"]
        del payload["k"]["s"]
        event = map_binance_kline_event(payload)
        assert event.raw_symbol == "ETHUSDT"

    # 8. timeframe from k.i
    def test_timeframe_from_k_i(self) -> None:
        event = map_binance_kline_event(_kline_payload(i="1h"))
        assert event.timeframe == "1h"

    # 9. open_time_ms from k.t
    def test_open_time_ms_from_k_t(self) -> None:
        event = map_binance_kline_event(_kline_payload(t=1710000000000))
        assert event.open_time_ms == 1710000000000

    # 10. close_time_ms from k.T
    def test_close_time_ms_from_k_T(self) -> None:
        event = map_binance_kline_event(_kline_payload(T=1710000899999))
        assert event.close_time_ms == 1710000899999

    # 11. open_price from k.o
    def test_open_price_from_k_o(self) -> None:
        event = map_binance_kline_event(_kline_payload(o="3100.00"))
        assert event.open_price == Decimal("3100.00")

    # 12. high_price from k.h
    def test_high_price_from_k_h(self) -> None:
        event = map_binance_kline_event(_kline_payload(h="3130.00"))
        assert event.high_price == Decimal("3130.00")

    # 13. low_price from k.l
    def test_low_price_from_k_l(self) -> None:
        event = map_binance_kline_event(_kline_payload(l="3090.00"))
        assert event.low_price == Decimal("3090.00")

    # 14. close_price from k.c
    def test_close_price_from_k_c(self) -> None:
        event = map_binance_kline_event(_kline_payload(c="3120.00"))
        assert event.close_price == Decimal("3120.00")

    # 15. volume from k.v
    def test_volume_from_k_v(self) -> None:
        event = map_binance_kline_event(_kline_payload(v="123.45"))
        assert event.volume == Decimal("123.45")

    # 16. is_closed=True from k.x true
    def test_is_closed_true_from_k_x_true(self) -> None:
        event = map_binance_kline_event(_kline_payload(x=True))
        assert event.is_closed is True

    # 17. is_closed=False from k.x false
    def test_is_closed_false_from_k_x_false(self) -> None:
        event = map_binance_kline_event(_kline_payload(x=False))
        assert event.is_closed is False

    # 18. raw payload preserved
    def test_raw_payload_preserved(self) -> None:
        payload = _kline_payload()
        event = map_binance_kline_event(payload)
        assert event.raw == payload
        assert event.raw["e"] == "kline"
        assert event.raw["k"]["o"] == "3100.00"


class TestBinanceKlineMapperMissingFields:
    # 19. missing k raises ValueError
    def test_missing_k_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]
        with pytest.raises(ValueError, match="missing k"):
            map_binance_kline_event(payload)

    # 20. non-dict k raises ValueError
    def test_non_dict_k_raises_value_error(self) -> None:
        payload: dict = {"e": "kline", "k": "not_a_dict"}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="missing k"):
            map_binance_kline_event(payload)

    # 21. missing k.i raises ValueError
    def test_missing_k_i_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["i"]
        with pytest.raises(ValueError, match="missing k.i"):
            map_binance_kline_event(payload)

    # 22. missing k.t raises ValueError
    def test_missing_k_t_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["t"]
        with pytest.raises(ValueError, match="missing k.t"):
            map_binance_kline_event(payload)

    # 23. missing k.T raises ValueError
    def test_missing_k_T_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["T"]
        with pytest.raises(ValueError, match="missing k.T"):
            map_binance_kline_event(payload)

    # 24. missing k.o raises ValueError
    def test_missing_k_o_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["o"]
        with pytest.raises(ValueError, match="missing k.o"):
            map_binance_kline_event(payload)

    # 25. missing k.h raises ValueError
    def test_missing_k_h_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["h"]
        with pytest.raises(ValueError, match="missing k.h"):
            map_binance_kline_event(payload)

    # 26. missing k.l raises ValueError
    def test_missing_k_l_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["l"]
        with pytest.raises(ValueError, match="missing k.l"):
            map_binance_kline_event(payload)

    # 27. missing k.c raises ValueError
    def test_missing_k_c_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["c"]
        with pytest.raises(ValueError, match="missing k.c"):
            map_binance_kline_event(payload)

    # 28. missing k.v raises ValueError
    def test_missing_k_v_raises_value_error(self) -> None:
        payload = _kline_payload()
        del payload["k"]["v"]
        with pytest.raises(ValueError, match="missing k.v"):
            map_binance_kline_event(payload)

    # 29. invalid decimal raises ValueError
    def test_invalid_decimal_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid decimal k.o"):
            map_binance_kline_event(_kline_payload(o="not_a_number"))

    # 30. invalid int raises ValueError
    def test_invalid_int_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid int k.t"):
            map_binance_kline_event(_kline_payload(t="not_a_number"))

    # 31. invalid k.x raises ValueError
    def test_invalid_k_x_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="invalid bool k.x"):
            map_binance_kline_event(_kline_payload(x="not_bool"))

    # 32. close_time_ms < open_time_ms raises ValueError through MarketCandleEvent
    def test_close_time_before_open_time_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="close_time_ms"):
            map_binance_kline_event(_kline_payload(t=1710000900000, T=1710000000000))

    # 33. negative volume raises ValueError through MarketCandleEvent
    def test_negative_volume_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="volume"):
            map_binance_kline_event(_kline_payload(v="-1.0"))


class TestBinanceKlineMapperEdgeCases:
    def test_price_as_integer(self) -> None:
        event = map_binance_kline_event(_kline_payload(o=3100))
        assert event.open_price == Decimal("3100")

    def test_volume_as_integer(self) -> None:
        event = map_binance_kline_event(_kline_payload(v=123))
        assert event.volume == Decimal("123")

    def test_empty_string_decimal_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing k.o"):
            map_binance_kline_event(_kline_payload(o=""))

    def test_none_decimal_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing k.o"):
            map_binance_kline_event(_kline_payload(o=None))

    def test_empty_string_int_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing k.t"):
            map_binance_kline_event(_kline_payload(t=""))

    def test_none_int_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing k.t"):
            map_binance_kline_event(_kline_payload(t=None))
