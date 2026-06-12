from src.exchanges.capabilities import binance_usdm_capabilities, okx_capabilities
from src.exchanges.models import ExchangeName


def test_okx_capabilities_exchange_name():
    assert okx_capabilities().exchange == ExchangeName.OKX


def test_binance_usdm_capabilities_exchange_name():
    assert binance_usdm_capabilities().exchange == ExchangeName.BINANCE


def test_binance_usdm_reduce_only_is_not_supported_in_hedge_mode():
    assert binance_usdm_capabilities().supports_reduce_only_in_hedge_mode is False


def test_binance_usdm_market_trade_stream():
    capabilities = binance_usdm_capabilities()

    assert capabilities.market_trade_stream == "aggTrade"
    assert capabilities.market_trade_stream_interval_ms == 100
