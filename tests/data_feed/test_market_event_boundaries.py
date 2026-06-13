from pathlib import Path


def test_market_events_have_no_live_or_exchange_adapter_dependency() -> None:
    text = Path("src/data_feed/market_events.py").read_text(encoding="utf-8")

    forbidden = [
        "src.live",
        "src.execution",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "src.exchanges.okx",
        "src.exchanges.binance",
        "OKX_CONFIG",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "os.environ",
        "websockets",
        "aiohttp",
        "requests",
        "httpx",
        "wss://",
        "fstream",
        "okx",
        "binance",
    ]

    for token in forbidden:
        assert token not in text, (
            f"{token} should not appear in canonical market event models"
        )
