from pathlib import Path


def test_binance_kline_mapper_has_no_live_or_websocket_dependency() -> None:
    text = Path("src/data_feed/binance/kline_mapper.py").read_text(encoding="utf-8")

    forbidden = [
        "src.live",
        "src.execution",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "websockets",
        "aiohttp",
        "requests",
        "httpx",
        "wss://",
        "fstream",
        "os.environ",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "CvdTracker",
        "BollBandBreakoutMonitor",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance kline mapper"
