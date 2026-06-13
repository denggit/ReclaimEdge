from pathlib import Path


def test_binance_websocket_feed_not_wired_into_live_or_strategy() -> None:
    text = Path("src/data_feed/binance/websocket_feed.py").read_text(encoding="utf-8")

    forbidden = [
        "src.live",
        "src.execution",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "scripts.run_boll_cvd_live",
        "CvdTracker",
        "BollBandBreakoutMonitor",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "EXCHANGE_API_PASSPHRASE",
        "os.environ",
        "dotenv",
        "private",
        "user data",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in Binance websocket feed"


def test_binance_websocket_feed_not_referenced_by_live_entrypoint_yet() -> None:
    text = Path("scripts/run_boll_cvd_live.py").read_text(encoding="utf-8")
    assert "BinanceWebSocketMarketDataFeed" not in text
    assert "build_binance_combined_market_stream_url" not in text
