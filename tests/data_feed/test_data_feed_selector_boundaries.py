from __future__ import annotations

from pathlib import Path


def test_data_feed_selector_has_no_live_env_or_websocket_dependency() -> None:
    files = [
        "src/data_feed/base.py",
        "src/data_feed/selector.py",
        "src/data_feed/binance/adapter.py",
        "src/data_feed/okx/adapter.py",
    ]

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
        "os.environ",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "CvdTracker",
        "BollBandBreakoutMonitor",
        "scripts/run_boll_cvd_live",
    ]

    for file_name in files:
        text = Path(file_name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, (
                f"{token} should not appear in {file_name}"
            )
