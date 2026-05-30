from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    BreakoutSignal,
)
from src.notifications.email_notifier import EmailNotifier, EmailNotifierConfig  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


async def main() -> None:
    load_dotenv()

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    email_config = EmailNotifierConfig.from_env()
    notifier = EmailNotifier(email_config)

    async def on_signal(signal: BreakoutSignal) -> None:
        # SMTP is blocking IO. Run it in a thread so the websocket loop is not blocked.
        await asyncio.to_thread(notifier.send_signal, signal)

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        handlers=[on_signal],
    )
    await monitor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
