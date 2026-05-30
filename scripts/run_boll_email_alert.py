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
from src.utils.email_sender import EmailSender  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def build_alert_email(signal: BreakoutSignal) -> tuple[str, str]:
    direction_text = {
        "BREAK_UPPER": "价格从 BOLL 轨道内部向上穿出上轨",
        "BREAK_LOWER": "价格从 BOLL 轨道内部向下穿出下轨",
    }[signal.direction]

    subject = f"🚨 ReclaimEdge BOLL穿轨预警 | {signal.inst_id} | {signal.direction}"
    content = f"""
<h2>ReclaimEdge BOLL穿轨预警</h2>

<p><strong>合约:</strong> {signal.inst_id}</p>
<p><strong>方向:</strong> {signal.direction}</p>
<p><strong>说明:</strong> {direction_text}</p>

<h3>价格</h3>
<p><strong>当前价格:</strong> {signal.price}</p>
<p><strong>上一笔tick价格:</strong> {signal.previous_price}</p>

<h3>BOLL</h3>
<p><strong>上轨:</strong> {signal.upper}</p>
<p><strong>中轨:</strong> {signal.middle}</p>
<p><strong>下轨:</strong> {signal.lower}</p>
<p><strong>上轨到中轨距离:</strong> {signal.upper_distance_pct * 100:.4f}%</p>
<p><strong>下轨到中轨距离:</strong> {signal.lower_distance_pct * 100:.4f}%</p>

<h3>时间戳</h3>
<p><strong>15m K线时间戳(ms):</strong> {signal.candle_ts_ms}</p>
<p><strong>tick时间戳(ms):</strong> {signal.tick_ts_ms}</p>

<p><strong>报警冻结:</strong> {signal.freeze_seconds} 秒</p>
""".strip()
    return subject, content


async def main() -> None:
    load_dotenv()

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    email_sender = EmailSender()

    async def on_signal(signal: BreakoutSignal) -> None:
        subject, content = build_alert_email(signal)
        ok = await email_sender.send_email_async(subject, content, content_type="html")
        if not ok:
            logging.getLogger(__name__).error("Failed to send BOLL alert email")

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        handlers=[on_signal],
    )
    await monitor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
