from __future__ import annotations

import asyncio
import copy
import datetime as dt
import html
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.execution.trader import LiveTradeResult, Trader  # noqa: E402
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)
from src.risk.simple_position_sizer import (  # noqa: E402
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.strategies.boll_cvd_reclaim_strategy import (  # noqa: E402
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    TradeIntent,
)
from src.utils.email_sender import EmailSender  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def live_trading_enabled() -> bool:
    return os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}


def format_ts_ms(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def build_live_success_email(intent: TradeIntent, result: LiveTradeResult) -> tuple[str, str]:
    subject = f"LIVE order executed | ETH-USDT-SWAP | {intent.intent_type} | layer {intent.layer_index}"
    event_time = format_ts_ms(intent.ts_ms)
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>LIVE order executed</h2>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">intent_type</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.intent_type)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">side</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.side)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">layer</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.layer_index}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">contracts</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(result.contracts)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(result.tp_price)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">order_id</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(result.order_id or '')}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_order_id</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(result.tp_order_id or '')}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(intent.reason)}</p>
  <p><strong>Message:</strong> {html.escape(result.message)}</p>
  <p><strong>Event time:</strong> {html.escape(event_time)}</p>
</div>
""".strip()
    return subject, content


def build_live_failure_email(intent: TradeIntent, error: Exception) -> tuple[str, str]:
    subject = f"LIVE order failed | ETH-USDT-SWAP | {intent.intent_type} | layer {intent.layer_index}"
    event_time = format_ts_ms(intent.ts_ms)
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>LIVE order failed</h2>
  <p><strong>Strategy state has been rolled back.</strong></p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">intent_type</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.intent_type)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">side</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.side)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">layer</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.layer_index}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.tp_price:.4f}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(intent.reason)}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
  <p><strong>Event time:</strong> {html.escape(event_time)}</p>
</div>
""".strip()
    return subject, content


async def main() -> None:
    load_dotenv()
    if not live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig(
        fast_window_seconds=float(os.getenv("CVD_FAST_WINDOW_SECONDS", "5")),
        price_stall_seconds=float(os.getenv("PRICE_STALL_SECONDS", "2")),
        price_stall_tolerance_pct=float(os.getenv("PRICE_STALL_TOLERANCE_PCT", "0.0005")),
    )
    sizer = SimplePositionSizer(SimplePositionSizerConfig.from_env())
    strategy = BollCvdReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
    cvd = CvdTracker(cvd_config)
    email_sender = EmailSender()
    trader = Trader()
    await trader.initialize()

    async def on_market_tick(event: MarketTickEvent) -> None:
        if event.boll is None:
            return
        cvd_snapshot = cvd.update(
            side=event.tick.side,
            size=event.tick.size,
            price=event.tick.price,
            ts_ms=event.tick.ts_ms,
        )
        backup_state = copy.deepcopy(strategy.state)
        intents = strategy.on_tick(
            price=event.tick.price,
            ts_ms=event.tick.ts_ms,
            boll=event.boll,
            cvd=cvd_snapshot,
        )
        for intent in intents:
            try:
                result = await trader.execute_intent(intent)
                if not result.ok:
                    raise RuntimeError(result.message)
                logger.warning(
                    "LIVE execution success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s order_id=%s tp_order_id=%s",
                    intent.intent_type,
                    intent.side,
                    intent.layer_index,
                    intent.price,
                    result.contracts,
                    result.tp_price,
                    result.order_id,
                    result.tp_order_id,
                )
                subject, content = build_live_success_email(intent, result)
                ok = await email_sender.send_email_async(subject, content, content_type="html")
                if not ok:
                    logger.error("Failed to send live execution success email")
            except Exception as exc:
                strategy.state = backup_state
                logger.exception("LIVE execution failed; strategy state has been rolled back")
                subject, content = build_live_failure_email(intent, exc)
                ok = await email_sender.send_email_async(subject, content, content_type="html")
                if not ok:
                    logger.error("Failed to send live execution failure email")
                break

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        tick_handlers=[on_market_tick],
    )
    await monitor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
