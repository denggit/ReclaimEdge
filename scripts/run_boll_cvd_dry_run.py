from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

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
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)


def format_ts_ms(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def build_trade_intent_email(intent: TradeIntent) -> tuple[str, str]:
    is_long = intent.side == "LONG"
    side_emoji = "🟢" if is_long else "🔴"
    side_text = "开多/补多" if is_long else "开空/补空"
    action_map = {
        "OPEN_LONG": "开第一仓多单",
        "ADD_LONG": "补一仓多单",
        "OPEN_SHORT": "开第一仓空单",
        "ADD_SHORT": "补一仓空单",
        "UPDATE_TP": "更新止盈到BOLL中轨",
    }
    action_text = action_map[intent.intent_type]
    subject = f"{side_emoji} DRY-RUN {action_text} | ETH-USDT-SWAP | 第{intent.layer_index}层"
    event_time = format_ts_ms(intent.ts_ms)

    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <div style="border: 1px solid #eee; border-radius: 12px; overflow: hidden;">
    <div style="background: #111827; color: #fff; padding: 18px 20px;">
      <div style="font-size: 13px; opacity: 0.85;">ReclaimEdge · BOLL + CVD Reclaim · DRY-RUN</div>
      <div style="font-size: 24px; font-weight: 700; margin-top: 4px;">{side_emoji} {action_text}</div>
      <div style="font-size: 16px; margin-top: 8px;">ETH-USDT-SWAP · {side_text} · 第 {intent.layer_index} 层</div>
    </div>

    <div style="padding: 18px 20px;">
      <div style="background: #f0fdf4; border-left: 5px solid #22c55e; padding: 12px 14px; border-radius: 8px; margin-bottom: 16px;">
        <div style="font-size: 16px; font-weight: 700;">🧠 触发原因</div>
        <div style="margin-top: 6px;">{intent.reason}</div>
        <div style="margin-top: 6px; color: #555;">这是 dry-run 模拟信号，没有真实下单。</div>
      </div>

      <h3 style="margin: 18px 0 10px;">💰 模拟下单</h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">方向</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{intent.side}</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">触发价格</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{intent.price:.4f}</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">模拟保证金</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.size.margin_usdt:.2f} USDT</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">模拟名义价值</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.size.notional_usdt:.2f} USDT</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">模拟 ETH 数量</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.size.eth_qty:.6f} ETH</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">止盈目标</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{intent.tp_price:.4f}</td></tr>
      </table>

      <h3 style="margin: 18px 0 10px;">📊 BOLL</h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">上轨</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.boll_upper:.4f}</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">中轨</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{intent.boll_middle:.4f}</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">下轨</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.boll_lower:.4f}</td></tr>
      </table>

      <h3 style="margin: 18px 0 10px;">⚡ CVD 快窗口</h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">fast CVD</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{intent.fast_cvd:.4f}</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">previous fast CVD</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.previous_fast_cvd:.4f}</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">buy ratio</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.buy_ratio * 100:.2f}%</td></tr>
        <tr><td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">sell ratio</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.sell_ratio * 100:.2f}%</td></tr>
      </table>

      <div style="margin-top: 18px; padding: 12px 14px; background: #f9fafb; border-radius: 8px; color: #555; font-size: 13px;">
        ⏱️ 触发时间：{event_time}<br>
        🧭 intent={intent.intent_type}, ts_ms={intent.ts_ms}
      </div>

      <div style="margin-top: 16px; color: #888; font-size: 12px;">
        此邮件由 ReclaimEdge 自动发送。当前为 dry-run 模拟信号，不代表真实下单已经发生。
      </div>
    </div>
  </div>
</div>
""".strip()
    return subject, content


async def main() -> None:
    load_dotenv()

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig(
        fast_window_seconds=float(__import__("os").getenv("CVD_FAST_WINDOW_SECONDS", "5")),
        price_stall_seconds=float(__import__("os").getenv("PRICE_STALL_SECONDS", "2")),
        price_stall_tolerance_pct=float(__import__("os").getenv("PRICE_STALL_TOLERANCE_PCT", "0.0005")),
    )
    sizer = SimplePositionSizer(SimplePositionSizerConfig.from_env())
    strategy = BollCvdReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
    cvd = CvdTracker(cvd_config)
    email_sender = EmailSender()

    async def on_market_tick(event: MarketTickEvent) -> None:
        if event.boll is None:
            return
        cvd_snapshot = cvd.update(
            side=event.tick.side,
            size=event.tick.size,
            price=event.tick.price,
            ts_ms=event.tick.ts_ms,
        )
        intents = strategy.on_tick(
            price=event.tick.price,
            ts_ms=event.tick.ts_ms,
            boll=event.boll,
            cvd=cvd_snapshot,
        )
        for intent in intents:
            logger.warning(
                "DRY-RUN intent | type=%s side=%s layer=%s price=%.4f tp=%.4f fast_cvd=%.4f buy_ratio=%.2f sell_ratio=%.2f reason=%s",
                intent.intent_type,
                intent.side,
                intent.layer_index,
                intent.price,
                intent.tp_price,
                intent.fast_cvd,
                intent.buy_ratio,
                intent.sell_ratio,
                intent.reason,
            )
            subject, content = build_trade_intent_email(intent)
            ok = await email_sender.send_email_async(subject, content, content_type="html")
            if not ok:
                logger.error("Failed to send dry-run trade intent email")

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        tick_handlers=[on_market_tick],
    )
    await monitor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
