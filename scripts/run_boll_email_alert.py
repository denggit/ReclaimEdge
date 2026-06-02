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

from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    BreakoutSignal,
)
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)


def format_ts_ms(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def build_alert_email(signal: BreakoutSignal) -> tuple[str, str]:
    is_upper = signal.direction == "BREAK_UPPER"
    direction_emoji = "🚀" if is_upper else "🧊"
    direction_label = "向上穿出上轨" if is_upper else "向下穿出下轨"
    direction_text = (
        "价格从 BOLL 轨道内部向上穿出上轨，当前处于上轨外侧。"
        if is_upper
        else "价格从 BOLL 轨道内部向下穿出下轨，当前处于下轨外侧。"
    )
    action_hint = (
        "这只是第一层预警：后续重点观察上涨是否停滞、主动买量是否衰减、是否出现回归中轨机会。"
        if is_upper
        else "这只是第一层预警：后续重点观察下跌是否停滞、主动卖量是否衰减、是否出现回归中轨机会。"
    )

    upper_dist = signal.upper_distance_pct * 100
    lower_dist = signal.lower_distance_pct * 100
    tick_time = format_ts_ms(signal.tick_ts_ms)
    candle_time = format_ts_ms(signal.candle_ts_ms)

    subject = f"🚨 {direction_emoji} ReclaimEdge BOLL穿轨 | {signal.inst_id} | {direction_label}"
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 720px;">
  <div style="border: 1px solid #eee; border-radius: 12px; overflow: hidden;">
    <div style="background: #111827; color: #fff; padding: 18px 20px;">
      <div style="font-size: 13px; opacity: 0.85;">ReclaimEdge · BOLL Tick Radar</div>
      <div style="font-size: 24px; font-weight: 700; margin-top: 4px;">🚨 BOLL 穿轨预警</div>
      <div style="font-size: 16px; margin-top: 8px;">{direction_emoji} <strong>{signal.inst_id}</strong> · {direction_label}</div>
    </div>

    <div style="padding: 18px 20px;">
      <div style="background: #fff7ed; border-left: 5px solid #f97316; padding: 12px 14px; border-radius: 8px; margin-bottom: 16px;">
        <div style="font-size: 16px; font-weight: 700;">⚠️ 当前状态</div>
        <div style="margin-top: 6px;">{direction_text}</div>
        <div style="margin-top: 6px; color: #555;">{action_hint}</div>
      </div>

      <h3 style="margin: 18px 0 10px;">💰 价格信息</h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">当前价格</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: 700; text-align: right;">{signal.price:.4f}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">上一笔 Tick 价格</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{signal.previous_price:.4f}</td>
        </tr>
      </table>

      <h3 style="margin: 18px 0 10px;">📊 BOLL 轨道</h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">上轨 Upper</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{signal.upper:.4f}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">中轨 Middle</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{signal.middle:.4f}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">下轨 Lower</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{signal.lower:.4f}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">上轨到中轨距离</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{upper_dist:.4f}%</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">下轨到中轨距离</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; font-weight: 700;">{lower_dist:.4f}%</td>
        </tr>
      </table>

      <h3 style="margin: 18px 0 10px;">⏱️ 时间信息</h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">触发 Tick 时间</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{tick_time}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">BOLL 使用的15m K线</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{candle_time}</td>
        </tr>
        <tr>
          <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">报警冻结</td>
          <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{signal.freeze_seconds // 60} 分钟</td>
        </tr>
      </table>

      <div style="margin-top: 18px; padding: 12px 14px; background: #f9fafb; border-radius: 8px; color: #555; font-size: 13px;">
        🧭 原始数据：direction={signal.direction}，tick_ts_ms={signal.tick_ts_ms}，candle_ts_ms={signal.candle_ts_ms}
      </div>

      <div style="margin-top: 16px; color: #888; font-size: 12px;">
        此邮件由 ReclaimEdge 自动发送。当前信号是预警，不是直接开仓指令。
      </div>
    </div>
  </div>
</div>
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
            logger.error("Failed to send BOLL alert email")

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        handlers=[on_signal],
    )
    await monitor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
