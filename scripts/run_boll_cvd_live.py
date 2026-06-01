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

from src.execution.trader import PositionSnapshot, Trader  # noqa: E402
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
    StrategyPositionState,
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


def build_live_failure_email(intent: TradeIntent, error: Exception, rolled_back: bool, halted: bool) -> tuple[str, str]:
    subject = f"LIVE order failed | ETH-USDT-SWAP | {intent.intent_type} | layer {intent.layer_index}"
    event_time = format_ts_ms(intent.ts_ms)
    state_text = "Strategy state has been rolled back." if rolled_back else "Entry may be live. Strategy state was NOT rolled back."
    halt_text = "Trading has been halted. Please check OKX manually." if halted else "Trading is not halted."
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>LIVE order failed</h2>
  <p><strong>{html.escape(state_text)}</strong></p>
  <p><strong>{html.escape(halt_text)}</strong></p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">intent_type</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.intent_type)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">side</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.side)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">layer</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.layer_index}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.tp_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_mode</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.tp_mode)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">avg_entry</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.avg_entry_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">breakeven</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.breakeven_price:.4f}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(intent.reason)}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
  <p><strong>Event time:</strong> {html.escape(event_time)}</p>
</div>
""".strip()
    return subject, content


def restore_strategy_from_position(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    strategy.state = StrategyPositionState(
        side=position.side,
        layers=1,
        last_entry_price=position.avg_entry_price,
        tp_price=None,
        last_order_ts_ms=0,
        last_tp_update_ts_ms=0,
        total_entry_qty=position.eth_qty,
        total_entry_notional=position.avg_entry_price * position.eth_qty,
        avg_entry_price=position.avg_entry_price,
    )
    logger.warning(
        "Recovered existing position into strategy state | side=%s contracts=%s eth_qty=%.6f avg_entry=%.4f",
        position.side,
        position.contracts,
        position.eth_qty,
        position.avg_entry_price,
    )


def sync_strategy_cost_from_position(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    if strategy.state.side is None or strategy.state.side != position.side or strategy.state.layers <= 0:
        restore_strategy_from_position(strategy, position)
        return
    strategy.state.total_entry_qty = position.eth_qty
    strategy.state.total_entry_notional = position.avg_entry_price * position.eth_qty
    strategy.state.avg_entry_price = position.avg_entry_price
    strategy.state.last_entry_price = strategy.state.last_entry_price or position.avg_entry_price


def position_log_key(position: PositionSnapshot) -> tuple[str, str, float]:
    if not position.has_position or position.side is None:
        return ("FLAT", "0", 0.0)
    return (position.side, str(position.contracts), round(position.avg_entry_price, 2))


async def main() -> None:
    load_dotenv()
    if not live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig.from_env()
    email_sender = EmailSender()
    trader = Trader()
    await trader.initialize()

    sizer = SimplePositionSizer(SimplePositionSizerConfig.from_account_equity(trader.account_equity_usdt))
    strategy = BollCvdReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
    startup_position = await trader.fetch_position_snapshot()
    if startup_position.has_position:
        restore_strategy_from_position(strategy, startup_position)

    cvd = CvdTracker(cvd_config)
    trade_lock = asyncio.Lock()
    trading_halted = False
    position_sync_seconds = float(os.getenv("POSITION_SYNC_SECONDS", "5"))
    account_sync_seconds = float(os.getenv("ACCOUNT_SYNC_SECONDS", "60"))
    account_log_min_delta_usdt = float(os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT", "0.01"))
    last_account_sync = 0.0
    last_logged_equity = trader.account_equity_usdt
    last_logged_position_key = position_log_key(startup_position)

    async def account_position_sync_loop() -> None:
        nonlocal trading_halted, last_account_sync, last_logged_equity, last_logged_position_key
        while True:
            try:
                await asyncio.sleep(position_sync_seconds)
                async with trade_lock:
                    now = asyncio.get_running_loop().time()
                    if now - last_account_sync >= account_sync_seconds:
                        equity = await trader.fetch_usdt_equity()
                        trader.account_equity_usdt = equity
                        sizer.update_account_equity(equity)
                        last_account_sync = now
                        if abs(equity - last_logged_equity) >= account_log_min_delta_usdt:
                            logger.info(
                                "ACCOUNT_SYNC_CHANGED | equity=%.4f previous=%.4f layer_margin_pct=%.4f leverage=%.2f",
                                equity,
                                last_logged_equity,
                                sizer.config.layer_margin_pct,
                                sizer.config.leverage,
                            )
                            last_logged_equity = equity

                    position = await trader.fetch_position_snapshot()
                    current_position_key = position_log_key(position)
                    if not position.has_position and strategy.state.layers > 0:
                        logger.warning("POSITION_SYNC_CHANGED | flat_on_okx=true. Resetting strategy and trader state.")
                        strategy.state = StrategyPositionState()
                        trader.mark_flat()
                        trading_halted = False
                        last_logged_position_key = current_position_key
                    elif position.has_position:
                        trader.position_contracts = position.contracts
                        sync_strategy_cost_from_position(strategy, position)
                        if current_position_key != last_logged_position_key:
                            logger.info(
                                "POSITION_SYNC_CHANGED | side=%s contracts=%s avg_entry=%.4f eth_qty=%.6f strategy_layers=%s",
                                position.side,
                                position.contracts,
                                position.avg_entry_price,
                                position.eth_qty,
                                strategy.state.layers,
                            )
                            last_logged_position_key = current_position_key
            except Exception:
                logger.exception("Account/position sync loop failed")

    async def on_market_tick(event: MarketTickEvent) -> None:
        nonlocal trading_halted
        if event.boll is None:
            return

        async with trade_lock:
            cvd_snapshot = cvd.update(
                side=event.tick.side,
                size=event.tick.size,
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
            )
            if trading_halted:
                return

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
                        if result.entry_filled:
                            trading_halted = True
                            raise RuntimeError(result.message)
                        strategy.state = backup_state
                        raise RuntimeError(result.message)

                    if intent.intent_type == "UPDATE_TP":
                        logger.warning(
                            "LIVE TP update success | side=%s layer=%s price=%.4f contracts=%s new_tp_price=%s tp_mode=%s avg_entry=%.4f breakeven=%.4f tp_order_id=%s",
                            intent.side,
                            intent.layer_index,
                            intent.price,
                            result.contracts,
                            result.tp_price,
                            intent.tp_mode,
                            intent.avg_entry_price,
                            intent.breakeven_price,
                            result.tp_order_id,
                        )
                    else:
                        logger.warning(
                            "LIVE entry success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s",
                            intent.intent_type,
                            intent.side,
                            intent.layer_index,
                            intent.price,
                            result.contracts,
                            result.tp_price,
                            intent.tp_mode,
                            intent.avg_entry_price,
                            intent.breakeven_price,
                            result.order_id,
                            result.tp_order_id,
                        )
                except Exception as exc:
                    try:
                        position = await trader.fetch_position_snapshot()
                        contracts = position.contracts
                    except Exception:
                        contracts = trader.position_contracts

                    entry_may_be_live = contracts > 0
                    rolled_back = False
                    if entry_may_be_live:
                        trading_halted = True
                        trader.position_contracts = contracts
                        logger.exception("LIVE execution failed after/possibly after entry. Trading halted; strategy state NOT rolled back.")
                    else:
                        strategy.state = backup_state
                        rolled_back = True
                        logger.exception("LIVE execution failed before entry; strategy state has been rolled back")

                    subject, content = build_live_failure_email(intent, exc, rolled_back=rolled_back, halted=trading_halted)
                    ok = await email_sender.send_email_async(subject, content, content_type="html")
                    if not ok:
                        logger.error("Failed to send live execution failure email")
                    break

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        tick_handlers=[on_market_tick],
    )
    await asyncio.gather(
        account_position_sync_loop(),
        monitor.run_forever(),
    )


if __name__ == "__main__":
    asyncio.run(main())
