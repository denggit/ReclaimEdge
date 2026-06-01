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
from src.reporting.daily_trade_reporter import DailyTradeReporter  # noqa: E402
from src.reporting.live_state_store import LiveStateStore  # noqa: E402
from src.reporting.trade_journal import LiveTradeJournal  # noqa: E402
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
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402
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


def parse_daily_report_time(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid DAILY_REPORT_TIME={value}")
    return hour, minute


def next_daily_report_time(hour: int, minute: int) -> dt.datetime:
    now = dt.datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


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


def restore_strategy_from_saved_state(strategy: BollCvdReclaimStrategy, saved_state) -> None:  # type: ignore[no-untyped-def]
    strategy.state = StrategyPositionState(
        side=saved_state.side,
        layers=saved_state.layers,
        last_entry_price=saved_state.last_entry_price,
        tp_price=saved_state.tp_price,
        last_order_ts_ms=saved_state.last_order_ts_ms,
        last_tp_update_ts_ms=saved_state.last_tp_update_ts_ms,
        last_tp_update_candle_ts_ms=saved_state.last_tp_update_candle_ts_ms,
        total_entry_qty=saved_state.total_entry_qty,
        total_entry_notional=saved_state.total_entry_notional,
        avg_entry_price=saved_state.avg_entry_price,
        breakeven_price=saved_state.breakeven_price,
        tp_mode=saved_state.tp_mode,
    )
    logger.warning(
        "Recovered strategy state from local disk | position_id=%s side=%s layers=%s avg_entry=%.4f tp=%s",
        saved_state.position_id,
        saved_state.side,
        saved_state.layers,
        saved_state.avg_entry_price,
        saved_state.tp_price,
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


async def fetch_usdt_cash_balance(trader: Trader) -> float:
    res = await trader.request("GET", "/api/v5/account/balance?ccy=USDT")
    data = res.get("data", [])
    if not data:
        return 0.0
    for item in data[0].get("details", []):
        if item.get("ccy") == "USDT":
            return float(item.get("cashBal") or item.get("availBal") or item.get("availEq") or item.get("eq") or 0.0)
    return float(data[0].get("totalEq") or 0.0)


async def main() -> None:
    load_dotenv()
    if not live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig.from_env()
    email_sender = EmailSender()
    journal = LiveTradeJournal()
    state_store = LiveStateStore()
    reporter = DailyTradeReporter(journal, email_sender)
    trader = Trader()
    await trader.initialize()

    sizer = SimplePositionSizer(SimplePositionSizerConfig.from_account_equity(trader.account_equity_usdt))
    strategy = BollCvdShockReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
    startup_position = await trader.fetch_position_snapshot()
    startup_cash = await fetch_usdt_cash_balance(trader)
    current_position_id: str | None = None
    cash_before_position: float | None = None

    saved_state = state_store.load()
    if startup_position.has_position:
        if saved_state and saved_state.side == startup_position.side and saved_state.layers > 0:
            restore_strategy_from_saved_state(strategy, saved_state)
            current_position_id = saved_state.position_id
            cash_before_position = saved_state.cash_before_position
        else:
            restore_strategy_from_position(strategy, startup_position)
            current_position_id = journal.new_position_id(trader.symbol, startup_position.side or "UNKNOWN")
            cash_before_position = startup_cash
            journal.record_startup_recovery(
                position_id=current_position_id,
                symbol=trader.symbol,
                side=startup_position.side or "UNKNOWN",
                contracts=str(startup_position.contracts),
                eth_qty=startup_position.eth_qty,
                avg_entry=startup_position.avg_entry_price,
                cash=startup_cash,
                equity=trader.account_equity_usdt,
            )
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
    else:
        state_store.clear()

    cvd = CvdTracker(cvd_config)
    trade_lock = asyncio.Lock()
    trading_halted = False
    position_sync_seconds = float(os.getenv("POSITION_SYNC_SECONDS", "5"))
    account_sync_seconds = float(os.getenv("ACCOUNT_SYNC_SECONDS", "60"))
    cash_log_min_delta_usdt = float(os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT", "0.01"))
    market_tick_heartbeat_seconds = float(os.getenv("MARKET_TICK_HEARTBEAT_SECONDS", "10"))
    last_account_sync = 0.0
    last_logged_cash = startup_cash
    last_logged_position_key = position_log_key(startup_position)
    last_market_tick_heartbeat = 0.0

    async def daily_report_loop() -> None:
        raw_time = os.getenv("DAILY_REPORT_TIME", "09:00")
        hour, minute = parse_daily_report_time(raw_time)
        logger.info("Daily trade report loop started | DAILY_REPORT_TIME=%s", raw_time)
        while True:
            target = next_daily_report_time(hour, minute)
            sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
            await asyncio.sleep(sleep_seconds)
            try:
                ok = await reporter.send_last_24h_report()
                if ok:
                    logger.info("Daily trade report sent successfully")
                else:
                    logger.error("Daily trade report failed")
            except Exception:
                logger.exception("Daily trade report loop failed")

    async def account_position_sync_loop() -> None:
        nonlocal trading_halted, last_account_sync, last_logged_cash, last_logged_position_key, current_position_id, cash_before_position
        while True:
            try:
                await asyncio.sleep(position_sync_seconds)
                async with trade_lock:
                    now = asyncio.get_running_loop().time()
                    equity = trader.account_equity_usdt
                    if now - last_account_sync >= account_sync_seconds:
                        equity = await trader.fetch_usdt_equity()
                        trader.account_equity_usdt = equity
                        sizer.update_account_equity(equity)

                        cash = await fetch_usdt_cash_balance(trader)
                        last_account_sync = now
                        if abs(cash - last_logged_cash) >= cash_log_min_delta_usdt:
                            logger.info(
                                "CASH_SYNC_CHANGED | cash=%.4f previous=%.4f equity=%.4f layer_margin_pct=%.4f leverage=%.2f",
                                cash,
                                last_logged_cash,
                                equity,
                                sizer.config.layer_margin_pct,
                                sizer.config.leverage,
                            )
                            last_logged_cash = cash

                    position = await trader.fetch_position_snapshot()
                    current_position_key = position_log_key(position)
                    if not position.has_position and strategy.state.layers > 0:
                        cash_after = await fetch_usdt_cash_balance(trader)
                        equity_after = await trader.fetch_usdt_equity()
                        journal.record_flat(
                            position_id=current_position_id,
                            symbol=trader.symbol,
                            side=strategy.state.side,
                            cash_before_position=cash_before_position,
                            cash_after=cash_after,
                            equity_after=equity_after,
                            reason="OKX position is flat. TP filled or manual close detected.",
                            layers=strategy.state.layers,
                            avg_entry_price=strategy.state.avg_entry_price,
                            last_tp_price=strategy.state.tp_price,
                        )
                        logger.warning("POSITION_SYNC_CHANGED | flat_on_okx=true. Resetting strategy and trader state.")
                        strategy.state = StrategyPositionState()
                        trader.mark_flat()
                        trading_halted = False
                        current_position_id = None
                        cash_before_position = None
                        state_store.clear()
                        last_logged_cash = cash_after
                        last_logged_position_key = current_position_key
                    elif position.has_position:
                        trader.position_contracts = position.contracts
                        sync_strategy_cost_from_position(strategy, position)
                        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
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
        nonlocal trading_halted, current_position_id, cash_before_position, last_market_tick_heartbeat
        if event.boll is None:
            return

        async with trade_lock:
            cvd_snapshot = cvd.update(
                side=event.tick.side,
                size=event.tick.size,
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
            )
            now = asyncio.get_running_loop().time()
            if now - last_market_tick_heartbeat >= market_tick_heartbeat_seconds:
                last_market_tick_heartbeat = now
                logger.info(
                    "MARKET_TICK_HEARTBEAT | price=%.4f tick_ts_ms=%s side=%s size=%.8f boll_lower=%.4f boll_middle=%.4f boll_upper=%.4f switch=%s fast_cvd=%.8f previous_fast_cvd=%.8f buy_ratio=%.4f sell_ratio=%.4f burst_net_move_pct=%.6f move_ratio=%.2f volume_ratio=%.2f burst_range_pct=%.6f baseline_range_pct=%.6f burst_volume=%.8f baseline_volume=%.8f up_burst=%s down_burst=%s",
                    event.tick.price,
                    event.tick.ts_ms,
                    event.tick.side,
                    event.tick.size,
                    event.boll.lower,
                    event.boll.middle,
                    event.boll.upper,
                    event.boll.alert_switch_on,
                    cvd_snapshot.fast_cvd,
                    cvd_snapshot.previous_fast_cvd,
                    cvd_snapshot.buy_ratio,
                    cvd_snapshot.sell_ratio,
                    cvd_snapshot.burst_net_move_pct,
                    cvd_snapshot.burst_move_ratio,
                    cvd_snapshot.burst_volume_ratio,
                    cvd_snapshot.burst_range_pct,
                    cvd_snapshot.baseline_range_pct,
                    cvd_snapshot.burst_volume,
                    cvd_snapshot.baseline_volume,
                    cvd_snapshot.up_burst,
                    cvd_snapshot.down_burst,
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
                    entry_cash_before = cash_before_position
                    if intent.intent_type != "UPDATE_TP" and current_position_id is None:
                        entry_cash_before = await fetch_usdt_cash_balance(trader)

                    result = await trader.execute_intent(intent)
                    if not result.ok:
                        if result.entry_filled:
                            trading_halted = True
                            raise RuntimeError(result.message)
                        strategy.state = backup_state
                        raise RuntimeError(result.message)

                    if intent.intent_type == "UPDATE_TP":
                        journal.record_tp_update(position_id=current_position_id, intent=intent, result=result, equity=trader.account_equity_usdt)
                        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
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
                        if current_position_id is None:
                            current_position_id = journal.new_position_id(trader.symbol, intent.side, intent.ts_ms)
                            cash_before_position = entry_cash_before
                        journal.record_entry(
                            position_id=current_position_id,
                            intent=intent,
                            result=result,
                            cash_before_position=cash_before_position,
                            equity=trader.account_equity_usdt,
                            extra={"symbol": trader.symbol},
                        )
                        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
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

                    try:
                        journal.record_error(position_id=current_position_id, intent=intent, error=exc, rolled_back=rolled_back, halted=trading_halted)
                    except Exception:
                        logger.exception("Failed to write trade journal error event")

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
        daily_report_loop(),
        monitor.run_forever(),
    )


if __name__ == "__main__":
    asyncio.run(main())
