from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.execution.paper_trader import PaperTrader
from src.execution.trader import Trader
from src.execution.trader_types import TraderInstrumentMetadata, TraderMarketSettings
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig
from src.live.heartbeat_writer import HeartbeatWriter, HeartbeatWriterConfig
from src.live.live_app_config import LiveAppConfig, LiveHeartbeatConfig
from src.live.runtime_paths import RuntimePaths
from src.live.runtime_types import TradeCommand
from src.monitors.boll_band_breakout_monitor import (
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)
from src.reporting.daily_trade_reporter import DailyTradeReporter
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.rolling_loss_guard import RollingLossGuard
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender


@dataclass(frozen=True)
class SymbolWorkerPersistence:
    journal: LiveTradeJournal
    state_store: LiveStateStore
    rolling_loss_guard: RollingLossGuard
    reporter: DailyTradeReporter


@dataclass(frozen=True)
class SymbolWorkerStrategyObjects:
    sizer: SimplePositionSizer
    strategy: BollCvdShockReclaimStrategy


@dataclass(frozen=True)
class SymbolWorkerQueues:
    strategy_tick_queue: asyncio.Queue[MarketTickEvent]
    execution_queue: asyncio.Queue[TradeCommand]


class SymbolWorkerFactory:
    """Factory that creates live single-symbol runtime objects.

    This factory is a **pure construction layer**.  It never:
    * starts or initializes a trader
    * runs a monitor forever
    * gathers async tasks
    * loads or saves state store data
    * loads or initializes the rolling loss guard
    * performs legacy runtime file handoff
    * reads environment variables
    * opens, reads, or writes files
    * sends email
    * makes OKX network requests
    """

    def create_email_sender(self) -> EmailSender:
        return EmailSender()

    def create_trader(
        self,
        *,
        trader_mode: str = "live",
        instrument_metadata: TraderInstrumentMetadata | None = None,
        market_settings: TraderMarketSettings | None = None,
    ) -> Trader | PaperTrader:
        if trader_mode == "live":
            return Trader(
                instrument_metadata=instrument_metadata,
                market_settings=market_settings,
            )
        if trader_mode == "paper":
            return self.create_paper_trader_from_env()
        raise RuntimeError(
            f"Invalid RECLAIM_WORKER_MODE: {trader_mode!r}. Must be 'live' or 'paper'."
        )

    def create_paper_trader_from_env(self) -> PaperTrader:
        """Create a PaperTrader for dry-run mode.

        PaperTrader reads its own configuration from environment variables
        (OKX_INST_ID, RECLAIM_PAPER_SYMBOLS, PAPER_ACCOUNT_EQUITY_USDT)
        and validates that only BTC-USDT-SWAP is allowed.
        """
        return PaperTrader()

    def create_runtime_paths(self, *, runtime_dir: str | Path, inst_id: str) -> RuntimePaths:
        return RuntimePaths(runtime_dir=runtime_dir, inst_id=inst_id)

    def create_persistence(
        self,
        *,
        runtime_paths: RuntimePaths,
        email_sender: EmailSender,
    ) -> SymbolWorkerPersistence:
        journal = LiveTradeJournal.from_runtime_paths(runtime_paths)
        state_store = LiveStateStore.from_runtime_paths(runtime_paths)
        rolling_loss_guard = RollingLossGuard.from_runtime_paths(runtime_paths)
        reporter = DailyTradeReporter(journal, email_sender)
        return SymbolWorkerPersistence(
            journal=journal,
            state_store=state_store,
            rolling_loss_guard=rolling_loss_guard,
            reporter=reporter,
        )

    def create_strategy_objects(
        self,
        *,
        strategy_config: BollCvdReclaimStrategyConfig,
        position_sizer_config: SimplePositionSizerConfig,
    ) -> SymbolWorkerStrategyObjects:
        sizer = SimplePositionSizer(position_sizer_config)
        strategy = BollCvdShockReclaimStrategy(strategy_config, sizer)
        return SymbolWorkerStrategyObjects(sizer=sizer, strategy=strategy)

    def create_cvd_tracker(self, config: CvdTrackerConfig) -> CvdTracker:
        return CvdTracker(config)

    def create_queues(self, app_config: LiveAppConfig) -> SymbolWorkerQueues:
        return SymbolWorkerQueues(
            strategy_tick_queue=asyncio.Queue(maxsize=app_config.strategy_tick_queue_maxsize),
            execution_queue=asyncio.Queue(maxsize=app_config.execution_queue_maxsize),
        )

    def create_heartbeat_writer(
        self,
        *,
        runtime_paths: RuntimePaths,
        heartbeat_config: LiveHeartbeatConfig,
    ) -> HeartbeatWriter:
        return HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=HeartbeatWriterConfig(
                enabled=heartbeat_config.enabled,
                interval_seconds=heartbeat_config.interval_seconds,
                stale_after_seconds=heartbeat_config.stale_after_seconds,
            ),
        )

    def create_monitor(
        self,
        *,
        config: BollBandBreakoutMonitorConfig,
        tick_handlers: Sequence[Callable[[MarketTickEvent], object]],
    ) -> BollBandBreakoutMonitor:
        return BollBandBreakoutMonitor(
            config=config,
            tick_handlers=list(tick_handlers),
        )
