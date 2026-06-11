#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C02 unit tests for ``src.live.symbol_worker_factory``.

These tests verify:
* Runtime paths are correctly generated.
* Persistence objects use the correct paths from RuntimePaths.
* Queues use the sizes from LiveAppConfig.
* Dataclasses are frozen.
* Factory source has no runtime side effects.
* Factory does not import workers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.live.live_app_config import (
    DailyReportConfig,
    LiveAppConfig,
    LiveHeartbeatConfig,
    WeeklySummaryConfig,
)
from src.live.runtime_paths import RuntimePaths
from src.live.symbol_worker_factory import (
    SymbolWorkerFactory,
    SymbolWorkerPersistence,
    SymbolWorkerQueues,
    SymbolWorkerStrategyObjects,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FACTORY_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_factory.py"


def _factory_source() -> str:
    return _FACTORY_MODULE.read_text()


# ---------------------------------------------------------------------------
# 1. test_create_runtime_paths
# ---------------------------------------------------------------------------


def test_create_runtime_paths(tmp_path: Path) -> None:
    factory = SymbolWorkerFactory()
    paths = factory.create_runtime_paths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP"
    )
    assert paths.state_file == tmp_path / "runtime" / "state" / "live_state_ETH-USDT-SWAP.json"
    assert paths.rolling_loss_guard_state_file == tmp_path / "runtime" / "risk" / "rolling_loss_guard_state.json"


# ---------------------------------------------------------------------------
# 2. test_create_persistence_uses_runtime_paths
# ---------------------------------------------------------------------------


def test_create_persistence_uses_runtime_paths(tmp_path: Path) -> None:
    factory = SymbolWorkerFactory()
    runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
    email_sender = factory.create_email_sender()
    persistence = factory.create_persistence(
        runtime_paths=runtime_paths,
        email_sender=email_sender,
    )
    assert persistence.journal.path == runtime_paths.journal_file
    assert persistence.journal.summary_path == runtime_paths.trade_summary_file
    assert persistence.state_store.path == runtime_paths.state_file
    assert persistence.rolling_loss_guard.state_path == runtime_paths.rolling_loss_guard_state_file
    assert persistence.reporter.journal is persistence.journal


# ---------------------------------------------------------------------------
# 3. test_create_queues_uses_app_config_sizes
# ---------------------------------------------------------------------------


def test_create_queues_uses_app_config_sizes() -> None:
    app_config = LiveAppConfig(
        strategy_tick_queue_maxsize=123,
        execution_queue_maxsize=45,
        position_sync_seconds=5.0,
        account_sync_seconds=60.0,
        cash_log_min_delta_usdt=0.01,
        market_tick_heartbeat_seconds=60.0,
        account_snapshot_stale_warn_seconds=30.0,
        strategy_tick_lag_warn_seconds=2.0,
        execution_queue_backlog_log_seconds=30.0,
        daily_report=DailyReportConfig(raw_time="09:00", hour=9, minute=0),
        weekly_summary=WeeklySummaryConfig(
            enabled=True,
            raw_time="10:00",
            raw_weekday="0",
            weekday=0,
            hour=10,
            minute=0,
            compact_after_success=False,
        ),
        heartbeat=LiveHeartbeatConfig(
            enabled=True, interval_seconds=10.0, stale_after_seconds=30.0
        ),
    )
    factory = SymbolWorkerFactory()
    queues = factory.create_queues(app_config)
    assert queues.strategy_tick_queue.maxsize == 123
    assert queues.execution_queue.maxsize == 45


# ---------------------------------------------------------------------------
# 4. test_dataclasses_are_frozen
# ---------------------------------------------------------------------------


def test_dataclasses_are_frozen() -> None:
    """SymbolWorkerPersistence, SymbolWorkerStrategyObjects, and
    SymbolWorkerQueues must be frozen=True dataclasses."""

    # We cannot easily construct full objects for all three without real
    # dependencies, so we verify via source inspection and by checking
    # that the classes are truly frozen dataclasses.

    # Verify frozen=True is in the source.
    source = _factory_source()
    assert "@dataclass(frozen=True)" in source, (
        "factory dataclasses must use frozen=True"
    )

    # Verify the dataclass fields are read-only by attempting to mutate
    # a partially-constructed instance.  SymbolWorkerQueues is the
    # easiest to construct without real dependencies.
    q = SymbolWorkerQueues(
        strategy_tick_queue=asyncio.Queue(maxsize=10),
        execution_queue=asyncio.Queue(maxsize=5),
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError or similar
        q.strategy_tick_queue = asyncio.Queue(maxsize=20)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. test_factory_source_has_no_runtime_side_effects
# ---------------------------------------------------------------------------


def test_factory_source_has_no_runtime_side_effects() -> None:
    """The factory source must NOT contain any runtime side-effect calls."""
    source = _factory_source()

    forbidden = [
        ".start(",
        ".initialize(",
        "run_forever(",
        "asyncio.gather",
        "state_store.load(",
        "load_or_initialize(",
        "handoff_legacy_runtime_files(",
        "os.getenv(",
        "load_dotenv",
        "fetch_position_snapshot",
        "fetch_usdt_equity",
        "request(",
    ]
    for token in forbidden:
        assert token not in source, (
            f"factory must not contain runtime side-effect: {token!r}"
        )


# ---------------------------------------------------------------------------
# 6. test_factory_does_not_import_workers
# ---------------------------------------------------------------------------


def test_factory_does_not_import_workers() -> None:
    """The factory must not import or reference live workers."""
    source = _factory_source()

    forbidden = [
        "src.live.workers",
        "account_position_sync_worker",
        "execution_worker",
        "strategy_tick_worker",
    ]
    for token in forbidden:
        assert token not in source, (
            f"factory must not import workers: {token!r}"
        )


# ---------------------------------------------------------------------------
# 7. test_factory_creates_strategy_objects_correctly
# ---------------------------------------------------------------------------


def test_factory_source_creates_strategy_objects_correctly() -> None:
    """Verify the factory constructs SimplePositionSizer and
    BollCvdShockReclaimStrategy with the expected signatures."""
    source = _factory_source()

    assert "SimplePositionSizer(position_sizer_config)" in source, (
        "factory must construct SimplePositionSizer(position_sizer_config)"
    )
    assert "BollCvdShockReclaimStrategy(strategy_config, sizer)" in source, (
        "factory must construct BollCvdShockReclaimStrategy(strategy_config, sizer)"
    )


# ---------------------------------------------------------------------------
# 8. test_create_heartbeat_writer_uses_runtime_paths_and_config
# ---------------------------------------------------------------------------


def test_create_heartbeat_writer_uses_runtime_paths_and_config(tmp_path: Path) -> None:
    factory = SymbolWorkerFactory()
    runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
    heartbeat_config = LiveHeartbeatConfig(
        enabled=True, interval_seconds=2.0, stale_after_seconds=6.0
    )
    writer = factory.create_heartbeat_writer(
        runtime_paths=runtime_paths, heartbeat_config=heartbeat_config
    )

    assert writer.path == runtime_paths.heartbeat_file
    assert writer.config.enabled is True
    assert writer.config.interval_seconds == 2.0
    assert writer.config.stale_after_seconds == 6.0


# ---------------------------------------------------------------------------
# 9. test_factory_source_does_not_start_heartbeat
# ---------------------------------------------------------------------------


def test_factory_source_does_not_start_heartbeat() -> None:
    """The factory source must NOT start the heartbeat writer — it only constructs."""
    source = _factory_source()

    forbidden = [
        "run_until_cancelled(",
        "write_once(",
        "asyncio.create_task",
    ]
    for token in forbidden:
        assert token not in source, (
            f"factory must not start heartbeat: {token!r}"
        )


# ---------------------------------------------------------------------------
# 10. G09c: create_trader passes through metadata and market_settings
# ---------------------------------------------------------------------------


class TestCreateTraderG09c:
    def test_live_mode_passes_metadata_and_market_settings(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """live mode create_trader must forward metadata and market_settings
        to the Trader constructor."""
        from decimal import Decimal
        from unittest.mock import patch

        from src.execution.trader_types import (
            TraderInstrumentMetadata,
            TraderMarketSettings,
        )
        from src.live.symbol_worker_factory import SymbolWorkerFactory

        # Set env vars required by Trader.__init__
        monkeypatch.setenv("OKX_INST_ID", "BTC-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_ALLOWED_LIVE_SYMBOLS", "BTC-USDT-SWAP")
        monkeypatch.setenv("LIVE_TRADING", "true")
        monkeypatch.setenv("OKX_API_KEY", "test-key")
        monkeypatch.setenv("OKX_SECRET_KEY", "test-secret")
        monkeypatch.setenv("OKX_PASSPHRASE", "test-pass")

        metadata = TraderInstrumentMetadata(
            inst_id="BTC-USDT-SWAP",
            contract_multiplier=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        settings = TraderMarketSettings(
            inst_id="BTC-USDT-SWAP",
            td_mode="cross",
            pos_side_mode="long_short",
            leverage=Decimal("5"),
        )

        factory = SymbolWorkerFactory()
        trader = factory.create_trader(
            trader_mode="live",
            instrument_metadata=metadata,
            market_settings=settings,
        )

        assert trader.symbol == "BTC-USDT-SWAP"
        assert trader.instrument_metadata is metadata
        assert trader.contract_multiplier == Decimal("0.01")
        assert trader.td_mode == "cross"
        assert trader.pos_side_mode == "long_short"
        assert trader.leverage == "5"

    def test_live_mode_without_metadata_uses_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When metadata and market_settings are None, Trader uses env defaults."""
        from decimal import Decimal

        from src.live.symbol_worker_factory import SymbolWorkerFactory

        monkeypatch.setenv("OKX_INST_ID", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_ALLOWED_LIVE_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("LIVE_TRADING", "true")
        monkeypatch.setenv("OKX_API_KEY", "test-key")
        monkeypatch.setenv("OKX_SECRET_KEY", "test-secret")
        monkeypatch.setenv("OKX_PASSPHRASE", "test-pass")
        monkeypatch.setenv("OKX_TD_MODE", "isolated")
        monkeypatch.setenv("LEVERAGE", "50")
        monkeypatch.setenv("OKX_POS_SIDE_MODE", "net")

        factory = SymbolWorkerFactory()
        trader = factory.create_trader(trader_mode="live")

        assert trader.symbol == "ETH-USDT-SWAP"
        assert trader.td_mode == "isolated"
        assert trader.leverage == "50"
        assert trader.pos_side_mode == "net"
        # ETH default metadata should be used
        assert trader.contract_multiplier == Decimal("0.1")

    def test_paper_mode_ignores_metadata_and_settings(self) -> None:
        """Paper mode should ignore metadata/market_settings (no crash)."""
        from decimal import Decimal
        from unittest.mock import patch

        from src.execution.trader_types import (
            TraderInstrumentMetadata,
            TraderMarketSettings,
        )
        from src.live.symbol_worker_factory import SymbolWorkerFactory

        metadata = TraderInstrumentMetadata(
            inst_id="BTC-USDT-SWAP",
            contract_multiplier=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        settings = TraderMarketSettings(
            inst_id="BTC-USDT-SWAP",
            td_mode="cross",
            pos_side_mode="long_short",
            leverage=Decimal("5"),
        )

        factory = SymbolWorkerFactory()
        # Paper mode should not crash when metadata/settings are passed
        trader = factory.create_trader(
            trader_mode="paper",
            instrument_metadata=metadata,
            market_settings=settings,
        )
        assert trader is not None

