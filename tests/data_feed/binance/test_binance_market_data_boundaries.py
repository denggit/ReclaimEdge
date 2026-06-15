#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_market_data_boundaries.py
@Description: Boundary tests for BinanceMarketDataClient.

              The source must NOT contain forbidden imports or patterns.
              Binance-specific code must only appear in allowed directories.
              Strategy, monitor, and live runtime must NOT import Binance
              concrete classes.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


# ======================================================================
# Source paths
# ======================================================================

_MARKET_DATA_CLIENT_PATH = ROOT / "src" / "data_feed" / "binance" / "market_data_client.py"
_MAPPERS_PATH = ROOT / "src" / "data_feed" / "binance" / "mappers.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


class TestFileExistenceAndCompilation:
    def test_market_data_client_file_exists(self) -> None:
        assert _MARKET_DATA_CLIENT_PATH.exists()
        assert _MARKET_DATA_CLIENT_PATH.is_file()

    def test_market_data_client_compiles(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        compile(text, str(_MARKET_DATA_CLIENT_PATH), "exec")

    def test_mappers_file_exists(self) -> None:
        assert _MAPPERS_PATH.exists()
        assert _MAPPERS_PATH.is_file()

    def test_mappers_compiles(self) -> None:
        text = _read(_MAPPERS_PATH)
        compile(text, str(_MAPPERS_PATH), "exec")


# ======================================================================
# Import-ability
# ======================================================================


class TestImportability:
    def test_binance_market_data_client_importable(self) -> None:
        from src.data_feed.binance.market_data_client import BinanceMarketDataClient  # noqa: F401

    def test_binance_mappers_importable(self) -> None:
        from src.data_feed.binance.mappers import (
            map_binance_agg_trade_to_market_trade_snapshot,
            map_binance_rest_kline_to_candle_snapshot,
        )  # noqa: F401

    def test_binance_market_data_client_in_init(self) -> None:
        from src.data_feed.binance import BinanceMarketDataClient  # noqa: F401


# ======================================================================
# Implements MarketDataClientPort
# ======================================================================


class TestImplementsMarketDataClientPort:
    def test_has_all_port_methods(self) -> None:
        from src.data_feed.binance.market_data_client import BinanceMarketDataClient

        required = {"fetch_recent_klines", "stream_market_events", "close"}
        actual = {
            name
            for name in dir(BinanceMarketDataClient)
            if not name.startswith("_") and callable(getattr(BinanceMarketDataClient, name, None))
        }
        missing = required - actual
        assert not missing, f"BinanceMarketDataClient is missing methods: {missing}"


# ======================================================================
# Forbidden in market_data_client.py
# ======================================================================


class TestNoForbiddenImportsInClient:
    def test_no_strategy_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "src.strategies" not in text

    def test_no_execution_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "src.execution" not in text

    def test_no_live_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "src.live" not in text

    def test_no_risk_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "src.risk" not in text

    def test_no_reporting_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "src.reporting" not in text

    def test_no_okx_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "src.data_feed.okx" not in text
        assert "src.exchanges.okx" not in text

    def test_no_env_read(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "os.getenv" not in text
        assert "load_dotenv" not in text

    def test_no_monitor_instantiation(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "BollBandBreakoutMonitor(" not in text
        assert "BollBandBreakoutMonitorConfig" not in text

    def test_no_scripts_import(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "scripts." not in text

    def test_no_execution_queue_write(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "execution_queue" not in text


# ======================================================================
# Forbidden patterns — no tick-path violations
# ======================================================================


class TestNoTickPathViolations:
    def test_no_file_io_in_tick_path(self) -> None:
        """The on_event callback path must not do file I/O."""
        text = _read(_MARKET_DATA_CLIENT_PATH)
        # open() / write() only appear in logging context and close()
        assert "open(" not in text
        assert ".write(" not in text

    def test_no_pandas_in_tick_path(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "pandas" not in text
        assert "pd." not in text

    def test_no_unbounded_list_in_tick_path(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "all_ticks" not in text

    def test_no_full_sort_in_tick_path(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert ".sort(" not in text


# ======================================================================
# Forbidden in mappers.py
# ======================================================================


class TestNoForbiddenImportsInMappers:
    def test_no_strategy_import(self) -> None:
        text = _read(_MAPPERS_PATH)
        assert "src.strategies" not in text

    def test_no_execution_import(self) -> None:
        text = _read(_MAPPERS_PATH)
        assert "src.execution" not in text

    def test_no_monitor_import(self) -> None:
        text = _read(_MAPPERS_PATH)
        assert "BollBandBreakoutMonitor" not in text


# ======================================================================
# Boundary scan — strategy / monitor / live must NOT import Binance
# ======================================================================


class TestStrategyMonitorLiveDoNotImportBinance:
    """Verify that strategy, monitor, and live runtime files never import
    Binance-specific concrete classes or modules.
    """

    FORBIDDEN_TOKENS: list[str] = [
        "BinanceMarketDataClient",
        "src.data_feed.binance",
        "src.exchanges.binance",
    ]

    ALLOWED_PATHS: set[str] = {
        "src/data_feed/binance",
        "src/exchanges/binance",
        "tests/data_feed/binance",
        "tests/exchanges/binance",
    }

    def _scan_directory(self, directory: Path) -> list[str]:
        violations: list[str] = []
        for py_file in sorted(directory.rglob("*.py")):
            rel = py_file.relative_to(ROOT).as_posix()
            # Skip allowed directories
            if any(rel.startswith(allowed) for allowed in self.ALLOWED_PATHS):
                continue
            # Skip __pycache__
            if "__pycache__" in rel:
                continue
            # Skip test files that are explicitly allowed (but we keep them out anyway)
            text = py_file.read_text(encoding="utf-8")
            for token in self.FORBIDDEN_TOKENS:
                if token in text:
                    for i, line in enumerate(text.split("\n"), 1):
                        if token in line and not line.strip().startswith("#"):
                            violations.append(f"{rel}:{i}: {line.strip()}")
                            break  # One violation per file per token is enough
        return violations

    def test_src_strategies_no_binance_imports(self) -> None:
        strat_dir = ROOT / "src" / "strategies"
        if not strat_dir.exists():
            return
        violations = self._scan_directory(strat_dir)
        assert not violations, (
            "src/strategies must NOT import Binance-specific code:\n"
            + "\n".join(violations)
        )

    def test_src_monitors_no_binance_imports(self) -> None:
        mon_dir = ROOT / "src" / "monitors"
        if not mon_dir.exists():
            return
        violations = self._scan_directory(mon_dir)
        assert not violations, (
            "src/monitors must NOT import Binance-specific code:\n"
            + "\n".join(violations)
        )

    def test_src_live_no_binance_imports(self) -> None:
        live_dir = ROOT / "src" / "live"
        if not live_dir.exists():
            return
        violations = self._scan_directory(live_dir)
        assert not violations, (
            "src/live must NOT import Binance-specific code:\n"
            + "\n".join(violations)
        )

    def test_src_execution_trader_no_binance_imports(self) -> None:
        trader_path = ROOT / "src" / "execution" / "trader.py"
        if not trader_path.exists():
            return
        text = trader_path.read_text(encoding="utf-8")
        for token in self.FORBIDDEN_TOKENS:
            for i, line in enumerate(text.split("\n"), 1):
                if token in line and not line.strip().startswith("#"):
                    assert False, (
                        f"src/execution/trader.py:{i} must NOT reference {token}: "
                        f"{line.strip()}"
                    )

    def test_scripts_run_boll_cvd_live_no_binance_imports(self) -> None:
        script_path = ROOT / "scripts" / "run_boll_cvd_live.py"
        if not script_path.exists():
            return
        text = script_path.read_text(encoding="utf-8")
        for token in self.FORBIDDEN_TOKENS:
            for i, line in enumerate(text.split("\n"), 1):
                if token in line and not line.strip().startswith("#"):
                    assert False, (
                        f"scripts/run_boll_cvd_live.py:{i} must NOT reference {token}: "
                        f"{line.strip()}"
                    )


# ======================================================================
# ALLOWED patterns — aiohttp and /fapi are OK at the adapter layer
# ======================================================================


class TestAllowedAdapterPatterns:
    def test_aiohttp_is_allowed(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "import aiohttp" in text

    def test_fapi_is_allowed(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "/fapi" in text

    def test_fstream_is_allowed(self) -> None:
        text = _read(_MARKET_DATA_CLIENT_PATH)
        assert "fstream.binance.com" in text
