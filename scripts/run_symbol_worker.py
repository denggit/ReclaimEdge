from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

load_dotenv()

from src.live.worker_logging import configure_symbol_worker_logging_env  # noqa: E402


_MULTI_SYMBOL_ENV = "RECLAIM_" + "SYMBOLS"


def _worker_symbol_from_env() -> str:
    for env_name in ("OKX_INST_ID", "RECLAIM_SYMBOL"):
        symbol = os.getenv(env_name, "").strip()
        if symbol:
            return symbol

    symbol = os.getenv(_MULTI_SYMBOL_ENV, "").split(",")[0].strip()
    return symbol or "UNKNOWN"


configure_symbol_worker_logging_env(
    symbol=_worker_symbol_from_env(),
    force=True,
)

from src.live import config_helpers as live_config_helpers  # noqa: E402
from src.live.symbol_worker_app import SymbolWorkerApp  # noqa: E402
from src.live.worker_shutdown import (  # noqa: E402
    WorkerShutdownController,
    install_symbol_worker_signal_handlers,
)


async def main() -> None:
    mode = os.getenv("RECLAIM_WORKER_MODE", "live").strip().lower()
    if mode not in ("live", "paper"):
        raise RuntimeError(
            f"Invalid RECLAIM_WORKER_MODE: {mode!r}. Must be 'live' or 'paper'."
        )
    if mode != "paper":
        if not live_config_helpers.live_trading_enabled():
            raise RuntimeError("LIVE_TRADING is not true. Refusing to start symbol worker.")
    shutdown_controller = WorkerShutdownController()
    install_symbol_worker_signal_handlers(shutdown_controller)
    app = SymbolWorkerApp.from_env(shutdown_controller=shutdown_controller)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
