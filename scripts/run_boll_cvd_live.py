from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.live import config_helpers as live_config_helpers  # noqa: E402
from src.live.symbol_worker_app import SymbolWorkerApp  # noqa: E402


async def main() -> None:
    load_dotenv()
    if not live_config_helpers.live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")
    app = SymbolWorkerApp.from_env()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
