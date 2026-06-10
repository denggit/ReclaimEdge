from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from config.env_runtime_config import load_env_runtime_config  # noqa: E402
from src.live import config_helpers as live_config_helpers  # noqa: E402
from src.live.supervisor import (  # noqa: E402
    ReclaimSupervisor,
    ReclaimSupervisorConfig,
    install_supervisor_signal_handlers,
    ChildEventReader,
    AlertDeduper,
    AlertPolicy,
    SupervisorEmailPublisher,
    SupervisorEventPipeline,
    WorkerEventOutboxRetention,
    select_enabled_supervisor_symbols,
    require_single_enabled_symbol,
)
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)


def build_parent_event_pipeline(supervisor: ReclaimSupervisor) -> SupervisorEventPipeline:
    """Assemble the parent event pipeline from existing components.

    Uses ``supervisor.runtime_paths()`` for deterministic outbox, cursor and
    dedupe paths — never duplicates path logic.  ``EmailSender()`` is
    constructed here (fail-fast on missing email config).  The pipeline is
    returned but **not** started — ``run_forever`` drives it via
    ``process_child_events_once()``.
    """
    runtime_paths = supervisor.runtime_paths()

    reader = ChildEventReader(
        outbox_path=runtime_paths.worker_event_outbox_file,
        cursor_path=runtime_paths.state_dir / f"worker_event_cursor_{runtime_paths.symbol_slug}.json",
    )

    deduper = AlertDeduper(
        state_path=runtime_paths.state_dir / f"supervisor_alert_dedupe_{runtime_paths.symbol_slug}.json",
    )

    email_sender = EmailSender()
    publisher = SupervisorEmailPublisher(email_sender=email_sender)

    retention = WorkerEventOutboxRetention(
        outbox_path=runtime_paths.worker_event_outbox_file,
        cursor_path=runtime_paths.state_dir / f"worker_event_cursor_{runtime_paths.symbol_slug}.json",
    )

    return SupervisorEventPipeline(
        reader=reader,
        deduper=deduper,
        publisher=publisher,
        alert_policy=AlertPolicy(),
        outbox_retention=retention,
    )


def _resolve_selected_symbol_and_child_env() -> tuple[str, dict[str, str]]:
    """Resolve which single symbol the supervisor should launch.

    Returns:
        (selected_symbol, child_env) where *child_env* overrides
        ``RECLAIM_SYMBOLS`` and ``OKX_INST_ID`` in the child process so
        that only the selected symbol is visible.

    Raises:
        RuntimeError: If no enabled symbols are found, or more than one
            enabled symbol is found (single-child invariant).
    """
    env_runtime = load_env_runtime_config()

    # -- legacy path: TOML disabled, only ETH is allowed -------------------
    if not env_runtime.use_symbol_toml:
        if env_runtime.symbols != ("ETH-USDT-SWAP",):
            raise RuntimeError(
                "RECLAIM_USE_SYMBOL_TOML is false but RECLAIM_SYMBOLS is not "
                "the single supported legacy symbol ETH-USDT-SWAP. "
                f"Got: {env_runtime.symbols!r}"
            )
        selected = "ETH-USDT-SWAP"
        child_env = {
            "RECLAIM_SYMBOLS": selected,
            "OKX_INST_ID": selected,
        }
        logger.info(
            "RECLAIM_SUPERVISOR_SYMBOL_SELECTION | legacy_toml_disabled "
            "selected=%s",
            selected,
        )
        return selected, child_env

    # -- TOML path: select enabled symbols from config files ---------------
    selection = select_enabled_supervisor_symbols(
        symbols=env_runtime.symbols,
        symbol_config_dir=env_runtime.symbol_config_dir,
    )
    selected = require_single_enabled_symbol(selection)

    logger.warning(
        "RECLAIM_SUPERVISOR_SYMBOL_SELECTION | requested=%s enabled=%s "
        "skipped_disabled=%s selected=%s",
        list(selection.requested_symbols),
        list(selection.enabled_symbols),
        list(selection.skipped_disabled_symbols),
        selected,
    )

    # Child env override — only the selected symbol is visible to the child.
    child_env = {
        "RECLAIM_SYMBOLS": selected,
        "OKX_INST_ID": selected,
    }
    return selected, child_env


async def main() -> None:
    load_dotenv()
    if not live_config_helpers.live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start reclaim supervisor.")

    selected_symbol, child_env = _resolve_selected_symbol_and_child_env()

    # Build supervisor config with the selected symbol and child env override.
    base_supervisor = ReclaimSupervisor.from_env()
    supervisor_config = replace(
        base_supervisor.config,
        child_name=selected_symbol,
        child_env=child_env,
    )

    # Build the event pipeline based on the selected symbol's runtime paths.
    temp_supervisor_for_paths = ReclaimSupervisor(config=supervisor_config)
    event_pipeline = build_parent_event_pipeline(temp_supervisor_for_paths)

    supervisor = ReclaimSupervisor(
        config=supervisor_config,
        event_pipeline=event_pipeline,
    )
    install_supervisor_signal_handlers(supervisor)
    await supervisor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
