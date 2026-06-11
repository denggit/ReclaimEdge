from __future__ import annotations

import asyncio
import os
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
    MultiSymbolSupervisor,
    install_supervisor_signal_handlers,
    ChildEventReader,
    AlertDeduper,
    AlertPolicy,
    SupervisorEmailPublisher,
    SupervisorEventPipeline,
    WorkerEventOutboxRetention,
    SupervisorSymbolSelection,
    select_enabled_supervisor_symbols,
    require_single_enabled_symbol,
    build_symbol_worker_plans,
)
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)


def _resolve_enabled_symbols_and_runtime() -> tuple[tuple[str, ...], Path]:
    """Resolve enabled supervisor symbols and runtime directory from env/TOML."""
    env_runtime = load_env_runtime_config()

    # -- legacy path: TOML disabled, only ETH is allowed -------------------
    if not env_runtime.use_symbol_toml:
        if env_runtime.symbols != ("ETH-USDT-SWAP",):
            raise RuntimeError(
                "RECLAIM_USE_SYMBOL_TOML is false but RECLAIM_SYMBOLS is not "
                "the single supported legacy symbol ETH-USDT-SWAP. "
                f"Got: {env_runtime.symbols!r}"
            )
        logger.info(
            "RECLAIM_SUPERVISOR_SYMBOL_SELECTION | legacy_toml_disabled "
            "selected=%s runtime_dir=%s",
            "ETH-USDT-SWAP",
            env_runtime.runtime_dir,
        )
        return ("ETH-USDT-SWAP",), env_runtime.runtime_dir

    # -- TOML path: select enabled symbols from config files ---------------
    selection = select_enabled_supervisor_symbols(
        symbols=env_runtime.symbols,
        symbol_config_dir=env_runtime.symbol_config_dir,
    )
    enabled_symbols = selection.enabled_symbols
    if len(enabled_symbols) == 0:
        raise RuntimeError(
            "No enabled symbols selected for supervisor. "
            f"requested_symbols={selection.requested_symbols!r}, "
            f"enabled_symbols={selection.enabled_symbols!r}, "
            f"skipped_disabled_symbols={selection.skipped_disabled_symbols!r}"
        )

    logger.warning(
        "RECLAIM_SUPERVISOR_SYMBOL_SELECTION | requested=%s enabled=%s "
        "skipped_disabled=%s selected=%s",
        list(selection.requested_symbols),
        list(selection.enabled_symbols),
        list(selection.skipped_disabled_symbols),
        list(enabled_symbols),
    )
    return enabled_symbols, env_runtime.runtime_dir


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


def _resolve_selected_symbol_and_child_env() -> tuple[str, dict[str, str], Path]:
    """Resolve which single symbol the supervisor should launch.

    Returns:
        (selected_symbol, child_env, runtime_dir) where *child_env*
        overrides ``RECLAIM_SYMBOLS`` and ``OKX_INST_ID`` in the child
        process so that only the selected symbol is visible, and
        *runtime_dir* comes from ``RECLAIM_RUNTIME_DIR``.

    Raises:
        RuntimeError: If no enabled symbols are found, or more than one
            enabled symbol is found (single-child invariant).
    """
    enabled_symbols, runtime_dir = _resolve_enabled_symbols_and_runtime()
    selection = SupervisorSymbolSelection(
        requested_symbols=enabled_symbols,
        enabled_symbols=enabled_symbols,
        skipped_disabled_symbols=(),
    )
    selected = require_single_enabled_symbol(selection)
    plans = build_symbol_worker_plans(
        (selected,),
        base_env=os.environ,
        runtime_dir=runtime_dir,
        heartbeat_dir=runtime_dir / "heartbeats",
        event_dir=runtime_dir / "events",
    )
    return selected, plans[0].child_env, runtime_dir


async def main() -> None:
    load_dotenv()
    if not live_config_helpers.live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start reclaim supervisor.")

    enabled_symbols, runtime_dir = _resolve_enabled_symbols_and_runtime()
    plans = build_symbol_worker_plans(
        enabled_symbols,
        base_env=os.environ,
        runtime_dir=runtime_dir,
        heartbeat_dir=runtime_dir / "heartbeats",
        event_dir=runtime_dir / "events",
    )

    base_supervisor = ReclaimSupervisor.from_env()

    supervisors: list[ReclaimSupervisor] = []
    for plan in plans:
        supervisor_config = replace(
            base_supervisor.config,
            child_name=plan.child_name,
            child_symbol=plan.symbol,
            runtime_dir=runtime_dir,
            child_env=plan.child_env,
        )

        temp_supervisor_for_paths = ReclaimSupervisor(config=supervisor_config)
        event_pipeline = build_parent_event_pipeline(temp_supervisor_for_paths)
        supervisors.append(
            ReclaimSupervisor(
                config=supervisor_config,
                event_pipeline=event_pipeline,
            )
        )

    if len(supervisors) == 1:
        supervisor = supervisors[0]
        install_supervisor_signal_handlers(supervisor)
        await supervisor.run_forever()
        return

    multi_supervisor = MultiSymbolSupervisor(supervisors)
    install_supervisor_signal_handlers(multi_supervisor)
    return_code = await multi_supervisor.run()
    if return_code != 0:
        raise RuntimeError(f"Multi-symbol supervisor exited with return_code={return_code}")


if __name__ == "__main__":
    asyncio.run(main())
