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
from src.live.supervisor import (  # noqa: E402
    ReclaimSupervisor,
    install_supervisor_signal_handlers,
    ChildEventReader,
    AlertDeduper,
    AlertPolicy,
    SupervisorEmailPublisher,
    SupervisorEventPipeline,
    WorkerEventOutboxRetention,
)
from src.utils.email_sender import EmailSender  # noqa: E402


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


async def main() -> None:
    load_dotenv()
    if not live_config_helpers.live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start reclaim supervisor.")
    base_supervisor = ReclaimSupervisor.from_env()
    event_pipeline = build_parent_event_pipeline(base_supervisor)
    supervisor = ReclaimSupervisor(
        config=base_supervisor.config,
        event_pipeline=event_pipeline,
    )
    install_supervisor_signal_handlers(supervisor)
    await supervisor.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
