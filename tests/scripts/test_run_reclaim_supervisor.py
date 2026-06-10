#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E05f-e wiring tests — verifies ``scripts/run_reclaim_supervisor.py`` correctly
assembles the parent event pipeline and injects it into ``ReclaimSupervisor``.

These tests use monkeypatch / fake objects only.  No real EmailSender, no real
SMTP, no real ``run_forever`` long‑running loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENTRY_SCRIPT = _PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py"


def _entry_source() -> str:
    return _ENTRY_SCRIPT.read_text(encoding="utf-8")


# ============================================================================
# Source guard — positive
# ============================================================================


def test_source_guard_must_contain_required_tokens() -> None:
    """E05f-e source must contain the pipeline assembly tokens."""
    source = _entry_source()

    required = [
        "build_parent_event_pipeline",
        "ChildEventReader",
        "AlertDeduper",
        "AlertPolicy",
        "SupervisorEmailPublisher",
        "SupervisorEventPipeline",
        "EmailSender",
        "worker_event_outbox_file",
        "worker_event_cursor_",
        "supervisor_alert_dedupe_",
    ]
    for token in required:
        assert token in source, (
            f"E05f-e run_reclaim_supervisor.py must contain {token!r}"
        )


# ============================================================================
# Source guard — negative
# ============================================================================


def test_source_guard_must_not_contain_forbidden_tokens() -> None:
    """E05f-e source must NOT contain trading / child-start / send-email tokens."""
    source = _entry_source()

    forbidden = [
        "Trader",
        "Strategy",
        "SymbolWorkerApp",
        "SymbolWorkerFactory",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "multiprocessing",
        "subprocess",
        "WorkerEventEmitter",
        "send_email_async(",
        "process_once(",
        "run_symbol_worker.py",
        "BTC-USDT-SWAP",
        "RECLAIM_SYMBOLS",
    ]
    for token in forbidden:
        assert token not in source, (
            f"E05f-e run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# Fake / helper classes
# ============================================================================


class FakeEmailSender:
    """A fake EmailSender that records construction and send calls."""

    instance_count = 0

    def __init__(self) -> None:
        type(self).instance_count += 1
        self.calls: list[tuple[str, str, str]] = []

    async def send_email_async(self, subject: str, content: str, content_type: str = "plain") -> bool:
        self.calls.append((subject, content, content_type))
        return True


# ============================================================================
# 1. build_parent_event_pipeline uses runtime_paths
# ============================================================================


def test_build_parent_event_pipeline_uses_runtime_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_parent_event_pipeline must use supervisor.runtime_paths() for
    outbox, cursor and dedupe paths."""
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.EmailSender", FakeEmailSender
    )

    from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor, ReclaimSupervisorConfig
    from src.live.supervisor.supervisor_event_pipeline import SupervisorEventPipeline
    from scripts.run_reclaim_supervisor import build_parent_event_pipeline

    config = ReclaimSupervisorConfig(
        project_root=tmp_path,
        runtime_dir=Path("runtime"),
    )
    supervisor = ReclaimSupervisor(config=config)
    pipeline = build_parent_event_pipeline(supervisor)

    assert isinstance(pipeline, SupervisorEventPipeline)

    rp = supervisor.runtime_paths()

    # outbox path
    assert pipeline._reader._outbox_path == rp.worker_event_outbox_file

    # cursor path
    expected_cursor = rp.state_dir / f"worker_event_cursor_{rp.symbol_slug}.json"
    assert pipeline._reader._cursor_path == expected_cursor

    # dedupe state path
    expected_dedupe = rp.state_dir / f"supervisor_alert_dedupe_{rp.symbol_slug}.json"
    assert pipeline._deduper._state_path == expected_dedupe


# ============================================================================
# 2. build_parent_event_pipeline does NOT do startup IO
# ============================================================================


def test_build_parent_event_pipeline_no_startup_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing reader/deduper/pipeline must NOT write files — only later
    process_once calls may create cursor/dedupe state."""
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.EmailSender", FakeEmailSender
    )

    from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor, ReclaimSupervisorConfig
    from scripts.run_reclaim_supervisor import build_parent_event_pipeline

    config = ReclaimSupervisorConfig(
        project_root=tmp_path,
        runtime_dir=Path("runtime"),
    )
    supervisor = ReclaimSupervisor(config=config)
    rp = supervisor.runtime_paths()

    build_parent_event_pipeline(supervisor)

    # runtime dir itself might not exist at all
    assert not (tmp_path / "runtime").exists(), (
        "build_parent_event_pipeline must not create the runtime directory"
    )

    # worker outbox must not exist
    assert not rp.worker_event_outbox_file.exists(), (
        "build_parent_event_pipeline must not create the worker outbox file"
    )

    # cursor must not exist
    expected_cursor = rp.state_dir / f"worker_event_cursor_{rp.symbol_slug}.json"
    assert not expected_cursor.exists(), (
        "build_parent_event_pipeline must not create the cursor file"
    )

    # dedupe state must not exist
    expected_dedupe = rp.state_dir / f"supervisor_alert_dedupe_{rp.symbol_slug}.json"
    assert not expected_dedupe.exists(), (
        "build_parent_event_pipeline must not create the dedupe state file"
    )


# ============================================================================
# 3. build_parent_event_pipeline creates EmailSender but does NOT send
# ============================================================================


def test_build_parent_event_pipeline_creates_email_sender_does_not_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EmailSender is constructed but no email is sent during pipeline assembly."""
    FakeEmailSender.instance_count = 0
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.EmailSender", FakeEmailSender
    )

    from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor, ReclaimSupervisorConfig
    from scripts.run_reclaim_supervisor import build_parent_event_pipeline

    config = ReclaimSupervisorConfig(
        project_root=tmp_path,
        runtime_dir=Path("runtime"),
    )
    supervisor = ReclaimSupervisor(config=config)
    pipeline = build_parent_event_pipeline(supervisor)

    # Exactly one EmailSender was constructed.
    assert FakeEmailSender.instance_count == 1, (
        f"Expected 1 EmailSender instance, got {FakeEmailSender.instance_count}"
    )

    # The underlying email_sender inside publisher must be our fake.
    email_sender = pipeline._publisher._email_sender
    assert isinstance(email_sender, FakeEmailSender)

    # No send calls were made.
    assert email_sender.calls == [], (
        f"EmailSender.send_email_async must not be called during assembly, "
        f"got {email_sender.calls}"
    )


# ============================================================================
# 4. main wiring injects pipeline into ReclaimSupervisor
# ============================================================================


def test_main_wiring_injects_pipeline_to_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must:
    1. Call build_parent_event_pipeline.
    2. Create a new ReclaimSupervisor with the pipeline injected.
    3. Pass that supervisor to install_supervisor_signal_handlers.
    4. Call run_forever on the new supervisor.
    """
    class SentinelPipeline:
        """A sentinel that satisfies the duck-typing check in ReclaimSupervisor.__init__."""
        async def process_once(self) -> None:
            pass

    sentinel_pipeline = SentinelPipeline()

    # -- track what is passed around ------------------------------------------
    signal_supervisor: object | None = None
    run_forever_called = 0
    run_event_pipeline: object | None = None

    # -- fakes -----------------------------------------------------------------
    def fake_build_pipeline(supervisor: object) -> object:
        return sentinel_pipeline

    def fake_install(supervisor: object) -> None:
        nonlocal signal_supervisor
        signal_supervisor = supervisor

    async def fake_run_forever(self: object) -> None:
        nonlocal run_forever_called, run_event_pipeline
        run_forever_called += 1
        run_event_pipeline = getattr(self, "event_pipeline", None)

    # -- apply monkeypatches ---------------------------------------------------
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
        lambda: True,
    )
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.build_parent_event_pipeline",
        fake_build_pipeline,
    )
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.install_supervisor_signal_handlers",
        fake_install,
    )
    monkeypatch.setattr(
        "src.live.supervisor.reclaim_supervisor.ReclaimSupervisor.run_forever",
        fake_run_forever,
    )

    from scripts.run_reclaim_supervisor import main
    import asyncio

    asyncio.run(main())

    # -- assertions ------------------------------------------------------------
    assert signal_supervisor is not None, (
        "install_supervisor_signal_handlers must be called"
    )
    assert getattr(signal_supervisor, "event_pipeline", None) is sentinel_pipeline, (
        "Supervisor passed to signal handlers must have the injected event_pipeline"
    )
    assert run_forever_called == 1, (
        f"run_forever must be called exactly once, got {run_forever_called}"
    )
    assert run_event_pipeline is sentinel_pipeline, (
        "run_forever must see the injected event_pipeline"
    )


# ============================================================================
# 5. LIVE_TRADING false gate still works
# ============================================================================


def test_main_live_trading_false_no_pipeline_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """When LIVE_TRADING is false, main() must raise RuntimeError and must NOT
    call build_parent_event_pipeline."""
    build_called = False

    def fake_build_pipeline(supervisor: object) -> object:
        nonlocal build_called
        build_called = True
        raise AssertionError("build_parent_event_pipeline must not be called")

    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
        lambda: False,
    )
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.build_parent_event_pipeline",
        fake_build_pipeline,
    )

    from scripts.run_reclaim_supervisor import main
    import asyncio

    with pytest.raises(RuntimeError, match="LIVE_TRADING is not true"):
        asyncio.run(main())

    assert not build_called, (
        "build_parent_event_pipeline must not be called when LIVE_TRADING is false"
    )


# ============================================================================
# 6. main does NOT send real email
# ============================================================================


def test_main_does_not_send_real_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """The main wiring test with a real build_parent_event_pipeline (but fake
    EmailSender) must not trigger any real SMTP send."""
    FakeEmailSender.instance_count = 0
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.EmailSender", FakeEmailSender
    )
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
        lambda: True,
    )
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)

    async def fake_run_forever(self: object) -> None:
        pass  # do not loop

    monkeypatch.setattr(
        "src.live.supervisor.reclaim_supervisor.ReclaimSupervisor.run_forever",
        fake_run_forever,
    )

    from scripts.run_reclaim_supervisor import main
    import asyncio

    asyncio.run(main())

    # Verify an EmailSender was constructed (fail‑fast behaviour).
    assert FakeEmailSender.instance_count >= 1, (
        "EmailSender must be constructed (fail‑fast for missing email config)"
    )
