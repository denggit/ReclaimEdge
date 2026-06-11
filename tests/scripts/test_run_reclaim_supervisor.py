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
    """E05f-e + F04 source must contain the pipeline assembly and symbol selection tokens."""
    source = _entry_source()

    required = [
        "build_parent_event_pipeline",
        "ChildEventReader",
        "AlertDeduper",
        "AlertPolicy",
        "SupervisorEmailPublisher",
        "SupervisorEventPipeline",
        "EmailSender",
        "MultiSymbolSupervisor",
        "build_symbol_worker_plans",
        "worker_event_outbox_file",
        "worker_event_cursor_",
        "supervisor_alert_dedupe_",
        # F04 symbol selection tokens
        "select_enabled_supervisor_symbols",
        "require_single_enabled_symbol",
        "OKX_INST_ID",
        "RECLAIM_SYMBOLS",
    ]
    for token in required:
        assert token in source, (
            f"E05f-e / F04 run_reclaim_supervisor.py must contain {token!r}"
        )


# ============================================================================
# Source guard — negative
# ============================================================================


def test_source_guard_must_not_contain_forbidden_tokens() -> None:
    """E05f-e + F04 source must NOT contain trading / child-start / send-email tokens.
    ``RECLAIM_SYMBOLS`` is allowed (used in child_env construction);
    ``BTC-USDT-SWAP`` is allowed in tests/config but must not be hard-coded as
    selected symbol in the entry.
    """
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
    ]
    for token in forbidden:
        assert token not in source, (
            f"E05f-e / F04 run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# Helpers
# ============================================================================


def _patch_entry_load_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``load_dotenv`` in the supervisor entry module namespace.

    The entry uses ``from dotenv import load_dotenv``, so patching
    ``dotenv.load_dotenv`` is NOT sufficient — the module-level binding
    must be replaced.
    """
    monkeypatch.setattr("scripts.run_reclaim_supervisor.load_dotenv", lambda: None)


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
    _patch_entry_load_dotenv(monkeypatch)
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
    _patch_entry_load_dotenv(monkeypatch)
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
    _patch_entry_load_dotenv(monkeypatch)

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


# ============================================================================
# E07: Retention wiring tests
# ============================================================================


def test_build_pipeline_has_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """build_parent_event_pipeline must wire WorkerEventOutboxRetention into the pipeline."""
    monkeypatch.setattr(
        "scripts.run_reclaim_supervisor.EmailSender", FakeEmailSender
    )

    from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor, ReclaimSupervisorConfig
    from src.live.supervisor.supervisor_event_pipeline import SupervisorEventPipeline
    from src.live.supervisor.outbox_retention import WorkerEventOutboxRetention
    from scripts.run_reclaim_supervisor import build_parent_event_pipeline

    config = ReclaimSupervisorConfig(
        project_root=tmp_path,
        runtime_dir=Path("runtime"),
    )
    supervisor = ReclaimSupervisor(config=config)
    pipeline = build_parent_event_pipeline(supervisor)

    assert isinstance(pipeline, SupervisorEventPipeline)
    assert hasattr(pipeline, "_outbox_retention"), (
        "pipeline must have _outbox_retention attribute"
    )
    assert pipeline._outbox_retention is not None
    assert isinstance(pipeline._outbox_retention, WorkerEventOutboxRetention)


def test_retention_source_guard() -> None:
    """run_reclaim_supervisor.py must contain WorkerEventOutboxRetention and outbox_retention=."""
    source = _entry_source()

    required = [
        "WorkerEventOutboxRetention",
        "outbox_retention=",
    ]
    for token in required:
        assert token in source, (
            f"E07 run_reclaim_supervisor.py must contain {token!r}"
        )


def test_retention_no_forbidden_tokens() -> None:
    """E07 run_reclaim_supervisor.py must NOT contain env/retention-control tokens."""
    source = _entry_source()

    forbidden = [
        "OUTBOX_RETENTION_MAX_BYTES",
        "OUTBOX_RETENTION_KEEP_ARCHIVES",
    ]
    for token in forbidden:
        assert token not in source, (
            f"E07 run_reclaim_supervisor.py must NOT contain {token!r}"
        )


def test_main_wiring_still_intact() -> None:
    """main() must still contain: build pipeline, install handlers, run_forever."""
    source = _entry_source()
    # build_parent_event_pipeline call
    assert "build_parent_event_pipeline(" in source
    # install_supervisor_signal_handlers call
    assert "install_supervisor_signal_handlers(" in source
    # run_forever call
    assert "run_forever()" in source


# ============================================================================
# F04 — symbol selection integration tests
# ============================================================================


def _write_minimal_toml(dir_path: Path, inst_id: str, *, enabled: bool = True) -> Path:
    """Write a minimal, validator-compliant TOML file for *inst_id*."""
    toml_path = dir_path / f"{inst_id}.toml"
    content = f"""\
[symbol]
inst_id = "{inst_id}"
enabled = {str(enabled).lower()}
live_trading = false

[market]
bar = "15m"
td_mode = "isolated"
pos_side_mode = "net"
contract_value = "0.1"
min_contracts = "0.01"
contract_precision = "0.01"
price_precision = "0.01"
boll_window = 20
boll_std_multiplier = "2.0"
boll_distance_threshold_pct = "0.005"
tp_boll_window = 15
min_outside_pct = "0.0005"

[capital]
dry_run_equity_usdt = "1000"
layer_margin_pct = "0.03"
leverage = "50"
max_layers = 3
layer_multiplier_step = "0.15"

[entry]
add_gap_mode = "linear"
add_gap_base_pct = "0.006"
add_gap_step_pct = "0.001"
first_add_block_seconds = 3600
add_min_interval_seconds = 1800
alert_freeze_seconds = 3600

[cvd]
fast_window_seconds = "5"
price_stall_seconds = "2"
price_stall_tolerance_pct = "0.0005"
burst_window_seconds = "3"
burst_baseline_seconds = "60"
burst_min_move_ratio = "2.5"
burst_min_volume_ratio = "2.0"
burst_min_abs_range_pct = "0.0015"

[tp]
tp_min_net_profit_pct = "0.002"
tp_boll_enabled = true
three_stage_runner_enabled = true
three_stage_tp1_ratio = "0.70"
three_stage_tp2_ratio = "0.20"
three_stage_runner_ratio = "0.10"
three_stage_tp2_use_structure_boll = true
middle_runner_enabled = false
split_tp_enabled = false

[middle_bucket_split]
enabled = false
fast_ratio = "0.60"
fast_sl_enabled = true
fast_sl_fee_buffer_pct = "0.001"

[sidecar]
enabled = false
margin_pct = "0.01"
tp_pct = "0.004"
skip_first_layer = true
max_legs = 10
order_status_check_seconds = "5"
tp_place_retry_count = 3
tp_place_retry_interval_seconds = "0.8"
tp_place_retry_backoff_multiplier = "1.5"
tp_rate_limit_fail_action = "HALT_ONLY"

[risk]
rolling_loss_guard_enabled = true
rolling_loss_warn_pct = "0.50"
rolling_loss_soft_halt_pct = "0.10"
order_failure_market_exit_delay_seconds = 1800

[execution]
private_write_min_interval_seconds = "0.6"
max_order_retries = 3

[runtime]
strategy_tick_queue_maxsize = 20000
execution_queue_maxsize = 1000
position_sync_seconds = "5"
account_sync_seconds = "60"
market_tick_heartbeat_seconds = "60"
account_snapshot_stale_warn_seconds = "30"
strategy_tick_lag_warn_seconds = "2"
execution_backlog_log_seconds = "30"
"""
    toml_path.write_text(content, encoding="utf-8")
    return toml_path


class TestResolveSelectedSymbolAndChildEnv:
    """Integration tests for ``_resolve_selected_symbol_and_child_env()``."""

    def test_eth_only_selected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """RECLAIM_SYMBOLS=ETH-USDT-SWAP → selected ETH, child_env ETH only."""
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env

        selected, child_env, runtime_dir = _resolve_selected_symbol_and_child_env()

        assert selected == "ETH-USDT-SWAP"
        assert child_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"
        assert child_env["OKX_INST_ID"] == "ETH-USDT-SWAP"
        assert runtime_dir == Path("runtime")

    def test_eth_and_btc_skips_disabled_btc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """RECLAIM_SYMBOLS=ETH,BTC with BTC disabled → selected ETH, child_env ETH only."""
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)
        _write_minimal_toml(tmp_path, "BTC-USDT-SWAP", enabled=False)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP,BTC-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env

        selected, child_env, runtime_dir = _resolve_selected_symbol_and_child_env()

        assert selected == "ETH-USDT-SWAP"
        assert child_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"
        assert child_env["OKX_INST_ID"] == "ETH-USDT-SWAP"
        assert "BTC-USDT-SWAP" not in child_env["RECLAIM_SYMBOLS"]
        assert isinstance(runtime_dir, Path)

    def test_btc_only_raises_runtime_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """RECLAIM_SYMBOLS=BTC only with BTC disabled → RuntimeError (no enabled)."""
        _write_minimal_toml(tmp_path, "BTC-USDT-SWAP", enabled=False)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "BTC-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env

        with pytest.raises(RuntimeError, match="No enabled symbols"):
            _resolve_selected_symbol_and_child_env()

    def test_legacy_toml_disabled_eth_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RECLAIM_USE_SYMBOL_TOML=false with ETH → selected ETH."""
        monkeypatch.setenv("RECLAIM_USE_SYMBOL_TOML", "false")
        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env

        selected, child_env, runtime_dir = _resolve_selected_symbol_and_child_env()

        assert selected == "ETH-USDT-SWAP"
        assert child_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"
        assert child_env["OKX_INST_ID"] == "ETH-USDT-SWAP"
        assert isinstance(runtime_dir, Path)

    def test_legacy_toml_disabled_with_btc_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RECLAIM_USE_SYMBOL_TOML=false with ETH,BTC → RuntimeError."""
        monkeypatch.setenv("RECLAIM_USE_SYMBOL_TOML", "false")
        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP,BTC-USDT-SWAP")

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env

        with pytest.raises(RuntimeError, match="RECLAIM_USE_SYMBOL_TOML is false"):
            _resolve_selected_symbol_and_child_env()

    def test_selected_symbol_flow_passes_to_supervisor_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that the selected symbol makes it into the supervisor config
        via the full _resolve flow (ETH only, TOML enabled)."""
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env
        from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor, ReclaimSupervisorConfig
        from dataclasses import replace

        selected, child_env, runtime_dir = _resolve_selected_symbol_and_child_env()
        base_supervisor = ReclaimSupervisor.from_env()
        supervisor_config = replace(
            base_supervisor.config,
            child_name=selected,
            runtime_dir=runtime_dir,
            child_env=child_env,
        )

        assert supervisor_config.child_name == "ETH-USDT-SWAP"
        assert supervisor_config.child_env is not None
        assert supervisor_config.child_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"
        assert supervisor_config.child_env["OKX_INST_ID"] == "ETH-USDT-SWAP"


class TestRunReclaimSupervisorMultiSymbol:
    def test_single_symbol_main_does_not_construct_multi_supervisor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)
        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
            lambda: True,
        )
        _patch_entry_load_dotenv(monkeypatch)

        def fail_multi(*args: object, **kwargs: object) -> object:
            raise AssertionError("single-symbol main must not construct MultiSymbolSupervisor")

        monkeypatch.setattr("scripts.run_reclaim_supervisor.MultiSymbolSupervisor", fail_multi)

        captured_supervisors: list[object] = []

        def fake_install(supervisor: object) -> None:
            captured_supervisors.append(supervisor)

        monkeypatch.setattr("scripts.run_reclaim_supervisor.install_supervisor_signal_handlers", fake_install)

        def fake_build_pipeline(supervisor: object) -> object:
            sentinel = type("SentinelPipeline", (), {})()
            sentinel.process_once = lambda: None  # type: ignore[attr-defined]
            return sentinel

        monkeypatch.setattr("scripts.run_reclaim_supervisor.build_parent_event_pipeline", fake_build_pipeline)

        async def fake_run_forever(self: object) -> None:
            pass

        monkeypatch.setattr(
            "src.live.supervisor.reclaim_supervisor.ReclaimSupervisor.run_forever",
            fake_run_forever,
        )

        from scripts.run_reclaim_supervisor import main
        import asyncio

        asyncio.run(main())

        assert len(captured_supervisors) == 1
        config = captured_supervisors[0].config
        assert config.child_symbol == "ETH-USDT-SWAP"
        assert config.child_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"

    def test_multi_symbol_main_builds_two_live_worker_plans(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP,BTC-USDT-SWAP")
        monkeypatch.setenv(
            "RECLAIM_WORKER_MODES",
            "ETH-USDT-SWAP:live,BTC-USDT-SWAP:live",
        )
        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor._resolve_enabled_symbols_and_runtime",
            lambda: (("ETH-USDT-SWAP", "BTC-USDT-SWAP"), tmp_path / "runtime"),
        )
        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.require_single_enabled_symbol",
            lambda selection: (_ for _ in ()).throw(AssertionError("must not require single symbol")),
        )
        _patch_entry_load_dotenv(monkeypatch)

        def fake_build_pipeline(supervisor: object) -> object:
            sentinel = type("SentinelPipeline", (), {})()
            sentinel.process_once = lambda: None  # type: ignore[attr-defined]
            return sentinel

        monkeypatch.setattr("scripts.run_reclaim_supervisor.build_parent_event_pipeline", fake_build_pipeline)
        monkeypatch.setattr("scripts.run_reclaim_supervisor.install_supervisor_signal_handlers", lambda supervisor: None)

        captured_supervisors: list[object] = []

        class FakeMultiSymbolSupervisor:
            def __init__(self, supervisors: object) -> None:
                captured_supervisors.extend(supervisors)

            def request_stop(self) -> None:
                pass

            async def run(self) -> int:
                return 0

        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.MultiSymbolSupervisor",
            FakeMultiSymbolSupervisor,
        )

        from scripts.run_reclaim_supervisor import main
        import asyncio

        asyncio.run(main())

        assert len(captured_supervisors) == 2
        configs = [supervisor.config for supervisor in captured_supervisors]
        child_names = {config.child_name for config in configs}
        heartbeat_paths = {supervisor.heartbeat_path for supervisor in captured_supervisors}

        assert len(child_names) == 2
        assert len(heartbeat_paths) == 2

        by_symbol = {config.child_symbol: config for config in configs}
        eth_env = by_symbol["ETH-USDT-SWAP"].child_env
        btc_env = by_symbol["BTC-USDT-SWAP"].child_env

        assert eth_env["OKX_INST_ID"] == "ETH-USDT-SWAP"
        assert eth_env["RECLAIM_SYMBOL"] == "ETH-USDT-SWAP"
        assert eth_env["RECLAIM_SYMBOLS"] == "ETH-USDT-SWAP"
        assert eth_env["RECLAIM_WORKER_MODE"] == "live"

        assert btc_env["OKX_INST_ID"] == "BTC-USDT-SWAP"
        assert btc_env["RECLAIM_SYMBOL"] == "BTC-USDT-SWAP"
        assert btc_env["RECLAIM_SYMBOLS"] == "BTC-USDT-SWAP"
        assert btc_env["RECLAIM_WORKER_MODE"] == "live"


# ============================================================================
# F04b — runtime_dir from EnvRuntimeConfig applied to supervisor config
# ============================================================================


class TestRuntimeDirAppliedToSupervisorConfig:
    """F04b: RECLAIM_RUNTIME_DIR must be applied to supervisor config."""

    def test_main_applies_custom_runtime_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() must pass RECLAIM_RUNTIME_DIR into supervisor_config."""
        custom_runtime = tmp_path / "custom_runtime"
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("RECLAIM_RUNTIME_DIR", str(custom_runtime))
        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
            lambda: True,
        )
        _patch_entry_load_dotenv(monkeypatch)

        # Capture the supervisor config passed to build_parent_event_pipeline.
        captured_config: object | None = None

        def fake_build_pipeline(supervisor: object) -> object:
            nonlocal captured_config
            captured_config = supervisor.config
            # Return a sentinel pipeline that duck-types correctly.
            sentinel = type("SentinelPipeline", (), {})()
            sentinel.process_once = lambda: None  # type: ignore[attr-defined]
            return sentinel

        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.build_parent_event_pipeline",
            fake_build_pipeline,
        )

        async def fake_run_forever(self: object) -> None:
            pass

        monkeypatch.setattr(
            "src.live.supervisor.reclaim_supervisor.ReclaimSupervisor.run_forever",
            fake_run_forever,
        )

        from scripts.run_reclaim_supervisor import main
        import asyncio

        asyncio.run(main())

        assert captured_config is not None, (
            "build_parent_event_pipeline must be called"
        )
        assert hasattr(captured_config, "runtime_dir"), (
            "supervisor config must have runtime_dir"
        )
        assert captured_config.runtime_dir == custom_runtime, (
            f"Expected runtime_dir={custom_runtime}, "
            f"got {captured_config.runtime_dir}"
        )

    def test_main_uses_default_runtime_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When RECLAIM_RUNTIME_DIR is not set, main() uses the default."""
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.live_config_helpers.live_trading_enabled",
            lambda: True,
        )
        _patch_entry_load_dotenv(monkeypatch)

        captured_config: object | None = None

        def fake_build_pipeline(supervisor: object) -> object:
            nonlocal captured_config
            captured_config = supervisor.config
            sentinel = type("SentinelPipeline", (), {})()
            sentinel.process_once = lambda: None  # type: ignore[attr-defined]
            return sentinel

        monkeypatch.setattr(
            "scripts.run_reclaim_supervisor.build_parent_event_pipeline",
            fake_build_pipeline,
        )

        async def fake_run_forever(self: object) -> None:
            pass

        monkeypatch.setattr(
            "src.live.supervisor.reclaim_supervisor.ReclaimSupervisor.run_forever",
            fake_run_forever,
        )

        from scripts.run_reclaim_supervisor import main
        import asyncio

        asyncio.run(main())

        assert captured_config is not None
        assert captured_config.runtime_dir == Path("runtime"), (
            f"Expected default runtime_dir=Path('runtime'), "
            f"got {captured_config.runtime_dir}"
        )

    def test_runtime_dir_in_resolve_return_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_resolve_selected_symbol_and_child_env must return runtime_dir."""
        custom_runtime = tmp_path / "my_runtime"
        _write_minimal_toml(tmp_path, "ETH-USDT-SWAP", enabled=True)

        monkeypatch.setenv("RECLAIM_SYMBOLS", "ETH-USDT-SWAP")
        monkeypatch.setenv("RECLAIM_SYMBOL_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("RECLAIM_RUNTIME_DIR", str(custom_runtime))

        from scripts.run_reclaim_supervisor import _resolve_selected_symbol_and_child_env

        selected, child_env, runtime_dir = _resolve_selected_symbol_and_child_env()

        assert selected == "ETH-USDT-SWAP"
        assert runtime_dir == custom_runtime


# ============================================================================
# F04b — source guard
# ============================================================================


def test_source_guard_f04b_runtime_dir_in_replace() -> None:
    """F04b: run_reclaim_supervisor.py must apply runtime_dir in replace()."""
    source = _entry_source()

    assert "runtime_dir=runtime_dir" in source, (
        "F04b run_reclaim_supervisor.py must pass runtime_dir=env_runtime.runtime_dir "
        "to replace()"
    )
