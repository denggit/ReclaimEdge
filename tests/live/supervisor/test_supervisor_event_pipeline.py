from __future__ import annotations

from pathlib import Path

import pytest

from src.live.supervisor.alert_deduper import AlertDedupeDecision, AlertDeduper
from src.live.supervisor.child_event_reader import (
    ChildEvent,
    ChildEventReadError,
    ChildEventReadResult,
    ChildEventReader,
)
from src.live.supervisor.supervisor_event_pipeline import (
    READ_ERROR_EVENT_TYPE,
    LIFECYCLE_EVENT_TYPES,
    SupervisorAlert,
    SupervisorAlertPublisher,
    SupervisorEventPipeline,
    SupervisorEventPipelineResult,
)

# ============================================================================
# Source path for guards
# ============================================================================

_PIPELINE_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "live"
    / "supervisor"
    / "supervisor_event_pipeline.py"
)


# ============================================================================
# Fake implementations for testing
# ============================================================================


class FakeReader:
    """A fake ChildEventReader that returns a pre-configured result."""

    def __init__(self, result: ChildEventReadResult) -> None:
        self.result = result

    def read_new_events(self) -> ChildEventReadResult:
        return self.result


class FakeDeduper:
    """A fake AlertDeduper that always returns the configured decision.

    Records all ``should_send`` calls for inspection.
    """

    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[dict] = []

    def should_send(self, **kwargs) -> AlertDedupeDecision:
        self.calls.append(kwargs)
        return AlertDedupeDecision(
            allowed=self.allowed,
            dedupe_key="test-key",
            last_sent_ts_ms=None if self.allowed else 1000,
            next_allowed_ts_ms=2000 if not self.allowed else None,
            now_ts_ms=kwargs.get("now_ms", 1000),
            reason=kwargs.get("reason"),
        )


class FakePublisher:
    """A fake SupervisorAlertPublisher.

    Records published alerts and returns a configurable result.
    """

    def __init__(self, result: bool = True, raises: bool = False) -> None:
        self.alerts: list[SupervisorAlert] = []
        self.result = result
        self.raises = raises

    async def publish_alert(self, alert: SupervisorAlert) -> bool:
        self.alerts.append(alert)
        if self.raises:
            raise RuntimeError("publish failed")
        return self.result


# ============================================================================
# Helpers
# ============================================================================


def _make_child_event(
    event_type: str = "WORKER_STARTED",
    payload: dict | None = None,
    ts_ms: int = 1000,
    source_path: str = "runtime/events/x.jsonl",
) -> ChildEvent:
    if payload is None:
        payload = {"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {}}
    return ChildEvent(
        ts_ms=ts_ms,
        event_type=event_type,
        payload=payload,
        source_path=source_path,
    )


def _make_read_error(
    error_type: str = "BAD_JSON",
    message: str = "bad line",
    raw_preview: str = "{bad",
    source_path: str = "runtime/events/x.jsonl",
) -> ChildEventReadError:
    return ChildEventReadError(
        ts_ms=1000,
        error_type=error_type,
        message=message,
        source_path=source_path,
        offset_start=0,
        offset_end=10,
        raw_preview=raw_preview,
    )


# ============================================================================
# 1. Processes lifecycle child event and publishes alert
# ============================================================================


class TestProcessesLifecycleEvent:
    @pytest.mark.asyncio
    async def test_publishes_alert(self) -> None:
        event = _make_child_event(
            event_type="WORKER_STARTED",
            payload={"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.events_seen == 1
        assert result.alerts_built == 1
        assert result.alerts_allowed == 1
        assert result.alerts_published == 1
        assert result.alerts_suppressed == 0
        assert result.publish_failures == 0
        assert len(publisher.alerts) == 1

        alert = publisher.alerts[0]
        assert "ReclaimEdge" in alert.subject
        assert "INFO" in alert.subject
        assert "ETH-USDT-SWAP" in alert.subject
        assert "WORKER_STARTED" in alert.subject


# ============================================================================
# 2. Suppresses duplicate when deduper disallows
# ============================================================================


class TestSuppressesDuplicate:
    @pytest.mark.asyncio
    async def test_deduper_disallows(self) -> None:
        event = _make_child_event(
            event_type="WORKER_STARTED",
            payload={"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=False)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.alerts_suppressed == 1
        assert result.alerts_published == 0
        assert len(publisher.alerts) == 0


# ============================================================================
# 3. Ignores non-lifecycle event
# ============================================================================


class TestIgnoresNonLifecycleEvent:
    @pytest.mark.asyncio
    async def test_ignores_order_filled(self) -> None:
        event = _make_child_event(
            event_type="ORDER_FILLED",
            payload={"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.events_seen == 1
        assert result.alerts_built == 0
        assert len(publisher.alerts) == 0


# ============================================================================
# 4. Converts read error to alert
# ============================================================================


class TestConvertsReadErrorToAlert:
    @pytest.mark.asyncio
    async def test_read_error_alert(self) -> None:
        error = _make_read_error(
            error_type="BAD_JSON",
            message="bad line",
            raw_preview="{bad",
            source_path="runtime/events/x.jsonl",
        )
        reader = FakeReader(
            ChildEventReadResult(events=[], errors=[error])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.read_errors_seen == 1
        assert result.alerts_built == 1
        assert len(publisher.alerts) == 1

        alert = publisher.alerts[0]
        assert alert.event_type == "CHILD_EVENT_READ_FAILED"
        assert alert.symbol == "SUPERVISOR"
        assert alert.severity == "ERROR"
        assert alert.reason == "BAD_JSON"


# ============================================================================
# 5. Publisher false counts as failure
# ============================================================================


class TestPublisherFalseCountsFailure:
    @pytest.mark.asyncio
    async def test_publisher_returns_false(self) -> None:
        event = _make_child_event()
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=False)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.publish_failures == 1
        assert result.alerts_published == 0


# ============================================================================
# 6. Publisher exception is caught
# ============================================================================


class TestPublisherExceptionIsCaught:
    @pytest.mark.asyncio
    async def test_publisher_raises(self) -> None:
        event = _make_child_event()
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True, raises=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        # Must NOT raise.
        result = await pipeline.process_once(now_ms=1000)

        assert result.publish_failures == 1
        assert result.alerts_published == 0


# ============================================================================
# 7. max_alerts_per_cycle limits processing
# ============================================================================


class TestMaxAlertsPerCycle:
    @pytest.mark.asyncio
    async def test_limits_processing(self) -> None:
        events = [
            _make_child_event(
                event_type=f"WORKER_STARTED",
                payload={"symbol": f"SYM-{i}", "severity": "INFO", "data": {}},
                source_path=f"path/{i}.jsonl",
            )
            for i in range(5)
        ]
        reader = FakeReader(
            ChildEventReadResult(events=events, errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
            max_alerts_per_cycle=2,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.alerts_built == 5
        assert result.alerts_allowed == 2
        assert result.dropped_due_to_cycle_limit == 3
        assert len(publisher.alerts) == 2
        assert len(deduper.calls) == 2


# ============================================================================
# 8. Severity fallback to INFO
# ============================================================================


class TestSeverityFallback:
    @pytest.mark.asyncio
    async def test_missing_severity(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH-USDT-SWAP", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.severity == "INFO"

    @pytest.mark.asyncio
    async def test_whitespace_severity(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH-USDT-SWAP", "severity": "   ", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.severity == "INFO"


# ============================================================================
# 9. Severity normalized uppercase
# ============================================================================


class TestSeverityNormalized:
    @pytest.mark.asyncio
    async def test_uppercase(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH-USDT-SWAP", "severity": "warning", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.severity == "WARNING"


# ============================================================================
# 10. Symbol fallback UNKNOWN
# ============================================================================


class TestSymbolFallback:
    @pytest.mark.asyncio
    async def test_missing_symbol(self) -> None:
        event = _make_child_event(
            payload={"severity": "INFO", "data": {}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.symbol == "UNKNOWN"


# ============================================================================
# 11. Reason extraction from payload top-level reason
# ============================================================================


class TestReasonExtraction:
    @pytest.mark.asyncio
    async def test_top_level_reason(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH", "severity": "ERROR", "reason": "abc"},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.reason == "abc"

    @pytest.mark.asyncio
    async def test_halt_reason(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH", "halt_reason": "rolling_loss"},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.reason == "rolling_loss"

    @pytest.mark.asyncio
    async def test_error_type_reason(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH", "error_type": "BAD_JSON"},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.reason == "BAD_JSON"

    @pytest.mark.asyncio
    async def test_data_reason(self) -> None:
        event = _make_child_event(
            payload={"symbol": "ETH", "data": {"reason": "nested"}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert alert.reason == "nested"


# ============================================================================
# 15. Explicit HTML escaping
# ============================================================================


class TestHtmlEscaping:
    @pytest.mark.asyncio
    async def test_escaping(self) -> None:
        event = _make_child_event(
            payload={
                "symbol": "<ETH>",
                "severity": "INFO",
                "reason": "<script>alert(1)</script>",
                "data": {},
            },
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert "<script>" not in alert.body
        assert "&lt;script&gt;" in alert.body
        assert "<ETH>" not in alert.body
        assert "&lt;ETH&gt;" in alert.body


# ============================================================================
# 16. Invalid constructor args
# ============================================================================


class TestInvalidConstructorArgs:
    def test_reader_without_read_new_events(self) -> None:
        class BadReader:
            pass

        with pytest.raises(ValueError):
            SupervisorEventPipeline(
                reader=BadReader(),
                deduper=FakeDeduper(),
                publisher=FakePublisher(),
            )

    def test_deduper_without_should_send(self) -> None:
        class BadDeduper:
            pass

        with pytest.raises(ValueError):
            SupervisorEventPipeline(
                reader=FakeReader(ChildEventReadResult()),
                deduper=BadDeduper(),
                publisher=FakePublisher(),
            )

    def test_publisher_without_publish_alert(self) -> None:
        class BadPublisher:
            pass

        with pytest.raises(ValueError):
            SupervisorEventPipeline(
                reader=FakeReader(ChildEventReadResult()),
                deduper=FakeDeduper(),
                publisher=BadPublisher(),
            )

    def test_max_alerts_per_cycle_zero(self) -> None:
        with pytest.raises(ValueError):
            SupervisorEventPipeline(
                reader=FakeReader(ChildEventReadResult()),
                deduper=FakeDeduper(),
                publisher=FakePublisher(),
                max_alerts_per_cycle=0,
            )

    def test_max_alerts_per_cycle_negative(self) -> None:
        with pytest.raises(ValueError):
            SupervisorEventPipeline(
                reader=FakeReader(ChildEventReadResult()),
                deduper=FakeDeduper(),
                publisher=FakePublisher(),
                max_alerts_per_cycle=-1,
            )

    def test_max_alerts_per_cycle_bool(self) -> None:
        with pytest.raises(ValueError):
            SupervisorEventPipeline(
                reader=FakeReader(ChildEventReadResult()),
                deduper=FakeDeduper(),
                publisher=FakePublisher(),
                max_alerts_per_cycle=True,
            )


# ============================================================================
# 17. Invalid now_ms
# ============================================================================


class TestInvalidNowMs:
    @pytest.mark.asyncio
    async def test_now_ms_negative(self) -> None:
        pipeline = SupervisorEventPipeline(
            reader=FakeReader(ChildEventReadResult()),
            deduper=FakeDeduper(),
            publisher=FakePublisher(),
        )
        with pytest.raises(ValueError):
            await pipeline.process_once(now_ms=-1)

    @pytest.mark.asyncio
    async def test_now_ms_bool(self) -> None:
        pipeline = SupervisorEventPipeline(
            reader=FakeReader(ChildEventReadResult()),
            deduper=FakeDeduper(),
            publisher=FakePublisher(),
        )
        with pytest.raises(ValueError):
            await pipeline.process_once(now_ms=True)


# ============================================================================
# 18. Deduper called with payload None
# ============================================================================


class TestDeduperPayloadNone:
    @pytest.mark.asyncio
    async def test_payload_is_none(self) -> None:
        event = _make_child_event()
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        assert len(deduper.calls) == 1
        call = deduper.calls[0]
        assert call["payload"] is None


# ============================================================================
# 19. No full payload in alert body
# ============================================================================


class TestNoFullPayloadInBody:
    @pytest.mark.asyncio
    async def test_large_data_not_in_body(self) -> None:
        huge = "x" * 1000
        event = _make_child_event(
            payload={"symbol": "ETH", "severity": "INFO", "data": {"huge": huge}},
        )
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        await pipeline.process_once(now_ms=1000)

        alert = publisher.alerts[0]
        assert huge not in alert.body


# ============================================================================
# E05a-b: cycle limit enforced before building alerts
# ============================================================================


class TestCycleLimitDoesNotBuildDroppedAlerts:
    @pytest.mark.asyncio
    async def test_dropped_alert_not_built(self) -> None:
        """Dropped candidates must not have their content appear in any published alert."""
        events = [
            _make_child_event(
                event_type="WORKER_STARTED",
                payload={"symbol": "ETH", "severity": "INFO", "reason": "first"},
                source_path="path/0.jsonl",
            ),
            _make_child_event(
                event_type="WORKER_STOPPED",
                payload={
                    "symbol": "ETH",
                    "severity": "INFO",
                    "reason": "DROPPED_REASON_SHOULD_NOT_APPEAR",
                    "data": {"huge": "X" * 10000},
                },
                source_path="path/1.jsonl",
            ),
        ]
        reader = FakeReader(
            ChildEventReadResult(events=events, errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
            max_alerts_per_cycle=1,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.alerts_built == 2
        assert result.alerts_allowed == 1
        assert result.alerts_published == 1
        assert result.dropped_due_to_cycle_limit == 1
        assert len(deduper.calls) == 1
        assert len(publisher.alerts) == 1
        assert "DROPPED_REASON_SHOULD_NOT_APPEAR" not in publisher.alerts[0].body
        assert "X" * 10000 not in publisher.alerts[0].body


class TestCycleLimitAppliesToReadErrorsWithoutBuildingAll:
    @pytest.mark.asyncio
    async def test_read_errors_limited(self) -> None:
        """Cycle limit applies to read errors — dropped errors are not built."""
        errors = [
            _make_read_error(error_type="BAD_JSON"),
            _make_read_error(error_type="LINE_TOO_LONG"),
            _make_read_error(error_type="INVALID_EVENT_OBJECT"),
        ]
        reader = FakeReader(
            ChildEventReadResult(events=[], errors=errors)
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
            max_alerts_per_cycle=1,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.read_errors_seen == 3
        assert result.alerts_built == 3
        assert result.alerts_allowed == 1
        assert result.alerts_published == 1
        assert result.dropped_due_to_cycle_limit == 2
        assert len(deduper.calls) == 1
        assert len(publisher.alerts) == 1


# ============================================================================
# 20. Source guard
# ============================================================================


class TestSourceGuard:
    def test_no_forbidden_imports(self) -> None:
        """Ensure the pipeline source does not import any forbidden modules."""
        forbidden = [
            "Trader",
            "Strategy",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "EmailSender",
            "os.getenv",
            "load_dotenv",
            "src.live.workers",
            "src.trader",
            "src.strategies",
            "src.live.symbol_worker_app",
            "src.live.symbol_worker_factory",
            "execution_worker",
            "strategy_tick_worker",
            "account_position_sync_worker",
        ]
        source_text = _PIPELINE_SOURCE.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source_text, (
                f"Forbidden import/usage '{token}' found in supervisor_event_pipeline.py"
            )


# ============================================================================
# 21. No history / jsonl / state guard
# ============================================================================


class TestNoHistoryJsonlState:
    def test_no_forbidden_patterns(self) -> None:
        """Ensure the pipeline source does not contain forbidden patterns."""
        forbidden = [
            "JsonlOutbox",
            "write_json_atomic",
            "read_json_or_none",
            "history",
            "suppression_history",
            '.open("a"',
            "create_task",
            "sleep(",
        ]
        source_text = _PIPELINE_SOURCE.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source_text, (
                f"Forbidden pattern '{token}' found in supervisor_event_pipeline.py"
            )

    def test_no_full_alert_list_build(self) -> None:
        """Ensure process_once does not build a full alert list before the cycle limit."""
        source_text = _PIPELINE_SOURCE.read_text(encoding="utf-8")
        assert "alerts: list[SupervisorAlert]" not in source_text, (
            "Pipeline source must not pre-build a full alert list — "
            "cycle limit must be enforced before building each SupervisorAlert"
        )
        assert "alerts.append(" not in source_text, (
            "Pipeline source must not append to a pre-built alert list — "
            "cycle limit must be enforced before building each SupervisorAlert"
        )


# ============================================================================
# 22. Additional edge case: now_ms=None generates timestamp internally
# ============================================================================


class TestNowMsNone:
    @pytest.mark.asyncio
    async def test_generates_internal_timestamp(self) -> None:
        event = _make_child_event()
        reader = FakeReader(
            ChildEventReadResult(events=[event], errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once()  # now_ms=None

        # Should complete without error.
        assert result.events_seen == 1
        assert result.alerts_published == 1
        # The deduper should have received a now_ms that is a positive int.
        assert len(deduper.calls) == 1
        call = deduper.calls[0]
        assert isinstance(call["now_ms"], int)
        assert call["now_ms"] > 0


# ============================================================================
# 23. Additional edge case: mix of lifecycle and non-lifecycle events
# ============================================================================


class TestMixedEvents:
    @pytest.mark.asyncio
    async def test_filters_non_lifecycle(self) -> None:
        events = [
            _make_child_event(event_type="WORKER_STARTED"),
            _make_child_event(event_type="ORDER_FILLED"),
            _make_child_event(event_type="WORKER_STOPPED"),
            _make_child_event(event_type="TRADE_EXECUTED"),
        ]
        reader = FakeReader(
            ChildEventReadResult(events=events, errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.events_seen == 4
        assert result.alerts_built == 2  # only WORKER_STARTED and WORKER_STOPPED
        assert result.alerts_published == 2
        assert len(publisher.alerts) == 2

        published_types = {a.event_type for a in publisher.alerts}
        assert published_types == {"WORKER_STARTED", "WORKER_STOPPED"}
        assert "ORDER_FILLED" not in published_types
        assert "TRADE_EXECUTED" not in published_types


# ============================================================================
# 24. All lifecycle event types produce alerts
# ============================================================================


class TestAllLifecycleEventTypes:
    @pytest.mark.asyncio
    async def test_all_lifecycle_types(self) -> None:
        events = [
            _make_child_event(
                event_type=et,
                payload={"symbol": "ETH-USDT-SWAP", "data": {}},
            )
            for et in sorted(LIFECYCLE_EVENT_TYPES)
        ]
        reader = FakeReader(
            ChildEventReadResult(events=events, errors=[])
        )
        deduper = FakeDeduper(allowed=True)
        publisher = FakePublisher(result=True)

        pipeline = SupervisorEventPipeline(
            reader=reader,
            deduper=deduper,
            publisher=publisher,
        )
        result = await pipeline.process_once(now_ms=1000)

        assert result.alerts_built == len(LIFECYCLE_EVENT_TYPES)
        assert result.alerts_published == len(LIFECYCLE_EVENT_TYPES)
        published_types = {a.event_type for a in publisher.alerts}
        assert published_types == LIFECYCLE_EVENT_TYPES


# ============================================================================
# 25. SupervisorAlert dataclass is frozen
# ============================================================================


class TestSupervisorAlertFrozen:
    def test_frozen(self) -> None:
        alert = SupervisorAlert(
            symbol="TEST",
            event_type="WORKER_STARTED",
            severity="INFO",
            reason=None,
            subject="test subject",
            body="<html></html>",
        )
        with pytest.raises(Exception):
            alert.symbol = "CHANGED"  # type: ignore[misc]


# ============================================================================
# 26. SupervisorEventPipelineResult dataclass is frozen
# ============================================================================


class TestPipelineResultFrozen:
    def test_frozen(self) -> None:
        result = SupervisorEventPipelineResult(
            events_seen=0,
            read_errors_seen=0,
            alerts_built=0,
            alerts_allowed=0,
            alerts_suppressed=0,
            alerts_published=0,
            publish_failures=0,
            dropped_due_to_cycle_limit=0,
        )
        with pytest.raises(Exception):
            result.events_seen = 5  # type: ignore[misc]
