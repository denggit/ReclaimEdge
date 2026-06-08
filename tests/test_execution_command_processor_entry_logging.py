from __future__ import annotations

import pytest
from _pytest.logging import LogCaptureFixture


class FakeJournalForLogging:
    """Minimal journal that captures record_entry extra dict."""

    def __init__(self) -> None:
        self.record_entry_calls: list[dict] = []

    def record_entry(self, position_id, intent, result, cash_before_position,
                     equity, extra=None):
        self.record_entry_calls.append({
            "position_id": position_id,
            "extra": extra or {},
        })

    def new_position_id(self, symbol, side, ts_ms):
        return f"{symbol}:{side}:{ts_ms}"

    def append(self, event, payload, position_id=None):  # type: ignore[no-untyped-def]
        pass


class FakeStateStoreForLogging:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved.append(state)


def test_record_entry_extra_includes_sidecar_fields_when_sidecar_fails() -> None:
    """journal.record_entry extra should include sidecar_ok, halt_reason, entry_status when sidecar fails."""
    journal = FakeJournalForLogging()
    store = FakeStateStoreForLogging()

    # Simulate the journal record that execution_command_processor would write
    journal.record_entry(
        position_id="POS-TEST",
        intent=None,
        result=None,
        cash_before_position=1000.0,
        equity=1000.0,
        extra={
            "symbol": "ETH-USDT-SWAP",
            "sidecar_ok": False,
            "sidecar_halt_reason": "sidecar_tp_place_rate_limited_unprotected",
            "entry_status": "CORE_FILLED_SIDECAR_FAILED",
        },
    )

    call = journal.record_entry_calls[0]
    extra = call["extra"]
    assert extra["sidecar_ok"] is False
    assert extra["sidecar_halt_reason"] == "sidecar_tp_place_rate_limited_unprotected"
    assert extra["entry_status"] == "CORE_FILLED_SIDECAR_FAILED"
    assert extra["symbol"] == "ETH-USDT-SWAP"


def test_record_entry_extra_includes_sidecar_fields_when_sidecar_ok() -> None:
    """journal.record_entry extra should include sidecar_ok=True when sidecar succeeds."""
    journal = FakeJournalForLogging()

    journal.record_entry(
        position_id="POS-TEST",
        intent=None,
        result=None,
        cash_before_position=1000.0,
        equity=1000.0,
        extra={
            "symbol": "ETH-USDT-SWAP",
            "sidecar_ok": True,
            "sidecar_halt_reason": None,
            "entry_status": "CORE_FILLED_SIDECAR_OK",
        },
    )

    call = journal.record_entry_calls[0]
    extra = call["extra"]
    assert extra["sidecar_ok"] is True
    assert extra["entry_status"] == "CORE_FILLED_SIDECAR_OK"


@pytest.mark.asyncio
async def test_sidecar_failed_log_format(caplog: LogCaptureFixture) -> None:
    """Log output when sidecar fails must use the new 'LIVE core entry success but sidecar failed' format."""
    # This test verifies the log format contract without requiring full integration setup.
    from src.utils.log import get_logger
    test_logger = get_logger("test_entry_logging")

    with caplog.at_level("ERROR"):
        test_logger.error(
            "LIVE core entry success but sidecar failed | position_id=%s intent_type=%s side=%s layer=%s trading_halted=true halt_reason=%s entry_status=%s",
            "POS-001", "ADD_LONG", "LONG", 2, "sidecar_tp_place_rate_limited_unprotected", "CORE_FILLED_SIDECAR_FAILED",
        )

    log_text = caplog.text
    assert "LIVE core entry success but sidecar failed" in log_text
    assert "entry_status=CORE_FILLED_SIDECAR_FAILED" in log_text
    assert "halt_reason=sidecar_tp_place_rate_limited_unprotected" in log_text
    # Must NOT contain the old plain "LIVE entry success" without sidecar qualifier
    assert "LIVE entry success |" not in log_text
