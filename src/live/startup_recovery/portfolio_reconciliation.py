from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.execution.trader_types import PositionSnapshot
from src.portfolio.capital_ledger import CapitalLedgerSnapshot, SymbolCapitalState
from src.reporting.live_state_store import LivePositionState

ReconciliationSeverity = Literal["OK", "WARN", "CRITICAL"]
ReconciliationAction = Literal["NONE", "HALT_NEW_RISK"]


@dataclass(frozen=True)
class StartupReconciliationIssue:
    code: str
    severity: ReconciliationSeverity
    message: str


@dataclass(frozen=True)
class StartupReconciliationResult:
    inst_id: str
    severity: ReconciliationSeverity
    action: ReconciliationAction
    okx_has_position: bool
    saved_has_position: bool
    ledger_is_active: bool
    okx_side: str | None
    saved_side: str | None
    ledger_side: str | None
    saved_layers: int
    ledger_used_layers: int
    ledger_plan_exists: bool
    issues: tuple[StartupReconciliationIssue, ...]

    @property
    def should_halt_new_risk(self) -> bool:
        return self.action == "HALT_NEW_RISK"


def _side_to_str(side: object) -> str | None:
    if side is None:
        return None
    value = getattr(side, "value", side)
    if value is None:
        return None
    text = str(value).upper()
    if text in {"LONG", "SHORT"}:
        return text
    return text


def _aggregate_severity(
    issues: list[StartupReconciliationIssue],
) -> ReconciliationSeverity:
    if any(i.severity == "CRITICAL" for i in issues):
        return "CRITICAL"
    if any(i.severity == "WARN" for i in issues):
        return "WARN"
    return "OK"


def _issue(
    code: str,
    severity: ReconciliationSeverity,
    message: str,
) -> StartupReconciliationIssue:
    return StartupReconciliationIssue(
        code=code,
        severity=severity,
        message=message,
    )


def _ledger_state_for_symbol(
    *, inst_id: str, ledger_snapshot: CapitalLedgerSnapshot
) -> SymbolCapitalState:
    return ledger_snapshot.symbols.get(inst_id, SymbolCapitalState())


def reconcile_startup_state(
    *,
    inst_id: str,
    position: PositionSnapshot,
    saved_state: LivePositionState | None,
    ledger_snapshot: CapitalLedgerSnapshot,
) -> StartupReconciliationResult:
    okx_has_position = bool(position.has_position)
    okx_side = _side_to_str(position.side) if okx_has_position else None

    saved_layers = (
        int(getattr(saved_state, "layers", 0) or 0)
        if saved_state is not None
        else 0
    )
    saved_side = (
        _side_to_str(getattr(saved_state, "side", None))
        if saved_state is not None
        else None
    )
    saved_has_position = (
        saved_state is not None and saved_layers > 0 and saved_side is not None
    )

    ledger_state = _ledger_state_for_symbol(
        inst_id=inst_id,
        ledger_snapshot=ledger_snapshot,
    )
    ledger_state_text = str(ledger_state.state or "").upper()
    ledger_used_layers = int(ledger_state.used_layers or 0)
    ledger_side = _side_to_str(ledger_state.side)
    ledger_is_active = (
        ledger_state_text != "FLAT"
        or ledger_used_layers > 0
        or ledger_side is not None
    )
    ledger_plan_exists = (
        ledger_state.position_plan_id is not None
        and len(ledger_state.planned_main_contracts) > 0
        and int(ledger_state.plan_max_layers or 0) >= 1
    )

    issues: list[StartupReconciliationIssue] = []

    if okx_has_position and not saved_has_position:
        issues.append(
            _issue(
                "OKX_POSITION_WITHOUT_SAVED_STATE",
                "WARN",
                f"{inst_id} has an exchange position but no active saved live state.",
            )
        )

    if okx_has_position and (ledger_state_text == "FLAT" or ledger_used_layers <= 0):
        issues.append(
            _issue(
                "OKX_POSITION_LEDGER_FLAT",
                "CRITICAL",
                f"{inst_id} has an exchange position but ledger is flat or has no used layers.",
            )
        )

    if ledger_is_active and ledger_used_layers > 0 and not okx_has_position:
        issues.append(
            _issue(
                "LEDGER_ACTIVE_BUT_OKX_FLAT",
                "CRITICAL",
                f"{inst_id} ledger is active but exchange position is flat.",
            )
        )

    if saved_has_position and not okx_has_position:
        issues.append(
            _issue(
                "SAVED_STATE_ACTIVE_BUT_OKX_FLAT",
                "CRITICAL",
                f"{inst_id} saved live state is active but exchange position is flat.",
            )
        )

    sides = [
        side
        for side in (
            okx_side,
            saved_side if saved_has_position else None,
            ledger_side if ledger_is_active else None,
        )
        if side is not None
    ]
    if len(sides) >= 2 and len(set(sides)) > 1:
        issues.append(
            _issue(
                "SIDE_MISMATCH",
                "CRITICAL",
                f"{inst_id} startup sides disagree between exchange, saved state, and ledger.",
            )
        )

    if (
        saved_layers > 0
        and ledger_used_layers > 0
        and saved_layers != ledger_used_layers
    ):
        issues.append(
            _issue(
                "LAYER_MISMATCH",
                "CRITICAL",
                f"{inst_id} saved layers differ from ledger used layers.",
            )
        )

    if ledger_is_active and not ledger_plan_exists:
        issues.append(
            _issue(
                "LEDGER_ACTIVE_MISSING_PLAN",
                "CRITICAL",
                f"{inst_id} ledger is active but does not contain a complete position plan.",
            )
        )

    if ledger_is_active and len(ledger_state.planned_main_contracts) != int(
        ledger_state.plan_max_layers or 0
    ):
        issues.append(
            _issue(
                "LEDGER_PLAN_LENGTH_MISMATCH",
                "CRITICAL",
                f"{inst_id} ledger planned_main_contracts length does not match plan_max_layers.",
            )
        )

    severity = _aggregate_severity(issues)
    action: ReconciliationAction = "NONE" if severity == "OK" else "HALT_NEW_RISK"

    return StartupReconciliationResult(
        inst_id=inst_id,
        severity=severity,
        action=action,
        okx_has_position=okx_has_position,
        saved_has_position=saved_has_position,
        ledger_is_active=ledger_is_active,
        okx_side=okx_side,
        saved_side=saved_side,
        ledger_side=ledger_side,
        saved_layers=saved_layers,
        ledger_used_layers=ledger_used_layers,
        ledger_plan_exists=ledger_plan_exists,
        issues=tuple(issues),
    )
