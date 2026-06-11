from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


VALID_WORKER_MODES = frozenset({"live", "paper"})
SUPPORTED_SUPERVISOR_SYMBOLS = frozenset({
    "ETH-USDT-SWAP",
    "BTC-USDT-SWAP",
})


@dataclass(frozen=True)
class SymbolWorkerPlan:
    symbol: str
    worker_mode: str
    child_name: str
    child_env: dict[str, str]
    heartbeat_path: Path
    event_outbox_path: Path | None


def validate_supported_supervisor_symbol(symbol: str) -> str:
    normalized = symbol.strip()
    if not normalized:
        raise ValueError("symbol must not be empty")
    if normalized not in SUPPORTED_SUPERVISOR_SYMBOLS:
        raise ValueError(
            f"unsupported supervisor symbol {normalized!r}; "
            f"supported={sorted(SUPPORTED_SUPERVISOR_SYMBOLS)!r}"
        )
    return normalized


def _normalize_worker_mode(mode: str, *, source: str) -> str:
    normalized = mode.strip().lower()
    if not normalized:
        raise ValueError(f"{source} mode must not be empty")
    if normalized not in VALID_WORKER_MODES:
        raise ValueError(
            f"{source} mode must be one of {sorted(VALID_WORKER_MODES)!r}, "
            f"got {mode!r}"
        )
    return normalized


def _default_worker_mode(base_env: Mapping[str, str]) -> str:
    raw = str(base_env.get("RECLAIM_WORKER_MODE", "")).strip()
    if not raw:
        return "live"
    return _normalize_worker_mode(raw, source="RECLAIM_WORKER_MODE")


def parse_worker_modes(base_env: Mapping[str, str]) -> dict[str, str]:
    """Parse RECLAIM_WORKER_MODES into a per-symbol mode mapping.

    Format: ``SYMBOL:live,SYMBOL:paper``.  Symbols and modes are trimmed;
    modes are lower-cased.  The fallback default is handled by
    :func:`worker_mode_for_symbol`.
    """
    raw = str(base_env.get("RECLAIM_WORKER_MODES", "")).strip()
    if not raw:
        return {}

    parsed: dict[str, str] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "RECLAIM_WORKER_MODES entries must use SYMBOL:MODE format, "
                f"got {item!r}"
            )
        raw_symbol, raw_mode = item.split(":", 1)
        symbol = raw_symbol.strip()
        if not symbol:
            raise ValueError("RECLAIM_WORKER_MODES symbol must not be empty")
        if symbol in parsed:
            raise ValueError(f"RECLAIM_WORKER_MODES contains duplicate symbol {symbol!r}")
        mode = _normalize_worker_mode(raw_mode, source=f"RECLAIM_WORKER_MODES[{symbol}]")
        parsed[symbol] = mode
    return parsed


def worker_mode_for_symbol(symbol: str, base_env: Mapping[str, str]) -> str:
    modes = parse_worker_modes(base_env)
    return modes.get(symbol, _default_worker_mode(base_env))


def build_symbol_worker_plans(
    symbols: Sequence[str],
    *,
    base_env: Mapping[str, str],
    runtime_dir: Path,
    heartbeat_dir: Path,
    event_dir: Path,
) -> list[SymbolWorkerPlan]:
    if not symbols:
        raise ValueError("symbols must not be empty")

    modes = parse_worker_modes(base_env)
    default_mode = _default_worker_mode(base_env)
    plans: list[SymbolWorkerPlan] = []
    seen: set[str] = set()

    for raw_symbol in symbols:
        symbol = validate_supported_supervisor_symbol(str(raw_symbol))
        if symbol in seen:
            raise ValueError(f"duplicate symbol {symbol!r}")
        seen.add(symbol)

        mode = modes.get(symbol, default_mode)
        child_env = {str(key): str(value) for key, value in base_env.items()}
        child_env.update(
            {
                "OKX_INST_ID": symbol,
                "RECLAIM_SYMBOL": symbol,
                "RECLAIM_SYMBOLS": symbol,
                "RECLAIM_WORKER_MODE": mode,
            }
        )

        plans.append(
            SymbolWorkerPlan(
                symbol=symbol,
                worker_mode=mode,
                child_name=f"reclaim-worker-{symbol}",
                child_env=child_env,
                heartbeat_path=Path(heartbeat_dir) / f"{symbol}.heartbeat.json",
                event_outbox_path=Path(event_dir) / f"worker_events_{symbol}.jsonl",
            )
        )

    # Keep the parameter visible to callers and linters; path ownership remains
    # in ReclaimSupervisorConfig/RuntimePaths.
    Path(runtime_dir)
    return plans
