from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from src.strategies.boll_cvd_reclaim_strategy import PositionSide


def _decimal_from_metadata_value(value: object, *, field_name: str) -> Decimal:
    """Convert a metadata constructor value to Decimal with validation.

    - ``Decimal`` is returned as-is
    - ``str`` / ``int`` / ``float`` are converted
    - ``bool`` is rejected (booleans are a subclass of int)
    - ``None`` or any other type raises ``ValueError``
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be a boolean, got {value!r}")
    if isinstance(value, (str, int, float)):
        return Decimal(str(value))
    raise ValueError(
        f"{field_name} must be Decimal, str, int, or float, got {type(value).__name__}: {value!r}"
    )


@dataclass(frozen=True)
class TraderInstrumentMetadata:
    """Immutable per-instrument contract parameters for live trading.

    This is intentionally self-contained: it does not read env, call the
    network, or depend on ``config/*``.  It exists so that different
    instruments (ETH vs BTC) can supply different multiplier / precision /
    min‑contract values without changing any trading logic.
    """

    inst_id: str
    contract_multiplier: Decimal
    contract_precision: Decimal
    min_contracts: Decimal

    def __post_init__(self) -> None:
        # --- inst_id ---
        if not isinstance(self.inst_id, str) or not self.inst_id.strip():
            raise ValueError(f"inst_id must be a non-empty string, got {self.inst_id!r}")

        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "inst_id", self.inst_id.strip())

        # --- numeric fields ---
        multiplier = _decimal_from_metadata_value(self.contract_multiplier, field_name="contract_multiplier")
        if multiplier <= 0:
            raise ValueError(f"contract_multiplier must be > 0, got {multiplier}")
        object.__setattr__(self, "contract_multiplier", multiplier)

        precision = _decimal_from_metadata_value(self.contract_precision, field_name="contract_precision")
        if precision <= 0:
            raise ValueError(f"contract_precision must be > 0, got {precision}")
        object.__setattr__(self, "contract_precision", precision)

        min_cts = _decimal_from_metadata_value(self.min_contracts, field_name="min_contracts")
        if min_cts <= 0:
            raise ValueError(f"min_contracts must be > 0, got {min_cts}")
        object.__setattr__(self, "min_contracts", min_cts)


@dataclass(frozen=True)
class TraderMarketSettings:
    """Immutable per-instrument market / leverage parameters for live trading.

    This is intentionally self-contained: it does not read env, call the
    network, or depend on ``config/*``.  It exists so that different
    instruments (ETH vs BTC) can supply different td_mode / pos_side_mode /
    leverage values without changing any trading logic.

    When *not* provided to ``Trader``, the current env-based defaults
    (``OKX_TD_MODE``, ``LEVERAGE``, ``OKX_POS_SIDE_MODE``) are used.
    """

    inst_id: str
    td_mode: str
    pos_side_mode: str
    leverage: Decimal

    def __post_init__(self) -> None:
        # --- inst_id ---
        if not isinstance(self.inst_id, str) or not self.inst_id.strip():
            raise ValueError(f"inst_id must be a non-empty string, got {self.inst_id!r}")
        object.__setattr__(self, "inst_id", self.inst_id.strip())

        # --- td_mode ---
        if not isinstance(self.td_mode, str) or not self.td_mode.strip():
            raise ValueError(f"td_mode must be a non-empty string, got {self.td_mode!r}")
        object.__setattr__(self, "td_mode", self.td_mode.strip())

        # --- pos_side_mode ---
        if not isinstance(self.pos_side_mode, str) or not self.pos_side_mode.strip():
            raise ValueError(
                f"pos_side_mode must be a non-empty string, got {self.pos_side_mode!r}"
            )
        object.__setattr__(self, "pos_side_mode", self.pos_side_mode.strip())

        # --- leverage ---
        lev = _decimal_from_metadata_value(self.leverage, field_name="leverage")
        if lev <= 0:
            raise ValueError(f"leverage must be > 0, got {lev}")
        object.__setattr__(self, "leverage", lev)


@dataclass(frozen=True)
class LiveTradeResult:
    ok: bool
    action: str
    order_id: Optional[str]
    tp_order_id: Optional[str]
    contracts: str
    tp_price: str
    message: str
    entry_filled: bool = False
    tp_ok: bool = False
    tp_order_ids: tuple[str, ...] = ()
    protective_sl_order_id: Optional[str] = None
    protective_sl_price: str = ""
    protective_sl_ok: bool = False
    contracts_before: str = ""
    contracts_reduced: str = ""
    contracts_after: str = ""
    near_tp_exit_all: bool = False
    reduce_filled: bool = False
    middle_bucket_split_executed: bool | None = None
    middle_bucket_split_disabled_reason: str | None = None
    middle_bucket_split_actual_order_mode: str | None = None


@dataclass(frozen=True)
class PositionSnapshot:
    side: Optional[PositionSide]
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal

    @property
    def has_position(self) -> bool:
        return self.side is not None and self.contracts > 0
