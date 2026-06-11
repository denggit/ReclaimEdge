from __future__ import annotations

import math
import os
import time as _time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.execution import order_specs
from src.execution.trader_types import (
    LiveTradeResult,
    PositionSnapshot,
    TraderInstrumentMetadata,
)
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)


# ── BTC instrument metadata (hardcoded, not from TOML) ───────────────────────
# OKX BTC-USDT-SWAP public instruments:
#   ctVal = 0.01, lotSz = 0.01, minSz = 0.01
_BTC_CONTRACT_MULTIPLIER = Decimal("0.01")
_BTC_CONTRACT_PRECISION = Decimal("0.01")
_BTC_MIN_CONTRACTS = Decimal("0.01")

_BTC_METADATA = TraderInstrumentMetadata(
    inst_id="BTC-USDT-SWAP",
    contract_multiplier=_BTC_CONTRACT_MULTIPLIER,
    contract_precision=_BTC_CONTRACT_PRECISION,
    min_contracts=_BTC_MIN_CONTRACTS,
)


@dataclass(frozen=True)
class PaperTraderConfig:
    """Immutable configuration for a PaperTrader instance."""
    symbol: str
    account_equity_usdt: float
    td_mode: str = "isolated"
    leverage: str = "20"
    pos_side_mode: str = "net"
    contract_multiplier: Decimal = _BTC_CONTRACT_MULTIPLIER
    contract_precision: Decimal = _BTC_CONTRACT_PRECISION
    min_contracts: Decimal = _BTC_MIN_CONTRACTS


def _parse_paper_symbols(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return ("BTC-USDT-SWAP",)
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return ("BTC-USDT-SWAP",)
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return tuple(result)


class PaperTrader:
    """Paper/dry-run trader that simulates order execution locally.

    PaperTrader implements the same duck-typed interface as ``Trader``
    but **never** sends a real OKX private order.  It is used for
    BTC-USDT-SWAP dry-run mode (G08).

    Rules:
    * No OKX API calls, no aiohttp.
    * No private write rate limiter.
    * Does NOT require ``LIVE_TRADING=true``.
    * Only ``BTC-USDT-SWAP`` is allowed (hard gate).
    * Sidecar methods are no-ops that return safe defaults.
    """

    def __init__(self) -> None:
        # ── symbol & paper gate ──────────────────────────────────────────
        self.symbol = os.getenv("OKX_INST_ID", "").strip()
        if not self.symbol:
            self.symbol = "BTC-USDT-SWAP"
        if self.symbol != "BTC-USDT-SWAP":
            raise RuntimeError(
                f"PaperTrader only supports BTC-USDT-SWAP, got {self.symbol!r}"
            )

        paper_symbols_raw = os.getenv("RECLAIM_PAPER_SYMBOLS", "BTC-USDT-SWAP")
        paper_symbols = _parse_paper_symbols(paper_symbols_raw)
        if self.symbol not in paper_symbols:
            raise RuntimeError(
                f"Symbol {self.symbol!r} is not in RECLAIM_PAPER_SYMBOLS: {paper_symbols!r}"
            )

        # ── instrument metadata ──────────────────────────────────────────
        self.contract_multiplier = _BTC_CONTRACT_MULTIPLIER
        self.contract_precision = _BTC_CONTRACT_PRECISION
        self.min_contracts = _BTC_MIN_CONTRACTS
        self.instrument_metadata = _BTC_METADATA

        # ── trading params ───────────────────────────────────────────────
        self.td_mode = "isolated"
        self.leverage = "20"
        self.pos_side_mode = "net"
        self.live_trading = False
        self.paper_trading = True

        # ── account ──────────────────────────────────────────────────────
        equity_raw = os.getenv("PAPER_ACCOUNT_EQUITY_USDT", "1000")
        try:
            self.account_equity_usdt = float(equity_raw)
        except ValueError:
            self.account_equity_usdt = 1000.0

        # ── position tracking ────────────────────────────────────────────
        self.position_contracts = Decimal("0")
        self._current_side: PositionSide | None = None
        self.tp_order_id: str | None = None
        self.near_tp_protective_sl_order_id: str | None = None
        self.middle_runner_protective_sl_order_id: str | None = None
        self.three_stage_post_tp1_protective_sl_order_id: str | None = None
        self.trend_runner_sl_order_id: str | None = None
        self.middle_bucket_fast_sl_order_id: str | None = None

        logger.warning(
            "PAPER_TRADER_ENABLED | symbol=%s equity=%.1f no_okx_private_orders=true",
            self.symbol,
            self.account_equity_usdt,
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        equity = await self.fetch_usdt_equity()
        self.account_equity_usdt = equity
        position = await self.fetch_position_snapshot()
        self.position_contracts = position.contracts
        logger.warning(
            "PAPER trader initialized | symbol=%s equity=%.4f existing_side=%s existing_contracts=%s",
            self.symbol,
            equity,
            position.side,
            self.position_contracts,
        )

    # ── equity / position ────────────────────────────────────────────────

    async def fetch_usdt_equity(self) -> float:
        return self.account_equity_usdt

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        if self.position_contracts <= 0:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))
        contracts = self.position_contracts
        eth_qty = float(contracts * self.contract_multiplier)
        return PositionSnapshot(
            side=self._current_side,
            contracts=contracts,
            avg_entry_price=0.0,
            eth_qty=eth_qty,
            raw_pos=contracts if self._current_side == "LONG" else -contracts,
        )

    async def fetch_position_contracts(self) -> Decimal:
        return self.position_contracts

    # ── execution ────────────────────────────────────────────────────────

    async def execute_intent(self, intent: TradeIntent) -> LiveTradeResult:
        if intent.intent_type == "NEAR_TP_REDUCE":
            return await self.execute_near_tp_reduce(intent)
        if intent.intent_type == "MARKET_EXIT_RUNNER":
            return await self.execute_market_exit_runner(intent)
        if intent.intent_type == "UPDATE_TP":
            return await self.replace_take_profit(intent)

        # OPEN / ADD
        ts_ms = int(_time.time() * 1000)
        layer_idx = getattr(intent, "layer_index", 0) or 0
        contracts = self.eth_qty_to_contracts(Decimal(str(intent.size.eth_qty)))

        self._current_side = intent.side
        self.position_contracts += contracts

        order_id = f"paper-entry-{ts_ms}-{layer_idx}"

        logger.warning(
            "PAPER_ORDER_SIMULATED | symbol=%s intent_type=%s side=%s layer=%s contracts=%s",
            self.symbol,
            intent.intent_type,
            intent.side,
            layer_idx,
            self.decimal_to_str(contracts),
        )

        try:
            tp = await self.replace_take_profit(intent)
            if not tp.ok:
                return LiveTradeResult(
                    ok=False,
                    action=intent.intent_type,
                    order_id=order_id,
                    tp_order_id=tp.tp_order_id,
                    contracts=self.decimal_to_str(contracts),
                    tp_price=tp.tp_price,
                    message=f"entry_filled_but_tp_failed: {tp.message}",
                    entry_filled=True,
                    tp_ok=False,
                    tp_order_ids=tp.tp_order_ids,
                    protective_sl_order_id=tp.protective_sl_order_id,
                    protective_sl_price=tp.protective_sl_price,
                    protective_sl_ok=tp.protective_sl_ok,
                )
            return LiveTradeResult(
                ok=True,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=tp.tp_order_id,
                contracts=self.decimal_to_str(contracts),
                tp_price=tp.tp_price,
                message="paper order placed and take-profit protected",
                entry_filled=True,
                tp_ok=True,
                tp_order_ids=tp.tp_order_ids,
                protective_sl_order_id=tp.protective_sl_order_id,
                protective_sl_price=tp.protective_sl_price,
                protective_sl_ok=tp.protective_sl_ok,
            )
        except Exception as exc:
            logger.exception("Paper entry failed during TP simulation")
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_filled_but_tp_exception: {exc}",
                entry_filled=True,
                tp_ok=False,
            )

    async def replace_take_profit(self, intent: TradeIntent) -> LiveTradeResult:
        ts_ms = int(_time.time() * 1000)
        layer_idx = getattr(intent, "layer_index", 0) or 0
        tp_id = f"paper-tp-{ts_ms}-{layer_idx}"
        self.tp_order_id = tp_id
        return LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id=tp_id,
            contracts=self.decimal_to_str(self.position_contracts),
            tp_price=self.price_to_str(intent.tp_price),
            message="paper take-profit placed",
            entry_filled=False,
            tp_ok=True,
        )

    async def execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        reduce_qty = Decimal(str(intent.size.eth_qty))
        reduce_contracts = self.eth_qty_to_contracts(reduce_qty)
        contracts_before = self.position_contracts
        self.position_contracts = max(Decimal("0"), self.position_contracts - reduce_contracts)
        if self.position_contracts <= 0:
            self.mark_flat()
        contracts_after = self.position_contracts

        logger.warning(
            "PAPER_NEAR_TP_REDUCE | symbol=%s contracts_before=%s contracts_reduced=%s contracts_after=%s",
            self.symbol,
            self.decimal_to_str(contracts_before),
            self.decimal_to_str(reduce_contracts),
            self.decimal_to_str(contracts_after),
        )

        return LiveTradeResult(
            ok=True,
            action="NEAR_TP_REDUCE",
            order_id=None,
            tp_order_id=self.tp_order_id,
            contracts=self.decimal_to_str(reduce_contracts),
            tp_price=self.price_to_str(intent.tp_price),
            message="paper near-tp reduce executed",
            reduce_filled=True,
            contracts_before=self.decimal_to_str(contracts_before),
            contracts_reduced=self.decimal_to_str(reduce_contracts),
            contracts_after=self.decimal_to_str(contracts_after),
        )

    async def execute_market_exit_runner(self, intent: TradeIntent) -> LiveTradeResult:
        contracts_before = self.position_contracts
        self.mark_flat()

        logger.warning(
            "PAPER_MARKET_EXIT_RUNNER | symbol=%s contracts_exited=%s",
            self.symbol,
            self.decimal_to_str(contracts_before),
        )

        return LiveTradeResult(
            ok=True,
            action="MARKET_EXIT_RUNNER",
            order_id=None,
            tp_order_id=None,
            contracts=self.decimal_to_str(contracts_before),
            tp_price="",
            message="paper market exit runner executed",
            near_tp_exit_all=True,
            contracts_before=self.decimal_to_str(contracts_before),
            contracts_reduced=self.decimal_to_str(contracts_before),
            contracts_after="0",
        )

    # ── sidecar (no-ops that return safe defaults) ───────────────────────

    async def place_sidecar_market_order(self, *, side: PositionSide, eth_qty: float) -> dict[str, Any]:
        ts_ms = int(_time.time() * 1000)
        contracts = self.eth_qty_to_contracts(Decimal(str(eth_qty)))
        return {
            "order_id": f"paper-sidecar-{ts_ms}",
            "contracts": self.decimal_to_str(contracts),
            "qty": float(contracts * self.contract_multiplier),
        }

    async def place_sidecar_fixed_take_profit(
        self,
        *,
        side: PositionSide,
        contracts: str | Decimal,
        tp_price: float,
        client_order_id: str | None = None,
    ) -> str:
        ts_ms = int(_time.time() * 1000)
        return f"paper-sidecar-tp-{ts_ms}"

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        return True

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "OPEN", "filled_qty": None, "avg_fill_price": None}

    # ── order queries (return empty lists) ───────────────────────────────

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        return []

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        return []

    async def cancel_existing_reduce_only_orders(self) -> None:
        pass

    # ── state management ─────────────────────────────────────────────────

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")
        self._current_side = None
        self.tp_order_id = None
        self.near_tp_protective_sl_order_id = None
        self.middle_runner_protective_sl_order_id = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None
        self.middle_bucket_fast_sl_order_id = None

    # ── conversion helpers ───────────────────────────────────────────────

    def eth_qty_to_contracts(self, eth_qty: Decimal) -> Decimal:
        raw_contracts = eth_qty / self.contract_multiplier
        contracts = self.round_contracts_down(raw_contracts)
        if contracts < self.min_contracts:
            raise RuntimeError(f"Order size {contracts} contracts is below minimum {self.min_contracts}")
        return contracts

    def round_contracts_down(self, contracts: Decimal) -> Decimal:
        return order_specs.round_contracts_down(
            contracts=contracts, contract_precision=self.contract_precision
        )

    def pos_side(self, side: str) -> str | None:
        return order_specs.pos_side_for_mode(side=side, pos_side_mode=self.pos_side_mode)

    @staticmethod
    def decimal_to_str(value: Decimal | str | int | float) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"

    # ── protective stop stubs (safe no-ops) ─────────────────────────────

    async def set_leverage(self) -> None:
        pass

    # request() is deliberately omitted — PaperTrader does not make HTTP calls.

    # ── protective stop methods (used by tp_sl_execution_manager) ────────
    # PaperTrader does NOT have a TpSlExecutionManager, so all protective
    # stop methods are safe no-ops that return simulated success.

    async def place_near_tp_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal | str | int | float,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ts_ms = int(_time.time() * 1000)
        order_id = f"paper-protective-sl-{ts_ms}"
        self.near_tp_protective_sl_order_id = order_id
        return True, order_id, ""

    async def place_middle_runner_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ts_ms = int(_time.time() * 1000)
        order_id = f"paper-middle-runner-sl-{ts_ms}"
        self.middle_runner_protective_sl_order_id = order_id
        return True, order_id, ""

    async def place_middle_bucket_fast_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ts_ms = int(_time.time() * 1000)
        order_id = f"paper-middle-bucket-fast-sl-{ts_ms}"
        self.middle_bucket_fast_sl_order_id = order_id
        return True, order_id, ""

    async def place_trend_runner_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ts_ms = int(_time.time() * 1000)
        order_id = f"paper-trend-runner-sl-{ts_ms}"
        self.trend_runner_sl_order_id = order_id
        return True, order_id, ""

    async def place_three_stage_post_tp1_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ts_ms = int(_time.time() * 1000)
        order_id = f"paper-three-stage-post-tp1-sl-{ts_ms}"
        self.three_stage_post_tp1_protective_sl_order_id = order_id
        return True, order_id, ""

    async def verify_near_tp_protective_stop(
        self, algo_id: str, side: PositionSide, contracts: Decimal, stop_price: float
    ) -> bool:
        return True

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_middle_bucket_fast_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def market_exit_remaining_position_with_retries(
        self,
        side: PositionSide,
        retry_count: int,
        *,
        context: str = "generic",
        retry_interval_seconds: float | None = None,
    ) -> tuple[bool, str]:
        contracts_before = self.position_contracts
        self.mark_flat()
        logger.warning(
            "PAPER_MARKET_EXIT_REMAINING | symbol=%s side=%s contracts=%s context=%s",
            self.symbol,
            side,
            self.decimal_to_str(contracts_before),
            context,
        )
        return True, f"paper-market-exit-{int(_time.time() * 1000)}"

    # ── TP/SL helpers used by execution path ─────────────────────────────

    async def _cleanup_after_market_exit(self) -> None:
        pass

    async def _cleanup_after_near_tp_market_exit(self) -> None:
        pass

    async def _cancel_existing_take_profit_orders_for_intent(self, intent: TradeIntent) -> None:
        pass

    async def _cancel_stale_runner_protective_stops_for_degrade(self, intent: TradeIntent) -> None:
        pass

    def _protected_order_ids_from_intent(self, intent: TradeIntent) -> set[str]:
        return set()

    @staticmethod
    def _split_order_ids(value: str | None) -> set[str]:
        return set()

    def _managed_core_contracts_from_intent(self, intent: TradeIntent) -> Decimal | None:
        return None

    def _build_take_profit_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return []

    def _build_three_stage_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        return []

    def _trend_runner_sl_contracts(self, intent: TradeIntent, net_contracts_for_sl: Decimal) -> Decimal:
        return net_contracts_for_sl

    async def _place_reduce_only_take_profit_orders(
        self, intent: TradeIntent, specs: list[tuple[str, Decimal, float]]
    ) -> list[str]:
        return [f"paper-tp-spec-{int(_time.time() * 1000)}" for _ in specs]

    def _reduce_only_tp_order_body(self, side: PositionSide, contracts: Decimal, price: float) -> dict[str, Any]:
        return {}

    def _reduce_only_market_order_body(self, side: PositionSide, contracts: Decimal) -> dict[str, Any]:
        return {}

    def _near_tp_protective_sl_algo_body(self, side: PositionSide, contracts: Decimal, stop_price: float) -> dict[str, Any]:
        return {}

    def _near_tp_fallback_conditional_close_body(self, side: PositionSide, contracts: Decimal, stop_price: float) -> dict[str, Any]:
        return {}

    def _near_tp_protective_stop_matches(
        self, item: dict[str, Any], algo_id: str, side: PositionSide, contracts: Decimal, stop_price: float
    ) -> bool:
        return True

    def _tp_price_summary(self, specs: list[tuple[str, Decimal, float]]) -> str:
        return "paper"

    async def _cancel_unverified_near_tp_algo(self, algo_id: str, *, phase: str) -> None:
        pass

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        return f"paper-extract-{int(_time.time() * 1000)}"

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        return f"paper-algo-{int(_time.time() * 1000)}"

    @staticmethod
    def _to_decimal(value: Decimal | str | int | float) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
