#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_live_trader.py
@Description: Binance live trader for the main strategy path.

Satisfies ``LiveTraderProtocol``.  Uses ``BinanceBrokerClient``,
``BinanceBrokerSemanticExecutor``, and ``BinanceAlgoOrderClient`` to
execute real orders on Binance USD-M Futures.

No dry-run.  No shadow.  Risk is controlled by:
- Small account balance
- ``LIVE_MAX_ORDER_NOTIONAL_USDT``
- ``LIVE_MAX_POSITION_NOTIONAL_USDT``
- ``MAX_LIVE_EQUITY_USDT``
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from decimal import Decimal, ROUND_DOWN
from typing import Any

from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIENT_ORDER_ID_PREFIX = "RE_MAIN_"
SYMBOL = "ETHUSDT"
_ETH_CONTRACT_MULTIPLIER = Decimal("0.1")

# Default protective SL percentage (as fraction of entry price)
_DEFAULT_PROTECTIVE_SL_PCT = Decimal("0.006")


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------


def _unique_client_order_id(tag: str) -> str:
    """Generate a unique client order ID with ts_ns + counter.

    Format: RE_MAIN_<tag>_<ns>_<counter>
    """
    _unique_client_order_id._counter += 1  # type: ignore[attr-defined]
    return f"{CLIENT_ORDER_ID_PREFIX}{tag}_{time.time_ns()}_{_unique_client_order_id._counter}"  # type: ignore[attr-defined]


_unique_client_order_id._counter = 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# BinanceLiveTrader
# ---------------------------------------------------------------------------


class BinanceLiveTrader:
    """Binance live trader implementing ``LiveTraderProtocol``.

    Production construction reads ``EXCHANGE_API_KEY`` and
    ``EXCHANGE_API_SECRET`` from the environment and wires up
    ``AiohttpBinanceTransport`` + ``BinanceBrokerClient``.

    For testing, inject fake ``broker_client``, ``semantic_executor``,
    and ``algo_client``.
    """

    # ── Protocol attributes ────────────────────────────────────────────
    symbol = SYMBOL
    account_equity_usdt: float = 0.0
    position_contracts: Decimal = Decimal("0")
    contract_multiplier: Decimal = Decimal("0.1")
    contract_precision: Decimal = Decimal("0.01")
    min_contracts: Decimal = Decimal("0.01")
    leverage: int = 1

    @property
    def broker_exchange_name(self) -> str:
        return "binance"

    def __init__(
        self,
        *,
        broker_client: Any | None = None,
        semantic_executor: Any | None = None,
        algo_client: Any | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._broker_client = broker_client
        self._semantic_executor = semantic_executor
        self._algo_client = algo_client
        self._env = env if env is not None else os.environ
        self._transport: Any | None = None
        self._own_transport: bool = False

        # Managed order tracking (Fix 9: split by type)
        self.tp_order_id: str | None = None
        self.tp_order_ids: tuple[str, ...] = ()
        self._protective_sl_order_id: str | None = None
        self._protective_sl_client_algo_id: str | None = None
        # Regular TP order IDs only — entry IDs never go here
        self._managed_tp_order_ids: set[str] = set()
        # Algo SL clientAlgoIds
        self._managed_sl_client_algo_ids: set[str] = set()

        # Limit caps (parsed in initialize)
        self.max_live_equity_usdt: float = 30.0
        self._max_order_notional: Decimal = Decimal("0")
        self._max_position_notional: Decimal = Decimal("0")

        # Protective SL percentage
        raw_pct = self._env.get("LIVE_PROTECTIVE_SL_PCT", "0.006").strip()
        self._protective_sl_pct: Decimal = Decimal(raw_pct) if raw_pct else _DEFAULT_PROTECTIVE_SL_PCT

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        """Start the transport if it needs a session."""
        if self._transport is not None and hasattr(self._transport, "start"):
            await self._transport.start()

    async def close(self) -> None:
        """Close the transport / aiohttp session (best-effort)."""
        if self._transport is not None and hasattr(self._transport, "close"):
            try:
                await self._transport.close()
            except Exception:
                logger.exception("Error closing Binance transport")

    async def initialize(self) -> None:
        """Run preflight checks and wire up production clients if needed."""
        # ── 1. Run preflight ──────────────────────────────────────────
        from src.live.binance_live_preflight import (
            build_binance_live_preflight_report,
            format_binance_live_blocked_message,
        )

        report = build_binance_live_preflight_report(
            self._env,
            orders_globally_enabled=True,  # BinanceLiveTrader IS the wiring
        )
        if not report.ok:
            raise RuntimeError(format_binance_live_blocked_message(report))

        # ── 2. Parse limit caps ────────────────────────────────────────
        config = report.config
        self.leverage = config.leverage or 1
        self._max_order_notional = config.max_order_notional_usdt or Decimal("0")
        self._max_position_notional = config.max_position_notional_usdt or Decimal("0")
        self.max_live_equity_usdt = float(
            self._env.get("MAX_LIVE_EQUITY_USDT", "30")
        )

        # ── 3. Wire production clients if not injected ─────────────────
        if self._broker_client is None:
            await self._wire_production_clients()

        if self._broker_client is None or self._semantic_executor is None:
            raise RuntimeError(
                "BinanceLiveTrader: broker_client and semantic_executor are required"
            )

        # ── 4. Fetch equity (fail-fast) ─────────────────────────────────
        equity = await self.fetch_usdt_equity()
        self.account_equity_usdt = equity
        if equity > self.max_live_equity_usdt:
            raise RuntimeError(
                f"USDT equity {equity:.4f} > MAX_LIVE_EQUITY_USDT "
                f"{self.max_live_equity_usdt:.4f}. Refusing live trading."
            )

        # ── 5. Fetch position (fail-fast) ──────────────────────────────
        pos = await self.fetch_position_snapshot()
        self.position_contracts = pos.contracts

        # ── 6. Verify position mode / margin mode / leverage ───────────
        await self._verify_config()

        logger.warning(
            "BinanceLiveTrader initialized | symbol=%s leverage=%s equity=%.4f "
            "existing_side=%s existing_contracts=%s existing_avg=%.4f "
            "contract_multiplier=%s min_contracts=%s",
            self.symbol,
            self.leverage,
            equity,
            pos.side,
            self.position_contracts,
            pos.avg_entry_price,
            self.contract_multiplier,
            self.min_contracts,
        )

    async def _wire_production_clients(self) -> None:
        """Build real Binance clients from environment credentials."""
        api_key = self._env.get("EXCHANGE_API_KEY", "").strip()
        api_secret = self._env.get("EXCHANGE_API_SECRET", "").strip()

        if not api_key or not api_secret:
            raise RuntimeError(
                "EXCHANGE_API_KEY and EXCHANGE_API_SECRET must be set "
                "for Binance live trading"
            )

        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
        from src.exchanges.binance.algo_orders import BinanceAlgoOrderClient
        from src.exchanges.binance.client import BinanceBrokerClient
        from src.exchanges.binance.semantic_executor import BinanceBrokerSemanticExecutor

        self._transport = AiohttpBinanceTransport()
        self._own_transport = True

        self._broker_client = BinanceBrokerClient(
            api_key=api_key,
            api_secret=api_secret,
            transport=self._transport,
        )

        self._algo_client = BinanceAlgoOrderClient(
            api_key=api_key,
            api_secret=api_secret,
            transport=self._transport,
        )

        self._semantic_executor = BinanceBrokerSemanticExecutor(
            self._broker_client,
            algo_client=self._algo_client,
        )

    # ==================================================================
    # Fix 2: Real _verify_config
    # ==================================================================

    async def _verify_config(self) -> None:
        """Verify one-way / net mode, isolated margin, and leverage.

        Performs signed REST calls to confirm live account configuration
        matches requirements.  Does NOT auto-set anything.
        """
        if self._transport is None or self._broker_client is None:
            raise RuntimeError(
                "Cannot verify Binance config without transport and broker client"
            )

        # ── 1. Verify position mode (one-way / net) ──────────────────────
        await self._verify_position_mode()

        # ── 2. Fetch positionRisk item (used for margin + leverage) ──────
        risk = await self._fetch_position_risk_item()

        # ── 3. Verify margin mode ────────────────────────────────────────
        self._verify_margin_mode_from_risk(risk)

        # ── 4. Verify leverage (even when position is flat) ──────────────
        self._verify_leverage_from_risk(risk)

        logger.warning(
            "Binance config verified | position_mode=net margin_mode=isolated leverage=%s",
            self.leverage,
        )

    def _api_credentials(self) -> tuple[str, str]:
        """Return (api_key, api_secret) from env or broker client."""
        api_key = self._env.get("EXCHANGE_API_KEY", "").strip()
        api_secret = self._env.get("EXCHANGE_API_SECRET", "").strip()
        if (not api_key or not api_secret) and self._broker_client is not None:
            api_key = getattr(self._broker_client, "_api_key", "") or ""
            api_secret = getattr(self._broker_client, "_api_secret", "") or ""
        return api_key, api_secret

    async def _fetch_position_risk_item(self) -> dict[str, Any]:
        """Fetch the ETHUSDT item from ``GET /fapi/v2/positionRisk``.

        Returns the raw dict for the ETHUSDT symbol.  Raises RuntimeError
        on failure or if the item cannot be found.
        """
        if self._transport is None:
            raise RuntimeError("transport not configured for positionRisk fetch")

        from src.exchanges.binance.signing import (
            BINANCE_USDM_BASE_URL,
            build_signed_request,
        )

        api_key, api_secret = self._api_credentials()

        signed = build_signed_request(
            method="GET",
            path="/fapi/v2/positionRisk",
            params={"symbol": self.symbol},
            api_key=api_key,
            api_secret=api_secret,
            base_url=BINANCE_USDM_BASE_URL,
        )
        response = await self._transport.send(signed)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Binance positionRisk HTTP {response.status_code}: "
                f"{response.payload}"
            )

        if not isinstance(response.payload, list):
            raise RuntimeError(
                f"Binance positionRisk payload is not a list: "
                f"{type(response.payload)}"
            )

        for item in response.payload:
            if isinstance(item, dict) and item.get("symbol") == self.symbol:
                return item

        raise RuntimeError(
            f"No {self.symbol} item found in Binance positionRisk response"
        )

    def _verify_margin_mode_from_risk(self, risk: dict[str, Any]) -> None:
        """Verify margin mode is isolated from a positionRisk item."""
        margin_type = str(risk.get("marginType") or "").lower()
        if margin_type == "cross":
            raise RuntimeError(
                f"Binance margin mode is CROSS for {self.symbol}. "
                f"Only isolated mode is supported."
            )
        if margin_type != "isolated":
            raise RuntimeError(
                f"Binance margin mode is unrecognized for {self.symbol}: "
                f"{margin_type!r}. Only isolated mode is supported."
            )

    def _verify_leverage_from_risk(self, risk: dict[str, Any]) -> None:
        """Verify leverage matches ``LIVE_LEVERAGE``, even when flat."""
        actual_leverage = risk.get("leverage")
        if actual_leverage is None:
            raise RuntimeError(
                f"Binance positionRisk item missing 'leverage' for {self.symbol}"
            )
        if int(actual_leverage) != self.leverage:
            raise RuntimeError(
                f"Binance leverage mismatch for {self.symbol}: "
                f"actual={actual_leverage} expected={self.leverage}"
            )

    async def _verify_position_mode(self) -> None:
        """Verify position mode is one-way / net via Binance REST.

        Raises RuntimeError on HTTP errors, malformed payloads,
        missing keys, or hedge mode.
        """
        if self._transport is None:
            raise RuntimeError("transport not configured for position mode check")

        from src.exchanges.binance.signing import (
            BINANCE_USDM_BASE_URL,
            build_signed_request,
        )

        api_key, api_secret = self._api_credentials()

        signed = build_signed_request(
            method="GET",
            path="/fapi/v1/positionSide/dual",
            params={},
            api_key=api_key,
            api_secret=api_secret,
            base_url=BINANCE_USDM_BASE_URL,
        )
        response = await self._transport.send(signed)

        # ── HTTP error check ────────────────────────────────────────────
        if response.status_code >= 400:
            raise RuntimeError(
                f"Binance positionSide/dual HTTP {response.status_code}: "
                f"{response.payload}"
            )

        # ── Payload type check ──────────────────────────────────────────
        if not isinstance(response.payload, dict):
            raise RuntimeError(
                f"Binance positionSide/dual payload is not a dict: "
                f"{type(response.payload)}"
            )

        # ── Key existence check ─────────────────────────────────────────
        if "dualSidePosition" not in response.payload:
            raise RuntimeError(
                f"Binance positionSide/dual response missing 'dualSidePosition' key"
            )

        # ── Value check ─────────────────────────────────────────────────
        dual = response.payload.get("dualSidePosition")
        if dual is True:
            raise RuntimeError(
                "Binance position mode is HEDGE (dualSidePosition=true). "
                "Only one-way / net mode is supported."
            )

        # If dual is False or falsy (not True), net mode is active — OK.

    # ==================================================================
    # Fix 3 & 4: Position snapshot (fail-fast, enum mapping)
    # ==================================================================

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        """Fetch and map Binance position to ``PositionSnapshot``.

        Raises RuntimeError on fetch failure — never returns flat silently.
        """
        if self._broker_client is None:
            raise RuntimeError("broker_client not configured for position fetch")

        pos = await self._broker_client.fetch_position(self.symbol)

        if pos is None or pos.quantity <= 0:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        eth_qty = float(pos.quantity)
        contracts = self.eth_qty_to_contracts(pos.quantity)
        avg_entry = float(pos.average_entry_price or 0.0)

        # Fix 4: compare enum properly
        if pos.position_side == BrokerPositionSide.LONG:
            side: PositionSide | None = "LONG"
        elif pos.position_side == BrokerPositionSide.SHORT:
            side = "SHORT"
        elif pos.position_side == BrokerPositionSide.UNKNOWN:
            raise RuntimeError(
                f"Binance position side is UNKNOWN for {self.symbol}"
            )
        else:
            raise RuntimeError(
                f"Unsupported Binance position side: {pos.position_side}"
            )

        return PositionSnapshot(
            side=side,
            contracts=contracts,
            avg_entry_price=avg_entry,
            eth_qty=eth_qty,
            raw_pos=contracts,
        )

    async def fetch_usdt_equity(self) -> float:
        """Fetch USDT balance from Binance.  Raises on failure."""
        if self._transport is None:
            raise RuntimeError("transport not configured for equity fetch")

        from src.exchanges.binance.signing import (
            BINANCE_USDM_BASE_URL,
            build_signed_request,
        )

        api_key = self._env.get("EXCHANGE_API_KEY", "").strip()
        api_secret = self._env.get("EXCHANGE_API_SECRET", "").strip()

        if not api_key or not api_secret:
            if self._broker_client is not None:
                api_key = getattr(self._broker_client, "_api_key", "") or ""
                api_secret = getattr(self._broker_client, "_api_secret", "") or ""

        if not api_key or not api_secret:
            raise RuntimeError("API credentials not available for equity fetch")

        signed = build_signed_request(
            method="GET",
            path="/fapi/v2/balance",
            params={},
            api_key=api_key,
            api_secret=api_secret,
            base_url=BINANCE_USDM_BASE_URL,
        )
        response = await self._transport.send(signed)

        if isinstance(response.payload, list):
            for item in response.payload:
                if isinstance(item, dict) and item.get("asset") == "USDT":
                    balance = item.get("balance") or item.get("availableBalance")
                    if balance is not None:
                        return float(balance)
            raise RuntimeError(
                f"No USDT balance row found in Binance balance response"
            )

        raise RuntimeError(
            f"Binance balance response malformed: {type(response.payload)}"
        )

    # ==================================================================
    # Quantity conversion
    # ==================================================================

    def eth_qty_to_contracts(self, eth_qty: Decimal) -> Decimal:
        raw = eth_qty / _ETH_CONTRACT_MULTIPLIER
        return self._round_contracts(raw)

    def contracts_to_eth_qty(self, contracts: Decimal) -> Decimal:
        return contracts * _ETH_CONTRACT_MULTIPLIER

    def _round_contracts(self, contracts: Decimal) -> Decimal:
        precision = self.contract_precision
        return (contracts / precision).quantize(Decimal("1"), rounding=ROUND_DOWN) * precision

    @staticmethod
    def decimal_to_str(value: Decimal) -> str:
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"

    # ==================================================================
    # execute_intent — main entry point
    # ==================================================================

    async def execute_intent(self, intent: TradeIntent) -> LiveTradeResult:
        intent_type = intent.intent_type

        if intent_type in ("OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"):
            return await self._execute_open_or_add(intent)

        if intent_type == "UPDATE_TP":
            return await self._execute_update_tp(intent)

        if intent_type == "NEAR_TP_REDUCE":
            return await self._execute_near_tp_reduce(intent)

        if intent_type in ("MARKET_EXIT", "MARKET_EXIT_RUNNER"):
            return await self._execute_market_exit(intent)

        return LiveTradeResult(
            ok=False,
            action=intent_type,
            order_id=None,
            tp_order_id=None,
            contracts="0",
            tp_price="",
            message=f"unsupported_binance_intent: {intent_type}",
        )

    # ==================================================================
    # Fix 5, 7, 9: OPEN / ADD with TP + SL, unique IDs, split tracking
    # ==================================================================

    async def _execute_open_or_add(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

        side: PositionSide = intent.side
        eth_qty = Decimal(str(intent.size.eth_qty))
        contracts = self.eth_qty_to_contracts(eth_qty)
        contracts_str = self.decimal_to_str(contracts)

        # ── Notional check: single order ───────────────────────────────
        mark_price = Decimal(str(intent.price))
        order_notional = eth_qty * mark_price
        if self._max_order_notional > 0 and order_notional > self._max_order_notional:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"live_max_order_notional_exceeded: {order_notional} > {self._max_order_notional}",
            )

        # ── Notional check: projected position ─────────────────────────
        current_eth = self.contracts_to_eth_qty(self.position_contracts)
        projected_notional = (current_eth + eth_qty) * mark_price
        if self._max_position_notional > 0 and projected_notional > self._max_position_notional:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"live_max_position_notional_exceeded: {projected_notional} > {self._max_position_notional}",
            )

        is_add = intent.intent_type.startswith("ADD_")
        pos_side = BrokerPositionSide.LONG if side == "LONG" else BrokerPositionSide.SHORT
        action = BrokerSemanticAction.ADD_POSITION if is_add else BrokerSemanticAction.OPEN_POSITION

        # Fix 7: unique client_order_id
        entry_client_order_id = _unique_client_order_id("entry")

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=action,
            role=BrokerSemanticOrderRole.ENTRY if not is_add else BrokerSemanticOrderRole.ADD,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            client_order_id=entry_client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            logger.exception("Entry order failed")
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_failed: {exc}",
            )

        if not result.ok:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=result.order_id,
                tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_failed: {result.message}",
            )

        order_id = result.order_id

        # Fix 9: entry order ID does NOT go into managed cancel set
        # Only TP and SL IDs go into their respective sets.

        # ── Refresh position after entry ────────────────────────────────
        try:
            pos = await self.fetch_position_snapshot()
            self.position_contracts = pos.contracts if pos.contracts > 0 else self.position_contracts + contracts
        except Exception:
            logger.exception("Failed to refresh position after entry")
            self.position_contracts += contracts

        # ── Place TP (Fix 5) ────────────────────────────────────────────
        tp_ok = True
        tp_message = ""
        tp_order_id_val: str | None = None
        tp_order_ids_val: tuple[str, ...] = ()

        try:
            tp_result = await self._place_tp_for_intent(intent, contracts, pos_side)
            tp_ok = tp_result.get("ok", False)
            tp_message = tp_result.get("message", "")
            tp_order_id_val = tp_result.get("tp_order_id")
            tp_order_ids_val = tp_result.get("tp_order_ids", ())
        except Exception as exc:
            tp_ok = False
            tp_message = str(exc)

        # ── Place protective SL (Fix 5) ──────────────────────────────────
        sl_ok = False
        sl_order_id: str | None = None
        sl_price: str = ""
        sl_message: str = ""

        try:
            sl_price_float = self._resolve_sl_price(intent, float(mark_price))
            sl_price = self.price_to_str(sl_price_float)
            sl_result = await self._place_protective_sl(pos_side, contracts, Decimal(str(sl_price_float)))
            sl_ok = sl_result.get("ok", False)
            sl_order_id = sl_result.get("sl_order_id")
            sl_message = sl_result.get("message", "")
        except Exception as exc:
            sl_ok = False
            sl_message = str(exc)

        # ── Build result ────────────────────────────────────────────────
        if not tp_ok:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=order_id, tp_order_id=tp_order_id_val,
                contracts=contracts_str, tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_filled_but_tp_failed: {tp_message}",
                entry_filled=True, tp_ok=False,
                tp_order_ids=tp_order_ids_val,
                protective_sl_order_id=sl_order_id,
                protective_sl_price=sl_price,
                protective_sl_ok=sl_ok,
            )

        if not sl_ok:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=order_id, tp_order_id=tp_order_id_val,
                contracts=contracts_str, tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_filled_but_sl_failed: {sl_message}",
                entry_filled=True, tp_ok=True,
                tp_order_ids=tp_order_ids_val,
                protective_sl_order_id=sl_order_id,
                protective_sl_price=sl_price,
                protective_sl_ok=False,
            )

        return LiveTradeResult(
            ok=True, action=intent.intent_type,
            order_id=order_id, tp_order_id=tp_order_id_val,
            contracts=contracts_str, tp_price=self.price_to_str(intent.tp_price),
            message="market order placed and TP+SL protected",
            entry_filled=True, tp_ok=True,
            tp_order_ids=tp_order_ids_val,
            protective_sl_order_id=sl_order_id,
            protective_sl_price=sl_price,
            protective_sl_ok=True,
        )

    def _resolve_sl_price(self, intent: TradeIntent, entry_price: float) -> float:
        """Determine protective SL trigger price.

        Priority:
        1. Intent fields: protective_sl_price / sl_price / stop_loss_price
        2. Fallback: entry_price * (1 ± LIVE_PROTECTIVE_SL_PCT) depending on side
        """
        sl_candidate = (
            getattr(intent, "protective_sl_price", None)
            or getattr(intent, "sl_price", None)
            or getattr(intent, "stop_loss_price", None)
        )
        if sl_candidate is not None and sl_candidate > 0:
            return float(sl_candidate)

        pct = float(self._protective_sl_pct)
        if intent.side == "LONG":
            return entry_price * (1.0 - pct)
        else:
            return entry_price * (1.0 + pct)

    async def _place_protective_sl(
        self,
        pos_side: Any,
        contracts: Decimal,
        trigger_price: Decimal,
    ) -> dict[str, Any]:
        """Place a protective stop-loss via Binance Algo Order API."""
        exec = self._semantic_executor
        if exec is None:
            return {"ok": False, "message": "semantic_executor_not_configured"}

        from src.exchanges.semantic_models import (
            BrokerSemanticAction,
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
        )

        # Fix 7: unique client_algo_id
        sl_client_algo_id = _unique_client_order_id("sl")

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            trigger_price=trigger_price,
            reduce_only=True,
            client_order_id=sl_client_algo_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

        # Fix 6: result.order_id is clientAlgoId (unified in semantic executor)
        sl_order_id = result.order_id or sl_client_algo_id
        if sl_order_id:
            self._protective_sl_order_id = sl_order_id
            self._protective_sl_client_algo_id = sl_client_algo_id
            # Fix 9: SL IDs go into separate set
            self._managed_sl_client_algo_ids.add(sl_client_algo_id)

        return {
            "ok": result.ok,
            "sl_order_id": sl_order_id,
            "message": result.message,
        }

    async def _place_tp_for_intent(
        self,
        intent: TradeIntent,
        contracts: Decimal,
        pos_side: Any,
    ) -> dict[str, Any]:
        """Place a single reduce-only LIMIT TP order."""
        exec = self._semantic_executor
        if exec is None:
            return {"ok": False, "message": "semantic_executor_not_configured"}

        from src.exchanges.models import BrokerQuantityUnit
        from src.exchanges.semantic_models import (
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
            BrokerSemanticAction,
        )

        # Fix 7: unique client_order_id
        tp_client_order_id = _unique_client_order_id("tp")

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal(str(intent.tp_price)),
            reduce_only=True,
            client_order_id=tp_client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

        tp_order_id = result.order_id
        if tp_order_id:
            self.tp_order_id = tp_order_id
            self.tp_order_ids = (tp_order_id,)
            # Fix 9: TP IDs go into TP set only
            self._managed_tp_order_ids.add(tp_order_id)

        return {
            "ok": result.ok,
            "tp_order_id": tp_order_id,
            "tp_order_ids": (tp_order_id,) if tp_order_id else (),
            "message": result.message,
        }

    # ==================================================================
    # UPDATE_TP
    # ==================================================================

    async def _execute_update_tp(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action="UPDATE_TP",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        tp_price_str = self.price_to_str(intent.tp_price)

        # ── Cancel existing TP(s) ──────────────────────────────────────
        for oid in self._managed_tp_order_ids:
            if oid:
                try:
                    await exec.execute(self._make_cancel_tp_request(oid))
                except Exception:
                    logger.exception("Failed to cancel old TP: %s", oid)
        self._managed_tp_order_ids.clear()

        # ── Determine current position side and contracts ──────────────
        pos = await self.fetch_position_snapshot()
        if not pos.has_position or pos.side is None:
            return LiveTradeResult(
                ok=False, action="UPDATE_TP",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price=tp_price_str,
                message="no_position_for_tp_update",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

        # Fix 4: compare string side from PositionSnapshot
        pos_side = BrokerPositionSide.LONG if pos.side == "LONG" else BrokerPositionSide.SHORT
        contracts = self.position_contracts if self.position_contracts > 0 else pos.contracts
        # Fix 7: unique client_order_id
        tp_client_order_id = _unique_client_order_id("tp")

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal(str(intent.tp_price)),
            reduce_only=True,
            client_order_id=tp_client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return LiveTradeResult(
                ok=False, action="UPDATE_TP",
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price=tp_price_str,
                message=f"tp_update_failed: {exc}",
            )

        tp_order_id = result.order_id
        if tp_order_id:
            self.tp_order_id = tp_order_id
            self.tp_order_ids = (tp_order_id,)
            self._managed_tp_order_ids.add(tp_order_id)

        return LiveTradeResult(
            ok=result.ok, action="UPDATE_TP",
            order_id=None, tp_order_id=tp_order_id,
            contracts=self.decimal_to_str(contracts),
            tp_price=tp_price_str,
            tp_order_ids=(tp_order_id,) if tp_order_id else (),
            message=result.message or "tp_updated",
        )

    def _make_cancel_tp_request(self, order_id: str) -> Any:
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest
        return BrokerSemanticRequest(
            exchange=self._semantic_executor.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            order_id=order_id,
        )

    def _make_cancel_sl_request(self, client_algo_id: str) -> Any:
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest
        return BrokerSemanticRequest(
            exchange=self._semantic_executor.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            order_id=client_algo_id,
        )

    # ==================================================================
    # NEAR_TP_REDUCE
    # ==================================================================

    async def _execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        pos = await self.fetch_position_snapshot()
        if not pos.has_position or pos.side is None:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="no_position_for_reduce",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

        pos_side = BrokerPositionSide.LONG if pos.side == "LONG" else BrokerPositionSide.SHORT
        reduce_ratio = Decimal(str(intent.near_tp_reduce_ratio)) if intent.near_tp_reduce_ratio > 0 else Decimal("1")
        reduce_contracts = self._round_contracts(self.position_contracts * reduce_ratio)
        if reduce_contracts > self.position_contracts:
            reduce_contracts = self.position_contracts

        if reduce_contracts <= 0:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="zero_reduce_contracts",
            )

        reduce_eth = self.contracts_to_eth_qty(reduce_contracts)
        mark_price = Decimal(str(intent.price))
        if self._max_order_notional > 0 and reduce_eth * mark_price > self._max_order_notional:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(reduce_contracts),
                tp_price="",
                message="live_max_order_notional_exceeded",
            )

        # Fix 7: unique ID
        client_order_id = _unique_client_order_id("reduce")

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.MARKET_EXIT,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=pos_side,
            quantity=reduce_contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            reduce_only=True,
            client_order_id=client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(reduce_contracts),
                tp_price="",
                message=f"near_tp_reduce_failed: {exc}",
            )

        if result.ok:
            self.position_contracts -= reduce_contracts

        return LiveTradeResult(
            ok=result.ok, action="NEAR_TP_REDUCE",
            order_id=result.order_id, tp_order_id=None,
            contracts=self.decimal_to_str(reduce_contracts), tp_price="",
            contracts_before=self.decimal_to_str(self.position_contracts + reduce_contracts),
            contracts_reduced=self.decimal_to_str(reduce_contracts),
            contracts_after=self.decimal_to_str(self.position_contracts),
            near_tp_exit_all=(reduce_ratio >= Decimal("1")),
            reduce_filled=result.ok,
            message=result.message or "near_tp_reduced",
        )

    # ==================================================================
    # Fix 9: MARKET_EXIT with split cancel
    # ==================================================================

    async def _execute_market_exit(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        pos = await self.fetch_position_snapshot()
        if not pos.has_position or pos.side is None or pos.contracts <= 0:
            await self._cancel_managed_orders()
            self.position_contracts = Decimal("0")
            return LiveTradeResult(
                ok=True, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="already_flat",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import BrokerSemanticAction, BrokerSemanticOrderRole, BrokerSemanticRequest

        pos_side = BrokerPositionSide.LONG if pos.side == "LONG" else BrokerPositionSide.SHORT
        contracts = pos.contracts
        # Fix 7: unique ID
        client_order_id = _unique_client_order_id("exit")

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.MARKET_EXIT
                if intent.intent_type == "MARKET_EXIT"
                else BrokerSemanticAction.MARKET_EXIT_RUNNER,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            reduce_only=True,
            client_order_id=client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price="",
                message=f"market_exit_failed: {exc}",
            )

        # Fix 9: cancel TP and SL separately
        await self._cancel_managed_orders()

        if result.ok:
            self.position_contracts = Decimal("0")

        return LiveTradeResult(
            ok=result.ok, action=intent.intent_type,
            order_id=result.order_id, tp_order_id=None,
            contracts=self.decimal_to_str(contracts), tp_price="",
            message=result.message or "market_exit_executed",
        )

    async def _cancel_managed_orders(self) -> None:
        """Cancel all known managed TP and SL orders separately.

        Fix 9: TP via CANCEL_REDUCE_ONLY_TP, SL via CANCEL_PROTECTIVE_STOP.
        Does NOT cancel entry orders or unknown orders.
        """
        exec = self._semantic_executor
        if exec is None:
            return

        # Cancel TP orders
        for oid in list(self._managed_tp_order_ids):
            if not oid:
                continue
            try:
                await exec.execute(self._make_cancel_tp_request(oid))
            except Exception:
                logger.exception("Failed to cancel managed TP: %s", oid)
            self._managed_tp_order_ids.discard(oid)

        # Cancel SL orders (via algo cancel)
        for cid in list(self._managed_sl_client_algo_ids):
            if not cid:
                continue
            try:
                await exec.execute(self._make_cancel_sl_request(cid))
            except Exception:
                logger.exception("Failed to cancel managed SL: %s", cid)
            self._managed_sl_client_algo_ids.discard(cid)

        self.tp_order_id = None
        self.tp_order_ids = ()
        self._protective_sl_order_id = None
        self._protective_sl_client_algo_id = None

    # ==================================================================
    # Recovery helpers
    # ==================================================================

    async def fetch_broker_open_orders(self) -> tuple[Any, ...]:
        if self._semantic_executor is None:
            return ()
        from src.exchanges.semantic_models import BrokerSemanticRequest, BrokerSemanticAction, BrokerSemanticOrderRole
        result = await self._semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self._semantic_executor.exchange,
                symbol=self.symbol,
                action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
                role=BrokerSemanticOrderRole.RECOVERY,
            )
        )
        return tuple(result.orders)

    async def fetch_broker_algo_orders(self) -> tuple[Any, ...]:
        if self._semantic_executor is None:
            return ()
        from src.exchanges.semantic_models import BrokerSemanticRequest, BrokerSemanticAction, BrokerSemanticOrderRole
        result = await self._semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self._semantic_executor.exchange,
                symbol=self.symbol,
                action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
                role=BrokerSemanticOrderRole.RECOVERY,
            )
        )
        return tuple(result.orders)

    async def recover_broker_open_orders(self) -> tuple[Any, ...]:
        if self._semantic_executor is None:
            return ()
        from src.exchanges.semantic_models import BrokerSemanticRequest, BrokerSemanticAction, BrokerSemanticOrderRole
        result = await self._semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self._semantic_executor.exchange,
                symbol=self.symbol,
                action=BrokerSemanticAction.RECOVER_OPEN_ORDERS,
                role=BrokerSemanticOrderRole.RECOVERY,
            )
        )
        return tuple(result.orders)

    async def fetch_broker_position(self) -> Any | None:
        if self._broker_client is None:
            return None
        return await self._broker_client.fetch_position(self.symbol)

    async def fetch_broker_open_order_raws(self) -> list[dict[str, Any]]:
        orders = await self.fetch_broker_open_orders()
        return [dict(getattr(o, "raw", {})) for o in orders]

    async def fetch_broker_algo_order_raws(self) -> list[dict[str, Any]]:
        orders = await self.fetch_broker_algo_orders()
        return [dict(getattr(o, "raw", {})) for o in orders]

    async def recover_broker_open_order_raws(self) -> list[dict[str, Any]]:
        orders = await self.recover_broker_open_orders()
        return [dict(getattr(o, "raw", {})) for o in orders]

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        return await self.fetch_broker_open_order_raws()

    async def fetch_order_status(self, order_id: str) -> dict[str, Any]:
        try:
            if self._broker_client is None:
                return {"order_id": order_id, "status": "UNKNOWN"}
            open_orders = await self._broker_client.fetch_open_orders(self.symbol)
            for o in open_orders:
                if o.order_id == order_id:
                    return {"order_id": order_id, "status": "OPEN"}
            return {"order_id": order_id, "status": "NOT_FOUND"}
        except Exception:
            return {"order_id": order_id, "status": "UNKNOWN"}

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        return await self.fetch_order_status(order_id)

    def mark_flat(self) -> None:
        """Reset all position / order tracking state."""
        self.position_contracts = Decimal("0")
        self.tp_order_id = None
        self.tp_order_ids = ()
        self._protective_sl_order_id = None
        self._protective_sl_client_algo_id = None
        self._managed_tp_order_ids.clear()
        self._managed_sl_client_algo_ids.clear()
