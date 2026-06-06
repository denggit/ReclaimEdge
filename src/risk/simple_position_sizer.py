from __future__ import annotations

import os
from dataclasses import dataclass

from src.position_management.sidecar.model import calculate_core_margin_pct


@dataclass(frozen=True)
class SimplePositionSizerConfig:
    dry_run_equity_usdt: float = 1000.0
    layer_margin_pct: float = 0.03
    leverage: float = 50.0
    layer_multiplier_step: float = 0.15
    sidecar_enabled: bool = False
    sidecar_margin_pct: float = 0.01
    sidecar_tp_pct: float = 0.004
    sidecar_close_when_core_flat: bool = True
    sidecar_order_status_check_seconds: float = 5.0
    sidecar_max_legs: int = 10
    sidecar_skip_first_layer: bool = True

    @classmethod
    def from_env(cls) -> "SimplePositionSizerConfig":
        config = cls(
            dry_run_equity_usdt=float(os.getenv("DRY_RUN_EQUITY_USDT", "1000")),
            layer_margin_pct=float(os.getenv("LAYER_MARGIN_PCT", "0.03")),
            leverage=float(os.getenv("LEVERAGE", "50")),
            sidecar_enabled=_env_bool("SIDECAR_ENABLED", False),
            sidecar_margin_pct=float(os.getenv("SIDECAR_MARGIN_PCT", "0.01")),
            sidecar_tp_pct=float(os.getenv("SIDECAR_TP_PCT", "0.004")),
            sidecar_close_when_core_flat=_env_bool("SIDECAR_CLOSE_WHEN_CORE_FLAT", True),
            sidecar_order_status_check_seconds=float(os.getenv("SIDECAR_ORDER_STATUS_CHECK_SECONDS", "5")),
            sidecar_max_legs=int(os.getenv("SIDECAR_MAX_LEGS", "10")),
            sidecar_skip_first_layer=_env_bool("SIDECAR_SKIP_FIRST_LAYER", True),
        )
        config.validate_sidecar()
        return config

    @classmethod
    def from_account_equity(cls, account_equity_usdt: float) -> "SimplePositionSizerConfig":
        config = cls(
            dry_run_equity_usdt=account_equity_usdt,
            layer_margin_pct=float(os.getenv("LAYER_MARGIN_PCT", "0.03")),
            leverage=float(os.getenv("LEVERAGE", "50")),
            sidecar_enabled=_env_bool("SIDECAR_ENABLED", False),
            sidecar_margin_pct=float(os.getenv("SIDECAR_MARGIN_PCT", "0.01")),
            sidecar_tp_pct=float(os.getenv("SIDECAR_TP_PCT", "0.004")),
            sidecar_close_when_core_flat=_env_bool("SIDECAR_CLOSE_WHEN_CORE_FLAT", True),
            sidecar_order_status_check_seconds=float(os.getenv("SIDECAR_ORDER_STATUS_CHECK_SECONDS", "5")),
            sidecar_max_legs=int(os.getenv("SIDECAR_MAX_LEGS", "10")),
            sidecar_skip_first_layer=_env_bool("SIDECAR_SKIP_FIRST_LAYER", True),
        )
        config.validate_sidecar()
        return config

    @property
    def core_margin_pct(self) -> float:
        return calculate_core_margin_pct(self.layer_margin_pct, self.sidecar_enabled, self.sidecar_margin_pct)

    def validate_sidecar(self) -> None:
        if not self.sidecar_enabled:
            return
        if self.sidecar_margin_pct <= 0:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MARGIN_PCT > 0")
        if self.sidecar_margin_pct >= self.layer_margin_pct:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MARGIN_PCT < LAYER_MARGIN_PCT")
        if self.sidecar_tp_pct <= 0:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_TP_PCT > 0")
        if self.sidecar_max_legs < 1:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MAX_LEGS >= 1")
        max_layers = int(os.getenv("MAX_LAYERS", "3"))
        if self.sidecar_max_legs < max_layers:
            raise RuntimeError("SIDECAR_ENABLED=true requires SIDECAR_MAX_LEGS >= MAX_LAYERS")


@dataclass(frozen=True)
class PositionSize:
    margin_usdt: float
    notional_usdt: float
    eth_qty: float
    layer_index: int
    layer_multiplier: float


class SimplePositionSizer:
    def __init__(self, config: SimplePositionSizerConfig):
        self.config = config

    def update_account_equity(self, account_equity_usdt: float) -> None:
        self.config = SimplePositionSizerConfig(
            dry_run_equity_usdt=account_equity_usdt,
            layer_margin_pct=self.config.layer_margin_pct,
            leverage=self.config.leverage,
            layer_multiplier_step=self.config.layer_multiplier_step,
            sidecar_enabled=self.config.sidecar_enabled,
            sidecar_margin_pct=self.config.sidecar_margin_pct,
            sidecar_tp_pct=self.config.sidecar_tp_pct,
            sidecar_close_when_core_flat=self.config.sidecar_close_when_core_flat,
            sidecar_order_status_check_seconds=self.config.sidecar_order_status_check_seconds,
            sidecar_max_legs=self.config.sidecar_max_legs,
            sidecar_skip_first_layer=self.config.sidecar_skip_first_layer,
        )

    @property
    def account_equity_usdt(self) -> float:
        return self.config.dry_run_equity_usdt

    def calculate(self, price: float, layer_index: int = 1) -> PositionSize:
        safe_layer_index = max(int(layer_index), 1)
        multiplier = 1.0 + (safe_layer_index - 1) * self.config.layer_multiplier_step
        base_margin = self.config.dry_run_equity_usdt * self.config.core_margin_pct
        margin = base_margin * multiplier
        notional = margin * self.config.leverage
        eth_qty = notional / price if price > 0 else 0.0
        return PositionSize(
            margin_usdt=margin,
            notional_usdt=notional,
            eth_qty=eth_qty,
            layer_index=safe_layer_index,
            layer_multiplier=multiplier,
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
