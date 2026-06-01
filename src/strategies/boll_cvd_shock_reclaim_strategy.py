from __future__ import annotations

import logging

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    TradeIntent,
)

logger = logging.getLogger(__name__)


class BollCvdShockReclaimStrategy(BollCvdReclaimStrategy):
    """BOLL reclaim strategy gated by a relative shock before armed state.

    Two-stage design:
    1. Shock stage only opens/refreshes armed state.
       - LOWER_ARMED requires price below lower band and cvd.down_burst.
       - UPPER_ARMED requires price above upper band and cvd.up_burst.
    2. Reclaim stage opens/adds only after armed, near the extreme, with CVD
       reclaim/stall confirmation inherited from BollCvdReclaimStrategy.

    This prevents slow grinding moves along the BOLL band from arming entries.
    """

    def __init__(self, config: BollCvdReclaimStrategyConfig, sizer: SimplePositionSizer):
        super().__init__(config, sizer)

    def on_tick(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> list[TradeIntent]:
        intents: list[TradeIntent] = []

        # Existing armed state still needs expiry/middle reset even if BOLL width
        # switch turns off later. But new armed state must respect switch=True.
        self._expire_armed_state(ts_ms)
        if boll.alert_switch_on:
            self._update_shock_armed_state(price, ts_ms, boll, cvd)
        else:
            self._reset_armed_if_middle_reclaimed(price, boll)

        tp_intent = self._maybe_update_tp(price, ts_ms, boll, cvd)
        if tp_intent is not None:
            intents.append(tp_intent)

        if not boll.alert_switch_on:
            return intents

        if not self._cooldown_ok(ts_ms):
            return intents

        if self._long_setup(price, cvd):
            intent = self._maybe_open_or_add_long(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        if self._short_setup(price, cvd):
            intent = self._maybe_open_or_add_short(price, ts_ms, boll, cvd)
            if intent is not None:
                intents.append(intent)

        return intents

    def _update_shock_armed_state(self, price: float, ts_ms: int, boll: BollSnapshot, cvd: CvdSnapshot) -> None:
        if price < boll.lower:
            if not cvd.down_burst:
                if self.state.lower_armed:
                    self.state.lower_extreme_price = min(self.state.lower_extreme_price or price, price)
                return
            if not self.state.lower_armed:
                self.state.lower_armed = True
                self.state.lower_armed_ts_ms = ts_ms
                self.state.lower_extreme_price = price
                logger.info(
                    "LOWER_ARMED | reason=down_shock price=%.4f lower=%.4f middle=%.4f move_ratio=%.2f volume_ratio=%.2f burst_range=%.5f baseline_range=%.5f max_entry_distance=%.4f%% max_armed=%ss",
                    price,
                    boll.lower,
                    boll.middle,
                    cvd.burst_move_ratio,
                    cvd.burst_volume_ratio,
                    cvd.burst_range_pct,
                    cvd.baseline_range_pct,
                    self.config.max_entry_distance_from_extreme_pct * 100,
                    self.config.max_armed_seconds,
                )
            else:
                old_extreme = self.state.lower_extreme_price or price
                self.state.lower_extreme_price = min(old_extreme, price)
                if self.state.lower_extreme_price < old_extreme:
                    logger.info(
                        "LOWER_ARMED_EXTREME_UPDATED | reason=down_shock extreme=%.4f price=%.4f move_ratio=%.2f volume_ratio=%.2f",
                        self.state.lower_extreme_price,
                        price,
                        cvd.burst_move_ratio,
                        cvd.burst_volume_ratio,
                    )
            self.state.lower_last_burst_ts_ms = ts_ms
            if self.state.upper_armed:
                logger.info("UPPER_ARMED_RESET | reason=opposite_lower_shock price=%.4f", price)
            self._reset_upper_armed()
            return

        if price > boll.upper:
            if not cvd.up_burst:
                if self.state.upper_armed:
                    self.state.upper_extreme_price = max(self.state.upper_extreme_price or price, price)
                return
            if not self.state.upper_armed:
                self.state.upper_armed = True
                self.state.upper_armed_ts_ms = ts_ms
                self.state.upper_extreme_price = price
                logger.info(
                    "UPPER_ARMED | reason=up_shock price=%.4f upper=%.4f middle=%.4f move_ratio=%.2f volume_ratio=%.2f burst_range=%.5f baseline_range=%.5f max_entry_distance=%.4f%% max_armed=%ss",
                    price,
                    boll.upper,
                    boll.middle,
                    cvd.burst_move_ratio,
                    cvd.burst_volume_ratio,
                    cvd.burst_range_pct,
                    cvd.baseline_range_pct,
                    self.config.max_entry_distance_from_extreme_pct * 100,
                    self.config.max_armed_seconds,
                )
            else:
                old_extreme = self.state.upper_extreme_price or price
                self.state.upper_extreme_price = max(old_extreme, price)
                if self.state.upper_extreme_price > old_extreme:
                    logger.info(
                        "UPPER_ARMED_EXTREME_UPDATED | reason=up_shock extreme=%.4f price=%.4f move_ratio=%.2f volume_ratio=%.2f",
                        self.state.upper_extreme_price,
                        price,
                        cvd.burst_move_ratio,
                        cvd.burst_volume_ratio,
                    )
            self.state.upper_last_burst_ts_ms = ts_ms
            if self.state.lower_armed:
                logger.info("LOWER_ARMED_RESET | reason=opposite_upper_shock price=%.4f", price)
            self._reset_lower_armed()
            return

        self._reset_armed_if_middle_reclaimed(price, boll)

    def _reset_armed_if_middle_reclaimed(self, price: float, boll: BollSnapshot) -> None:
        if self.state.lower_armed and price >= boll.middle:
            logger.info("LOWER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_lower_armed()
        if self.state.upper_armed and price <= boll.middle:
            logger.info("UPPER_ARMED_RESET | reason=middle_reclaimed price=%.4f middle=%.4f", price, boll.middle)
            self._reset_upper_armed()

    def _long_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.lower_armed or self.state.lower_extreme_price is None:
            return False
        if self.state.lower_last_burst_ts_ms <= self.state.last_order_ts_ms:
            return False
        return super()._long_setup(price, cvd)

    def _short_setup(self, price: float, cvd: CvdSnapshot) -> bool:
        if not self.state.upper_armed or self.state.upper_extreme_price is None:
            return False
        if self.state.upper_last_burst_ts_ms <= self.state.last_order_ts_ms:
            return False
        return super()._short_setup(price, cvd)

    def _reset_lower_armed(self) -> None:
        super()._reset_lower_armed()
        self.state.lower_last_burst_ts_ms = 0

    def _reset_upper_armed(self) -> None:
        super()._reset_upper_armed()
        self.state.upper_last_burst_ts_ms = 0
