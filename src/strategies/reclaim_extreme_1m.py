"""1m closed-candle fractal extreme detection for Reclaim V2.

Replaces tick-level confirmed swing extremes with 1-minute fractal extremes
(2-left, 2-right).  A fractal extreme is only emitted after the two
candles on the right side have closed, giving a confirmation delay of
approximately 2-3 minutes.

Definitions
-----------
LOWER_EXTREME:
    The middle candle among 5 consecutive closed 1m candles whose **low**
    is the lowest of all 5.

UPPER_EXTREME:
    The middle candle among 5 consecutive closed 1m candles whose **high**
    is the highest of all 5.

This module is pure logic:

- No env reads
- No exchange calls
- No TradeIntent creation
- No strategy state access
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExtremeSide = Literal["LOWER", "UPPER"]


# ---------------------------------------------------------------------------
# value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReclaimOneMinuteCandle:
    """A single completed 1-minute OHLC candle with anchored CVD.

    Attributes
    ----------
    start_ts_ms : int
        Millisecond timestamp of the minute boundary (floor).
    end_ts_ms : int
        Millisecond timestamp of the next minute boundary (exclusive).
    open / high / low / close : float
        OHLC prices over the 1m interval.
    anchored_cvd_open / anchored_cvd_high / anchored_cvd_low / anchored_cvd_close : float
        Anchored cumulative CVD sampled at the respective OHLC points.
    high_ts_ms / low_ts_ms : int
        Exact tick timestamps when high/low were recorded.
    """

    start_ts_ms: int
    end_ts_ms: int
    open: float
    high: float
    low: float
    close: float

    anchored_cvd_open: float
    anchored_cvd_high: float
    anchored_cvd_low: float
    anchored_cvd_close: float

    high_ts_ms: int
    low_ts_ms: int


@dataclass(frozen=True)
class ReclaimFractalExtreme:
    """A confirmed 1m fractal swing extreme event.

    Produced when a 2L/2R fractal pattern completes — the middle candle
    of 5 consecutive closed 1m candles is an extreme.

    Attributes
    ----------
    side : Literal["LOWER", "UPPER"]
    price : float
        The extreme price (low for LOWER, high for UPPER).
    anchored_cvd : float
        Anchored CVD at the time of the extreme.
        LOWER → ``candle.anchored_cvd_low``; UPPER → ``candle.anchored_cvd_high``.
    candle_start_ts_ms / candle_end_ts_ms : int
        Bounds of the candidate (middle) candle.
    extreme_ts_ms : int
        Exact tick timestamp of the extreme within the candle.
    confirm_ts_ms : int
        Timestamp when the fractal was confirmed (when the rightmost
        candle closed).
    confirm_reason : str
        Always ``"fractal_1m_2l2r"``.
    """

    side: ExtremeSide
    price: float
    anchored_cvd: float
    candle_start_ts_ms: int
    candle_end_ts_ms: int
    extreme_ts_ms: int
    confirm_ts_ms: int
    confirm_reason: str  # "fractal_1m_2l2r"


# ---------------------------------------------------------------------------
# 1m candle builder
# ---------------------------------------------------------------------------


class ReclaimOneMinuteCandleBuilder:
    """Aggregates ticks into 1-minute OHLC candles with anchored CVD.

    When ``update()`` detects that a tick has crossed into a new minute,
    it returns the **previous** minute's completed candle and begins
    building the new one.

    Usage::

        builder = ReclaimOneMinuteCandleBuilder()
        for tick in ticks:
            candle = builder.update(price=tick.price, anchored_cvd=tick.cvd, ts_ms=tick.ts)
            if candle is not None:
                # feed to fractal tracker
                ...
    """

    def __init__(self) -> None:
        self._candle_start_ts_ms: int = 0
        self._open: float = 0.0
        self._high: float = float("-inf")
        self._low: float = float("inf")
        self._close: float = 0.0

        self._anchored_cvd_open: float = 0.0
        self._anchored_cvd_high: float = 0.0
        self._anchored_cvd_low: float = 0.0
        self._anchored_cvd_close: float = 0.0

        self._high_ts_ms: int = 0
        self._low_ts_ms: int = 0

        self._has_data: bool = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state (e.g. on strategy reset)."""
        self._candle_start_ts_ms = 0
        self._open = 0.0
        self._high = float("-inf")
        self._low = float("inf")
        self._close = 0.0
        self._anchored_cvd_open = 0.0
        self._anchored_cvd_high = 0.0
        self._anchored_cvd_low = 0.0
        self._anchored_cvd_close = 0.0
        self._high_ts_ms = 0
        self._low_ts_ms = 0
        self._has_data = False

    def update(
        self,
        *,
        price: float,
        anchored_cvd: float,
        ts_ms: int,
    ) -> ReclaimOneMinuteCandle | None:
        """Process one tick.

        Returns the **previous** minute's closed candle if the tick has
        advanced into a new minute boundary; ``None`` otherwise.
        """
        minute_start = (ts_ms // 60_000) * 60_000

        # First tick ever
        if not self._has_data:
            self._start_new_candle(minute_start, price, anchored_cvd, ts_ms)
            return None

        # Same minute — update OHLC in-place
        if minute_start == self._candle_start_ts_ms:
            self._update_ohlc(price, anchored_cvd, ts_ms)
            return None

        # New minute — close previous candle and start new one
        closed = self._close_candle(minute_start)
        self._start_new_candle(minute_start, price, anchored_cvd, ts_ms)
        return closed

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _start_new_candle(
        self,
        minute_start: int,
        price: float,
        anchored_cvd: float,
        ts_ms: int,
    ) -> None:
        self._candle_start_ts_ms = minute_start
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._anchored_cvd_open = anchored_cvd
        self._anchored_cvd_high = anchored_cvd
        self._anchored_cvd_low = anchored_cvd
        self._anchored_cvd_close = anchored_cvd
        self._high_ts_ms = ts_ms
        self._low_ts_ms = ts_ms
        self._has_data = True

    def _update_ohlc(self, price: float, anchored_cvd: float, ts_ms: int) -> None:
        self._close = price
        self._anchored_cvd_close = anchored_cvd

        if price > self._high:
            self._high = price
            self._anchored_cvd_high = anchored_cvd
            self._high_ts_ms = ts_ms

        if price < self._low:
            self._low = price
            self._anchored_cvd_low = anchored_cvd
            self._low_ts_ms = ts_ms

    def _close_candle(self, next_minute_start: int) -> ReclaimOneMinuteCandle:
        return ReclaimOneMinuteCandle(
            start_ts_ms=self._candle_start_ts_ms,
            end_ts_ms=next_minute_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            anchored_cvd_open=self._anchored_cvd_open,
            anchored_cvd_high=self._anchored_cvd_high,
            anchored_cvd_low=self._anchored_cvd_low,
            anchored_cvd_close=self._anchored_cvd_close,
            high_ts_ms=self._high_ts_ms,
            low_ts_ms=self._low_ts_ms,
        )


# ---------------------------------------------------------------------------
# 1m fractal extreme tracker
# ---------------------------------------------------------------------------


class ReclaimFractalExtremeTracker:
    """Tracks consecutive 1m closed candles and emits fractal extremes.

    Maintains a sliding window of the most recent 5 closed candles.
    When a new candle closes, checks whether the middle candle (index 2)
    is a fractal extreme (2-left, 2-right).

    A given candidate candle is only emitted once — the window slides
    forward, so the same middle position never repeats.

    Usage::

        tracker = ReclaimFractalExtremeTracker(side="LOWER")
        for candle in closed_candles:
            extreme = tracker.on_closed_candle(candle, confirm_ts_ms=candle.end_ts_ms)
            if extreme is not None:
                print(f"Fractal {extreme.side} at {extreme.price}")
    """

    def __init__(
        self,
        *,
        side: ExtremeSide,
        window_left: int = 2,
        window_right: int = 2,
    ) -> None:
        if side not in ("LOWER", "UPPER"):
            raise ValueError(f"side must be LOWER or UPPER, got {side!r}")
        self.side: ExtremeSide = side
        self._window_left = window_left
        self._window_right = window_right
        self._required = window_left + 1 + window_right  # total window size
        self._candles: list[ReclaimOneMinuteCandle] = []
        # Track which candle start_ts_ms has already been emitted as an extreme
        self._emitted_candle_start_ts_ms: set[int] = set()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear accumulated candles and emission memory."""
        self._candles.clear()
        self._emitted_candle_start_ts_ms.clear()

    def on_closed_candle(
        self,
        candle: ReclaimOneMinuteCandle,
        *,
        confirm_ts_ms: int,
    ) -> ReclaimFractalExtreme | None:
        """Feed a newly closed 1m candle.

        Returns a :class:`ReclaimFractalExtreme` when a 2L/2R fractal
        pattern completes; ``None`` otherwise.
        """
        self._candles.append(candle)

        # Keep only the most recent `_required` candles
        while len(self._candles) > self._required:
            self._candles.pop(0)

        if len(self._candles) < self._required:
            return None

        # The candidate is the middle candle (index 2 for 2L/2R)
        candidate = self._candles[self._window_left]

        # Prevent duplicate emission for the same candle
        if candidate.start_ts_ms in self._emitted_candle_start_ts_ms:
            return None

        if self.side == "LOWER":
            result = self._check_lower_fractal(candidate, confirm_ts_ms)
        else:
            result = self._check_upper_fractal(candidate, confirm_ts_ms)

        if result is not None:
            self._emitted_candle_start_ts_ms.add(candidate.start_ts_ms)

        return result

    # ------------------------------------------------------------------
    # internal — LOWER fractal
    # ------------------------------------------------------------------

    def _check_lower_fractal(
        self,
        candidate: ReclaimOneMinuteCandle,
        confirm_ts_ms: int,
    ) -> ReclaimFractalExtreme | None:
        candles = self._candles
        # candidate is candles[2]

        is_lowest = all(candidate.low <= c.low for c in candles)
        if not is_lowest:
            return None

        strict_left = (
            candidate.low < candles[0].low
            and candidate.low < candles[1].low
        )
        strict_right = (
            candidate.low <= candles[3].low
            and candidate.low <= candles[4].low
        )

        if not (strict_left and strict_right):
            return None

        return ReclaimFractalExtreme(
            side="LOWER",
            price=candidate.low,
            anchored_cvd=candidate.anchored_cvd_low,
            candle_start_ts_ms=candidate.start_ts_ms,
            candle_end_ts_ms=candidate.end_ts_ms,
            extreme_ts_ms=candidate.low_ts_ms,
            confirm_ts_ms=confirm_ts_ms,
            confirm_reason="fractal_1m_2l2r",
        )

    # ------------------------------------------------------------------
    # internal — UPPER fractal
    # ------------------------------------------------------------------

    def _check_upper_fractal(
        self,
        candidate: ReclaimOneMinuteCandle,
        confirm_ts_ms: int,
    ) -> ReclaimFractalExtreme | None:
        candles = self._candles
        # candidate is candles[2]

        is_highest = all(candidate.high >= c.high for c in candles)
        if not is_highest:
            return None

        strict_left = (
            candidate.high > candles[0].high
            and candidate.high > candles[1].high
        )
        strict_right = (
            candidate.high >= candles[3].high
            and candidate.high >= candles[4].high
        )

        if not (strict_left and strict_right):
            return None

        return ReclaimFractalExtreme(
            side="UPPER",
            price=candidate.high,
            anchored_cvd=candidate.anchored_cvd_high,
            candle_start_ts_ms=candidate.start_ts_ms,
            candle_end_ts_ms=candidate.end_ts_ms,
            extreme_ts_ms=candidate.high_ts_ms,
            confirm_ts_ms=confirm_ts_ms,
            confirm_reason="fractal_1m_2l2r",
        )
