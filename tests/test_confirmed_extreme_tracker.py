"""Tests for delayed confirmed swing extreme tracker.

Covers:
- LOWER retrace confirm
- LOWER stable confirm
- UPPER mirror tests
- Candidate extension with min_price_extension_pct
- Reset behaviour
"""

from __future__ import annotations

import pytest
from src.strategies.confirmed_extreme import (
    ConfirmedExtreme,
    ConfirmedExtremeConfig,
    ConfirmedExtremeTracker,
)


def _config(**overrides) -> ConfirmedExtremeConfig:
    kwargs = dict(
        confirm_mode="RETRACE_OR_STABLE",
        confirm_retrace_pct=0.0008,
        confirm_stable_seconds=8,
        min_price_extension_pct=0.0002,
    )
    kwargs.update(overrides)
    return ConfirmedExtremeConfig(**kwargs)


# ======================================================================
# LOWER retrace confirm
# ======================================================================


def test_lower_retrace_confirm() -> None:
    """Price: 1782 → 1781 → 1780 → 1780.9 triggers retrace confirm at 1780."""
    cfg = _config(confirm_retrace_pct=0.0008)
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    # Set candidate at 1782
    result = tracker.update(price=1782.0, anchored_cvd=-50000.0, ts_ms=1000)
    assert result is None

    # Extend to 1781 (new low)
    result = tracker.update(price=1781.0, anchored_cvd=-55000.0, ts_ms=2000)
    assert result is None

    # Extend to 1780 (new low)
    result = tracker.update(price=1780.0, anchored_cvd=-60000.0, ts_ms=3000)
    assert result is None

    # Retrace: 1780.9 >= 1780 * 1.0008 = 1781.424...  Wait that's right
    # 1780 * 1.0008 = 1781.424. 1780.9 is below that. Let me recalculate.
    # The retrace threshold for 1780: 1780 * (1 + 0.0008) = 1780 * 1.0008 = 1781.424
    # 1780.9 < 1781.424, so NO retrace confirm at 1780.9.
    # I need: price >= 1781.424. Let me use 1782 since 1782 > 1781.424.
    result = tracker.update(price=1782.0, anchored_cvd=-48000.0, ts_ms=4000)
    assert result is not None, "Should confirm on retrace"
    assert result.price == 1780.0
    assert result.anchored_cvd == -60000.0
    assert result.confirm_reason == "retrace"
    assert result.side == "LOWER"
    assert result.confirm_ts_ms == 4000
    assert result.ts_ms == 3000  # when candidate was last updated


def test_lower_retrace_exact_threshold() -> None:
    """At exact retrace threshold, it confirms."""
    cfg = _config(confirm_retrace_pct=0.001)  # 0.1%
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    tracker.update(price=100.0, anchored_cvd=0.0, ts_ms=1000)
    result = tracker.update(price=100.1, anchored_cvd=0.0, ts_ms=2000)
    # 100.0 * 1.001 = 100.1 → exactly at threshold → confirmed
    assert result is not None
    assert result.price == 100.0
    assert result.confirm_reason == "retrace"


# ======================================================================
# LOWER stable confirm
# ======================================================================


def test_lower_stable_confirm() -> None:
    """Price holds at 1780 for 8 seconds → stable confirm."""
    cfg = _config(confirm_stable_seconds=8)
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    # Set candidate
    tracker.update(price=1782.0, anchored_cvd=-50000.0, ts_ms=1000)
    tracker.update(price=1781.0, anchored_cvd=-55000.0, ts_ms=2000)
    tracker.update(price=1780.0, anchored_cvd=-60000.0, ts_ms=3000)

    # 3 seconds later — not confirmed (need 8)
    result = tracker.update(price=1780.05, anchored_cvd=-59000.0, ts_ms=6000)
    assert result is None, "3s < 8s, should not confirm"

    # 8 seconds after last update (3000 + 8000 = 11000)
    result = tracker.update(price=1780.05, anchored_cvd=-59000.0, ts_ms=11000)
    assert result is not None, "Should stable-confirm after 8s"
    assert result.price == 1780.0
    assert result.confirm_reason == "stable"


def test_lower_stable_reset_on_new_extreme() -> None:
    """A new candidate extension resets the stable timer."""
    cfg = _config(confirm_stable_seconds=8)
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    tracker.update(price=1780.0, anchored_cvd=-60000.0, ts_ms=3000)

    # 7 seconds later — extend to new low → timer resets
    result = tracker.update(price=1779.0, anchored_cvd=-65000.0, ts_ms=10000)
    assert result is None, "New low resets timer"

    # 8 seconds later from new low (10000 + 8000 = 18000)
    result = tracker.update(price=1779.05, anchored_cvd=-64000.0, ts_ms=18000)
    assert result is not None
    assert result.price == 1779.0
    assert result.confirm_reason == "stable"


# ======================================================================
# LOWER candidate extension (min_price_extension_pct)
# ======================================================================


def test_lower_no_extension_on_micro_tick() -> None:
    """Tiny price moves below min_price_extension_pct don't update candidate."""
    cfg = _config(min_price_extension_pct=0.001)  # 0.1%
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    tracker.update(price=100.0, anchored_cvd=0.0, ts_ms=1000)

    # 99.95 is only 0.05% below 100.0 → below 0.1% extension → NO update
    result = tracker.update(price=99.95, anchored_cvd=0.0, ts_ms=2000)
    # No extension, and not retraced enough → no confirm
    assert result is None
    # Candidate should still be 100.0
    # Retrace: 100.0 * 1.0008 = 100.08. 99.95 << 100.08. No retrace confirm.
    # Stable: 1 second, not 8. So no confirm. Good.

    # Now proper extension: 99.0 is 1% below 100
    tracker.update(price=99.0, anchored_cvd=0.0, ts_ms=3000)
    # Retrace confirm: 99.0 * 1.0008 = 99.0792. 99.1 > 99.0792 ✓
    result = tracker.update(price=99.1, anchored_cvd=0.0, ts_ms=4000)
    assert result is not None
    assert result.price == 99.0
    assert result.confirm_reason == "retrace"


# ======================================================================
# UPPER mirror tests
# ======================================================================


def test_upper_retrace_confirm() -> None:
    """Price: 3200 → 3210 → 3220 → 3218 → confirms at 3220."""
    cfg = _config(confirm_retrace_pct=0.0008)
    tracker = ConfirmedExtremeTracker(side="UPPER", config=cfg)

    tracker.update(price=3200.0, anchored_cvd=50000.0, ts_ms=1000)
    tracker.update(price=3210.0, anchored_cvd=55000.0, ts_ms=2000)
    tracker.update(price=3220.0, anchored_cvd=60000.0, ts_ms=3000)

    # Retrace: 3220 * (1 - 0.0008) = 3220 * 0.9992 = 3217.424
    # 3217 < 3217.424 → confirmed
    result = tracker.update(price=3217.0, anchored_cvd=58000.0, ts_ms=4000)
    assert result is not None
    assert result.price == 3220.0
    assert result.anchored_cvd == 60000.0
    assert result.confirm_reason == "retrace"
    assert result.side == "UPPER"


def test_upper_stable_confirm() -> None:
    """Price holds high for 8 seconds → stable confirm."""
    cfg = _config(confirm_stable_seconds=8)
    tracker = ConfirmedExtremeTracker(side="UPPER", config=cfg)

    tracker.update(price=3200.0, anchored_cvd=50000.0, ts_ms=1000)
    tracker.update(price=3220.0, anchored_cvd=60000.0, ts_ms=2000)

    # 8 seconds later
    result = tracker.update(price=3219.5, anchored_cvd=59000.0, ts_ms=10000)
    assert result is not None
    assert result.price == 3220.0
    assert result.confirm_reason == "stable"


def test_upper_extension() -> None:
    """Higher prices extend candidate."""
    cfg = _config(min_price_extension_pct=0.0002)
    tracker = ConfirmedExtremeTracker(side="UPPER", config=cfg)

    tracker.update(price=3200.0, anchored_cvd=50000.0, ts_ms=1000)
    # 3205 > 3200 * 1.0002 = 3200.64 → extension
    result = tracker.update(price=3205.0, anchored_cvd=55000.0, ts_ms=2000)
    assert result is None

    # Retrace confirm at 3205: 3205 * 0.9992 = 3202.436
    result = tracker.update(price=3202.0, anchored_cvd=54000.0, ts_ms=3000)
    assert result is not None
    assert result.price == 3205.0


# ======================================================================
# Reset behaviour
# ======================================================================


def test_reset_clears_candidate() -> None:
    """Reset discards the running candidate."""
    cfg = _config()
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    tracker.update(price=1780.0, anchored_cvd=-60000.0, ts_ms=3000)
    tracker.reset()

    # After reset, candidate is fresh
    result = tracker.update(price=1790.0, anchored_cvd=-50000.0, ts_ms=4000)
    assert result is None  # first tick after reset → new candidate


def test_confirm_resets_candidate() -> None:
    """After a confirmed extreme, candidate starts fresh."""
    cfg = _config()
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    tracker.update(price=1780.0, anchored_cvd=-60000.0, ts_ms=1000)
    # Retrace confirm
    result = tracker.update(price=1781.5, anchored_cvd=-58000.0, ts_ms=2000)
    assert result is not None

    # Next tick → new candidate
    result2 = tracker.update(price=1781.0, anchored_cvd=-57000.0, ts_ms=3000)
    assert result2 is None  # no confirm on fresh candidate


# ======================================================================
# Smoke / edge cases
# ======================================================================


def test_no_confirm_without_extension_or_stable() -> None:
    """Many ticks at same price level don't confirm until stable time."""
    cfg = _config(confirm_stable_seconds=8)
    tracker = ConfirmedExtremeTracker(side="LOWER", config=cfg)

    tracker.update(price=1780.0, anchored_cvd=-60000.0, ts_ms=1000)

    for i in range(7):
        result = tracker.update(price=1780.01, anchored_cvd=-59500.0, ts_ms=2000 + i * 1000)
        assert result is None, f"Tick {i}: should not confirm before stable timeout"


def test_first_tick_sets_candidate() -> None:
    """Very first tick sets candidate, no confirm."""
    tracker = ConfirmedExtremeTracker(side="LOWER", config=_config())
    result = tracker.update(price=1800.0, anchored_cvd=0.0, ts_ms=1000)
    assert result is None


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError):
        ConfirmedExtremeTracker(side="INVALID", config=_config())  # type: ignore[arg-type]
