"""Outside sweep volume profile — tracks tick-by-tick volume during an
outside-band excursion to identify the Point of Control (POC).

The POC is the price bucket with the highest accumulated volume during
the sweep.  It is used by Reclaim V2 to choose between a POC-based
entry protective stop and a classic extreme-based stop.

This module is pure logic; it has no dependency on strategy state or
exchange adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SweepVolumeProfile:
    """Tracks volume per price bucket during an outside-band sweep.

    ``bucket_pct`` defines the bucket size as a fraction of price
    (e.g. 0.0002 = 0.02 %).  Each tick's volume is assigned to a
    bucket by rounding the price to the nearest bucket boundary.

    Usage::

        profile = SweepVolumeProfile(bucket_pct=0.0002)
        profile.add(price=1900.12, volume=1.5)
        poc = profile.poc_price()  # => 1900.10 (or None if empty)
        profile.reset()
    """

    bucket_pct: float
    buckets: dict[float, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bucket_pct <= 0:
            raise ValueError(
                f"sweep profile bucket_pct={self.bucket_pct} must be > 0"
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def add(self, price: float, volume: float) -> None:
        """Record ``volume`` at ``price`` in the profile.

        ``volume`` should be >= 0.  Negative values are silently
        clamped to 0.
        """
        if volume < 0:
            volume = 0.0
        if volume <= 0:
            return
        if price <= 0:
            return
        bp = self._bucket_price(price)
        self.buckets[bp] = self.buckets.get(bp, 0.0) + volume

    def poc_price(self) -> float | None:
        """Return the price bucket with the highest accumulated volume.

        Returns ``None`` when no volume has been recorded.
        When multiple buckets share the same max volume the **middle**
        of the tied bucket prices is returned (midpoint between min
        and max tied bucket).
        """
        if not self.buckets:
            return None
        max_vol = max(self.buckets.values())
        tied = [bp for bp, vol in self.buckets.items() if vol == max_vol]
        if len(tied) == 1:
            return tied[0]
        return (min(tied) + max(tied)) / 2.0

    def reset(self) -> None:
        """Clear all accumulated volume data."""
        self.buckets.clear()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _bucket_price(self, price: float) -> float:
        """Round *price* to the nearest bucket boundary."""
        bucket_size = price * self.bucket_pct
        if bucket_size <= 0:
            return price
        return round(price / bucket_size) * bucket_size
