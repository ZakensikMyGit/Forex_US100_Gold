"""Pomiar ostatniej fali i zmiana roli poziomu na zamkniętych świecach."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from . import data, harmonics, indicators

POF_RATIOS = (0.382, 0.5, 0.618, 0.786, 0.886)
ZONE_ATR = 0.1


@dataclass(frozen=True)
class WaveMeasurement:
    start: harmonics.Pivot
    end: harmonics.Pivot

    def levels(self) -> tuple[tuple[float, float], ...]:
        leg = self.end.price - self.start.price
        return tuple((ratio, self.end.price - ratio * leg) for ratio in POF_RATIOS)


@dataclass(frozen=True)
class PolarityZone:
    pivot: harmonics.Pivot
    width: float
    broken_at: pd.Timestamp
    retested_at: pd.Timestamp | None

    @property
    def role(self) -> str:
        return "Wsparcie" if self.pivot.high else "Opór"

    @property
    def bounds(self) -> tuple[float, float]:
        return self.pivot.price - self.width, self.pivot.price + self.width


def last_wave(bars: pd.DataFrame, interval: str, now: pd.Timestamp) -> WaveMeasurement | None:
    latest = next(reversed(list(harmonics.confirmed_waves(bars, now, interval=interval))), None)
    return WaveMeasurement(*latest) if latest else None


def polarity_zones(bars: pd.DataFrame, interval: str, now: pd.Timestamp) -> list[PolarityZone]:
    """Znajdź świeże zmiany ról. Powrót zamknięcia za strefę unieważnia poprzedni retest."""
    ends = pd.DatetimeIndex(pd.to_datetime([data.bar_end(interval, t) for t in bars.index], utc=True))
    count = int(ends.searchsorted(now, side="right"))
    closed = bars.iloc[:count]
    if closed.empty:
        return []
    ends = ends[:count]
    atr = indicators.atr(closed).to_numpy()
    close, low, high = (closed[name].to_numpy() for name in ("close", "low", "high"))
    candidates: list[PolarityZone] = []
    for pivot in harmonics.confirmed_pivots(closed, now, interval=interval):
        first = int(ends.searchsorted(pivot.confirmed_at, side="right"))
        width = float(atr[first - 1]) * ZONE_ATR
        if not isfinite(width) or width <= 0:
            continue
        price = pivot.price
        signed = close[first:] - price if pivot.high else price - close[first:]
        returns = np.flatnonzero(signed < -width)
        run = int(returns[-1]) + 1 if returns.size else 0
        breaks = np.flatnonzero(signed[run:] > width)
        if not breaks.size:
            continue
        breakout = first + run + int(breaks[0])
        after = slice(breakout + 1, None)
        touches = (low[after] <= price + width) & (high[after] >= price - width)
        held = close[after] >= price if pivot.high else close[after] <= price
        retests = np.flatnonzero(touches & held)
        retest = ends[breakout + 1 + int(retests[0])] if retests.size else None
        candidates.append(PolarityZone(pivot, width, ends[breakout], retest))

    nearest = sorted(candidates, key=lambda zone: (
        abs(zone.pivot.price - close[-1]), zone.retested_at is None, -zone.broken_at.timestamp(),
    ))
    # Jedna najbliższa strefa każdego rodzaju ogranicza nakładanie się linii.
    return [zone for polarity in (True, False)
            if (zone := next((z for z in nearest if z.pivot.high == polarity), None)) is not None]
