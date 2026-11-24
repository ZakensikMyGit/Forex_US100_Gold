"""Projekcja równego AB=CD i obserwacja strefy D na zamkniętych świecach."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from .data import TZ_NY, bar_end, bars_between
from .indicators import swing_points
from .instruments import price_text
from .signals import last_closed, rejects_level

PIVOT_BARS = 2
BC_MIN = 0.382
BC_MAX = 0.886
D_TOLERANCE = 0.03  # odchylenie od projekcji po każdej stronie, jako część długości AB
ALERT_SETTING = "powiadomienia AB=CD"


@dataclass(frozen=True)
class Pivot:
    time: pd.Timestamp
    confirmed_at: pd.Timestamp
    price: float
    high: bool


@dataclass(frozen=True)
class Projection:
    a: Pivot
    b: Pivot
    c: Pivot

    @property
    def id(self) -> str:
        return "-".join(str(int(p.time.timestamp())) for p in (self.a, self.b, self.c))

    @property
    def leg(self) -> float:
        return self.b.price - self.a.price

    @property
    def d(self) -> float:
        return self.c.price + self.leg

    @property
    def zone(self) -> tuple[float, float]:
        margin = abs(self.leg) * D_TOLERANCE
        return self.d - margin, self.d + margin

    @property
    def reaction_direction(self) -> str:
        return "short" if self.leg > 0 else "long"


@dataclass(frozen=True)
class Status:
    stage: str
    at: pd.Timestamp
    close: float | None = None


def confirmed_pivots(bars: pd.DataFrame, now: pd.Timestamp, *, interval: str = "H4") -> Iterator[Pivot]:
    last = last_closed(bars, interval, now)
    if last is None:
        return
    closed = bars.loc[:last.name]
    highs, lows = swing_points(closed, k=PIVOT_BARS)
    for time in highs.index.union(lows.index).sort_values():
        high = time in highs.index
        if high and time in lows.index:
            continue  # świeca z oboma ekstremami nie ujawnia ich kolejności
        price = float(highs[time] if high else lows[time])
        confirmation_bar = closed.index[closed.index.get_loc(time) + PIVOT_BARS]
        yield Pivot(time, bar_end(interval, confirmation_bar), price, high)


def confirmed_waves(
    bars: pd.DataFrame, now: pd.Timestamp, *, interval: str = "H4",
) -> Iterator[tuple[Pivot, Pivot]]:
    """Kolejne fale wyznaczone z potwierdzonych, naprzemiennych ekstremów."""
    swing: list[Pivot] = []
    for pivot in confirmed_pivots(bars, now, interval=interval):
        if swing and swing[-1].high == pivot.high:
            if (pivot.price <= swing[-1].price if pivot.high else pivot.price >= swing[-1].price):
                continue
            swing[-1] = pivot
        else:
            swing.append(pivot)
        if len(swing) >= 2:
            yield swing[-2], swing[-1]


def projections(bars: pd.DataFrame, now: pd.Timestamp, *, interval: str = "H4") -> list[Projection]:
    """Kolejność odpowiada chwilom, w których potwierdzono punkty C."""
    pivots: list[Pivot] = []
    found = []
    for point in confirmed_pivots(bars, now, interval=interval):
        if pivots and pivots[-1].high == point.high:
            if (point.price <= pivots[-1].price if point.high else point.price >= pivots[-1].price):
                continue
            pivots[-1] = point
        else:
            pivots.append(point)
        if len(pivots) < 3:
            continue
        a, b, c = pivots[-3:]
        ab, bc = b.price - a.price, c.price - b.price
        if (b.high and ab > 0 or not b.high and ab < 0) and BC_MIN <= -bc / ab <= BC_MAX:
            found.append(Projection(a, b, c))
    return found


def track(projection: Projection, bars: pd.DataFrame, now: pd.Timestamp, *, interval: str = "H1") -> Status:
    """Oceń kolejne świece po potwierdzeniu C, do ostatniej zamkniętej."""
    lower, upper = projection.zone
    upward = projection.leg > 0
    direction = projection.reaction_direction
    status = Status("building", projection.c.confirmed_at)
    last = last_closed(bars, interval, now)
    if last is None:
        return status

    if interval == "D1":
        # Indeks D1 zawiera daty sesji w Nowym Jorku, a nie godziny ich rozpoczęcia.
        session = projection.c.confirmed_at.tz_convert(TZ_NY).tz_localize(None).normalize()
        first = bars.index.searchsorted(session, side="right")
        window = bars.iloc[first:bars.index.get_loc(last.name) + 1]
    else:
        window = bars_between(bars, projection.c.confirmed_at, last.name)
    for bar in window[["open", "high", "low", "close"]].itertuples():
        closed_at = bar_end(interval, bar.Index)
        if (upward and bar.close < projection.c.price) or (not upward and bar.close > projection.c.price):
            return Status("invalid_c", closed_at, bar.close)
        if (upward and bar.close > upper) or (not upward and bar.close < lower):
            return Status("invalid_d", closed_at, bar.close)
        if bar.low <= upper and bar.high >= lower:
            if rejects_level(bar.open, bar.high, bar.low, bar.close, projection.d, direction):
                return Status("rejected", closed_at, bar.close)
            if status.stage == "building":
                status = Status("reached", closed_at, bar.close)
    return status


def describe(projection: Projection, status: Status, off: float, decimals: int) -> tuple[str, str]:
    lower, upper = projection.zone
    zone = f"{price_text(lower + off, decimals)}–{price_text(upper + off, decimals)}"
    if status.stage == "building":
        direction = "w górę" if projection.leg > 0 else "w dół"
        return "AB=CD · W budowie", f"Punkty A, B i C potwierdzone na H4. Projekcja prowadzi {direction} do strefy D {zone}."
    if status.stage == "reached":
        return "AB=CD · Cena dotarła do strefy D", f"Zakres zamkniętej H1 objął część strefy {zone}. Czekam na odrzucenie poziomu D."
    close = price_text(float(status.close) + off, decimals)
    if status.stage == "rejected":
        direction = projection.reaction_direction.upper()
        return f"AB=CD · Reakcja {direction} na H1", f"H1 zamknęła się na {close} po odrzuceniu poziomu D w strefie {zone}."
    if status.stage == "invalid_c":
        reason = f"H1 zamknęła się na {close} po przeciwnej stronie punktu C {price_text(projection.c.price + off, decimals)}."
    else:
        reason = f"H1 zamknęła się na {close} poza dalszą granicą strefy D {zone}."
    return "AB=CD · Układ zanegowany", reason
