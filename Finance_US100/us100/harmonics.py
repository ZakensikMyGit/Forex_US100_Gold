"""Projekcja równego AB=CD z potwierdzonych ekstremów H4 i obserwacja strefy D na H1."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from .data import bar_end
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


def confirmed_pivots(h4: pd.DataFrame, now: pd.Timestamp) -> Iterator[Pivot]:
    last = last_closed(h4, "H4", now)
    if last is None:
        return
    closed = h4.loc[:last.name]
    highs, lows = swing_points(closed, k=PIVOT_BARS)
    for time in highs.index.union(lows.index).sort_values():
        high = time in highs.index
        if high and time in lows.index:
            continue  # świeca z oboma ekstremami nie ujawnia ich kolejności
        price = float(highs[time] if high else lows[time])
        confirmation_bar = closed.index[closed.index.get_loc(time) + PIVOT_BARS]
        yield Pivot(time, bar_end("H4", confirmation_bar), price, high)


def projections(h4: pd.DataFrame, now: pd.Timestamp) -> list[Projection]:
    """Kolejność odpowiada chwilom, w których potwierdzono punkty C."""
    pivots: list[Pivot] = []
    found = []
    for point in confirmed_pivots(h4, now):
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


def track(projection: Projection, h1: pd.DataFrame, now: pd.Timestamp) -> Status:
    """Oceń kolejne zamknięte H1, które zaczęły się po ustaleniu projekcji."""
    lower, upper = projection.zone
    upward = projection.leg > 0
    direction = projection.reaction_direction
    status = Status("building", projection.c.confirmed_at)
    last = last_closed(h1, "H1", now)
    if last is None or last.name < projection.c.confirmed_at:
        return status

    for bar in h1.loc[projection.c.confirmed_at:last.name, ["open", "high", "low", "close"]].itertuples():
        closed_at = bar_end("H1", bar.Index)
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
