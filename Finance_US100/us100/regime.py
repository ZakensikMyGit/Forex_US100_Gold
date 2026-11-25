"""Reżim rynku na zamkniętych świecach H4: trend wzrostowy, trend spadkowy albo konsolidacja.

Reguły ustalone z góry, bez dopasowywania do wyników:
- trend wzrostowy: EMA50 > EMA200, EMA50 wyżej niż 10 świec temu i ADX(14) >= 20,
- trend spadkowy: odwrotnie, również przy ADX >= 20,
- konsolidacja: pozostałe przypadki,
- wysoka zmienność: ATR H4 w górnych 10% swoich wartości z ostatnich ok. 60 dni (360 świec H4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators
from .data import bar_end

ADX_TREND = 20.0
SLOPE_BARS = 10
VOL_WINDOW = 360
VOL_HIGH = 0.9

UP, DOWN, RANGE = "trend wzrostowy", "trend spadkowy", "konsolidacja"


def classify(h4: pd.DataFrame) -> pd.DataFrame:
    """Reżim po zamknięciu każdej świecy H4 (kolumny: reżim, adx, wysoka zmienność)."""
    ema50 = h4["ema50"] if "ema50" in h4 else indicators.ema(h4["close"], 50)
    ema200 = h4["ema200"] if "ema200" in h4 else indicators.ema(h4["close"], 200)
    atr = h4["atr"] if "atr" in h4 else indicators.atr(h4)
    strength = indicators.adx(h4)
    slope = ema50.diff(SLOPE_BARS)
    up = (ema50 > ema200) & (slope > 0) & (strength >= ADX_TREND)
    down = (ema50 < ema200) & (slope < 0) & (strength >= ADX_TREND)
    label = np.where(up, UP, np.where(down, DOWN, RANGE))
    label = np.where(strength.isna() | slope.isna(), "brak danych", label)
    volatility = atr.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 3).rank(pct=True)
    return pd.DataFrame({"reżim": label, "adx": strength, "wysoka zmienność": volatility > VOL_HIGH}, index=h4.index)


def at(h4: pd.DataFrame, moments: pd.DatetimeIndex) -> pd.DataFrame:
    """Reżim znany w danych chwilach: z ostatniej H4 zamkniętej najpóźniej w tej chwili."""
    table = classify(h4)
    ends = pd.DatetimeIndex(pd.to_datetime([bar_end("H4", t) for t in h4.index], utc=True))
    rows = ends.searchsorted(moments, side="right") - 1
    out = pd.DataFrame({"reżim": "brak danych", "adx": np.nan, "wysoka zmienność": False}, index=moments)
    known = rows >= 0
    if known.any():
        picked = table.iloc[rows[known]]
        out.loc[known, "reżim"] = picked["reżim"].to_numpy()
        out.loc[known, "adx"] = picked["adx"].to_numpy()
        out.loc[known, "wysoka zmienność"] = picked["wysoka zmienność"].to_numpy()
    return out


@dataclass(frozen=True)
class Current:
    label: str
    adx: float | None
    high_volatility: bool

    def text(self) -> str:
        adx = f", ADX {self.adx:.0f}" if self.adx is not None and np.isfinite(self.adx) else ""
        volatility = "wysoka" if self.high_volatility else "zwykła"
        return f"{self.label}{adx}, zmienność {volatility}"


def current(h4: pd.DataFrame, now: pd.Timestamp) -> Current:
    row = at(h4, pd.DatetimeIndex([now])).iloc[0]
    return Current(str(row["reżim"]), float(row["adx"]) if pd.notna(row["adx"]) else None, bool(row["wysoka zmienność"]))
