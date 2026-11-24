"""Wskaźniki techniczne liczone tak, jak w typowych platformach (xStation, TradingView)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618)
CLOSE_QUARTILE = 0.25


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI Wildera."""
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((loss == 0) & (gain > 0), 100.0)  # same wzrosty
    out = out.mask((loss == 0) & (gain == 0), 50.0)  # płaska seria: brak przewagi żadnej strony
    return out.where(loss.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def close_position(df: pd.DataFrame) -> pd.Series:
    """Położenie zamknięcia w zakresie świecy, od 0 dla minimum do 1 dla maksimum."""
    width = df["high"] - df["low"]
    return (df["close"] - df["low"]) / width.where(width > 0)


def relative_volume(volume: pd.Series) -> pd.Series:
    """Porównaj obrót z medianą poprzednich 20 świec przy min. 12 dodatnich odczytach."""
    valid = volume.where(volume.gt(0) & np.isfinite(volume))
    return valid / valid.shift().rolling(20, min_periods=12).median()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    for n in (9, 26, 50, 200):
        out[f"ema{n}"] = ema(c, n)
    out["rsi"] = rsi(c)
    out["macd"] = ema(c, 12) - ema(c, 26)
    out["macd_signal"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    out["bb_mid"], out["bb_up"], out["bb_low"] = mid, mid + 2 * sd, mid - 2 * sd
    out["atr"] = atr(out)
    return out


# ---------------------------------------------------------------- dołki i szczyty


def swing_points(df: pd.DataFrame, k: int = 5) -> tuple[pd.Series, pd.Series]:
    """Potwierdzone lokalne szczyty i dołki (ekstremum w oknie +-k świec)."""
    win = 2 * k + 1
    hi = df["high"]
    lo = df["low"]
    is_high = hi.eq(hi.rolling(win, center=True).max())
    is_low = lo.eq(lo.rolling(win, center=True).min())
    return hi[is_high], lo[is_low]


def detect_divergences(df: pd.DataFrame, k: int = 5, lookback: int = 80) -> list[dict]:
    """Dywergencje RSI i MACD między dwoma ostatnimi dołkami / szczytami.

    Zwraca też dywergencje "w budowie": gdy ostatnia świeca robi nowe ekstremum ceny,
    a wskaźnik nie potwierdza go względem poprzedniego potwierdzonego ekstremum.
    """
    recent = df.iloc[-lookback:]
    if recent.empty:
        return []
    highs, lows = swing_points(recent, k)
    found: list[dict] = []
    last_idx = recent.index[-1]

    def check(points: pd.Series, kind: str) -> None:
        bullish = kind == "low"
        col_price = "low" if bullish else "high"
        candidates = list(points.index)
        # dodaj bieżącą świecę jako kandydat "w budowie"
        last_extreme = recent[col_price].iloc[-1]
        prev_pts = [i for i in candidates if i != last_idx]
        pairs: list[tuple[pd.Timestamp, pd.Timestamp, bool]] = []
        if len(prev_pts) >= 2:
            pairs.append((prev_pts[-2], prev_pts[-1], True))
        if prev_pts:
            ref = prev_pts[-1]
            new_extreme = last_extreme < recent.at[ref, col_price] if bullish else last_extreme > recent.at[ref, col_price]
            if new_extreme:
                pairs.append((ref, last_idx, False))
        for a, b, confirmed in pairs:
            pa, pb = recent.at[a, col_price], recent.at[b, col_price]
            price_ok = pb < pa if bullish else pb > pa
            if not price_ok:
                continue
            for ind in ("rsi", "macd"):
                ia, ib = recent.at[a, ind], recent.at[b, ind]
                if pd.isna(ia) or pd.isna(ib):
                    continue
                ind_ok = ib > ia if bullish else ib < ia
                if ind_ok:
                    found.append(
                        {
                            "typ": "bycza" if bullish else "niedźwiedzia",
                            "wskaźnik": ind.upper(),
                            "od": a,
                            "do": b,
                            "cena od": float(pa),
                            "cena do": float(pb),
                            "status": "potwierdzona" if confirmed else "w budowie",
                        }
                    )

    check(lows, "low")
    check(highs, "high")
    return found


# ---------------------------------------------------------------- Fibonacci


def fib_levels(low: float, high: float, downtrend: bool = True) -> list[tuple[str, float]]:
    """Poziomy zniesień i rozszerzeń.

    downtrend=True: ruch od szczytu w dół, poziomy liczone od szczytu (jak w xStation
    przy rysowaniu od dołka do szczytu: 0 = szczyt, 1 = dołek, 1.618 = rozszerzenie).
    """
    rng = high - low
    out = []
    for r in FIB_RATIOS:
        lvl = high - r * rng if downtrend else low + r * rng
        out.append((f"{r:g}", lvl))
    return out


def auto_fib_decline(df: pd.DataFrame, lookback: int = 250) -> tuple[float, pd.Timestamp, float, pd.Timestamp]:
    """Spadek od najwyższego szczytu w oknie do najniższego dołka po nim.

    Zniesienia tego ruchu w górę (dołek + r * zakres) to opory dla odbicia.
    """
    recent = df.iloc[-lookback:]
    hi_t = recent["high"].idxmax()
    after = recent.loc[hi_t:]
    lo_t = after["low"].idxmin()
    return float(recent.at[hi_t, "high"]), hi_t, float(recent.at[lo_t, "low"]), lo_t


def auto_fib_swing(df: pd.DataFrame, lookback: int = 250) -> tuple[float, pd.Timestamp, float, pd.Timestamp]:
    """Najniższy dołek i najwyższy szczyt w oknie - główny ruch do zniesień."""
    recent = df.iloc[-lookback:]
    hi_t = recent["high"].idxmax()
    before = recent.loc[:hi_t]
    lo_t = before["low"].idxmin()
    return float(recent.at[lo_t, "low"]), lo_t, float(recent.at[hi_t, "high"]), hi_t
