"""Pobieranie notowań z Yahoo Finance i budowa świec D1 / H4 / H1.

US100 w XTB śledzi kontrakt terminowy Nasdaq 100 (NQ), dlatego domyślnym źródłem
jest NQ=F. Ewentualną różnicę względem kursu w xStation koryguje się w ustawieniach
(przesunięcie w punktach).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

TZ_LOCAL = "Europe/Warsaw"
TZ_NY = "America/New_York"

TICKER_NQ = "NQ=F"
TICKER_NDX = "^NDX"
TICKER_VIX = "^VIX"
TICKER_USDPLN = "USDPLN=X"

INTERVALS = ("D1", "H4", "H1")
CHART_INTRADAY = {"M15": "15m", "M5": "5m"}
BAR_LENGTH = {
    "H1": timedelta(hours=1), "H4": timedelta(hours=4),
    **{interval: pd.Timedelta(duration) for interval, duration in CHART_INTRADAY.items()},
}

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if not df.empty:
        df = df[_OHLCV].dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise RuntimeError(f"Brak danych z Yahoo Finance dla {ticker} ({interval}).")
    df.columns = [c.lower() for c in df.columns]
    return df


def load_daily(ticker: str = TICKER_NQ, period: str = "5y") -> pd.DataFrame:
    """Świece dzienne (domyślnie 5 lat - wystarczy na rozgrzanie EMA 200)."""
    df = _history(ticker, period, "1d")
    # Daty sesji (bez godziny) - sesja NQ kończy się o 17:00 czasu nowojorskiego.
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(TZ_NY).tz_localize(None)
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]


def load_hourly(ticker: str = TICKER_NQ, period: str = "730d") -> pd.DataFrame:
    """Świece godzinowe (domyślnie ok. 2 lata), w czasie polskim (jak w xStation)."""
    return _local_intraday(_history(ticker, period, "1h"))


def load_intraday(ticker: str, interval: str, period: str = "1mo") -> pd.DataFrame:
    """Krótkie świece z Yahoo Finance w czasie polskim."""
    return _local_intraday(_history(ticker, period, CHART_INTRADAY[interval])).sort_index()


def _local_intraday(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(TZ_LOCAL)
    return df[~df.index.duplicated(keep="last")]


def resample_h4(hourly: pd.DataFrame) -> pd.DataFrame:
    """Świece H4 o granicach 00/04/08/12/16/20 czasu polskiego - także po zmianie czasu.

    Grupujemy po godzinie na zegarze (a nie co stałe 4 godziny), bo zwykłe resample("4h")
    po przejściu na czas zimowy/letni przesuwa granice o godzinę względem xStation.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    idx = pd.DatetimeIndex(hourly.index)
    wall = idx.tz_localize(None) if idx.tz is not None else idx
    key = wall.floor("D") + pd.to_timedelta((wall.hour // 4) * 4, unit="h")
    out = hourly.groupby(key).agg(agg).dropna(subset=["open", "close"])
    out_idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        out_idx = out_idx.tz_localize(idx.tz, ambiguous=False, nonexistent="shift_forward")
    out.index = out_idx
    return out


def last_value(ticker: str, period: str = "5d", interval: str = "1h") -> tuple[float, float]:
    """Ostatnie zamknięcie i zamknięcie poprzedniego dnia (do wyliczenia zmiany)."""
    intraday = _history(ticker, period, interval)
    daily = _history(ticker, "1mo", "1d")
    last = float(intraday["close"].iloc[-1])
    prev = float(daily["close"].iloc[-2]) if len(daily) >= 2 else last
    return last, prev


def bar_closed(interval: str, bar_start: Any, now: Any = None) -> bool:
    """Czy świeca jest już zamknięta (sygnały liczymy tylko na zamkniętych świecach)."""
    now = now or datetime.now(timezone.utc)
    return now >= bar_end(interval, bar_start).to_pydatetime()


def bars_between(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Wybiera świece według czasu otwarcia, włącznie z granicami, z posortowanego indeksu."""
    first = frame.index.searchsorted(start, side="left")
    after_last = frame.index.searchsorted(end, side="right")
    return frame.iloc[first:after_last]


def _wall(naive: Any, tz) -> pd.Timestamp:
    return naive.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")


def local_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return _wall(stamp, TZ_LOCAL) if stamp.tzinfo is None else stamp.tz_convert(TZ_LOCAL)


def bar_end(interval: str, bar_start: Any) -> pd.Timestamp:
    """Koniec świecy liczony na zegarze lokalnym, więc w dniu zmiany czasu świeca 00:00 H4
    kończy się o 04:00 (a nie po 4 godzinach bezwzględnych, czyli o 03:00 lub 05:00)."""
    if interval == "D1":
        # sesja kończy się o 17:00 czasu nowojorskiego - także w dniu zmiany czasu w USA
        return _wall(pd.Timestamp(bar_start.date()) + pd.Timedelta(hours=17), TZ_NY)
    tz = bar_start.tz
    if interval == "H4" and tz is not None:
        return _wall(bar_start.tz_localize(None) + BAR_LENGTH["H4"], tz)
    end = bar_start + BAR_LENGTH[interval]
    return end if end.tz is not None else end.tz_localize("UTC")
