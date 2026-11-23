"""Wspólne ładowanie danych dla panelu i monitora."""

from __future__ import annotations

import time

import pandas as pd

from . import data, indicators as ind


def build_frames(daily: pd.DataFrame, hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "D1": ind.add_indicators(daily),
        "H4": ind.add_indicators(data.resample_h4(hourly)),
        "H1": ind.add_indicators(hourly),
    }


def load_frames(ticker: str = data.TICKER_NQ) -> dict[str, pd.DataFrame]:
    """Pełne pobranie historii (panel - z buforem Streamlit)."""
    return build_frames(data.load_daily(ticker), data.load_hourly(ticker))


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Nowe świece zastępują stare o tym samym czasie (ostatnia świeca mogła być niepełna)."""
    return pd.concat([old[~old.index.isin(new.index)], new]).sort_index()


class FrameCache:
    """Bufor monitora: pełna historia raz na dobę, w pozostałych cyklach tylko końcówka.

    Wskaźniki są zawsze liczone od nowa na całej scalonej historii, więc EMA i RSI
    są identyczne jak przy pełnym pobraniu.
    """

    FULL_RELOAD_S = 24 * 3600

    def __init__(self) -> None:
        self._raw: dict[str, tuple[pd.DataFrame, pd.DataFrame, float]] = {}
        self.stats = {"pełne": 0, "przyrostowe": 0}

    def frames(self, ticker: str) -> dict[str, pd.DataFrame]:
        cached = self._raw.get(ticker)
        if cached is None or time.time() - cached[2] > self.FULL_RELOAD_S:
            daily, hourly = data.load_daily(ticker), data.load_hourly(ticker)
            self._raw[ticker] = (daily, hourly, time.time())
            self.stats["pełne"] += 1
        else:
            daily_old, hourly_old, loaded = cached
            daily = _merge(daily_old, data.load_daily(ticker, "1mo"))
            hourly = _merge(hourly_old, data.load_hourly(ticker, "5d"))
            self._raw[ticker] = (daily, hourly, loaded)
            self.stats["przyrostowe"] += 1
        return build_frames(daily, hourly)
