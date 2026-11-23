"""Wspólne ładowanie danych dla panelu i monitora."""

from __future__ import annotations

import logging
import time

import pandas as pd

from . import archive, data, indicators as ind

LOG = logging.getLogger(__name__)


def build_frames(daily: pd.DataFrame, hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "D1": ind.add_indicators(daily),
        "H4": ind.add_indicators(data.resample_h4(hourly)),
        "H1": ind.add_indicators(hourly),
    }


def load_frames(ticker: str = data.TICKER_NQ) -> dict[str, pd.DataFrame]:
    """Pełne pobranie historii (panel - z buforem Streamlit)."""
    daily, hourly = _fetch_history(ticker)
    return build_frames(daily, hourly)


def load_archived_frames(ticker: str) -> dict[str, pd.DataFrame]:
    return build_frames(archive.load(ticker, "D1"), archive.load(ticker, "H1"))


def _save_history(ticker: str, interval: str, frame: pd.DataFrame, observed_at: pd.Timestamp) -> None:
    try:
        archive.save(ticker, interval, frame, observed_at)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        LOG.warning("Nie zapisano archiwum %s %s: %s", ticker, interval, exc)


def _with_archive(ticker: str, interval: str, frame: pd.DataFrame) -> pd.DataFrame:
    try:
        older = archive.load(ticker, interval)
    except FileNotFoundError:
        return frame
    except (OSError, ValueError, KeyError, ImportError) as exc:
        LOG.warning("Nie odczytano archiwum %s %s: %s", ticker, interval, exc)
        return frame
    if frame.empty:
        return frame
    return _merge(older.loc[:frame.index[-1]], frame)


def _fetch_history(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_at = pd.Timestamp.now(tz="UTC")
    daily = data.load_daily(ticker)
    _save_history(ticker, "D1", daily, observed_at)
    hourly = data.load_hourly(ticker)
    _save_history(ticker, "H1", hourly, observed_at)
    return _with_archive(ticker, "D1", daily), _with_archive(ticker, "H1", hourly)


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
            daily, hourly = _fetch_history(ticker)
            self._raw[ticker] = (daily, hourly, time.time())
            self.stats["pełne"] += 1
        else:
            daily_old, hourly_old, loaded = cached
            observed_at = pd.Timestamp.now(tz="UTC")
            daily_new = data.load_daily(ticker, "1mo")
            _save_history(ticker, "D1", daily_new, observed_at)
            hourly_new = data.load_hourly(ticker, "5d")
            _save_history(ticker, "H1", hourly_new, observed_at)
            daily, hourly = _merge(daily_old, daily_new), _merge(hourly_old, hourly_new)
            self._raw[ticker] = (daily, hourly, loaded)
            self.stats["przyrostowe"] += 1
        return build_frames(daily, hourly)
