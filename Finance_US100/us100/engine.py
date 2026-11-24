"""Wspólne ładowanie danych dla panelu i monitora."""

from __future__ import annotations

import logging
import time

import pandas as pd

from . import archive, data, indicators as ind
from .signals import last_closed

LOG = logging.getLogger(__name__)


def build_frames(daily: pd.DataFrame, hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    h4 = data.resample_h4(hourly)
    observed = hourly.attrs.get("pobrano")
    if observed is not None and not h4.empty:
        h4_end = data.bar_end("H4", h4.index[-1])
        if data.bar_end("H1", hourly.index[-1]) < h4_end <= observed:
            # Dostawca nie zwrócił ostatniej H1 potrzebnej do zamknięcia tej H4.
            h4 = h4.iloc[:-1]
        h4.attrs["pobrano"] = observed
    return {
        "D1": ind.add_indicators(daily),
        "H4": ind.add_indicators(h4),
        "H1": ind.add_indicators(hourly),
    }


def load_frames(ticker: str = data.TICKER_NQ) -> dict[str, pd.DataFrame]:
    """Pełne pobranie historii (panel - z buforem Streamlit)."""
    daily, hourly = _fetch_history(ticker)
    return build_frames(daily, hourly)


def load_archived_frames(ticker: str) -> dict[str, pd.DataFrame]:
    return build_frames(archive.load(ticker, "D1"), archive.load(ticker, "H1"))


def _latest_bar(frame: pd.DataFrame, interval: str, now: pd.Timestamp) -> dict:
    last = frame.iloc[-1]
    closed = last_closed(frame, interval, now)
    return {
        "pobrano": now.isoformat(),
        "początek": last.name.isoformat(),
        "zamknięta do": data.bar_end(interval, closed.name).isoformat() if closed is not None else None,
        "ohlcv": {col: float(last[col]) if pd.notna(last[col]) else None for col in archive.COLUMNS},
    }


def _with_latest(frame: pd.DataFrame, interval: str, info: dict) -> pd.DataFrame:
    if not info:
        return frame
    observed = pd.Timestamp(info["pobrano"])
    last = last_closed(frame, interval, observed)
    frame = frame.loc[:last.name] if last is not None else frame.iloc[:0]
    stamp = pd.Timestamp(info["początek"]).tz_convert(frame.index.tz)
    if frame.empty or stamp > frame.index[-1]:
        row = pd.DataFrame([info["ohlcv"]], index=pd.DatetimeIndex([stamp]))
        frame = pd.concat([frame, row])
    frame.attrs["pobrano"] = observed
    return frame


def load_monitor_frames(ticker: str, snapshot: dict) -> dict[str, pd.DataFrame]:
    hourly = _with_latest(archive.load(ticker, "H1"), "H1", snapshot.get("H1", {}))
    return build_frames(archive.load(ticker, "D1"), hourly)


def load_short_chart(ticker: str, interval: str, snapshot: dict | None = None) -> pd.DataFrame:
    bars = _with_latest(archive.load(ticker, "M5"), "M5", (snapshot or {}).get("M5", {}))
    if interval == "M15":
        grouped = bars.resample("15min")
        aggregated = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        complete = aggregated.loc[grouped["close"].count().eq(3)]
        complete.attrs = bars.attrs.copy()
        bars = complete
    return bars


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
    return (
        _with_archive(ticker, "D1", _download(ticker, "D1", data.load_daily)),
        _with_archive(ticker, "H1", _download(ticker, "H1", data.load_hourly)),
    )


def _download(ticker: str, interval: str, load, *args) -> pd.DataFrame:
    observed_at = pd.Timestamp.now(tz="UTC")
    frame = load(ticker, *args).copy(deep=False)
    _save_history(ticker, interval, frame, observed_at)
    frame.attrs["pobrano"] = observed_at
    return frame


def _closed_as_observed(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    closed = last_closed(frame, interval, frame.attrs["pobrano"])
    return frame.loc[:closed.name] if closed is not None else frame.iloc[:0]


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Nowe świece zastępują stare o tym samym czasie (ostatnia świeca mogła być niepełna)."""
    merged = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    merged.attrs = new.attrs.copy()
    return merged


class FrameCache:
    """Bufor monitora: pełna historia raz na dobę, w pozostałych cyklach tylko końcówka.

    Wskaźniki są zawsze liczone od nowa na całej scalonej historii, więc EMA i RSI
    są identyczne jak przy pełnym pobraniu.
    """

    FULL_RELOAD_S = 24 * 3600

    def __init__(self) -> None:
        self._raw: dict[str, tuple[pd.DataFrame, pd.DataFrame, float]] = {}
        self._daily_hour: dict[str, pd.Timestamp] = {}
        self.hourly_snapshot: dict[str, dict] = {}
        self.stats = {"pełne": 0, "przyrostowe": 0}

    def frames(self, ticker: str) -> dict[str, pd.DataFrame]:
        cached = self._raw.get(ticker)
        observed_at = pd.Timestamp.now(tz="UTC")
        hour = observed_at.floor("h")
        if cached is None or time.time() - cached[2] > self.FULL_RELOAD_S:
            daily, hourly = _fetch_history(ticker)
            self._raw[ticker] = (daily, hourly, time.time())
            self.stats["pełne"] += 1
            self._daily_hour[ticker] = hour
        else:
            daily, hourly_old, loaded = cached
            if self._daily_hour.get(ticker) != hour:
                try:
                    daily_new = _download(ticker, "D1", data.load_daily, "1mo")
                except Exception as exc:
                    LOG.warning("Używam poprzedniego D1 %s: %s", ticker, exc)
                else:
                    daily = _merge(_closed_as_observed(daily, "D1"), daily_new)
                    self._daily_hour[ticker] = hour
                    self._raw[ticker] = (daily, hourly_old, loaded)
            hourly_new = _download(ticker, "H1", data.load_hourly, "5d")
            hourly = _merge(_closed_as_observed(hourly_old, "H1"), hourly_new)
            self._raw[ticker] = (daily, hourly, loaded)
            self.stats["przyrostowe"] += 1
        observed_at = hourly.attrs["pobrano"]
        self.hourly_snapshot[ticker] = _latest_bar(hourly, "H1", observed_at)
        if not daily.empty and daily.attrs["pobrano"] < data.bar_end("D1", daily.index[-1]) <= observed_at:
            daily = _closed_as_observed(daily, "D1")
        return build_frames(daily, hourly)

    def short_snapshot(self, ticker: str) -> dict:
        period = "5d" if archive.path_for(ticker, "M5").exists() else "1mo"
        frame = _download(ticker, "M5", data.load_intraday, "M5", period)
        return _latest_bar(frame, "M5", frame.attrs["pobrano"])
