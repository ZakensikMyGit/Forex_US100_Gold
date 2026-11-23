"""Zachowuje zamknięte świece Yahoo w lokalnym archiwum Parquet."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pandas as pd

from .data import bar_end
from .delivery import DATA_DIR, file_lock

ARCHIVE_DIR = DATA_DIR / "notowania"
COLUMNS = ["open", "high", "low", "close", "volume"]
OBSERVED = "_pobrano_utc"


def path_for(ticker: str, interval: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9^=._-]+", ticker) or interval not in ("D1", "H1"):
        raise ValueError("Nieobsługiwany symbol lub interwał archiwum.")
    return ARCHIVE_DIR / f"{ticker}_{interval}.parquet"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans or not df.index.is_unique:
        raise ValueError(f"Niepoprawny indeks archiwum {path.name}.")
    return df[[*COLUMNS, OBSERVED]].sort_index()


def load(ticker: str, interval: str) -> pd.DataFrame:
    path = path_for(ticker, interval)
    result = _read(path)[COLUMNS]
    if result.empty:
        raise ValueError(f"Archiwum {ticker} {interval} nie zawiera zamkniętych notowań.")
    return result


def save(ticker: str, interval: str, frame: pd.DataFrame, observed_at: pd.Timestamp) -> None:
    if frame.empty:
        return
    ends = pd.to_datetime([bar_end(interval, t) for t in frame.index], utc=True)
    incoming = frame.loc[ends <= observed_at, COLUMNS].dropna(subset=COLUMNS[:4]).copy()
    if incoming.empty:
        return
    incoming[OBSERVED] = observed_at
    path = path_for(ticker, interval)
    with file_lock(path.with_suffix(".lock"), timeout=2):
        if path.exists():
            saved = _read(path)
            # Dwa procesy mogą skończyć pobieranie w innej kolejności, niż je rozpoczęły.
            joined = pd.concat([saved, incoming]).sort_values(OBSERVED, kind="stable")
            incoming = joined.loc[~joined.index.duplicated(keep="last")]
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            incoming.sort_index().to_parquet(temp, engine="pyarrow")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
