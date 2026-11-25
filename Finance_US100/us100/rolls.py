"""Zmiany kontraktu w ciągłych notowaniach Yahoo (GC=F, NQ=F) i wyrównanie historii.

GC=F i NQ=F to ciągłe serie: w dniu zmiany Yahoo przełącza je z kontraktu bieżącego na następny,
a różnica cen między kontraktami (złoto ok. 20-30 USD, US100 ok. 200-300 pkt) wygląda na wykresie
jak nagły ruch rynku. Fałszywa luka zamyka wirtualne pozycje, zmienia wskaźniki i wywołuje alerty.

Wykrywanie: w okolicy przewidywanej zmiany monitor porównuje ostatnie świece H1 serii ciągłej
z kontraktem bieżącym i następnym (np. GCZ26.CMX i GCG27.CMX). Pierwsza świeca, od której seria
pokrywa się z następnym kontraktem, wyznacza chwilę zmiany; skok to różnica cen kontraktów.

Wyrównanie (metoda „panama”): wszystkie świece zakończone przed zmianą przesuwamy o skok, więc
historia jest w skali nowego kontraktu i nie ma luki. Świecę H1 w chwili zmiany zastępujemy świecą
następnego kontraktu, a sesję D1 z dnia zmiany składamy na nowo z wyrównanych świec H1.
Archiwum zostaje surowe - wyrównanie dzieje się przy każdym odczycie.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import contracts, delivery
from .data import BAR_LENGTH, bar_end

REGISTRY_PATH = delivery.DATA_DIR / "zmiany_kontraktu.json"
PRICE_COLUMNS = ("open", "high", "low", "close")
WINDOW_DAYS = 7  # wykrywanie tylko w okolicy przewidywanej zmiany

SPECS = {  # klucz instrumentu: (ticker ciągły, prefiks, sufiks Yahoo, miesiące kontraktów)
    "US100": ("NQ=F", "NQ", ".CME", (3, 6, 9, 12)),
    "GOLD": ("GC=F", "GC", ".CMX", (2, 4, 6, 8, 10, 12)),
}
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


@dataclass(frozen=True)
class Roll:
    ticker: str
    at: pd.Timestamp  # początek pierwszej świecy H1 w skali nowego kontraktu
    gap: float  # cena nowego kontraktu minus cena starego
    old: str
    new: str
    bar: dict | None = None  # OHLC nowego kontraktu dla świecy H1 w chwili zmiany


# ------------------------------------------------------------------ rejestr

def load(ticker: str | None = None) -> list[Roll]:
    rows = delivery.read_json(REGISTRY_PATH, [])
    rolls = []
    for row in rows if isinstance(rows, list) else []:
        try:
            roll = Roll(row["ticker"], pd.Timestamp(row["od"]).tz_convert("UTC"), float(row["skok"]),
                        row.get("z", ""), row.get("na", ""), row.get("świeca"))
        except (KeyError, TypeError, ValueError):
            continue
        if ticker is None or roll.ticker == ticker:
            rolls.append(roll)
    return sorted(rolls, key=lambda r: r.at)


def save(roll: Roll) -> bool:
    """Dopisz zmianę kontraktu; zwraca False, jeśli ta para kontraktów jest już w rejestrze."""
    with delivery.file_lock(REGISTRY_PATH.with_suffix(".lock")):
        rows = delivery.read_json(REGISTRY_PATH, [])
        rows = rows if isinstance(rows, list) else []
        if any(r.get("ticker") == roll.ticker and r.get("z") == roll.old and r.get("na") == roll.new for r in rows):
            return False
        rows.append({"ticker": roll.ticker, "od": roll.at.isoformat(), "skok": roll.gap, "z": roll.old,
                     "na": roll.new, "świeca": roll.bar, "wykryto": pd.Timestamp.now(tz="UTC").isoformat()})
        delivery.write_json(REGISTRY_PATH, sorted(rows, key=lambda r: r["od"]))
    return True


# ------------------------------------------------------------------ kontrakty

def contract_pair(key: str, today: date) -> tuple[str, str]:
    """Symbole Yahoo kontraktu bieżącego i następnego, np. ("GCZ26.CMX", "GCG27.CMX")."""
    _, prefix, suffix, months = SPECS[key]
    listed = [(year, month) for year in (today.year, today.year + 1) for month in months]
    front = next(i for i, (year, month) in enumerate(listed) if (year, month) >= (today.year, today.month))
    if key == "US100":
        year, month = listed[front]
        if date(year, month, 1) <= today and today > contracts._third_friday(year, month):
            front += 1  # kontrakt już wygasł w tym miesiącu
    symbols = [f"{prefix}{MONTH_CODES[m]}{y % 100:02d}{suffix}" for y, m in listed[front:front + 2]]
    return symbols[0], symbols[1]


def near_roll(key: str, today: date) -> bool:
    last, upcoming = contracts.rolls_around(key, today)
    return any(d is not None and abs((today - d).days) <= WINDOW_DAYS for d in (last, upcoming))


def detect(ticker: str, continuous: pd.DataFrame, front: pd.DataFrame, following: pd.DataFrame,
           old: str, new: str) -> Roll | None:
    """Znajdź chwilę, od której seria ciągła pokrywa się z następnym kontraktem zamiast z bieżącym."""
    joined = pd.concat([continuous["close"].rename("c"), front["close"].rename("f"), following["close"].rename("n")],
                       axis=1, join="inner").dropna()
    if len(joined) < 4:
        return None
    spread = joined["n"] - joined["f"]
    gap = float(spread.median())
    if not np.isfinite(gap) or abs(gap) < 1e-9:
        return None
    tolerance = 0.25 * abs(gap)
    to_front, to_next = (joined["c"] - joined["f"]).abs(), (joined["c"] - joined["n"]).abs()
    regime = np.where((to_front <= tolerance) & (to_front < to_next), "f",
                      np.where((to_next <= tolerance) & (to_next < to_front), "n", ""))
    known = [(t, r) for t, r in zip(joined.index, regime) if r]
    if not known or known[-1][1] != "n" or not any(r == "f" for _, r in known):
        return None  # jeszcze przed zmianą albo zmiana poza zakresem danych
    first_new = max(i for i, (_, r) in enumerate(known) if r == "f") + 1
    at = known[first_new][0]
    around = spread.loc[max(at - pd.Timedelta(hours=3), spread.index[0]):at + pd.Timedelta(hours=3)]
    row = following.loc[at] if at in following.index else None
    bar = {col: float(row[col]) for col in PRICE_COLUMNS} if row is not None else None
    return Roll(ticker, pd.Timestamp(at).tz_convert("UTC"), round(float(around.median()), 4), old, new, bar)


def check(key: str, today: date | None = None, *, load_hourly=None) -> Roll | None:
    """Dla monitora: wykryj i zapisz nową zmianę kontraktu. Zwraca ją tylko, gdy jest nowa."""
    today = today or pd.Timestamp.now(tz="UTC").date()
    if key not in SPECS or not near_roll(key, today):
        return None
    ticker = SPECS[key][0]
    old, new = contract_pair(key, today - pd.Timedelta(days=WINDOW_DAYS))
    if any(r.old == old and r.new == new for r in load(ticker)):
        return None
    if load_hourly is None:
        from . import data
        load_hourly = data.load_hourly
    roll = detect(ticker, load_hourly(ticker, "5d"), load_hourly(old, "5d"), load_hourly(new, "5d"), old, new)
    return roll if roll is not None and save(roll) else None


# ------------------------------------------------------------------ wyrównanie

def _shift_before(frame: pd.DataFrame, ends: pd.DatetimeIndex, roll: Roll) -> pd.DataFrame:
    before = np.asarray(ends <= roll.at)
    if before.any():
        cols = [c for c in PRICE_COLUMNS if c in frame]
        frame.loc[before, cols] = frame.loc[before, cols] + roll.gap
    return frame


def adjust(frame: pd.DataFrame, ticker: str, interval: str, rolls: list[Roll] | None = None) -> pd.DataFrame:
    """Świece H1, M5 lub M15 w skali najnowszego kontraktu (bez sztucznych luk przy zmianach)."""
    rolls = load(ticker) if rolls is None else rolls
    if frame.empty or not rolls:
        return frame
    out = frame.copy()
    out.attrs = frame.attrs.copy()
    ends = pd.DatetimeIndex(out.index + BAR_LENGTH[interval]).tz_convert("UTC")
    starts = pd.DatetimeIndex(out.index).tz_convert("UTC")
    for roll in rolls:
        out = _shift_before(out, ends, roll)
        if interval == "H1" and roll.bar:
            hit = np.asarray(starts == roll.at)
            for col in PRICE_COLUMNS:
                out.loc[hit, col] = roll.bar[col]
    return out


def adjust_daily(daily: pd.DataFrame, ticker: str, hourly_adjusted: pd.DataFrame,
                 rolls: list[Roll] | None = None) -> pd.DataFrame:
    """D1: sesje zakończone przed zmianą przesunięte o skok, sesja ze zmianą złożona z wyrównanych H1."""
    rolls = load(ticker) if rolls is None else rolls
    if daily.empty or not rolls:
        return daily
    out = daily.copy()
    out.attrs = daily.attrs.copy()
    ends = pd.DatetimeIndex(pd.to_datetime([bar_end("D1", t) for t in out.index], utc=True))
    hourly_starts = pd.DatetimeIndex(hourly_adjusted.index).tz_convert("UTC")
    for roll in rolls:
        out = _shift_before(out, ends, roll)
        i = int(ends.searchsorted(roll.at, side="right"))
        if i >= len(out):
            continue
        session_start = ends[i - 1] if i > 0 else ends[i] - pd.Timedelta(days=1)
        inside = hourly_adjusted.loc[np.asarray((hourly_starts >= session_start) & (hourly_starts < ends[i]))]
        if inside.empty:
            continue
        label = out.index[i]
        out.loc[label, "open"] = float(inside["open"].iloc[0])
        out.loc[label, "high"] = float(inside["high"].max())
        out.loc[label, "low"] = float(inside["low"].min())
        if ends[i] <= hourly_starts[-1] + BAR_LENGTH["H1"]:
            out.loc[label, "close"] = float(inside["close"].iloc[-1])
    return out
