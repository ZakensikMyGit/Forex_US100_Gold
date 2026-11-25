"""Automatyczne przesunięcie względem xStation.

Ręczna kalibracja porównuje bieżącą cenę z xStation z opóźnioną świecą Yahoo, więc ruch rynku
z tych ok. 10-15 minut trafia do przesunięcia. Tu porównujemy ceny z TEJ SAMEJ chwili:

- złoto: cena spot XAU z gold-api (bieżąca, zapisywana przez monitor co 5 min) z zamknięciem
  świecy M5 kontraktu GC=F, która skończyła się w tej samej minucie,
- US100: indeks kasowy ^NDX z zamknięciem M5 kontraktu NQ=F na tej samej świecy
  (oba z Yahoo, z identycznym opóźnieniem; ^NDX notowany tylko w sesji kasowej USA).

Wynik to mediana ostatnich par, więc pojedyncze rozjazdy nie przesuwają cen. Zmiana kontraktu
(GC=F, NQ=F) przestaje być problemem: różnica jest mierzona na bieżąco.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import contracts, delivery, gold_spot
from .data import BAR_LENGTH
from .instruments import ALL

SETTING = "automatyczne przesunięcie"
RECORD = "przesunięcie auto"
CORRECTION = "korekta przesunięcia"
SPOT_PATH = delivery.DATA_DIR / "spot_zlota.json"
SAMPLES = 12  # ok. godzina świec M5
MIN_SAMPLES = 3
PAIR_TOLERANCE = pd.Timedelta(seconds=90)
KEEP_SPOT = timedelta(days=3)
NDX = "^NDX"


@dataclass(frozen=True)
class Estimate:
    value: float  # różnica cena referencyjna - kontrakt (bez korekty)
    samples: int
    spread: float  # rozrzut par (maks. - min.)
    at: pd.Timestamp  # czas ostatniej pary
    source: str
    price: float  # ostatnie zamknięcie kontraktu w parach (do kontroli wiarygodności)


def _estimate(differences: pd.Series, prices: pd.Series, source: str) -> Estimate | None:
    differences = differences.dropna().sort_index().iloc[-SAMPLES:]
    if len(differences) < MIN_SAMPLES:
        return None
    return Estimate(float(np.median(differences)), len(differences), float(differences.max() - differences.min()),
                    pd.Timestamp(differences.index[-1]), source, float(prices.loc[differences.index[-1]]))


# ------------------------------------------------------------------ złoto

def load_spots() -> list[dict]:
    rows = delivery.read_json(SPOT_PATH, [])
    return rows if isinstance(rows, list) else []


def record_spot(price: float, updated: datetime, now: datetime | None = None) -> None:
    """Dopisz notowanie spot (bez duplikatów) i usuń wpisy starsze niż 3 dni."""
    now = now or datetime.now(timezone.utc)
    stamp = updated.astimezone(timezone.utc).isoformat()
    rows = [r for r in load_spots() if datetime.fromisoformat(r["czas"]) >= now - KEEP_SPOT and r["czas"] != stamp]
    rows.append({"czas": stamp, "cena": float(price)})
    with delivery.file_lock(SPOT_PATH.with_suffix(".lock")):
        delivery.write_json(SPOT_PATH, sorted(rows, key=lambda r: r["czas"]))


def gold_estimate(spots: list[dict], m5: pd.DataFrame) -> Estimate | None:
    """Pary: notowanie spot i zamknięcie M5 GC=F kończącej się w tej samej chwili (±90 s)."""
    if m5.empty or not spots:
        return None
    ends = pd.DatetimeIndex(m5.index + BAR_LENGTH["M5"]).tz_convert("UTC")
    closes = m5["close"].to_numpy()
    differences, prices = {}, {}
    for row in spots:
        moment = pd.Timestamp(row["czas"]).tz_convert("UTC")
        i = int(ends.searchsorted(moment))
        best = min((j for j in (i - 1, i) if 0 <= j < len(ends)), key=lambda j: abs(ends[j] - moment), default=None)
        if best is not None and abs(ends[best] - moment) <= PAIR_TOLERANCE:
            differences[ends[best]] = float(row["cena"]) - float(closes[best])
            prices[ends[best]] = float(closes[best])
    return _estimate(pd.Series(differences, dtype=float), pd.Series(prices, dtype=float),
                     "spot XAU (gold-api) – GC=F, świece M5")


# ------------------------------------------------------------------ US100

def us100_estimate(ndx_m5: pd.DataFrame, nq_m5: pd.DataFrame) -> Estimate | None:
    """Pary: zamknięcia M5 indeksu ^NDX i kontraktu NQ=F z tej samej świecy (sesja kasowa USA)."""
    if ndx_m5.empty or nq_m5.empty:
        return None
    joined = pd.concat([ndx_m5["close"].rename("ndx"), nq_m5["close"].rename("nq")], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    joined.index = pd.DatetimeIndex(joined.index + BAR_LENGTH["M5"]).tz_convert("UTC")
    return _estimate(joined["ndx"] - joined["nq"], joined["nq"], "indeks ^NDX – NQ=F, świece M5")


# ------------------------------------------------------------------ zastosowanie

def apply(cfg: dict, key: str, estimate: Estimate | None) -> str | None:
    """Zapisz pomiar w ustawieniach i - gdy różnica jest istotna - nowe przesunięcie.

    Przesunięcie zmienia się dopiero, gdy nowa wartość różni się o co najmniej typowy koszt
    transakcji (US100 2 pkt, złoto 0,5 USD). Drobne wahania nie przestawiają więc poziomów i alertów.
    Zwraca opis zmiany albo None."""
    if estimate is None or not cfg.get(SETTING, True):
        return None
    inst = ALL[key]
    correction = float(cfg.get(CORRECTION, 0) or 0)
    value = round(estimate.value + correction, inst.decimals)
    previous = cfg.get(RECORD) or {}
    if previous.get("czas") and pd.Timestamp(previous["czas"]) >= estimate.at:
        return None  # ten pomiar już był
    cfg[RECORD] = {
        "wartość": value, "pomiar": round(estimate.value, inst.decimals), "próbki": estimate.samples,
        "rozrzut": round(estimate.spread, inst.decimals), "czas": estimate.at.isoformat(), "źródło": estimate.source,
    }
    if contracts.offset_problem(value, estimate.price):
        cfg[RECORD]["odrzucono"] = "nierealnie duża wartość"
        return None
    current = float(cfg.get("przesunięcie", 0) or 0)
    if abs(value - current) < inst.typical_cost:
        return None
    cfg["przesunięcie"] = value
    return f"przesunięcie {current:+g} -> {value:+g} ({estimate.source}, {estimate.samples} par)"


def measure(key: str, m5: pd.DataFrame, *, load_ndx=None, quote=None) -> Estimate | None:
    """Pomiar dla monitora: złoto zapisuje bieżący spot, US100 pobiera ^NDX."""
    if key == "GOLD":
        price, updated = (quote or gold_spot.load_quote)()
        record_spot(price, updated)
        return gold_estimate(load_spots(), m5)
    if key == "US100":
        from . import data
        ndx = (load_ndx or (lambda: data.load_intraday(NDX, "M5", "5d")))()
        return us100_estimate(ndx, m5)
    return None
