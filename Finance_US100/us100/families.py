"""Rodziny sygnałów wejścia i bramka dopuszczenia.

Każda rodzina to osobna hipoteza z regułami ustalonymi z góry. Zgłasza kandydatów tylko na
zamkniętych świecach, a wszystkie rodziny są rozliczane tak samo: wejście na otwarciu następnej
świecy H1, koszt, SL przed TP w tej samej świecy, limit czasu, jedna pozycja naraz.

Rodzina może trafić do alertów dopiero wtedy, gdy przejdzie bramkę:
- co najmniej ``MIN_TRADES`` zakończonych pozycji przed danymi nietkniętymi,
- średni wynik minus margines błędu (z korektą na liczbę sprawdzanych rodzin) powyżej zera
  i powyżej losowych wejść z tymi samymi odległościami SL/TP,
- dodatni średni wynik w co najmniej 2 z 3 kolejnych okien.

Ostatnie ``HOLDOUT_DAYS`` dni to dane nietknięte: nie wchodzą do bramki i są pokazywane
tylko na wyraźne żądanie, żeby nie dopasowywać reguł do wyniku.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import advisor
from .data import BAR_LENGTH, bar_end

HOLDOUT_DAYS = 120
WARMUP_BARS = 200  # świeże wskaźniki (EMA200, ATR) na początku historii
MIN_TRADES = 100
WINDOWS = 3
MIN_POSITIVE_WINDOWS = 2
MAX_LIVE_AGE = pd.Timedelta(hours=2)
HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class Candidate:
    family: str
    direction: str
    at: pd.Timestamp  # koniec świecy sygnałowej
    reference: float  # zamknięcie świecy sygnałowej (orientacyjna cena wejścia)
    risk: float  # odległość SL od wejścia
    reward: float  # odległość TP od wejścia
    limit_h: float
    note: str

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1

    @property
    def stop(self) -> float:
        return self.reference - self.sign * self.risk

    @property
    def target(self) -> float:
        return self.reference + self.sign * self.reward


@dataclass(frozen=True)
class Trade:
    candidate: Candidate
    opened: pd.Timestamp
    closed: pd.Timestamp
    outcome: str
    points: float


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp
    stats: advisor.History
    random: advisor.History


@dataclass(frozen=True)
class Evaluation:
    key: str
    name: str
    description: str
    windows: tuple[Window, ...]
    total: advisor.History
    random: advisor.History
    holdout: advisor.History
    holdout_start: pd.Timestamp
    status: str  # "przewaga", "brak przewagi", "za mało danych"
    verdict: tuple[str, str]
    latest: Candidate | None  # świeży kandydat do pokazania w panelu


@dataclass(frozen=True)
class Params:
    cost: float
    min_rr: float
    limit_h: float


@dataclass(frozen=True)
class Family:
    key: str
    name: str
    description: str
    candidates: Callable[[dict[str, pd.DataFrame], pd.Timestamp, Params], list[Candidate]]


# ------------------------------------------------------------------ wspólne narzędzia

def closed_bars(frame: pd.DataFrame, interval: str, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    ends = pd.DatetimeIndex(frame.index) + BAR_LENGTH[interval]
    observed = min(now, frame.attrs.get("pobrano", now))
    count = int(ends.searchsorted(observed, side="right"))
    return frame.iloc[:count], ends[:count]


def atr_exits(atr: float, sl_atr: float = 1.0, rr: float = 1.5) -> tuple[float, float]:
    """SL w odległości ``sl_atr`` × ATR, TP w odległości ``rr`` × ryzyko."""
    risk = sl_atr * atr
    return risk, rr * risk


def simulate(h1: pd.DataFrame, ends: pd.DatetimeIndex, candidates: list[Candidate], cost: float) -> list[Trade]:
    """Wejście na otwarciu pierwszej świecy po sygnale, jedna pozycja naraz."""
    prices = tuple(h1[col].to_numpy() for col in ("open", "high", "low", "close"))
    dates = h1.index
    trades: list[Trade] = []
    busy_until: pd.Timestamp | None = None
    for candidate in sorted(candidates, key=lambda c: c.at):
        k = int(dates.searchsorted(candidate.at))
        if k >= len(dates) or (busy_until is not None and dates[k] < busy_until):
            continue
        sign = candidate.sign
        entry = float(prices[0][k])
        exit_ = advisor._resolve(prices, dates, ends, k, sign, entry - sign * candidate.risk,
                                 entry + sign * candidate.reward, dates[k] + pd.Timedelta(hours=candidate.limit_h))
        if exit_ is None:
            break  # pozycja wciąż otwarta - kolejnych nie da się uczciwie rozliczyć
        label, price, closed = exit_
        busy_until = closed
        trades.append(Trade(candidate, dates[k], closed, label, (price - entry) * sign - cost))
    return trades


def summarize(trades: list[Trade]) -> advisor.History:
    if not trades:
        return advisor.History(0, 0, 0, 0, None)
    points = [t.points for t in trades]
    labels = [t.outcome for t in trades]
    return advisor.History(len(trades), labels.count("TP"), labels.count("SL"), labels.count("limit"),
                           float(np.mean(points)), float(np.std(points, ddof=1)) if len(points) > 1 else None)


def margin_z(tested: int) -> float:
    """Dwustronny próg 95% z korektą Bonferroniego na liczbę sprawdzanych rodzin."""
    return NormalDist().inv_cdf(1 - 0.025 / max(tested, 1))


# ------------------------------------------------------------------ rodzina 0: obecny Doradca

def _advisor_candidates(frames: dict[str, pd.DataFrame], now: pd.Timestamp, params: Params) -> list[Candidate]:
    market = advisor.Market(frames, now, history=True)
    found = []
    for i in range(len(market.bars)):
        outcome = market.assess_bar(i, params.cost, params.min_rr)
        proposal = outcome.proposal
        if proposal is None or outcome.reason or proposal.rr is None or proposal.rr < params.min_rr:
            continue
        found.append(Candidate(
            "odbicie", proposal.level.direction, market.ends[i], proposal.entry,
            abs(proposal.entry - proposal.stop), abs(proposal.target - proposal.entry), params.limit_h,
            f"{proposal.level.name} · SL za knotem świecy, TP z poziomu lub 2× ryzyko",
        ))
    return found


# ------------------------------------------------------------------ rodzina A: wybicie poziomu sesji

def _previous_session(frames: dict[str, pd.DataFrame], h1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dla każdej H1: indeks ostatniej zakończonej sesji D1 w chwili otwarcia świecy, jej maksimum i minimum."""
    d1 = frames["D1"]
    d1_ends = pd.DatetimeIndex(pd.to_datetime([bar_end("D1", t) for t in d1.index], utc=True))
    previous = d1_ends.searchsorted(h1.index, side="right") - 1
    return previous, d1["high"].to_numpy(), d1["low"].to_numpy()


def _pressure(highs: np.ndarray, lows: np.ndarray, i: int, sign: int, level: float, atr: float) -> bool:
    """Zapowiedź na niższym interwale: przed wybiciem H1 cena „ściska się” pod poziomem D1.

    Z 4 świec H1 przed wybiciem co najmniej 3 mają wyższe minimum (dla LONG) lub niższe
    maksimum (dla SHORT), a najdalsze ekstremum tych świec dochodzi na 0,5 ATR do poziomu."""
    if i < 5:
        return False
    if sign > 0:
        steps = int((np.diff(lows[i - 5:i]) > 0).sum())
        reach = level - highs[i - 4:i].max()
    else:
        steps = int((np.diff(highs[i - 5:i]) < 0).sum())
        reach = lows[i - 4:i].min() - level
    return steps >= 3 and reach <= 0.5 * atr


def _session_breaks(frames: dict[str, pd.DataFrame], now: pd.Timestamp, *, precursor: bool) -> list[Candidate]:
    h1, ends = closed_bars(frames["H1"], "H1", now)
    if h1.empty:
        return []
    previous, d_high, d_low = _previous_session(frames, h1)
    highs, lows, closes, atr = (h1[col].to_numpy() for col in ("high", "low", "close", "atr"))
    key = "wybicie_zapowiedz" if precursor else "wybicie"
    fired: set[tuple[int, str]] = set()
    found = []
    for i in range(1, len(h1)):
        p = previous[i]
        if p < 0 or not isfinite(atr[i]) or atr[i] <= 0:
            continue
        for direction, level in (("long", d_high[p]), ("short", d_low[p])):
            sign = 1 if direction == "long" else -1
            if not ((closes[i] - level) * sign > 0 >= (closes[i - 1] - level) * sign):
                continue
            if (p, direction) in fired:
                continue  # tylko pierwsze wybicie danego poziomu w sesji
            fired.add((p, direction))
            if precursor and not _pressure(highs, lows, i, sign, level, atr[i]):
                continue
            risk, reward = atr_exits(float(atr[i]))
            name = "maksimum" if sign > 0 else "minimum"
            found.append(Candidate(
                key, direction, ends[i], float(closes[i]), risk, reward, 8.0,
                f"Zamknięcie H1 za {name} poprzedniej sesji ({level:g})"
                + (" po ściskaniu pod poziomem" if precursor else ""),
            ))
    return found


FAMILIES: tuple[Family, ...] = (
    Family("odbicie", "Odbicie od poziomu (obecny Doradca)",
           "Odrzucenie poziomu AB=CD, Fibo H4 lub sesji D1 z filtrami trendu, momentum i wolumenu. "
           "SL za knotem świecy, TP z poziomu albo 2× ryzyko.", _advisor_candidates),
    Family("wybicie", "A. Wybicie poziomu sesji",
           "Pierwsze zamknięcie H1 za maksimum (LONG) lub minimum (SHORT) poprzedniej sesji D1. "
           "SL 1× ATR H1, TP 1,5× ryzyko, limit 8 godz.",
           lambda frames, now, _params: _session_breaks(frames, now, precursor=False)),
    Family("wybicie_zapowiedz", "A+. Wybicie z zapowiedzią na H1",
           "Jak A, ale tylko gdy 4 świece H1 przed wybiciem ściskały się pod poziomem: "
           "co najmniej 3 wyższe minima (LONG) lub niższe maksima (SHORT) i ekstremum w odległości do 0,5 ATR. "
           "Niższy interwał zapowiada ruch na poziomie wyższego.",
           lambda frames, now, _params: _session_breaks(frames, now, precursor=True)),
)


# ------------------------------------------------------------------ ocena i bramka

def evaluate(family: Family, frames: dict[str, pd.DataFrame], params: Params, now: pd.Timestamp,
             *, tested: int = len(FAMILIES), holdout_days: int = HOLDOUT_DAYS) -> Evaluation:
    h1, ends = closed_bars(frames["H1"], "H1", now)
    candidates = family.candidates(frames, now, params)
    trades = simulate(h1, ends, candidates, params.cost)
    empty = advisor.History(0, 0, 0, 0, None)
    holdout_start = (ends[-1] if len(ends) else now) - pd.Timedelta(days=holdout_days)
    research = [t for t in trades if t.closed < holdout_start]
    start = h1.index[min(WARMUP_BARS, len(h1) - 1)] if len(h1) else now
    edges = [start + (holdout_start - start) * part / WINDOWS for part in range(WINDOWS + 1)]
    windows = []
    for w, (left, right) in enumerate(zip(edges, edges[1:])):
        inside = [t for t in research if left <= t.opened < right]
        distances = [(t.candidate.risk, t.candidate.reward) for t in inside]
        limit = inside[0].candidate.limit_h if inside else params.limit_h
        chance = advisor.random_with_distances(h1, ends, distances, params.cost, limit, left, right, seed=w)
        windows.append(Window(left, right, summarize(inside), chance))
    total = summarize([t for t in research if t.opened >= start])
    chance_parts = [w.random for w in windows if w.random.count]
    chance_n = sum(h.count for h in chance_parts)
    chance = advisor.History(
        chance_n, sum(h.targets for h in chance_parts), sum(h.stops for h in chance_parts),
        sum(h.timeouts for h in chance_parts),
        sum(h.mean_points * h.count for h in chance_parts) / chance_n if chance_n else None,
    )
    z = margin_z(tested)
    verdict = advisor.against_chance(family.name, total, chance, z)
    positive = sum(1 for w in windows if w.stats.mean_points is not None and w.stats.mean_points > 0)
    if total.count < MIN_TRADES:
        status = "za mało danych"
        verdict = ("info", f"{family.name}: {total.count}/{MIN_TRADES} zakończonych pozycji przed danymi nietkniętymi "
                           "– za mało, żeby ocenić przewagę.")
    elif verdict[0] == "success" and positive >= MIN_POSITIVE_WINDOWS:
        status = "przewaga"
    else:
        status = "brak przewagi"
        if verdict[0] == "success":
            verdict = ("warning", f"{verdict[1]} Jednak dodatni wynik tylko w {positive}/{WINDOWS} oknach – "
                                  "przewaga jest niestabilna w czasie.")
    holdout = summarize([t for t in trades if t.opened >= holdout_start]) if trades else empty
    fresh = [c for c in candidates if now - c.at <= MAX_LIVE_AGE]
    return Evaluation(family.key, family.name, family.description, tuple(windows), total, chance, holdout,
                      holdout_start, status, verdict, fresh[-1] if fresh else None)


def evaluate_all(frames: dict[str, pd.DataFrame], params: Params, now: pd.Timestamp) -> list[Evaluation]:
    return [evaluate(family, frames, params, now) for family in FAMILIES]


def margin(stats: advisor.History, tested: int = len(FAMILIES)) -> float | None:
    if not stats.std_points or stats.count < 2:
        return None
    return margin_z(tested) * stats.std_points / sqrt(stats.count)
