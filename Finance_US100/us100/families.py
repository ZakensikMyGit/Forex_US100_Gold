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

from . import advisor, indicators, regime
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
    live_start: pd.Timestamp | None = None  # początek symulacji na żywo tej wersji reguł
    live: advisor.History | None = None
    live_trades: tuple[Trade, ...] = ()
    live_open: OpenPosition | None = None


@dataclass(frozen=True)
class OpenPosition:
    candidate: Candidate
    opened: pd.Timestamp
    entry: float

    @property
    def stop(self) -> float:
        return self.entry - self.candidate.sign * self.candidate.risk

    @property
    def target(self) -> float:
        return self.entry + self.candidate.sign * self.candidate.reward


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
    instruments: tuple[str, ...] = ("US100", "GOLD")


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
    return simulate_open(h1, ends, candidates, cost)[0]


def simulate_open(h1: pd.DataFrame, ends: pd.DatetimeIndex, candidates: list[Candidate],
                  cost: float) -> tuple[list[Trade], OpenPosition | None]:
    """Jak ``simulate``, ale zwraca też pozycję, która wciąż jest otwarta na ostatniej świecy."""
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
            # pozycja wciąż otwarta - kolejnych nie da się uczciwie rozliczyć
            return trades, OpenPosition(candidate, dates[k], entry)
        label, price, closed = exit_
        busy_until = closed
        trades.append(Trade(candidate, dates[k], closed, label, (price - entry) * sign - cost))
    return trades, None


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


# ------------------------------------------------------------------ rodzina B: korekta w trendzie

PULLBACK_LOOKBACK = 6


def pullback_rule(h1: pd.DataFrame, ends: pd.DatetimeIndex, regimes: np.ndarray, high_volatility: np.ndarray,
                  d1_bias: np.ndarray) -> list[Candidate]:
    """Korekta w trendzie: tylko w reżimie trendu H4 zgodnym z D1, przy zwykłej zmienności.

    LONG: w ostatnich 6 świecach H1 RSI spadł do 45 lub niżej (korekta), a teraz H1 zamyka się
    z powrotem nad EMA20 po zamknięciu pod nią. SHORT symetrycznie (RSI 55 lub wyżej, powrót pod EMA20).
    SL za dołkiem/szczytem korekty + 0,1 ATR, nie bliżej niż 1 ATR; sygnał pomijany, gdy SL > 2 ATR.
    TP 2× ryzyko, limit 24 godz."""
    closes, highs, lows, rsi, atr = (h1[col].to_numpy() for col in ("close", "high", "low", "rsi", "atr"))
    ema20 = indicators.ema(h1["close"], 20).to_numpy()
    found = []
    n = PULLBACK_LOOKBACK
    for i in range(max(n, 20), len(h1)):
        sign = 1 if regimes[i] == regime.UP else -1 if regimes[i] == regime.DOWN else 0
        if not sign or high_volatility[i] or d1_bias[i] != sign or not isfinite(atr[i]) or atr[i] <= 0:
            continue
        if not ((closes[i] - ema20[i]) * sign > 0 >= (closes[i - 1] - ema20[i - 1]) * sign):
            continue
        depth = rsi[i - n:i]
        if not np.isfinite(depth).any():
            continue
        if (np.nanmin(depth) > 45) if sign > 0 else (np.nanmax(depth) < 55):
            continue
        swing = lows[i - n:i + 1].min() if sign > 0 else highs[i - n:i + 1].max()
        risk = max(float(atr[i]), (closes[i] - swing) * sign + 0.1 * float(atr[i]))
        if risk > 2 * atr[i]:
            continue
        direction = "long" if sign > 0 else "short"
        found.append(Candidate(
            "korekta", direction, ends[i], float(closes[i]), risk, 2 * risk, 24.0,
            f"{regimes[i].capitalize()} H4 zgodny z D1 · powrót {'nad' if sign > 0 else 'pod'} EMA20 H1 po korekcie RSI",
        ))
    return found


def _pullbacks(frames: dict[str, pd.DataFrame], now: pd.Timestamp, _params: Params) -> list[Candidate]:
    h1, ends = closed_bars(frames["H1"], "H1", now)
    if h1.empty:
        return []
    known = regime.at(frames["H4"], ends)
    d1 = frames["D1"]
    d1_ends = pd.DatetimeIndex(pd.to_datetime([bar_end("D1", t) for t in d1.index], utc=True))
    rows = d1_ends.searchsorted(ends, side="right") - 1
    ema50 = d1["ema50"].to_numpy() if "ema50" in d1 else indicators.ema(d1["close"], 50).to_numpy()
    side = np.sign(d1["close"].to_numpy() - ema50)
    bias = np.where(rows >= 0, side[np.clip(rows, 0, None)], 0)
    return pullback_rule(h1, ends, known["reżim"].to_numpy(), known["wysoka zmienność"].to_numpy(dtype=bool), bias)


# ------------------------------------------------------------------ rodzina C: wybicie zakresu otwarcia (US100)

NY = "America/New_York"
RANGE_START, FIRST_SIGNAL, LAST_SIGNAL, SESSION_CLOSE = 9, 10, 14, 16  # pełne godziny czasu nowojorskiego


def opening_range_rule(h1: pd.DataFrame, ends: pd.DatetimeIndex) -> list[Candidate]:
    """Wybicie zakresu otwarcia sesji kasowej USA na świecach H1.

    Zakres: świeca H1 09:00-10:00 czasu nowojorskiego (15:00-16:00 czasu polskiego poza krótkimi
    okresami różnej zmiany czasu) - zawiera otwarcie sesji kasowej o 09:30. Sygnał: pierwsze zamknięcie
    H1 rozpoczętej między 10:00 a 14:00 NY za maksimum (LONG) lub minimum (SHORT) zakresu, najwyżej jeden
    na dzień. SL w połowie zakresu; dzień pomijany, gdy ryzyko < 0,5 ATR albo > 2 ATR. TP 1,5× ryzyko,
    wyjście najpóźniej na zamknięciu sesji kasowej o 16:00 NY."""
    local = h1.index.tz_convert(NY)
    hours = local.hour.to_numpy()
    days = local.normalize()
    highs, lows, closes, atr = (h1[col].to_numpy() for col in ("high", "low", "close", "atr"))
    found = []
    opening: tuple[pd.Timestamp, float, float] | None = None
    done = None
    for i in range(len(h1)):
        day = days[i]
        if hours[i] == RANGE_START and local[i].minute == 0:
            opening = (day, float(highs[i]), float(lows[i]))
            continue
        if opening is None or opening[0] != day or done == day or not FIRST_SIGNAL <= hours[i] <= LAST_SIGNAL:
            continue
        _, top, bottom = opening
        sign = 1 if closes[i] > top else -1 if closes[i] < bottom else 0
        if not sign:
            continue
        done = day  # tylko pierwsze wybicie w dniu, także gdy ryzyko jest poza zakresem
        middle = (top + bottom) / 2
        risk = (closes[i] - middle) * sign
        if not isfinite(atr[i]) or not 0.5 * atr[i] <= risk <= 2 * atr[i]:
            continue
        close_at = (day + pd.Timedelta(hours=SESSION_CLOSE)).tz_convert(h1.index.tz)
        limit = (close_at - ends[i]) / HOUR
        if limit <= 0:
            continue
        found.append(Candidate(
            "otwarcie", "long" if sign > 0 else "short", ends[i], float(closes[i]), float(risk), 1.5 * float(risk),
            float(limit), f"Zamknięcie H1 za {'maksimum' if sign > 0 else 'minimum'} zakresu 09:00-10:00 NY "
                          f"({top:g}-{bottom:g}) · wyjście najpóźniej o 16:00 NY",
        ))
    return found


def _opening_ranges(frames: dict[str, pd.DataFrame], now: pd.Timestamp, _params: Params) -> list[Candidate]:
    h1, ends = closed_bars(frames["H1"], "H1", now)
    return opening_range_rule(h1, ends) if not h1.empty else []


# ------------------------------------------------------------------ rodzina E: powrót do średniej w konsolidacji

def mean_reversion_rule(h1: pd.DataFrame, ends: pd.DatetimeIndex, regimes: np.ndarray,
                        high_volatility: np.ndarray) -> list[Candidate]:
    """Powrót do średniej, tylko gdy H4 jest w konsolidacji i zmienność jest zwykła.

    LONG: minimum H1 pod dolną wstęgą Bollingera, zamknięcie z powrotem nad nią i RSI H1 <= 40.
    SHORT: maksimum nad górną wstęgą, zamknięcie pod nią i RSI >= 60.
    TP na środkowej wstędze (średnia 20 H1). SL za ekstremum świecy + 0,1 ATR, nie bliżej niż 1 ATR;
    sygnał pomijany, gdy SL > 2 ATR albo odległość do TP jest mniejsza niż ryzyko. Limit 12 godz."""
    closes, highs, lows, rsi, atr, low_band, mid, up_band = (
        h1[col].to_numpy() for col in ("close", "high", "low", "rsi", "atr", "bb_low", "bb_mid", "bb_up"))
    found = []
    for i in range(len(h1)):
        if regimes[i] != regime.RANGE or high_volatility[i]:
            continue
        values = (closes[i], rsi[i], atr[i], low_band[i], mid[i], up_band[i])
        if not all(isfinite(v) for v in values) or atr[i] <= 0:
            continue
        if lows[i] < low_band[i] < closes[i] and rsi[i] <= 40:
            sign, extreme = 1, lows[i]
        elif highs[i] > up_band[i] > closes[i] and rsi[i] >= 60:
            sign, extreme = -1, highs[i]
        else:
            continue
        risk = max(float(atr[i]), (closes[i] - extreme) * sign + 0.1 * float(atr[i]))
        reward = (mid[i] - closes[i]) * sign
        if risk > 2 * atr[i] or reward < risk:
            continue
        found.append(Candidate(
            "powrot", "long" if sign > 0 else "short", ends[i], float(closes[i]), risk, float(reward), 12.0,
            f"Konsolidacja H4 · powrót {'nad dolną' if sign > 0 else 'pod górną'} wstęgę Bollingera H1, "
            f"RSI {rsi[i]:.0f}, cel: średnia 20 H1 ({mid[i]:g})",
        ))
    return found


def _mean_reversions(frames: dict[str, pd.DataFrame], now: pd.Timestamp, _params: Params) -> list[Candidate]:
    h1, ends = closed_bars(frames["H1"], "H1", now)
    if h1.empty:
        return []
    known = regime.at(frames["H4"], ends)
    return mean_reversion_rule(h1, ends, known["reżim"].to_numpy(), known["wysoka zmienność"].to_numpy(dtype=bool))


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
    Family("korekta", "B. Korekta w trendzie (filtr reżimu)",
           "Tylko w trendzie H4 (EMA50/EMA200, nachylenie EMA50, ADX ≥ 20) zgodnym z D1 (zamknięcie wobec EMA50) "
           "i przy zwykłej zmienności. Wejście, gdy po korekcie (RSI H1 ≤ 45 dla LONG, ≥ 55 dla SHORT w 6 świecach) "
           "H1 zamyka się z powrotem za EMA20. SL za korektą, min. 1 ATR, maks. 2 ATR; TP 2× ryzyko, limit 24 godz.",
           _pullbacks),
    Family("otwarcie", "C. Wybicie zakresu otwarcia sesji USA",
           "Zakres: świeca H1 09:00–10:00 czasu nowojorskiego (zawiera otwarcie sesji kasowej o 09:30). "
           "Pierwsze zamknięcie świecy H1 rozpoczętej między 10:00 a 14:00 NY za zakresem, najwyżej raz dziennie. "
           "SL w połowie zakresu (0,5–2 ATR), TP 1,5× ryzyko, wyjście najpóźniej o 16:00 NY.",
           _opening_ranges, instruments=("US100",)),
    Family("powrot", "E. Powrót do średniej w konsolidacji",
           "Tylko w konsolidacji H4 (brak trendu wg EMA50/EMA200 i ADX < 20) przy zwykłej zmienności. "
           "Wejście, gdy H1 wybija knotem poza wstęgę Bollingera i zamyka się z powrotem w środku "
           "(RSI ≤ 40 dla LONG, ≥ 60 dla SHORT). TP na średniej 20 H1, SL za knotem (1–2 ATR), "
           "tylko gdy TP jest co najmniej tak daleko jak SL. Limit 12 godz.",
           _mean_reversions, instruments=("US100",)),
)


def for_instrument(key: str) -> tuple[Family, ...]:
    return tuple(family for family in FAMILIES if key in family.instruments)


# ------------------------------------------------------------------ ocena i bramka

def evaluate(family: Family, frames: dict[str, pd.DataFrame], params: Params, now: pd.Timestamp,
             *, tested: int = 1, holdout_days: int = HOLDOUT_DAYS, live_start: pd.Timestamp | None = None,
             unit: str = "pkt") -> Evaluation:
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
    verdict = advisor.against_chance(family.name, total, chance, z, unit)
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
    live, live_trades, live_open = None, (), None
    if live_start is not None:
        # Symulacja na żywo: te same reguły i rozliczenie, ale tylko sygnały po starcie tej wersji reguł.
        forward, live_open = simulate_open(h1, ends, [c for c in candidates if c.at >= live_start], params.cost)
        live, live_trades = summarize(forward), tuple(forward)
    return Evaluation(family.key, family.name, family.description, tuple(windows), total, chance, holdout,
                      holdout_start, status, verdict, fresh[-1] if fresh else None,
                      live_start, live, live_trades, live_open)


def evaluate_all(frames: dict[str, pd.DataFrame], params: Params, now: pd.Timestamp,
                 instrument: str = "US100", live_starts: dict[str, pd.Timestamp] | None = None) -> list[Evaluation]:
    """Ocena rodzin danego instrumentu; próg uwzględnia, ile rodzin jest sprawdzanych."""
    chosen = for_instrument(instrument)
    starts = live_starts or {}
    unit = "USD" if instrument == "GOLD" else "pkt"
    return [evaluate(family, frames, params, now, tested=len(chosen), live_start=starts.get(family.key), unit=unit)
            for family in chosen]


# ------------------------------------------------------------------ symulacja na żywo

LIVE_MIN_DAYS = 28
LIVE_MIN_TRADES = 20


def live_starts(state: dict, instrument: str, now: pd.Timestamp) -> tuple[dict[str, pd.Timestamp], bool]:
    """Początek symulacji na żywo dla każdej rodziny i informacja, czy stan trzeba zapisać.

    ``state`` to ustawienie „symulacja rodzin”: {"start": iso, "wersje": {klucz: {"opis", "od"}}}.
    Zmiana opisu reguł rodziny oznacza nową wersję - jej wyniki na żywo liczą się od nowa."""
    if not state.get("start"):
        return {}, False
    start = pd.Timestamp(state["start"])
    versions = state.setdefault("wersje", {})
    changed = False
    starts = {}
    for family in for_instrument(instrument):
        version = versions.get(family.key)
        if not isinstance(version, dict) or version.get("opis") != family.description:
            version = {"opis": family.description, "od": max(start, now).isoformat()}
            versions[family.key] = version
            changed = True
        starts[family.key] = max(start, pd.Timestamp(version["od"]))
    return starts, changed


def readiness(evaluation: Evaluation, now: pd.Timestamp) -> list[tuple[str, bool]]:
    """Warunki, które rodzina musi spełnić, zanim dostanie alerty „OTWÓRZ”."""
    live = evaluation.live
    days = (now - evaluation.live_start).days if evaluation.live_start is not None else 0
    count = live.count if live else 0
    return [
        ("Przewaga w testach historycznych", evaluation.status == "przewaga"),
        (f"Symulacja na żywo co najmniej {LIVE_MIN_DAYS} dni (jest {days})", days >= LIVE_MIN_DAYS),
        (f"Co najmniej {LIVE_MIN_TRADES} zakończonych pozycji na żywo (jest {count})", count >= LIVE_MIN_TRADES),
        ("Średni wynik na żywo powyżej zera po kosztach",
         bool(live and live.mean_points is not None and live.mean_points > 0)),
    ]


def margin(stats: advisor.History, tested: int) -> float | None:
    if not stats.std_points or stats.count < 2:
        return None
    return margin_z(tested) * stats.std_points / sqrt(stats.count)
