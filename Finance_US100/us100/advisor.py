"""Propozycje pozycji i odtwarzanie tych samych reguł na zamkniętych świecach."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd

from . import harmonics, indicators, plan
from .data import BAR_LENGTH, bar_end, bars_between
from .signals import rejects_level, trend_series

MIN_HISTORY = 30
MAX_SIGNAL_AGE = pd.Timedelta(hours=2)
ALERT_SETTING = "powiadomienia doradcy"
MIN_VALIDATION_BARS = 250
ELEVATED_VOLUME = 1.5
RANDOM_DRAWS = 20  # losowych wejść na każdą pozycję wariantu bazowego


@dataclass(frozen=True)
class Level:
    family: str
    name: str
    direction: str
    price: float
    target: float | None = None


@dataclass(frozen=True)
class Proposal:
    level: Level
    at: pd.Timestamp
    entry: float
    stop: float
    target: float
    rr: float | None
    evidence: tuple[tuple[str, int], ...]
    additional_levels: tuple[str, ...] = ()
    @property
    def support_count(self) -> int:
        return sum(vote > 0 for _, vote in self.evidence)

    @property
    def technical_support_count(self) -> int:
        return sum(vote > 0 for _, vote in self.evidence[:3])

    @property
    def opposition_count(self) -> int:
        return sum(vote < 0 for _, vote in self.evidence)


@dataclass(frozen=True)
class Assessment:
    reason: str
    proposal: Proposal | None = None
    nearest: Level | None = None


@dataclass(frozen=True)
class History:
    count: int
    targets: int
    stops: int
    timeouts: int
    mean_points: float | None
    std_points: float | None = None  # odchylenie standardowe wyniku pojedynczej pozycji

    def description(self) -> str:
        if self.count < MIN_HISTORY or self.mean_points is None:
            return f"Za mała próba do oszacowania skuteczności ({self.count}/{MIN_HISTORY} przypadków)."
        rate = self.targets / self.count
        z = 1.96
        adjustment = z ** 2 / self.count
        center = (rate + adjustment / 2) / (1 + adjustment)
        half = z * sqrt((rate * (1 - rate) + adjustment / 4) / self.count) / (1 + adjustment)
        return (f"Historycznie {self.targets}/{self.count} ({rate:.1%}) osiągnęło TP przed SL i limitem czasu. "
                f"Orientacyjny przedział 95%: {max(0, center - half):.0%}–{min(1, center + half):.0%}. "
                f"SL: {self.stops}, limit: {self.timeouts}. Średni wynik po kosztach: {self.mean_points:+.2f} pkt.")


@dataclass(frozen=True)
class ReplayTrade:
    proposal: Proposal
    accepted: bool
    opened: pd.Timestamp
    closed: pd.Timestamp
    outcome: str
    points: float | None


@dataclass(frozen=True)
class Validation:
    cutoff: pd.Timestamp
    end: pd.Timestamp
    preparation_bars: int
    test_bars: int
    baseline: History
    advisor: History
    random: History | None = None  # losowe wejścia z odległościami SL/TP wariantu bazowego


@dataclass(frozen=True)
class WalkForward:
    total: Validation
    windows: tuple[Validation, ...]


@dataclass(frozen=True)
class TimingComparison:
    start: pd.Timestamp
    end: pd.Timestamp
    h1: History
    m5: History


class Market:
    """Dane wyrównane do świecy sygnałowej i zamkniętych świec kontekstu."""

    def __init__(self, frames: dict[str, pd.DataFrame], now: pd.Timestamp, *,
                 history: bool = False, trigger: str = "H1") -> None:
        self.trigger = trigger
        source = frames[trigger]
        self.ends = pd.DatetimeIndex(source.index) + BAR_LENGTH[trigger]
        observed_at = min(now, source.attrs.get("pobrano", now))
        count = int(self.ends.searchsorted(observed_at, side="right"))
        self.bars = source.iloc[:count]
        self.ends = self.ends[:count]
        self.relative_volume = indicators.relative_volume(
            self.bars.get("volume", pd.Series(index=self.bars.index, dtype=float)),
        ).to_numpy()
        self.close_position = indicators.close_position(self.bars).to_numpy()
        self.daily_lows = frames["D1"]["low"].to_numpy()
        self.daily_highs = frames["D1"]["high"].to_numpy()
        self.trends = {}
        for interval in ("H1", "H4", "D1"):
            if interval == trigger:
                self.trends[interval] = trend_series(self.bars).to_numpy()
                continue
            frame = frames[interval]
            ends = pd.DatetimeIndex(pd.to_datetime([bar_end(interval, t) for t in frame.index], utc=True))
            mapped = ends.searchsorted(self.ends, side="right") - 1
            trends = np.full(count, "brak danych", dtype=object)
            available = mapped >= 0
            if available.any():
                trends[available] = trend_series(frame).to_numpy()[mapped[available]]
            self.trends[interval] = trends
            if interval == "D1":
                self.daily_at_open = ends.searchsorted(self.bars.index, side="right") - 1

        self.waves = list(harmonics.confirmed_waves(frames["H4"], now))
        confirmations = pd.DatetimeIndex(pd.to_datetime([b.confirmed_at for _, b in self.waves], utc=True))
        self.wave_at_open = confirmations.searchsorted(self.bars.index, side="right") - 1

        self.rejected_d: dict[pd.Timestamp, Level] = {}
        self.latest_d: Level | None = None
        patterns = harmonics.projections(frames["H4"], now)
        if not history:
            # W chwili zamknięcia H1 mogła pojawić się nowa projekcja H4.
            # Poprzednia projekcja wciąż obowiązywała podczas tej H1.
            patterns = patterns[-2:]
        for i, pattern in enumerate(patterns):
            horizon = patterns[i + 1].c.confirmed_at if i + 1 < len(patterns) else now
            window = bars_between(self.bars, pattern.c.confirmed_at, horizon)
            reaction = harmonics.track(pattern, window, horizon, interval=trigger)
            if reaction.stage == "rejected":
                self.rejected_d[reaction.at] = Level(
                    "AB=CD", "Strefa D układu AB=CD", pattern.reaction_direction, pattern.d,
                    pattern.d - 0.382 * pattern.leg,
                )
            elif i == len(patterns) - 1 and reaction.stage in ("building", "reached"):
                self.latest_d = Level("AB=CD", "Strefa D układu AB=CD", pattern.reaction_direction, pattern.d)

    def levels(self, i: int) -> list[Level]:
        levels = []
        reaction = self.rejected_d.get(self.ends[i])
        if reaction:
            levels.append(reaction)
        wave_i = self.wave_at_open[i]
        if wave_i >= 0:
            a, b = self.waves[wave_i]
            leg = b.price - a.price
            if (leg > 0 and b.high) or (leg < 0 and not b.high):
                levels.append(Level("Fibo H4", "Fibo 61,8% fali H4", "long" if leg > 0 else "short",
                                    b.price - 0.618 * leg, b.price - 0.382 * leg))
        daily_i = self.daily_at_open[i]
        if daily_i >= 0:
            levels.extend((Level("Sesja D1", "Minimum poprzedniej sesji", "long", float(self.daily_lows[daily_i])),
                           Level("Sesja D1", "Maksimum poprzedniej sesji", "short", float(self.daily_highs[daily_i]))))
        return levels

    def assess_bar(self, i: int, cost: float, min_rr: float) -> Assessment:
        bar = self.bars.iloc[i]
        levels = self.levels(i)
        rejected = [level for level in levels if rejects_level(
            bar["open"], bar["high"], bar["low"], bar["close"], level.price, level.direction,
        )]
        if not rejected:
            nearest = min(levels, key=lambda level: abs(bar["close"] - level.price), default=None)
            return Assessment(f"Ostatnia zamknięta {self.trigger} nie potwierdziła odrzucenia obserwowanych poziomów.", nearest=nearest)
        level = rejected[0]
        if any(other.direction != level.direction for other in rejected[1:]):
            return Assessment("Na tej samej świecy wystąpiły sprzeczne reakcje LONG i SHORT.")
        if not isfinite(bar["atr"]) or bar["atr"] <= 0:
            return Assessment(f"Brakuje ATR {self.trigger} do wyznaczenia poziomu obrony.", nearest=level)

        direction = level.direction
        sign = 1 if direction == "long" else -1
        entry = float(bar["close"])
        stop = float(bar["low"] if sign > 0 else bar["high"]) - sign * 0.1 * float(bar["atr"])
        options = []
        for candidate in rejected:
            target = candidate.target if candidate.target is not None else entry + 2 * (entry - stop)
            options.append((candidate, target, plan.reward_to_risk(direction, entry, stop, target, cost)))
        # Preferuj AB=CD, następnie Fibo i sesję D1, jeśli cel spełnia warunki planu.
        chosen = next((option for option in options if option[2] is not None and option[2] >= min_rr), options[0])
        level, target, rr = chosen
        technical = self.evidence(i, sign)
        volume = self.volume_evidence(i, sign)
        evidence = (*technical, volume)
        additional_levels = tuple(other.name for other in rejected if other != level)
        proposal = Proposal(level, self.ends[i], entry, stop, target, rr, evidence, additional_levels)
        if rr is None or rr < min_rr:
            before_cost = plan.reward_to_risk(direction, entry, stop, target, 0)
            if rr is not None and before_cost is not None and before_cost >= min_rr:
                reason = (f"Koszt {cost:g} pkt ceny obniża zysk do ryzyka poniżej {min_rr:g}:1 "
                          f"(przed kosztami {before_cost:.2f}:1, po kosztach {rr:.2f}:1).")
            else:
                reason = f"Nawet przy zerowym koszcie układ nie osiąga zysku do ryzyka {min_rr:g}:1."
        elif proposal.technical_support_count < 2:
            reason = "Za mało zgodnych potwierdzeń z trendu, momentum i reakcji ceny."
        elif volume[1] < 0:
            reason = f"Podwyższony wolumen {self.trigger} przy zamknięciu świecy po stronie przeciwnej do kierunku wejścia."
        else:
            reason = ""
        return Assessment(reason, proposal)

    def volume_evidence(self, i: int, sign: int) -> tuple[str, int]:
        relative, position = self.relative_volume[i], self.close_position[i]
        if not isfinite(relative) or not isfinite(position):
            return f"Wolumen {self.trigger} · brak porównywalnych danych", 0
        if relative < ELEVATED_VOLUME:
            return f"Wolumen {self.trigger} · {relative:.2f}× mediany, poniżej progu {ELEVATED_VOLUME:g}×", 0
        band = indicators.CLOSE_QUARTILE
        direction = int(position >= 1 - band) - int(position <= band)
        place = {1: "górna ćwiartka", -1: "dolna ćwiartka", 0: "środek zakresu"}[direction]
        return f"Wolumen {self.trigger} · {relative:.2f}× mediany, zamknięcie: {place}", direction * sign

    def evidence(self, i: int, sign: int) -> tuple[tuple[str, int], ...]:
        bar = self.bars.iloc[i]
        previous = self.bars.iloc[i - 1] if i > 0 else bar
        trends = [self.trends[interval][i] for interval in ("H1", "H4", "D1")]
        trend_votes = [1 if trend == "wzrostowy" else -1 if trend == "spadkowy" else 0 for trend in trends]
        trend_vote = int(np.sign(sum(trend_votes) * sign))
        rsi_change = float(bar["rsi"] - previous["rsi"])
        macd_change = float(bar["macd_hist"] - previous["macd_hist"])
        momentum = sum(1 if value > 0 else -1 if value < 0 else 0 for value in (rsi_change, macd_change))
        momentum_vote = int(np.sign(momentum * sign))
        lower, upper = bar["bb_low"], bar["bb_up"]
        lower_return = bar["low"] < lower < bar["close"]
        upper_return = bar["high"] > upper > bar["close"]
        band_vote = int(lower_return) - int(upper_return)
        recent = self.bars.iloc[max(0, i - 79):i + 1]
        divergences = indicators.detect_divergences(recent)
        earliest = recent.index[max(0, len(recent) - 8)]
        divs = {d["typ"] for d in divergences if d["do"] >= earliest}
        divergence_vote = int("bycza" in divs) - int("niedźwiedzia" in divs)
        reaction_vote = int(np.sign((band_vote + divergence_vote) * sign))
        return (
            (f"Trend H1, H4 i D1 · {', '.join(trends)}", trend_vote),
            (f"Momentum · RSI {bar['rsi']:.1f}, zmiana {rsi_change:+.1f}, histogram MACD {macd_change:+.3f}", momentum_vote),
            (f"Bollinger · {'powrót od dołu' if band_vote > 0 else 'powrót od góry' if band_vote < 0 else 'brak reakcji'}, "
             f"dywergencje {self.trigger} · {', '.join(sorted(divs)) or 'brak'}", reaction_vote),
        )


def assess(frames: dict[str, pd.DataFrame], cost: float, min_rr: float,
           now: pd.Timestamp | None = None, *, live: bool = False) -> Assessment:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if frames["H1"].empty:
        return Assessment("Brak zamkniętych świec H1.")
    market = Market(frames, now)
    if market.bars.empty or now - market.ends[-1] > MAX_SIGNAL_AGE:
        return Assessment("Brak świeżej zamkniętej H1. Czekam na nowe notowania.")
    outcome = market.assess_bar(len(market.bars) - 1, cost, min_rr)
    proposal = outcome.proposal
    if live and proposal and not outcome.reason:
        later = bars_between(frames["H1"], proposal.at, now)
        if not later.empty:
            sign = 1 if proposal.level.direction == "long" else -1
            stop_hit = (later["low"] <= proposal.stop).any() if sign > 0 else (later["high"] >= proposal.stop).any()
            target_hit = (later["high"] >= proposal.target).any() if sign > 0 else (later["low"] <= proposal.target).any()
            if stop_hit or target_hit:
                return Assessment("Po sygnale cena osiągnęła już SL lub TP.", proposal)
            latest_price = float(later["close"].iloc[-1])
            current_rr = plan.reward_to_risk(proposal.level.direction, latest_price, proposal.stop, proposal.target, cost)
            if current_rr is None or current_rr < min_rr:
                return Assessment("Przy bieżącej cenie zysk do ryzyka jest już poniżej ustawionego minimum.", proposal)
    if proposal is None and outcome.nearest is not None and market.latest_d:
        price = float(market.bars["close"].iloc[-1])
        nearest = min((outcome.nearest, market.latest_d), key=lambda level: abs(price - level.price))
        return Assessment(outcome.reason, nearest=nearest)
    return outcome


def _trades(
    market: Market, cost: float, min_rr: float, limit_h: float, *, start: int = 0, include_baseline: bool = False,
    execution_m5: pd.DataFrame | None = None,
) -> Iterator[ReplayTrade]:
    tape = market.bars if execution_m5 is None else execution_m5
    opens, highs, lows, closes = (tape[col].to_numpy() for col in ("open", "high", "low", "close"))
    dates = tape.index
    ends = market.ends if execution_m5 is None else dates + BAR_LENGTH["M5"]
    entries = dates.get_indexer(market.ends[start:])
    for i, entry_i in enumerate(entries, start=start):
        if entry_i < 0:
            continue
        outcome = market.assess_bar(i, cost, min_rr)
        proposal = outcome.proposal
        if proposal is None or proposal.rr is None or proposal.rr < min_rr:
            continue
        if outcome.reason and not include_baseline:
            continue
        direction = proposal.level.direction
        actual_entry = opens[entry_i]
        actual_rr = plan.reward_to_risk(direction, actual_entry, proposal.stop, proposal.target, cost)
        if actual_rr is None or actual_rr < min_rr:
            continue
        sign = 1 if direction == "long" else -1
        deadline = dates[entry_i] + pd.Timedelta(hours=limit_h)
        exit_ = _resolve((opens, highs, lows, closes), dates, ends, entry_i, sign, proposal.stop, proposal.target, deadline)
        if exit_ is None:
            yield ReplayTrade(proposal, not outcome.reason, dates[entry_i], deadline, "otwarta", None)
            continue
        label, exit_price, resolved_at = exit_
        yield ReplayTrade(proposal, not outcome.reason, dates[entry_i], resolved_at,
                          label, (exit_price - actual_entry) * sign - cost)


def _resolve(
    prices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dates: pd.DatetimeIndex, ends: pd.DatetimeIndex,
    entry_i: int, sign: int, stop: float, target: float, deadline: pd.Timestamp,
) -> tuple[str, float, pd.Timestamp] | None:
    """Wyjście z pozycji: luka na otwarciu, potem SL przed TP w tej samej świecy, na końcu limit czasu."""
    opens, highs, lows, closes = prices
    for j in range(entry_i, len(dates)):
        start_at, end_at = dates[j], ends[j]
        if start_at >= deadline:
            return "limit", opens[j], start_at
        if (opens[j] - stop) * sign <= 0:
            return "SL", opens[j], start_at
        if (opens[j] - target) * sign >= 0:
            return "TP", target, start_at
        if (lows[j] <= stop if sign > 0 else highs[j] >= stop):
            return "SL", stop, end_at
        if (highs[j] >= target if sign > 0 else lows[j] <= target):
            return "TP", target, end_at
        if end_at >= deadline:
            return "limit", closes[j], end_at
    return None


def random_entries(
    market: Market, reference: list[ReplayTrade], cost: float, limit_h: float,
    start: pd.Timestamp, end: pd.Timestamp, *, draws: int = RANDOM_DRAWS, seed: int = 0,
) -> History:
    """Punkt odniesienia: losowa godzina wejścia i losowy kierunek, te same odległości SL i TP.

    Każda pozycja z ``reference`` jest powtarzana ``draws`` razy na losowej świecy z okna
    [start, end). Reguła, która nie bije tego wyniku, nie ma przewagi nad przypadkiem."""
    distances = [(abs(t.proposal.entry - t.proposal.stop), abs(t.proposal.target - t.proposal.entry))
                 for t in reference if t.points is not None]
    return random_with_distances(market.bars, market.ends, distances, cost, limit_h, start, end, draws=draws, seed=seed)


def random_with_distances(
    bars: pd.DataFrame, ends: pd.DatetimeIndex, distances: list[tuple[float, float]], cost: float, limit_h: float,
    start: pd.Timestamp, end: pd.Timestamp, *, draws: int = RANDOM_DRAWS, seed: int = 0,
) -> History:
    """Losowe wejścia dla par (odległość SL, odległość TP) - wspólne dla Doradcy i rodzin sygnałów."""
    prices = tuple(bars[col].to_numpy() for col in ("open", "high", "low", "close"))
    dates = bars.index
    candidates = np.flatnonzero((dates >= start) & (dates < end))
    if not len(candidates) or not distances:
        return History(0, 0, 0, 0, None)
    rng = np.random.default_rng(seed)
    hour = pd.Timedelta(hours=limit_h)
    outcomes: list[tuple[str, float]] = []
    for risk, reward in distances:
        for i, sign in zip(rng.choice(candidates, draws), rng.choice((-1, 1), draws)):
            entry = prices[0][i]
            exit_ = _resolve(prices, dates, ends, int(i), int(sign), entry - sign * risk, entry + sign * reward, dates[i] + hour)
            if exit_ is not None:
                outcomes.append((exit_[0], (exit_[1] - entry) * sign - cost))
    if not outcomes:
        return History(0, 0, 0, 0, None)
    labels = [label for label, _ in outcomes]
    points = [value for _, value in outcomes]
    return History(len(outcomes), labels.count("TP"), labels.count("SL"), labels.count("limit"),
                   float(np.mean(points)), float(np.std(points, ddof=1)) if len(points) > 1 else None)


def _summarize(trades: list[ReplayTrade]) -> History:
    finished = [trade for trade in trades if trade.points is not None]
    return History(
        len(finished), sum(t.outcome == "TP" for t in finished), sum(t.outcome == "SL" for t in finished),
        sum(t.outcome == "limit" for t in finished), float(np.mean([t.points for t in finished])) if finished else None,
        float(np.std([t.points for t in finished], ddof=1)) if len(finished) > 1 else None,
    )


def _training_history(trades: Iterator[ReplayTrade], as_of: pd.Timestamp | None = None) -> dict[tuple[str, str, int], History]:
    results: dict[tuple[str, str, int], list[ReplayTrade]] = {}
    busy: dict[tuple[str, str], pd.Timestamp] = {}
    for trade in trades:
        proposal = trade.proposal
        key = proposal.level.family, proposal.level.direction
        if key in busy and trade.opened < busy[key]:
            continue
        busy[key] = trade.closed
        if trade.points is None or (as_of is not None and trade.closed > as_of):
            continue
        results.setdefault((*key, proposal.technical_support_count), []).append(trade)
        results.setdefault((*key, 0), []).append(trade)
    return {key: _summarize(trades) for key, trades in results.items()}


def replay(frames: dict[str, pd.DataFrame], cost: float, min_rr: float,
           limit_h: float, now: pd.Timestamp | None = None) -> dict[tuple[str, str, int], History]:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if frames["H1"].empty:
        return {}
    market = Market(frames, now, history=True)
    return _training_history(_trades(market, cost, min_rr, limit_h))


def validate_later_period(
    frames: dict[str, pd.DataFrame], cost: float, min_rr: float, limit_h: float, now: pd.Timestamp | None = None,
) -> Validation | None:
    """Pierwsze 80% H1 tworzy statystyki porównawcze, ostatnie 20% testuje oba warianty."""
    if len(frames["H1"]) < MIN_VALIDATION_BARS:
        return None
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    market = Market(frames, now, history=True)
    count = len(market.bars)
    if count < MIN_VALIDATION_BARS:
        return None
    start = count * 4 // 5
    cutoff = market.bars.index[start]
    first_period = replay(frames, cost, min_rr, limit_h, cutoff)
    baseline, confirmed = [], []
    for trade in _trades(market, cost, min_rr, limit_h, start=start, include_baseline=True):
        if not baseline or trade.opened >= baseline[-1].closed:
            baseline.append(trade)
        if trade.accepted and (not confirmed or trade.opened >= confirmed[-1].closed):
            stats, _ = comparable(trade.proposal, first_period)
            if not history_hold(stats):
                confirmed.append(trade)
    return Validation(cutoff, market.ends[-1], start, count - start, _summarize(baseline), _summarize(confirmed))


def validate_walk_forward(
    frames: dict[str, pd.DataFrame], cost: float, min_rr: float, limit_h: float, now: pd.Timestamp | None = None,
) -> WalkForward | None:
    """Trzy kolejne okna po rozgrzewce, ze statystykami znanymi na początku każdego okna."""
    if len(frames["H1"]) < MIN_VALIDATION_BARS:
        return None
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    market = Market(frames, now, history=True)
    count = len(market.bars)
    if count < MIN_VALIDATION_BARS:
        return None
    edges = [count * part // 5 for part in (2, 3, 4, 5)]
    dates = market.bars.index
    trades = list(_trades(market, cost, min_rr, limit_h, include_baseline=True))
    hour = pd.Timedelta(hours=1)
    baseline, confirmed, windows = [], [], []
    for start, stop in zip(edges, edges[1:]):
        cutoff = dates[start]
        end = dates[stop] if stop < count else market.ends[-1]
        # Otwarte pozycje mogą blokować kolejne wejścia, lecz ich wynik nie trafia do statystyk.
        training = _training_history((t for t in trades if t.accepted and t.opened < cutoff), as_of=cutoff)
        baseline_start, confirmed_start = len(baseline), len(confirmed)
        for trade in trades:
            signal_start = trade.proposal.at - hour
            if not cutoff <= signal_start < end:
                continue
            if not baseline or trade.opened >= baseline[-1].closed:
                baseline.append(trade)
            if trade.accepted and (not confirmed or trade.opened >= confirmed[-1].closed):
                stats, _ = comparable(trade.proposal, training)
                if not history_hold(stats):
                    confirmed.append(trade)
        chance = random_entries(market, baseline[baseline_start:], cost, limit_h, cutoff, end, seed=len(windows))
        windows.append(Validation(cutoff, end, start, stop - start, _summarize(baseline[baseline_start:]),
                                  _summarize(confirmed[confirmed_start:]), chance))
    total_chance = [w.random for w in windows if w.random is not None and w.random.count]
    chance_count = sum(h.count for h in total_chance)
    total_random = History(
        chance_count, sum(h.targets for h in total_chance), sum(h.stops for h in total_chance),
        sum(h.timeouts for h in total_chance),
        sum(h.mean_points * h.count for h in total_chance) / chance_count if chance_count else None,
    )  # odchylenie nie jest potrzebne: losowa próba jest wielokrotnie większa od ocenianej
    total = Validation(dates[edges[0]], market.ends[-1], edges[0], count - edges[0],
                       _summarize(baseline), _summarize(confirmed), total_random)
    return WalkForward(total, tuple(windows))


def compare_entry_timing(
    frames: dict[str, pd.DataFrame], m5: pd.DataFrame, cost: float, min_rr: float, limit_h: float,
    now: pd.Timestamp,
) -> TimingComparison | None:
    """Porównaj sygnały H1 i M5, rozliczając oba na wspólnej taśmie M5."""
    if m5.empty or frames["H1"].empty:
        return None
    h1 = Market(frames, now, history=True)
    short = Market({**frames, "M5": indicators.add_indicators(m5)}, now, history=True, trigger="M5")
    if h1.bars.empty or short.bars.empty:
        return None
    start = max(h1.bars.index[0], short.bars.index[0])
    end = min(h1.ends[-1], short.ends[-1])
    hours = h1.bars.index[(h1.bars.index >= start) & (h1.ends <= end)]
    tape = short.bars.loc[short.ends <= end]
    counts = tape.index.searchsorted(hours + BAR_LENGTH["H1"]) - tape.index.searchsorted(hours)
    hours = hours[counts == BAR_LENGTH["H1"] // BAR_LENGTH["M5"]]
    if len(hours) < MIN_VALIDATION_BARS:
        return None
    edges = [hours[len(hours) * part // 5] for part in (2, 3, 4)] + [end]

    def compare(market: Market) -> History:
        first = int(market.bars.index.searchsorted(start))
        trades = list(_trades(market, cost, min_rr, limit_h, start=first, execution_m5=tape))
        selected: list[ReplayTrade] = []
        for left, right in zip(edges, edges[1:]):
            training = _training_history((t for t in trades if t.opened < left), as_of=left)
            for trade in trades:
                if not left <= trade.opened < right or (selected and trade.opened < selected[-1].closed):
                    continue
                stats, _ = comparable(trade.proposal, training)
                if not history_hold(stats):
                    selected.append(trade)
        return _summarize(selected)

    return TimingComparison(edges[0], end, compare(h1), compare(short))


def against_chance(label: str, stats: History, chance: History | None, z: float = 2.0,
                   unit: str = "pkt") -> tuple[str, str]:
    """Ocena wariantu względem losowych wejść: (rodzaj komunikatu, treść).

    Przewaga liczy się tylko wtedy, gdy średni wynik minus z błędów standardowych (domyślnie 2, ok. 95%)
    przekracza zarówno wynik losowych wejść, jak i zero."""
    if chance is None or chance.mean_points is None:
        return "info", f"{label}: brak losowych wejść do porównania."
    if stats.mean_points is None or stats.count < MIN_HISTORY:
        return "info", (f"{label}: za mało zakończonych pozycji ({stats.count}/{MIN_HISTORY}), "
                        "żeby porównać z losowymi wejściami.")
    bar = max(chance.mean_points, 0.0)
    margin = z * stats.std_points / sqrt(stats.count) if stats.std_points else 0.0
    numbers = (f"średnio {stats.mean_points:+.2f} {unit} (±{margin:.2f}) przy {stats.count} pozycjach, "
               f"losowe wejścia {chance.mean_points:+.2f} {unit}")
    if stats.mean_points - margin > bar:
        return "success", f"{label}: {numbers}. Przewaga nad przypadkiem i powyżej zera po kosztach."
    if stats.mean_points > bar:
        return "info", (f"{label}: {numbers}. Wynik jest lepszy, ale mieści się w granicach przypadku – "
                        "potrzeba więcej pozycji, zanim uznasz to za przewagę.")
    return "warning", (f"{label}: {numbers}. Brak przewagi po kosztach. "
                       "Nie traktuj tych sygnałów jako podstawy wejścia.")


def comparable(proposal: Proposal, history: dict[tuple[str, str, int], History]) -> tuple[History, str]:
    key = proposal.level.family, proposal.level.direction
    exact = history.get((*key, proposal.technical_support_count))
    if exact and exact.count >= MIN_HISTORY:
        return exact, f"{proposal.technical_support_count} zgodne grupy techniczne"
    return history.get((*key, 0), History(0, 0, 0, 0, None)), "wszystkie zgodne sygnały tej rodziny i kierunku"


def history_hold(history: History) -> str:
    if history.count >= MIN_HISTORY and history.mean_points is not None and history.mean_points <= 0:
        return "Podobne sygnały nie dawały dodatniego średniego wyniku po kosztach."
    return ""
