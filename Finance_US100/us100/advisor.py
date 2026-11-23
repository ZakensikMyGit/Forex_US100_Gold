"""Propozycje pozycji i odtwarzanie tych samych reguł na zamkniętych świecach."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np
import pandas as pd

from . import harmonics, indicators, plan
from .data import bar_end, bars_between
from .signals import rejects_level, trend_series

MIN_HISTORY = 30
MAX_SIGNAL_AGE = pd.Timedelta(hours=2)
ALERT_SETTING = "powiadomienia doradcy"
MIN_VALIDATION_BARS = 250


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


@dataclass(frozen=True)
class WalkForward:
    total: Validation
    windows: tuple[Validation, ...]


class Market:
    """Dane wyrównane do początku i zamknięcia każdej H1, współdzielone przez panel i replay."""

    def __init__(self, frames: dict[str, pd.DataFrame], now: pd.Timestamp, *, history: bool = False) -> None:
        source = frames["H1"]
        self.ends = pd.DatetimeIndex(source.index) + pd.Timedelta(hours=1)
        count = int(self.ends.searchsorted(now, side="right"))
        self.h1 = source.iloc[:count]
        self.ends = self.ends[:count]
        self.daily_lows = frames["D1"]["low"].to_numpy()
        self.daily_highs = frames["D1"]["high"].to_numpy()
        self.trends = {"H1": trend_series(self.h1).to_numpy()}
        for interval in ("H4", "D1"):
            frame = frames[interval]
            ends = pd.DatetimeIndex(pd.to_datetime([bar_end(interval, t) for t in frame.index], utc=True))
            mapped = ends.searchsorted(self.ends, side="right") - 1
            trends = np.full(count, "brak danych", dtype=object)
            available = mapped >= 0
            if available.any():
                trends[available] = trend_series(frame).to_numpy()[mapped[available]]
            self.trends[interval] = trends
            if interval == "D1":
                self.daily_at_open = ends.searchsorted(self.h1.index, side="right") - 1

        self.waves: list[tuple[harmonics.Pivot, harmonics.Pivot]] = []
        swing: list[harmonics.Pivot] = []
        for pivot in harmonics.confirmed_pivots(frames["H4"], now):
            if swing and swing[-1].high == pivot.high:
                if (pivot.price <= swing[-1].price if pivot.high else pivot.price >= swing[-1].price):
                    continue
                swing[-1] = pivot
            else:
                swing.append(pivot)
            if len(swing) >= 2:
                self.waves.append((swing[-2], swing[-1]))
        confirmations = pd.DatetimeIndex(pd.to_datetime([b.confirmed_at for _, b in self.waves], utc=True))
        self.wave_at_open = confirmations.searchsorted(self.h1.index, side="right") - 1

        self.rejected_d: dict[pd.Timestamp, Level] = {}
        self.latest_d: Level | None = None
        patterns = harmonics.projections(frames["H4"], now)
        if not history:
            # W chwili zamknięcia H1 mogła pojawić się nowa projekcja H4.
            # Poprzednia projekcja wciąż obowiązywała podczas tej H1.
            patterns = patterns[-2:]
        for i, pattern in enumerate(patterns):
            horizon = patterns[i + 1].c.confirmed_at if i + 1 < len(patterns) else now
            window = bars_between(self.h1, pattern.c.confirmed_at, horizon)
            reaction = harmonics.track(pattern, window, horizon)
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
        bar = self.h1.iloc[i]
        levels = self.levels(i)
        rejected = [level for level in levels if rejects_level(
            bar["open"], bar["high"], bar["low"], bar["close"], level.price, level.direction,
        )]
        if not rejected:
            nearest = min(levels, key=lambda level: abs(bar["close"] - level.price), default=None)
            return Assessment("Ostatnia zamknięta H1 nie potwierdziła odrzucenia obserwowanych poziomów.", nearest=nearest)
        level = rejected[0]
        if any(other.direction != level.direction for other in rejected[1:]):
            return Assessment("Na tej samej świecy wystąpiły sprzeczne reakcje LONG i SHORT.")
        if not isfinite(bar["atr"]) or bar["atr"] <= 0:
            return Assessment("Brakuje ATR H1 do wyznaczenia poziomu obrony.", nearest=level)

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
        evidence = self.evidence(i, sign)
        additional_levels = tuple(other.name for other in rejected if other != level)
        proposal = Proposal(level, self.ends[i], entry, stop, target, rr, evidence, additional_levels)
        if rr is None or rr < min_rr:
            reason = f"Zysk do ryzyka po kosztach jest poniżej minimum {min_rr:g}:1 lub cel jest już za ceną."
        elif proposal.support_count <= proposal.opposition_count or proposal.support_count < 2:
            reason = "Za mało zgodnych potwierdzeń z trendu, momentum i reakcji ceny."
        else:
            reason = ""
        return Assessment(reason, proposal)

    def evidence(self, i: int, sign: int) -> tuple[tuple[str, int], ...]:
        bar = self.h1.iloc[i]
        previous = self.h1.iloc[i - 1] if i > 0 else bar
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
        recent = self.h1.iloc[max(0, i - 79):i + 1]
        divergences = indicators.detect_divergences(recent)
        earliest = recent.index[max(0, len(recent) - 8)]
        divs = {d["typ"] for d in divergences if d["do"] >= earliest}
        divergence_vote = int("bycza" in divs) - int("niedźwiedzia" in divs)
        reaction_vote = int(np.sign((band_vote + divergence_vote) * sign))
        return (
            (f"Trend H1, H4 i D1 · {', '.join(trends)}", trend_vote),
            (f"Momentum · RSI {bar['rsi']:.1f}, zmiana {rsi_change:+.1f}, histogram MACD {macd_change:+.3f}", momentum_vote),
            (f"Bollinger · {'powrót od dołu' if band_vote > 0 else 'powrót od góry' if band_vote < 0 else 'brak reakcji'}, "
             f"dywergencje H1 · {', '.join(sorted(divs)) or 'brak'}", reaction_vote),
        )


def assess(frames: dict[str, pd.DataFrame], cost: float, min_rr: float,
           now: pd.Timestamp | None = None, *, live: bool = False) -> Assessment:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if frames["H1"].empty:
        return Assessment("Brak zamkniętych świec H1.")
    market = Market(frames, now)
    if market.h1.empty or now - market.ends[-1] > MAX_SIGNAL_AGE:
        return Assessment("Brak świeżej zamkniętej H1. Czekam na nowe notowania.")
    outcome = market.assess_bar(len(market.h1) - 1, cost, min_rr)
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
        price = float(market.h1["close"].iloc[-1])
        nearest = min((outcome.nearest, market.latest_d), key=lambda level: abs(price - level.price))
        return Assessment(outcome.reason, nearest=nearest)
    return outcome


def _trades(
    market: Market, cost: float, min_rr: float, limit_h: float, *, start: int = 0, include_baseline: bool = False,
) -> Iterator[ReplayTrade]:
    opens, highs, lows, closes = (market.h1[col].to_numpy() for col in ("open", "high", "low", "close"))
    dates = market.h1.index
    for i in range(start, len(dates) - 1):
        entry_i = i + 1
        if dates[entry_i] != market.ends[i]:
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
        stop, target = proposal.stop, proposal.target
        deadline = dates[entry_i] + pd.Timedelta(hours=limit_h)
        j, label, exit_price = entry_i, None, None
        while j < len(dates) and market.ends[j] <= deadline:
            if (opens[j] - stop) * sign <= 0:
                label, exit_price = "SL", opens[j]
            elif (opens[j] - target) * sign >= 0:
                label, exit_price = "TP", target
            elif (lows[j] <= stop if sign > 0 else highs[j] >= stop):
                label, exit_price = "SL", stop
            elif (highs[j] >= target if sign > 0 else lows[j] <= target):
                label, exit_price = "TP", target
            if label:
                break
            j += 1
        if label is None:
            if market.ends[-1] < deadline:
                yield ReplayTrade(proposal, not outcome.reason, dates[entry_i], deadline, "otwarta", None)
                continue
            if j == entry_i:
                continue
            j -= 1
            label, exit_price = "limit", closes[j]
        resolved_at = deadline if label == "limit" else market.ends[j]
        yield ReplayTrade(proposal, not outcome.reason, dates[entry_i], resolved_at,
                          label, (exit_price - actual_entry) * sign - cost)


def _summarize(trades: list[ReplayTrade]) -> History:
    finished = [trade for trade in trades if trade.points is not None]
    return History(
        len(finished), sum(t.outcome == "TP" for t in finished), sum(t.outcome == "SL" for t in finished),
        sum(t.outcome == "limit" for t in finished), float(np.mean([t.points for t in finished])) if finished else None,
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
        results.setdefault((*key, proposal.support_count), []).append(trade)
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
    count = len(market.h1)
    if count < MIN_VALIDATION_BARS:
        return None
    start = count * 4 // 5
    cutoff = market.h1.index[start]
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
    count = len(market.h1)
    if count < MIN_VALIDATION_BARS:
        return None
    edges = [count * part // 5 for part in (2, 3, 4, 5)]
    dates = market.h1.index
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
        windows.append(Validation(cutoff, end, start, stop - start, _summarize(baseline[baseline_start:]),
                                  _summarize(confirmed[confirmed_start:])))
    total = Validation(dates[edges[0]], market.ends[-1], edges[0], count - edges[0],
                       _summarize(baseline), _summarize(confirmed))
    return WalkForward(total, tuple(windows))


def comparable(proposal: Proposal, history: dict[tuple[str, str, int], History]) -> tuple[History, str]:
    key = proposal.level.family, proposal.level.direction
    exact = history.get((*key, proposal.support_count))
    if exact and exact.count >= MIN_HISTORY:
        return exact, f"{proposal.support_count} zgodne grupy wskaźników"
    return history.get((*key, 0), History(0, 0, 0, 0, None)), "wszystkie zgodne sygnały tej rodziny i kierunku"


def history_hold(history: History) -> str:
    if history.count >= MIN_HISTORY and history.mean_points is not None and history.mean_points <= 0:
        return "Podobne sygnały nie dawały dodatniego średniego wyniku po kosztach."
    return ""
