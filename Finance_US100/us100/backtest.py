"""Odtwarzanie alertów na historii i szacowanie czasu trzymania pozycji.

1. replay_alerts - te same reguły co monitor (reguły zamknięć, dotknięcia i luki poziomów,
   strefy RSI) uruchomione po kolei na historycznych świecach. Pokazuje, kiedy alert by
   przyszedł, i pozwala wyłapać alerty pominięte albo zbyt częste.
2. holding_times - "podobne sygnały": zamknięcie H1 za najwyższym szczytem (long) lub
   najniższym dołkiem (short) z ostatnich N świec. SL i TP ustawione w tej samej odległości
   w ATR H1 co w planie. Każdy przypadek jest śledzony świeca po świecy do TP, SL albo limitu.

Ograniczenia: świece H1 nie mówią, co było pierwsze w ich wnętrzu - gdy w jednej świecy padł
i SL, i TP, liczymy SL (wariant ostrożny). Wyniki nie uwzględniają spreadu ani poślizgu.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import bar_end
from .signals import NO_DATA, last_closed, level_hits, rejects_level, trend_series

MIN_SAMPLE = 20


# ---------------------------------------------------------------- 1. odtwarzanie alertów


def _closed_only(df: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    bar = last_closed(df, interval, now)
    return df.loc[:bar.name] if bar is not None else df.iloc[:0]


def replay_alerts(
    frames: dict[str, pd.DataFrame],
    cfg: dict,
    days: int = 30,
    cooldown_h: float = 4.0,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    off = float(cfg.get("przesunięcie", 0) or 0)
    now = now or pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=days)
    rows: list[dict] = []

    def in_range(ts: pd.Timestamp) -> bool:
        t = ts if ts.tzinfo else ts.tz_localize("UTC")
        return t >= start

    # reguły zamknięć: przejście z niespełnionej na spełnioną
    for rule in cfg.get("reguły", []):
        iv = rule.get("interwał", "H1")
        lvl = float(rule.get("poziom") or 0)
        df = _closed_only(frames.get(iv, pd.DataFrame()), iv, now)
        if not lvl or df.empty:
            continue
        c = df["close"] + off
        hit = (c > lvl) if "powyżej" in rule.get("warunek", "") else (c < lvl)
        trans = hit & ~hit.shift(1, fill_value=True)
        for ts in df.index[trans.to_numpy()]:
            if in_range(bar_end(iv, ts)):
                rows.append({"czas": bar_end(iv, ts), "źródło": "reguła", "interwał": iv, "próg": lvl,
                             "opis": f"{rule.get('nazwa', '')}: {rule.get('warunek', '')}", "cena": round(float(c.loc[ts]), 2)})

    # poziomy: dotknięcie w świecy H1 albo przeskok luką między świecami, z przerwą jak w monitorze
    h1 = _closed_only(frames["H1"], "H1", now)
    if not h1.empty:
        lo, hi, op, cl = (h1[x].to_numpy() + off for x in ("low", "high", "open", "close"))
        for lv in cfg.get("poziomy", []):
            lvl = float(lv.get("poziom") or 0)
            if not lvl:
                continue
            touched, gap = level_hits(lo, hi, op, np.r_[np.nan, cl[:-1]], lvl)
            last: pd.Timestamp | None = None
            for i in np.flatnonzero(touched | gap):
                ts = bar_end("H1", h1.index[i])
                if last is not None and ts - last < pd.Timedelta(hours=cooldown_h):
                    continue
                last = ts
                if in_range(ts):
                    kind = "dotknięcie" if touched[i] else "przeskok luką"
                    rows.append({"czas": ts, "źródło": "poziom", "interwał": "H1", "próg": lvl,
                                 "opis": f"{lv.get('nazwa', '')}: {kind}", "cena": round(float(cl[i]), 2)})

    # strefy RSI na zamkniętych H1 i H4
    for iv in ("H1", "H4"):
        df = _closed_only(frames[iv], iv, now)
        r = df["rsi"]
        prev = r.shift(1)
        events = {
            "wychodzi z wyprzedania": (prev < 30) & (r >= 30),
            "wychodzi z wykupienia": (prev > 70) & (r <= 70),
            "wchodzi w wyprzedanie": (prev >= 30) & (r < 30),
        }
        for desc, mask in events.items():
            for ts in df.index[mask.fillna(False).to_numpy()]:
                if in_range(bar_end(iv, ts)):
                    rows.append({"czas": bar_end(iv, ts), "źródło": "RSI", "interwał": iv, "próg": 30 if "wyprzed" in desc else 70,
                                 "opis": f"RSI {desc} ({prev.loc[ts]:.1f} → {r.loc[ts]:.1f})", "cena": round(float(df.loc[ts, 'close'] + off), 2)})

    out = pd.DataFrame(rows, columns=["czas", "źródło", "interwał", "próg", "opis", "cena"])
    if not out.empty:
        out["czas"] = pd.to_datetime(out["czas"], utc=True).dt.tz_convert("Europe/Warsaw")
        out = out.sort_values("czas", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------- 2. czas trzymania


@dataclass
class HoldingStats:
    przypadki: int
    tp: int
    sl: int
    bez_wyjscia: int
    tp_mediana_h: float | None
    tp_p25_h: float | None
    tp_p75_h: float | None
    sl_mediana_h: float | None
    sl_p25_h: float | None
    sl_p75_h: float | None

    @property
    def wystarczy(self) -> bool:
        return self.przypadki >= MIN_SAMPLE

    @property
    def odsetek_bez_wyjscia(self) -> float:
        return self.bez_wyjscia / self.przypadki if self.przypadki else 0.0

    @property
    def odsetek_tp(self) -> float:
        return self.tp / self.przypadki if self.przypadki else 0.0


def similar_signals(h1: pd.DataFrame, direction: str, lookback: int = 20) -> np.ndarray:
    """Indeksy świec, na których zamknięcie przebiło szczyt (long) / dołek (short) z N poprzednich świec."""
    if direction == "long":
        ref = h1["high"].rolling(lookback).max().shift(1)
        cross = (h1["close"] > ref) & (h1["close"].shift(1) <= ref.shift(1))
    else:
        ref = h1["low"].rolling(lookback).min().shift(1)
        cross = (h1["close"] < ref) & (h1["close"].shift(1) >= ref.shift(1))
    return np.flatnonzero(cross.fillna(False).to_numpy())


def simulate(h1: pd.DataFrame, direction: str, signals: np.ndarray, sl_atr: float, tp_atr: float, limit_h: float) -> pd.DataFrame:
    """Śledzi każdy sygnał świeca po świecy, do limitu liczonego w czasie zegarowym (jak w monitorze,
    więc weekend też się liczy). Sygnały nachodzące na trwającą pozycję są pomijane.

    Zegar: wejście następuje na ZAMKNIĘCIU świecy sygnałowej i od tej chwili liczymy limit i czas.
    Świeca j kończy się (h[j] - h[i]) godzin po wejściu, więc TP w pierwszej świecy po sygnale
    to 1 godzina, a przy limicie 1 godziny sprawdzana jest dokładnie jedna świeca."""
    atr = h1["atr"].to_numpy()
    cl = h1["close"].to_numpy()
    sign = 1 if direction == "long" else -1
    hours = _hours(h1)
    rows = []
    busy_until = -1
    for i in signals:
        if i <= busy_until or i + 1 >= len(h1) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry = cl[i]
        res = _track(h1, hours, int(i), sign, entry - sign * sl_atr * atr[i], entry + sign * tp_atr * atr[i], limit_h)
        if res is None:
            continue
        outcome, exit_i, held = res
        rows.append({"wejście": h1.index[i] + pd.Timedelta(hours=1), "wynik": outcome, "godziny": held})
        busy_until = exit_i
    return pd.DataFrame(rows, columns=["wejście", "wynik", "godziny"])


def _hours(h1: pd.DataFrame) -> np.ndarray:
    idx = pd.DatetimeIndex(h1.index)
    return ((idx - idx[0]) / pd.Timedelta(hours=1)).to_numpy(dtype=float)  # niezależnie od jednostki (ns/us)


def _track(h1: pd.DataFrame, hours: np.ndarray, i: int, sign: int, sl: float, tp: float, limit_h: float) -> tuple[str, int, float] | None:
    """Wspólny zegar symulacji (opis w simulate). Zwraca (wynik, indeks świecy wyjścia, godziny)
    albo None, gdy historia kończy się przed limitem bez TP i SL."""
    hi, lo = h1["high"].to_numpy(), h1["low"].to_numpy()
    n = len(h1)
    outcome, exit_i = None, i
    for j in range(i + 1, n):
        if hours[j] - hours[i] > limit_h:  # koniec świecy j względem wejścia
            outcome = "limit"
            break
        exit_i = j
        sl_hit = lo[j] <= sl if sign > 0 else hi[j] >= sl
        tp_hit = hi[j] >= tp if sign > 0 else lo[j] <= tp
        if sl_hit:  # przy obu w jednej świecy - wariant ostrożny
            outcome = "SL"
            break
        if tp_hit:
            outcome = "TP"
            break
    if outcome is None:
        # koniec historii bez TP/SL: wynik znany tylko, jeśli minął już pełny limit
        if hours[-1] - hours[i] < limit_h:
            return None
        outcome = "limit"
    held = limit_h if outcome == "limit" else hours[exit_i] - hours[i]
    return outcome, exit_i, float(held)


# ---------------------------------------------------------------- 3. plan z panelu na historii


def _trend_at(h1_end_utc: pd.DatetimeIndex, df: pd.DataFrame, interval: str) -> np.ndarray:
    """Trend z ostatniej świecy H4/D1 ZAMKNIĘTEJ do końca danej świecy H1 (bez przyszłości)."""
    if df.empty:
        return np.full(len(h1_end_utc), NO_DATA, dtype=object)
    ends = pd.DatetimeIndex([bar_end(interval, ts) for ts in df.index]).tz_convert("UTC")
    right = pd.DataFrame({"t": ends, "trend": trend_series(df).to_numpy()}).sort_values("t")
    left = pd.DataFrame({"t": h1_end_utc})
    merged = pd.merge_asof(left, right, on="t", direction="backward")
    return merged["trend"].fillna(NO_DATA).to_numpy(dtype=object)


@dataclass
class PlanHistory:
    sygnaly: int
    trades: pd.DataFrame
    powod: str | None = None  # dlaczego nie ma przypadków

    def stats(self) -> HoldingStats:
        return summarize_holding(self.trades)


def plan_history(
    frames: dict[str, pd.DataFrame],
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    off: float,
    cost_pts: float,
    min_rr: float,
    limit_h: float,
    now: pd.Timestamp | None = None,
) -> PlanHistory:
    """Dokładnie warunki z 'Planu pozycji' na historii H1:
    zamknięcie H1 za poziomem wejścia, trend H4 i D1 nie przeciwny (z zamkniętych świec),
    zysk/ryzyko po kosztach >= min. Wejście na zamknięciu świecy, w której komplet warunków
    pojawił się po raz pierwszy (przejście z niespełnionych). SL/TP to ceny z planu.
    Wynik w punktach po kosztach: TP i SL po cenie poziomu, limit po zamknięciu ostatniej świecy."""
    now = now or pd.Timestamp.now(tz="UTC")
    sign = 1 if direction == "long" else -1
    cols = ["wejście", "cena wejścia", "wynik", "godziny", "wynik [pkt]"]
    risk_pts = (entry - sl) * sign + cost_pts
    reward_pts = (tp - entry) * sign - cost_pts
    if (entry - sl) * sign <= 0 or (tp - entry) * sign <= 0:
        return PlanHistory(0, pd.DataFrame(columns=cols), "SL i TP muszą leżeć po przeciwnych stronach wejścia.")
    if risk_pts <= 0 or reward_pts / risk_pts < min_rr:
        return PlanHistory(0, pd.DataFrame(columns=cols), f"Warunek zysku do ryzyka (≥ {min_rr:g}) nie jest spełniony dla tych poziomów, więc plan nigdy nie dałby wejścia.")

    h1 = _closed_only(frames["H1"], "H1", now)
    if len(h1) < 2:
        return PlanHistory(0, pd.DataFrame(columns=cols), "Za krótka historia H1.")
    c = h1["close"].to_numpy() + off
    h1_end = pd.DatetimeIndex(h1.index + pd.Timedelta(hours=1)).tz_convert("UTC")
    against = "spadkowy" if direction == "long" else "wzrostowy"
    t4 = _trend_at(h1_end, frames["H4"], "H4")
    td = _trend_at(h1_end, frames["D1"], "D1")
    ok = ((c - entry) * sign > 0) & ~np.isin(t4, [against, NO_DATA]) & ~np.isin(td, [against, NO_DATA])
    risk, reward = (c - sl) * sign, (tp - c) * sign
    ok &= (risk > 0) & (reward > 0) & (reward - cost_pts >= min_rr * (risk + cost_pts))
    starts = np.flatnonzero(ok & ~np.r_[True, ok[:-1]])  # przejście na "komplet spełniony"

    hours = _hours(h1)
    rows = []
    busy_until = -1
    for i in starts:
        if i <= busy_until or i + 1 >= len(h1):
            continue
        px = c[i]
        res = _track(h1, hours, int(i), sign, sl - off, tp - off, limit_h)
        if res is None:
            continue
        outcome, exit_i, held = res
        exit_px = {"TP": tp, "SL": sl}.get(outcome, c[exit_i])
        rows.append({"wejście": h1.index[i] + pd.Timedelta(hours=1), "cena wejścia": round(float(px), 2), "wynik": outcome,
                     "godziny": held, "wynik [pkt]": round(float((exit_px - px) * sign - cost_pts), 2)})
        busy_until = exit_i
    return PlanHistory(len(starts), pd.DataFrame(rows, columns=cols), None if len(starts) else "Na historii H1 komplet warunków nie wystąpił ani razu.")


def rejection_history(
    frames: dict[str, pd.DataFrame], limit_h: float, cost_pts: float, now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """H1 na ekstremach poprzedniej zamkniętej sesji D1, z wejściem na następnym otwarciu.

    Każde dotknięcie oceniamy osobno. Odrzucenia są podzbiorem tej samej próby.
    Oba warianty używają SL 1 ATR i TP 2 ATR, liczonych względem ceny wejścia.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    columns = ["sygnał", "kierunek", "poziom", "odrzucenie", "wejście", "cena wejścia", "wynik", "wynik [pkt]"]
    h1, daily = frames["H1"], frames["D1"]
    if len(h1) < 2 or daily.empty:
        return pd.DataFrame(columns=columns)
    h1 = _closed_only(h1, "H1", now)
    if len(h1) < 2:
        return pd.DataFrame(columns=columns)

    daily_end = pd.DatetimeIndex([bar_end("D1", day) for day in daily.index])
    previous = daily_end.searchsorted(h1.index, side="right") - 1
    opens, highs, lows, closes, atr = (h1[name].to_numpy() for name in ("open", "high", "low", "close", "atr"))
    hours = _hours(h1)
    duration = float(limit_h)
    records = []
    for direction, sign, side in (("long", 1, "low"), ("short", -1, "high")):
        levels = np.where(previous >= 0, daily[side].to_numpy()[previous], np.nan)
        touch = (lows <= levels) & (highs >= levels) & ((opens - levels) * sign > 0)
        rejection = rejects_level(opens, highs, lows, closes, levels, direction)
        for signal_i in np.flatnonzero(touch[:-1] & np.isfinite(atr[:-1]) & (atr[:-1] > 0)):
            entry_i = signal_i + 1
            entry = opens[entry_i]
            sl = entry - sign * atr[signal_i]
            tp = entry + sign * 2 * atr[signal_i]
            deadline = hours[entry_i] + duration
            outcome, exit_price = None, None
            for j in range(entry_i, len(h1)):
                if hours[j] + 1 > deadline:
                    outcome, exit_price = "limit", closes[j - 1]
                    break
                if (opens[j] - sl) * sign <= 0:
                    outcome, exit_price = "SL", opens[j]
                    break
                if (opens[j] - tp) * sign >= 0:
                    outcome, exit_price = "TP", tp
                    break
                adverse, favorable = (lows[j], highs[j]) if sign > 0 else (highs[j], lows[j])
                if (adverse - sl) * sign <= 0:
                    outcome, exit_price = "SL", sl
                    break
                if (favorable - tp) * sign >= 0:
                    outcome, exit_price = "TP", tp
                    break
            if outcome is None and hours[-1] + 1 >= deadline:
                outcome, exit_price = "limit", closes[-1]
            if exit_price is not None:
                records.append({
                    "sygnał": bar_end("H1", h1.index[signal_i]),
                    "kierunek": direction.upper(), "poziom": levels[signal_i],
                    "odrzucenie": bool(rejection[signal_i]),
                    "wejście": h1.index[entry_i], "cena wejścia": entry, "wynik": outcome,
                    "wynik [pkt]": (exit_price - entry) * sign - cost_pts,
                })
    return pd.DataFrame(records, columns=columns).sort_values(["sygnał", "kierunek"]).reset_index(drop=True)


def summarize_holding(trades: pd.DataFrame) -> HoldingStats:
    def q(s: pd.Series, p: float) -> float | None:
        return float(s.quantile(p)) if len(s) else None

    tp = trades.loc[trades["wynik"] == "TP", "godziny"]
    sl = trades.loc[trades["wynik"] == "SL", "godziny"]
    return HoldingStats(
        przypadki=len(trades), tp=len(tp), sl=len(sl), bez_wyjscia=int((trades["wynik"] == "limit").sum()),
        tp_mediana_h=q(tp, 0.5), tp_p25_h=q(tp, 0.25), tp_p75_h=q(tp, 0.75),
        sl_mediana_h=q(sl, 0.5), sl_p25_h=q(sl, 0.25), sl_p75_h=q(sl, 0.75),
    )


def holding_times(h1: pd.DataFrame, direction: str, sl_atr: float, tp_atr: float, limit_h: float, lookback: int = 20) -> tuple[HoldingStats, pd.DataFrame]:
    trades = simulate(h1, direction, similar_signals(h1, direction, lookback), sl_atr, tp_atr, limit_h)
    return summarize_holding(trades), trades
