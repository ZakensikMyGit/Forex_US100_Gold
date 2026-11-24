"""Plan pozycji: porównanie scenariuszy long / short / czekaj oraz śledzenie otwartych planów.

Aplikacja nie wybiera kierunku. Pokazuje, które z ustawionych przez użytkownika warunków
są spełnione, gdzie wypada SL i TP oraz ile wynosi ryzyko i możliwy wynik w PLN.
Wszystkie ceny są w skali XTB (cena źródła + przesunięcie).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from . import indicators as ind
from .data import TZ_LOCAL, bar_end, local_timestamp
from .instruments import price_text
from .signals import NO_DATA, last_closed, rejects_level, trend_series

DIRECTIONS = ("long", "short")


def _r(x: float, dec: int) -> float:
    return round(float(x), dec)


def nearest_levels(levels: list[tuple[str, float]], price: float) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
    """Najbliższy poziom nad ceną i najbliższy pod ceną."""
    above = [lv for lv in levels if lv[1] > price]
    below = [lv for lv in levels if lv[1] < price]
    return (min(above, key=lambda x: x[1]) if above else None, max(below, key=lambda x: x[1]) if below else None)


def last_swing(df: pd.DataFrame, direction: str, ref: float, off: float, k: int = 3, lookback: int = 150) -> float | None:
    """Ostatni potwierdzony dołek pod ref (long) albo szczyt nad ref (short), w skali XTB."""
    highs, lows = ind.swing_points(df.iloc[-lookback:], k)
    pts = (lows + off) if direction == "long" else (highs + off)
    pts = pts[pts < ref] if direction == "long" else pts[pts > ref]
    return float(pts.iloc[-1]) if len(pts) else None


def default_scenario(
    direction: str, frames: dict[str, pd.DataFrame], levels: list[tuple[str, float]], price: float, off: float, dec: int
) -> dict:
    """Propozycja startowa do edycji: wejście = najbliższy poziom w kierunku pozycji
    (zamknięcie H1 za nim), SL = za ostatnim dołkiem/szczytem H4, TP = kolejny poziom."""
    h4 = frames["H4"]
    atr4 = float(h4["atr"].iloc[-1]) if len(h4) and pd.notna(h4["atr"].iloc[-1]) else 0.0
    buf = 0.1 * atr4
    above, below = nearest_levels(levels, price)
    if direction == "long":
        trigger = above[1] if above else price
        swing = last_swing(h4, "long", trigger, off)
        sl = (swing - buf) if swing is not None else trigger - 1.5 * (atr4 or price * 0.01)
        nxt, _ = nearest_levels(levels, trigger)
        tp = nxt[1] if nxt else trigger + 2 * (trigger - sl)
    else:
        trigger = below[1] if below else price
        swing = last_swing(h4, "short", trigger, off)
        sl = (swing + buf) if swing is not None else trigger + 1.5 * (atr4 or price * 0.01)
        _, nxt = nearest_levels(levels, trigger)
        tp = nxt[1] if nxt else trigger - 2 * (sl - trigger)
    return {"wejście": _r(trigger, dec), "sl": _r(sl, dec), "tp": _r(tp, dec)}


def money(points: float, volume: float, usd_per_point: float, usdpln: float | None) -> float | None:
    return None if usdpln is None else points * volume * usd_per_point * usdpln


def _trade_metrics(direction: str, entry: float, sl: float, tp: float, cost_pts: float) -> tuple[float, float, float | None, bool]:
    sign = 1 if direction == "long" else -1
    risk = (entry - sl) * sign + cost_pts
    reward = (tp - entry) * sign - cost_pts
    geometry_ok = (entry - sl) * sign > 0 and (tp - entry) * sign > 0
    return risk, reward, reward / risk if geometry_ok and risk > 0 else None, geometry_ok


def reward_to_risk(direction: str, entry: float, sl: float, tp: float, cost_pts: float) -> float | None:
    return _trade_metrics(direction, entry, sl, tp, cost_pts)[2]


def _entry_checks(
    direction: str, entry: float, frames: dict[str, pd.DataFrame], h1_close: float | None,
    signal_time: pd.Timestamp | None, rr: float | None, min_rr: float, *, include_evidence: bool = False,
) -> list[dict]:
    sign = 1 if direction == "long" else -1
    against = "spadkowy" if direction == "long" else "wzrostowy"
    side = "powyżej" if direction == "long" else "poniżej"
    checks = [{
        "Warunek": f"Zamknięcie H1 {side} {entry:,.2f}".replace(",", " "),
        "Spełniony": h1_close is not None and (h1_close - entry) * sign > 0,
        "Stan": f"ostatnie zamknięcie H1: {h1_close:,.2f}".replace(",", " ") if h1_close is not None else "brak zamkniętej świecy",
    }]
    for interval in ("H4", "D1"):
        bar = last_closed(frames[interval], interval, signal_time) if signal_time is not None else None
        trend = trend_series(bar.to_frame().T).iloc[0] if bar is not None else NO_DATA
        check = {"Warunek": f"Trend {interval} nie jest {against}",
                 "Spełniony": trend not in (against, NO_DATA), "Stan": f"{interval}: {trend}"}
        if include_evidence and bar is not None:
            check["dane"] = {
                "zamknięcie": bar_end(interval, bar.name).isoformat(),
                **{name: float(bar[name]) if math.isfinite(bar[name]) else None
                   for name in ("close", "ema9", "ema26", "ema50", "ema200")},
            }
        checks.append(check)
    checks.append({
        "Warunek": f"Zysk do ryzyka (po kosztach) ≥ {min_rr:g}",
        "Spełniony": rr is not None and rr >= min_rr,
        "Stan": f"{rr:.2f}" if rr is not None else "SL/TP po złej stronie wejścia",
    })
    return checks


def evaluate_scenario(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    frames: dict[str, pd.DataFrame],
    off: float,
    volume: float,
    usd_per_point: float,
    usdpln: float | None,
    cost_pts: float = 0.0,
    min_rr: float = 1.5,
) -> dict:
    """Lista kontrolna warunków, odległości i wynik w PLN dla jednego kierunku."""
    risk_pts, reward_pts, rr, geometry_ok = _trade_metrics(direction, entry, sl, tp, cost_pts)
    bar = last_closed(frames["H1"], "H1")
    h1_close = float(bar["close"]) + off if bar is not None else None
    signal_time = bar_end("H1", bar.name) if bar is not None else None
    checks = _entry_checks(direction, entry, frames, h1_close, signal_time, rr, min_rr)
    atr_h1 = float(frames["H1"]["atr"].iloc[-1]) if pd.notna(frames["H1"]["atr"].iloc[-1]) else None
    atr_d1 = float(frames["D1"]["atr"].iloc[-1]) if pd.notna(frames["D1"]["atr"].iloc[-1]) else None
    return {
        "kierunek": direction,
        "checks": checks,
        "spełnione": sum(1 for c in checks if c["Spełniony"]),
        "ryzyko_pkt": risk_pts,
        "zysk_pkt": reward_pts,
        "rr": rr,
        "sl_atr_h1": abs(entry - sl) / atr_h1 if atr_h1 else None,
        "sl_atr_d1": abs(entry - sl) / atr_d1 if atr_d1 else None,
        "strata_pln": money(risk_pts, volume, usd_per_point, usdpln),
        "zysk_pln": money(reward_pts, volume, usd_per_point, usdpln),
        "poprawny": geometry_ok,
    }


def new_watch(
    entry: float, sl: float, tp: float, volume: float, limit_h: float, cost: float, min_rr: float,
    created: pd.Timestamp, expires: pd.Timestamp, *, require_rejection: bool = False,
) -> dict:
    return {"id": uuid.uuid4().hex, "próg": entry, "sl": sl, "tp": tp, "wolumen": volume,
            "limit godzin": limit_h, "koszt pkt": cost, "min rr": min_rr,
            "utworzono": created.isoformat(), "ważny do": expires.isoformat(),
            "odrzucenie H1": require_rejection}


def watch_rejects_level(direction: str, watch: dict, bar: pd.Series, off: float) -> bool:
    return bool(rejects_level(bar["open"] + off, bar["high"] + off, bar["low"] + off,
                              bar["close"] + off, watch["próg"], direction))


def watch_rejection_text(direction: str, level: float, extreme: float, close: float, decimals: int) -> str:
    level_text = price_text(level, decimals)
    extreme_text = price_text(extreme, decimals)
    close_text = price_text(close, decimals)
    if direction == "long":
        return f"Cena zeszła pod wsparcie {level_text} do {extreme_text} i wróciła nad poziom. H1 zamknęła się na {close_text}."
    return f"Cena przebiła sufit {level_text}, doszła do {extreme_text} i wróciła pod poziom. H1 zamknęła się na {close_text}."


def check_entry_watches(
    watches: dict, frames: dict[str, pd.DataFrame], off: float, previous: dict, now: pd.Timestamp | None = None,
) -> tuple[dict, dict]:
    """Sygnał raz na obserwację, z ostatniej zamkniętej H1 po jej zapisaniu."""
    if not watches:
        return {}, {}
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    bar = last_closed(frames["H1"], "H1", now)
    sent = {}
    new_events = {}
    for direction, watch in watches.items():
        watch_id = watch["id"]
        if watch_id in previous:
            sent[watch_id] = previous[watch_id]
            continue
        if bar is None or now >= pd.Timestamp(watch["ważny do"]):
            continue
        when = bar_end("H1", bar.name)
        created = pd.Timestamp(watch["utworzono"])
        if when <= created:
            continue
        rejection = watch.get("odrzucenie H1", False)
        if rejection and (bar.name < created or not watch_rejects_level(direction, watch, bar, off)):
            continue
        signal = entry_signal(direction, watch, frames, float(bar["close"]) + off, when)
        if signal is not None:
            if rejection:
                if signal["rr"] < watch["min rr"]:
                    continue
                signal["ekstremum odrzucenia"] = float(bar["low" if direction == "long" else "high"]) + off
            new_events[direction] = sent[watch_id] = signal
    return new_events, sent


def entry_signal(
    direction: str, scenario: dict, frames: dict[str, pd.DataFrame], price: float, when: pd.Timestamp,
    *, include_evidence: bool = False,
) -> dict | None:
    """Wspólna ocena zamknięcia H1 dla alertu i symulacji na bieżących notowaniach."""
    entry, sl, tp = scenario["próg"], scenario["sl"], scenario["tp"]
    cost, min_rr = scenario["koszt pkt"], scenario["min rr"]
    _, _, planned_rr, _ = _trade_metrics(direction, entry, sl, tp, cost)
    _, _, signal_rr, _ = _trade_metrics(direction, price, sl, tp, cost)
    checks = _entry_checks(direction, entry, frames, price, when, planned_rr, min_rr, include_evidence=include_evidence)
    if signal_rr is None or not all(c["Spełniony"] for c in checks):
        return None
    signal = {"świeca": when.isoformat(), "cena": price, "rr": signal_rr,
              "trend H4": checks[1]["Stan"].removeprefix("H4: "), "trend D1": checks[2]["Stan"].removeprefix("D1: ")}
    if include_evidence:
        signal["warunki"] = checks
        signal["rr planu"] = planned_rr
    return signal


# ---------------------------------------------------------------- otwarte plany


def new_plan(direction: str, entry: float, sl: float, tp: float, volume: float, limit_h: float, trigger: float | None, note: str = "") -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "kierunek": direction,
        "cena wejścia": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "wolumen": float(volume),
        "limit godzin": float(limit_h),
        "warunek wejścia": float(trigger) if trigger else None,
        "otwarto": pd.Timestamp.now(tz=TZ_LOCAL).isoformat(timespec="minutes"),
        "notatka": note,
        "status": "otwarty",
    }


def validate_plan(direction: str, entry: float, sl: float, tp: float, volume: float, limit_h: float) -> list[str]:
    sign = 1 if direction == "long" else -1
    errs = []
    if direction not in DIRECTIONS:
        errs.append("kierunek musi być long albo short")
    if not all(math.isfinite(value) and value > 0 for value in (entry, sl, tp)):
        errs.append("ceny muszą być większe od zera")
    if (entry - sl) * sign <= 0:
        errs.append("SL musi być " + ("poniżej" if sign > 0 else "powyżej") + " ceny wejścia")
    if (tp - entry) * sign <= 0:
        errs.append("TP musi być " + ("powyżej" if sign > 0 else "poniżej") + " ceny wejścia")
    if not math.isfinite(volume) or volume <= 0:
        errs.append("wolumen musi być większy od zera")
    if not math.isfinite(limit_h) or limit_h <= 0:
        errs.append("limit czasu musi być większy od zera")
    return errs


def plan_status(plan: dict, price: float, usd_per_point: float, usdpln: float | None, now: datetime | None = None) -> dict:
    checked = pd.Timestamp.now(tz=TZ_LOCAL) if now is None else local_timestamp(now)
    sign = 1 if plan["kierunek"] == "long" else -1
    entry, sl, tp, vol = plan["cena wejścia"], plan["sl"], plan["tp"], plan["wolumen"]
    opened = local_timestamp(plan["otwarto"])
    left_h = plan["limit godzin"] - (checked - opened).total_seconds() / 3600
    points = (price - entry) * sign
    pnl_pln = money(points, vol, usd_per_point, usdpln)
    return {
        "Plan": plan["id"],
        "Kierunek": plan["kierunek"],
        "Wejście": entry,
        "Wolumen [lot]": vol,
        "Cena wyceny": price,
        "SL": sl,
        "TP": tp,
        "Do SL [pkt]": round((price - sl) * sign, 2),
        "Do TP [pkt]": round((tp - price) * sign, 2),
        "Wynik teraz [pkt]": round(points, 2),
        "Wynik teraz [USD]": round(points * vol * usd_per_point, 2),
        "Wynik teraz [PLN]": round(pnl_pln, 2) if pnl_pln is not None else None,
        "Otwarto": opened.strftime("%Y-%m-%d %H:%M"),
        "Zostało godzin": round(left_h, 1),
    }


def check_plan_events(
    plans: list[dict], frames: dict[str, pd.DataFrame], off: float, window: pd.DataFrame, flags: dict,
    now: datetime | None = None, *, decimals: int = 2,
) -> tuple[list[tuple[str, str]], dict]:
    """Zwraca tytuły i treść alertów planu."""
    alerts, new_flags = timed_plan_events(plans, frames, off, window, flags, now, decimals=decimals)
    return [(alert.title, alert.body) for alert in alerts], new_flags


@dataclass(frozen=True)
class PlanAlert:
    title: str
    body: str
    at: pd.Timestamp | None = None
    time_label: str = "Sygnał"


def timed_plan_events(
    plans: list[dict], frames: dict[str, pd.DataFrame], off: float, window: pd.DataFrame, flags: dict,
    now: datetime | None = None, *, decimals: int = 2,
) -> tuple[list[PlanAlert], dict]:
    """Zdarzenia dla otwartych planów. window = świece H1 od ostatniego sprawdzenia (ceny źródła).

    Każde zdarzenie zgłaszane jest raz na plan (flagi w stanie monitora).
    """
    checked = pd.Timestamp.now(tz=TZ_LOCAL) if now is None else local_timestamp(now)
    out: list[PlanAlert] = []
    new_flags: dict = {}
    lows, highs = window["low"] + off, window["high"] + off
    h1 = frames["H1"]
    price = float(h1["close"].iloc[-1]) + off
    closed = last_closed(h1, "H1", checked)
    latest_possible = min(checked, bar_end("H1", h1.index[-1]))
    for p in plans:
        if p.get("status", "otwarty") != "otwarty":
            continue
        f = dict(flags.get(p["id"], {}))
        sign = 1 if p["kierunek"] == "long" else -1
        direction = p["kierunek"].upper()
        entry, sl, tp = p["cena wejścia"], p["sl"], p["tp"]
        reference = f"Wejście {price_text(entry, decimals)} · plan {p['id']}."

        # Tylko ruch PO otwarciu: liczą się świece H1, które zaczęły się nie wcześniej niż wejście,
        # oraz bieżąca cena, jeśli ostatni odczyt obejmuje czas po otwarciu. Zakres świecy, w której otwarto
        # pozycję, jest pomijany - nie da się w niej oddzielić ruchu sprzed i po wejściu.
        opened_ts = local_timestamp(p["otwarto"])
        current = latest_possible > opened_ts
        after = window.index >= opened_ts
        lo_after, hi_after = lows[after], highs[after]
        sl_bars = lo_after <= sl if sign > 0 else hi_after >= sl
        tp_bars = hi_after >= tp if sign > 0 else lo_after <= tp
        sl_at = sl_bars.idxmax() if sl_bars.any() else None
        tp_at = tp_bars.idxmax() if tp_bars.any() else None
        if not f.get("sl") and (sl_at is not None or (current and (price - sl) * sign <= 0)):
            f["sl"] = True
            out.append(PlanAlert(
                f"{direction}: cena dotarła do stop loss (SL)",
                f"Dane panelu pokazały osiągnięcie SL {price_text(sl, decimals)}. "
                f"Sprawdź wykonanie zlecenia w xStation.\n{reference}",
                sl_at, "Świeca H1",
            ))
        if not f.get("tp") and (tp_at is not None or (current and (price - tp) * sign >= 0)):
            f["tp"] = True
            out.append(PlanAlert(
                f"{direction}: cena dotarła do celu zysku (TP)",
                f"Dane panelu pokazały osiągnięcie TP {price_text(tp, decimals)}. "
                f"Sprawdź wykonanie zlecenia w xStation.\n{reference}",
                tp_at, "Świeca H1",
            ))

        risk = abs(entry - sl)
        to_sl = (price - sl) * sign
        if current and risk and not f.get("sl") and to_sl <= 0.25 * risk and not f.get("blisko_sl"):
            f["blisko_sl"] = True
            out.append(PlanAlert(
                f"{direction}: blisko stop loss (SL)",
                f"Cena w panelu: {price_text(price, decimals)}. Do SL {price_text(sl, decimals)} zostało "
                f"{price_text(to_sl, decimals)} pkt.\n{reference}",
            ))

        trig = p.get("warunek wejścia")
        # tylko świeca zamknięta po otwarciu pozycji
        if trig and closed is not None and not f.get("warunek") and bar_end("H1", closed.name) > opened_ts:
            c = float(closed["close"]) + off
            if (c - trig) * sign < 0:
                f["warunek"] = True
                side = "poniżej" if sign > 0 else "powyżej"
                out.append(PlanAlert(
                    f"{direction}: warunek wejścia przestał obowiązywać",
                    f"Świeca H1 zamknęła się na {price_text(c, decimals)}, {side} progu "
                    f"{price_text(trig, decimals)}. Sprawdź otwarty plan.\n{reference}",
                    bar_end("H1", closed.name), "Zamknięcie H1",
                ))

        deadline = opened_ts + pd.Timedelta(hours=float(p["limit godzin"]))
        if checked >= deadline and not f.get("limit"):
            f["limit"] = True
            out.append(PlanAlert(
                f"{direction}: minął limit czasu",
                f"Minęło zaplanowane {p['limit godzin']:g} godz. od wejścia. "
                f"Sprawdź w xStation, czy pozycja jest nadal otwarta.\n{reference}",
                pd.Timestamp(deadline), "Limit czasu",
            ))
        new_flags[p["id"]] = f
    return out, new_flags
