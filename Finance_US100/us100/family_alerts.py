"""Powiadomienia rodzin sygnałów wysyłane przez monitor.

Dwa rodzaje wiadomości:
- „Symulacja · …” – rodzina jest w fazie sprawdzania. Informacja o wirtualnym wejściu i zamknięciu,
  wyraźnie opisana jako NIE-sygnał do handlu.
- „OTWÓRZ …” – tylko dla rodziny, która spełnia wszystkie warunki z ``families.readiness``:
  przewaga w testach historycznych, co najmniej 28 dni i 20 pozycji na żywo, dodatni wynik na żywo.

Ocena rodzin jest kosztowna, dlatego liczymy ją tylko po zamknięciu nowej świecy H1.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from . import delivery, families, notify
from .instruments import ALL, price_text

ALERT_SETTING = "powiadomienia symulacji rodzin"
SIGNAL_VALIDITY = pd.Timedelta(minutes=30)
KEEP_IDS = 300

Evaluator = Callable[..., list[families.Evaluation]]


def state_path(key: str):
    return notify.DATA_DIR / f"stan_rodzin_{key}.json"


def load_state(key: str) -> dict:
    return delivery.read_json(state_path(key), {})


def save_state(state: dict, key: str) -> None:
    delivery.write_json(state_path(key), state)


def _price(value: float, key: str) -> str:
    inst = ALL.get(key)
    return price_text(value, inst.decimals if inst else 2)


def _live_summary(ev: families.Evaluation, unit: str) -> str:
    live = ev.live
    if live is None or not live.count:
        return "Na żywo: brak zakończonych pozycji."
    return (f"Na żywo: {live.count} pozycji, TP {live.targets}, SL {live.stops}, limit {live.timeouts}, "
            f"średnio {live.mean_points:+.2f} {unit}.")


def _moment(identifier: str) -> str:
    return identifier.split("|", 1)[-1]


def _busy(ev: families.Evaluation, candidate: families.Candidate) -> bool:
    """Czy symulacja pominie kandydata, bo rodzina ma już pozycję z wcześniejszego sygnału."""
    if ev.live_open is not None and ev.live_open.candidate != candidate:
        return True
    return any(t.closed > candidate.at for t in ev.live_trades if t.candidate != candidate)


def check(key: str, cfg: dict, frames: dict[str, pd.DataFrame], state: dict, observed_at: pd.Timestamp,
          *, evaluate: Evaluator = families.evaluate_all) -> tuple[list[dict], dict, bool]:
    """Zwraca (nowe zdarzenia, nowy stan, czy trzeba zapisać ustawienia z nowymi wersjami reguł)."""
    simulation = cfg.get("symulacja rodzin") or {}
    if not simulation.get("start"):
        return [], state, False
    _, ends = families.closed_bars(frames["H1"], "H1", observed_at)
    if not len(ends):
        return [], state, False
    candle = ends[-1].isoformat()
    if state.get("świeca") == candle:
        return [], state, False

    starts, versions_changed = families.live_starts(simulation, key, observed_at)
    settings = cfg["plan_ustawienia"]
    params = families.Params(float(settings["koszt_pkt"]), float(settings["min_rr"]), float(settings["limit_h"]))
    report = evaluate(frames, params, observed_at, key, starts)

    unit = "USD" if key == "GOLD" else "pkt"
    off = float(cfg.get("przesunięcie", 0) or 0)
    enabled = bool(cfg.get(ALERT_SETTING, True))
    first_run = "świeca" not in state
    seen_signals = set(state.get("sygnały", []))
    seen_closed = set(state.get("zamknięte", []))
    events: list[dict] = []

    for ev in report:
        for trade in ev.live_trades:
            # wynik w identyfikatorze: po wyrównaniu historii (zmiana kontraktu) pozycja może zamknąć się inaczej
            trade_id = f"{ev.key}|{trade.opened.isoformat()}|{trade.outcome}|{trade.closed.isoformat()}"
            if trade_id in seen_closed:
                continue
            seen_closed.add(trade_id)
            if first_run or not enabled:
                continue
            opened = trade.opened.tz_convert(notify.TZ_LOCAL)
            event = notify.make_event(
                f"Symulacja · {ev.name} · zamknięto {trade.outcome}",
                f"Wirtualna pozycja {trade.candidate.direction.upper()} z {opened:%d.%m %H:%M}: "
                f"wynik {trade.points:+.2f} {unit} po kosztach.\n{_live_summary(ev, unit)}\n"
                "To wynik symulacji, nie transakcji.",
                key, at=trade.closed, time_label="Zamknięcie pozycji wirtualnej",
            )
            event["id"] = f"rodzina-{key}-{ev.key}-zamk-{int(trade.opened.timestamp())}"
            events.append(event)

        candidate = ev.latest
        if candidate is None or observed_at - candidate.at > families.MAX_LIVE_AGE:
            continue
        if ev.live_start is not None and candidate.at < ev.live_start:
            continue
        signal_id = f"{ev.key}|{candidate.at.isoformat()}"
        if signal_id in seen_signals:
            continue
        seen_signals.add(signal_id)
        if first_run:
            continue
        prices = (f"Wejście ok. {_price(candidate.reference + off, key)} na otwarciu kolejnej H1 · "
                  f"SL {_price(candidate.stop + off, key)} · TP {_price(candidate.target + off, key)} (ceny xStation).")
        ready = all(ok for _, ok in families.readiness(ev, observed_at))
        direction = candidate.direction.upper()
        if ready:
            valid_until = (candidate.at + SIGNAL_VALIDITY).tz_convert(notify.TZ_LOCAL)
            history = ev.total
            past = (f"Historia: {history.count} pozycji, średnio {history.mean_points:+.2f} {unit}. "
                    if history.mean_points is not None else "")
            event = notify.make_event(
                f"OTWÓRZ {direction} · {ev.name}",
                f"{prices}\n{candidate.note}.\n"
                f"{past}{_live_summary(ev, unit)}\n"
                f"Sygnał aktualny do {valid_until:%H:%M}. Później sprawdź, czy cena nie oddaliła się od wejścia.",
                key, at=candidate.at, time_label="Zamknięcie H1",
            )
            event["id"] = f"rodzina-{key}-{ev.key}-otworz-{int(candidate.at.timestamp())}"
            events.append(event)
        elif enabled and not _busy(ev, candidate):
            missing = [label for label, ok in families.readiness(ev, observed_at) if not ok]
            event = notify.make_event(
                f"Symulacja · {ev.name} · wirtualne wejście {direction}",
                f"To NIE jest sygnał do handlu – rodzina jest jeszcze sprawdzana.\n{prices}\n{candidate.note}.\n"
                f"{_live_summary(ev, unit)} Brakuje: {'; '.join(missing)}.",
                key, at=candidate.at, time_label="Zamknięcie H1",
            )
            event["id"] = f"rodzina-{key}-{ev.key}-sym-{int(candidate.at.timestamp())}"
            events.append(event)

    new_state = {
        "świeca": candle,
        "sygnały": sorted(seen_signals, key=_moment)[-KEEP_IDS:],
        "zamknięte": sorted(seen_closed, key=_moment)[-KEEP_IDS:],
        "sprawdzono": observed_at.isoformat(),
    }
    return events, new_state, versions_changed
