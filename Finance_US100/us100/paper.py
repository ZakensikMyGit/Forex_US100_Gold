"""Symulacja na kolejnych zamkniętych świecach H1 po włączeniu scenariusza."""

from __future__ import annotations

import math
from pathlib import Path
import uuid

import pandas as pd

from . import delivery, plan, storage

HOUR = pd.Timedelta(hours=1)
MODEL_VERSION = 2
VOLUME_BY_INSTRUMENT = {"US100": 0.003, "GOLD": 0.001}


def load(key: str) -> dict:
    return delivery.read_json(storage.DATA_DIR / f"symulacja_{key}.json", {})


def csv_path(key: str) -> Path:
    return storage.DATA_DIR / f"symulacja_{key}.csv"


def positions(state: dict) -> pd.DataFrame:
    rows = []
    for status, records in (("otwarta", state.get("otwarte", {}).values()), ("zamknięta", state.get("zamknięte", []))):
        for position in records:
            row = {"status": status, **position}
            row.setdefault("sl początkowy", row.get("sl"))
            row.setdefault("zmiany SL", 0)
            row.setdefault("description", None)
            rows.append(row)
    return pd.DataFrame(rows)


def csv_bytes(rows: pd.DataFrame) -> bytes:
    return rows.to_csv(index=False).encode("utf-8-sig")


def update(key: str, cfg: dict, frames: dict[str, pd.DataFrame]) -> None:
    previous = load(key)
    state = advance(key, cfg.get("symulacja", {}), frames, previous)
    if state != previous:
        delivery.write_json(storage.DATA_DIR / f"symulacja_{key}.json", state)
    if state.get("otwarte") or state.get("zamknięte"):
        path = csv_path(key)
        export = csv_bytes(positions(state))
        # Ponowna próba również bez nowych świec, np. po zamknięciu CSV w Excelu.
        if not path.exists() or path.read_bytes() != export:
            temporary = path.with_suffix(".csv.tmp")
            temporary.write_bytes(export)
            temporary.replace(path)


def new_scenario(key: str, entry: float, sl: float, tp: float, settings: dict, off: float) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "aktywacja": pd.Timestamp.now(tz="UTC").isoformat(),
        "próg": entry, "sl": sl, "tp": tp, "przesunięcie": off,
        "wolumen": VOLUME_BY_INSTRUMENT[key], "limit godzin": settings["limit_h"],
        "koszt pkt": settings["koszt_pkt"], "min rr": settings["min_rr"],
    }


def _exit(position: dict, bar: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, float, pd.Timestamp, str] | None:
    off = position["przesunięcie"]
    opened = pd.Timestamp(position["otwarto"])
    deadline = opened + pd.Timedelta(hours=position["limit godzin"])
    sign = 1 if position["kierunek"] == "long" else -1
    sl, tp = position["sl"], position["tp"]
    first = float(bar["open"]) + off
    if start >= deadline:
        return "limit", first, start, "limit upłynął najpóźniej na otwarciu świecy, przyjęto jej cenę otwarcia"
    if (first - sl) * sign <= 0:
        return "SL", first, start, "otwarcie na SL lub za SL, przyjęto cenę otwarcia świecy"
    if (first - tp) * sign >= 0:
        return "TP", tp, start, "otwarcie na TP lub za TP, przyjęto cenę TP"
    adverse = float(bar["low" if sign > 0 else "high"]) + off
    favorable = float(bar["high" if sign > 0 else "low"]) + off
    sl_hit = (adverse - sl) * sign <= 0
    tp_hit = (favorable - tp) * sign >= 0
    if sl_hit:
        detail = "SL i TP w tej samej świecy, pierwszeństwo SL" if tp_hit else "zakres świecy osiągnął SL"
        return "SL", sl, end, detail
    if tp_hit:
        return "TP", tp, end, "zakres świecy osiągnął TP"
    if end >= deadline:
        return "limit", float(bar["close"]) + off, end, "limit upłynął w tej świecy, przyjęto jej cenę zamknięcia"
    return None


def _entry_description(key: str, direction: str, scenario: dict, bar: pd.Series, signal: dict, first: bool) -> str:
    entry_state = "pierwsza ocena scenariusza po aktywacji" if first else "warunki ponownie spełnione po wcześniejszej niespełnionej ocenie"
    rows = [
        f"Wejście {key} {direction.upper()} na zamknięciu H1 {signal['świeca']}. Aktywacja {scenario['aktywacja']}. "
        f"Stan scenariusza: {entry_state}. Brak otwartej pozycji w tym kierunku.",
        f"Zamknięcie H1 z Yahoo Finance {bar['close']:g} + przesunięcie {scenario['przesunięcie']:+g} = {signal['cena']:g}. "
        f"Próg wejścia {scenario['próg']:g}, cena sygnału między SL {scenario['sl']:g} i TP {scenario['tp']:g}.",
        f"Zysk/ryzyko w planie {signal['rr planu']:.4f}, wymagane minimum {scenario['min rr']:g}, "
        f"przy cenie sygnału {signal['rr']:.4f}. Koszt {scenario['koszt pkt']:g} pkt, "
        f"wolumen {scenario['wolumen']:g} lot, limit {scenario['limit godzin']:g} godz.",
    ]
    for check in signal["warunki"]:
        row = f"Spełniony warunek: {check['Warunek']}. {check['Stan']}."
        if values := check.get("dane"):
            ema = ", ".join(f"{name.upper()} {values[name]:g}" if values[name] is not None else f"{name.upper()} brak"
                            for name in ("ema9", "ema26", "ema50", "ema200"))
            row += f" Świeca zamknięta {values['zamknięcie']}, cena zamknięcia {values['close']:g}, {ema} (skala źródła)."
        rows.append(row)
    return "\n".join(rows)


def _exit_description(position: dict, bar: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                      reason: str, price: float, when: pd.Timestamp, detail: str, pnl: float, risk: float) -> str:
    offset = position["przesunięcie"]
    o, h, l, c = (float(bar[name]) + offset for name in ("open", "high", "low", "close"))
    deadline = pd.Timestamp(position["otwarto"]) + pd.Timedelta(hours=position["limit godzin"])
    gap = "tak" if position.get("przerwa H1") else "nie"
    return (
        f"Wyjście {reason} po {price:g} o {when.isoformat()} — {detail}.\n"
        f"Świeca H1 {start.isoformat()} – {end.isoformat()}, po przesunięciu: otwarcie {o:g}, maksimum {h:g}, "
        f"minimum {l:g}, zamknięcie {c:g}. SL {position['sl']:g}, TP {position['tp']:g}, "
        f"SL początkowy {position.get('sl początkowy', position['sl']):g}, przesunięcia SL {position.get('zmiany SL', 0)}, "
        f"limit do {deadline.isoformat()}, przerwa H1 przed wyjściem: {gap}.\n"
        f"Wynik {pnl:+.6g} pkt po koszcie {position['koszt pkt']:g} pkt, "
        f"początkowe ryzyko {risk:g} pkt, wynik {pnl / risk:+.6g} R."
    )


def _update_open_position(position: dict, price: float, when: pd.Timestamp) -> dict:
    entry, sl, cost = position["cena wejścia"], position["sl"], position["koszt pkt"]
    initial_sl = position.get("sl początkowy", sl)
    sign = 1 if position["kierunek"] == "long" else -1
    pnl = (price - entry) * sign - cost
    previous_best = position.get("maks. wynik pkt", -cost)
    changes = position.get("zmiany SL", 0)
    updated = {**position, "sl początkowy": initial_sl, "zmiany SL": changes,
               "wynik otwarty pkt": round(pnl, 6), "maks. wynik pkt": max(pnl, previous_best)}
    new_sl = price - sign * abs(entry - initial_sl)
    if pnl > 0 and pnl > previous_best and (new_sl - sl) * sign > 0:
        text = position.get("description") or "Opis decyzji o wejściu nie był zapisywany w tej wersji symulacji."
        updated.update({
            "sl": new_sl,
            "zmiany SL": changes + 1,
            "description": (f"{text}\n\nPo zamknięciu H1 {when.isoformat()} po {price:g} wynik po kosztach wzrósł do "
                            f"{pnl:+g} pkt (poprzednie maksimum {previous_best:+g} pkt). "
                            f"SL przesunięto z {sl:g} na {new_sl:g}, zachowując początkowy dystans "
                            f"{abs(entry - initial_sl):g} pkt od ceny. Nowy SL obowiązuje od następnej świecy. "
                            f"TP {position['tp']:g}."),
        })
    return updated


def advance(
    key: str, scenarios: dict, frames: dict[str, pd.DataFrame], previous: dict,
    now: pd.Timestamp | None = None,
) -> dict:
    """Przetwórz kolejne świece raz. Stan pozycji, dziennik i kursor zapisują się razem."""
    scenarios = {
        direction: scenario for direction, scenario in scenarios.items()
        if all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
               for value in (scenario.get("sl"), scenario.get("tp")))
    }
    opened = dict(previous.get("otwarte", {}))
    if not scenarios and not opened:
        return previous
    h1 = frames["H1"]
    if h1.empty:
        return previous
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if start_at := previous.get("sprawdzono"):
        cursor = pd.Timestamp(start_at)
    else:
        cursor = min(pd.Timestamp(s["aktywacja"]) for s in scenarios.values())
    index = pd.DatetimeIndex(h1.index)
    bars = h1[(index + HOUR > cursor) & (index + HOUR <= now)]
    if bars.empty:
        return previous

    closed = list(previous.get("zamknięte", []))
    ready = {s_id: value for s_id, value in previous.get("warunki", {}).items()
             if any(s["id"] == s_id for s in scenarios.values())}
    for start, bar in bars.iterrows():
        end = start + HOUR
        for direction, pos in list(opened.items()):
            if start > cursor:
                pos = opened[direction] = {**pos, "przerwa H1": True}
            result = _exit(pos, bar, start, end)
            if result is None:
                price = float(bar["close"]) + pos["przesunięcie"]
                opened[direction] = _update_open_position(pos, price, end)
                continue
            reason, price, when, detail = result
            sign = 1 if direction == "long" else -1
            pnl = (price - pos["cena wejścia"]) * sign - pos["koszt pkt"]
            initial_sl = pos.get("sl początkowy", pos["sl"])
            risk = abs(pos["cena wejścia"] - initial_sl) + pos["koszt pkt"]
            exit_text = _exit_description(pos, bar, start, end, reason, price, when, detail, pnl, risk)
            entry_text = pos.get("description") or "Opis decyzji o wejściu nie był zapisywany w tej wersji symulacji."
            closed.append({**pos, "description": f"{entry_text}\n\n{exit_text}",
                           "zamknięto": when.isoformat(), "cena wyjścia": price,
                           "sl początkowy": initial_sl, "wynik otwarty pkt": None,
                           "powód": reason, "wynik pkt": round(pnl, 6), "wynik R": round(pnl / risk, 6),
                           "godziny": (when - pd.Timestamp(pos["otwarto"])).total_seconds() / 3600})
            del opened[direction]

        for direction, scenario in scenarios.items():
            if end <= pd.Timestamp(scenario["aktywacja"]):
                continue
            price = float(bar["close"]) + scenario["przesunięcie"]
            signal = plan.entry_signal(direction, scenario, frames, price, end, include_evidence=True)
            first = scenario["id"] not in ready
            was_ready = ready.get(scenario["id"], False)
            ready[scenario["id"]] = signal is not None
            if signal is None or was_ready or direction in opened:
                continue
            features = {name: float(bar[name]) if math.isfinite(bar[name]) else None
                        for name in ("rsi", "atr", "macd", "ema9", "ema26", "ema50", "ema200")}
            entry_params = {**scenario, "wolumen": VOLUME_BY_INSTRUMENT[key]}
            opened[direction] = {
                **entry_params, "instrument": key, "wersja symulacji": MODEL_VERSION,
                "kierunek": direction, "otwarto": end.isoformat(), "cena wejścia": price,
                "sl początkowy": scenario["sl"], "zmiany SL": 0,
                "wynik otwarty pkt": -scenario["koszt pkt"], "maks. wynik pkt": -scenario["koszt pkt"],
                "rr wejścia": signal["rr"], "przerwa H1": False,
                "trend H4": signal["trend H4"], "trend D1": signal["trend D1"], **features,
                "description": _entry_description(key, direction, entry_params, bar, signal, first),
            }
        cursor = end
    return {"sprawdzono": cursor.isoformat(), "otwarte": opened, "zamknięte": closed, "warunki": ready}
