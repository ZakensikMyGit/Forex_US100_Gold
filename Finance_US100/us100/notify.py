"""Wykrywanie zdarzeń (reguły, dywergencje, poziomy, RSI) dla monitora.

Stan między sprawdzeniami trzymany jest w dane/stan_monitora*.json, żeby to samo zdarzenie
nie było zgłaszane wielokrotnie. Wysyłka i ponawianie: moduł delivery.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import delivery
from . import harmonics
from . import indicators as ind
from . import plan
from .instruments import ALL, price_text
from .data import bar_end
from .signals import evaluate_rules, last_closed, level_hits

DATA_DIR = delivery.DATA_DIR
STATE_PATH = DATA_DIR / "stan_monitora.json"

LEVEL_COOLDOWN = timedelta(hours=4)

load_telegram = delivery.load_telegram
save_telegram = delivery.save_telegram


# ---------------------------------------------------------------- stan


def _state_path(key: str) -> Path:
    return STATE_PATH if key == "US100" else DATA_DIR / f"stan_monitora_{key}.json"


def load_state(key: str = "US100") -> dict:
    return delivery.read_json(_state_path(key), {})


def save_state(state: dict, key: str = "US100") -> None:
    delivery.write_json(_state_path(key), state)


def read_log(limit: int = 200, instrument: str | None = None) -> tuple[pd.DataFrame, int]:
    """Historia (od najnowszych) i liczba uszkodzonych wierszy, które pominięto."""
    rows, bad = delivery.read_log(limit, instrument)
    cols = ["czas", "instrument", "tytuł", "treść", "windows", "telegram"]
    df = pd.DataFrame(rows)
    return (df[[c for c in cols if c in df.columns]] if not df.empty else pd.DataFrame(columns=cols)), bad


def pending(instrument: str | None = None) -> list[dict]:
    q = delivery.load_queue()
    return [e for e in q if instrument is None or e.get("instrument") == instrument]

# ---------------------------------------------------------------- wykrywanie zdarzeń


def make_event(title: str, body: str, instrument: str = "US100") -> dict:
    return {"czas": datetime.now().strftime("%Y-%m-%d %H:%M"), "instrument": instrument, "tytuł": title, "treść": body}


def _fmt(x: float, instrument: str = "US100") -> str:
    inst = ALL.get(instrument)
    return price_text(x, inst.decimals if inst is not None else 2)


def _divergence_event(divergence: dict, frame: pd.DataFrame, interval: str, off: float, instrument: str) -> dict:
    bullish = divergence["typ"] == "bycza"
    indicator = divergence["wskaźnik"]
    before, after = (_fmt(divergence[f"cena {end}"] + off, instrument) for end in ("od", "do"))
    a, b = (float(frame.at[divergence[end], indicator.lower()]) for end in ("od", "do"))
    pattern = "niższy dołek" if bullish else "wyższy szczyt"
    change = "wzrósł" if bullish else "spadł"
    direction = "Możliwe odbicie" if bullish else "Możliwe wyhamowanie wzrostu"
    status = "Sygnał wstępny" if divergence["status"] == "w budowie" else "Układ potwierdzony"
    decimals = 1 if indicator == "RSI" else 3
    body = (
        f"Cena zrobiła {pattern}: z {before} na {after}. "
        f"{indicator} {change} z {a:.{decimals}f} do {b:.{decimals}f}.\n"
        f"{status}. Zobacz wykres {interval}."
    )
    return make_event(f"{direction} ({indicator}, {interval})", body, instrument)


def check_events(
    frames: dict[str, pd.DataFrame], cfg: dict, state: dict, instrument: str = "US100"
) -> tuple[list[dict], dict]:
    """Porównuje bieżący stan z poprzednim i zwraca nowe zdarzenia oraz nowy stan."""
    off = float(cfg.get("przesunięcie", 0) or 0)
    first_run = not state
    events: list[dict] = []
    new_state: dict = {"reguły": {}, "dywergencje": [], "rsi": {}, "dotknięcia": dict(state.get("dotknięcia", {}))}

    # 1. Reguły - zgłaszamy przejście z "niespełniona" na "spełniona"
    rules = evaluate_rules(cfg.get("reguły", []), frames, off)
    prev_rules = state.get("reguły", {})
    for _, r in rules.iterrows():
        key = f"{r['Reguła']}|{r['Warunek']}"
        hit = bool(r["Spełniona"])
        new_state["reguły"][key] = hit
        if hit and not prev_rules.get(key, False) and not first_run:
            body = (f"Cena zamknięcia {_fmt(r['Ostatnie zamknięcie'], instrument)} spełniła warunek "
                    f"„{r['Warunek']}”.\nZobacz zakładkę „Reguły”.")
            events.append(make_event(f"Spełniono regułę: {r['Reguła']}", body))

    # 2. Dywergencje - nowe wpisy
    prev_div = set(state.get("dywergencje", []))
    for iv, df in frames.items():
        for d in ind.detect_divergences(df):
            key = f"{iv}|{d['typ']}|{d['wskaźnik']}|{d['do']}|{d['status']}"
            new_state["dywergencje"].append(key)
            if key not in prev_div and not first_run:
                events.append(_divergence_event(d, df, iv, off, instrument))

    # 3. RSI wychodzi ze strefy wyprzedania / wykupienia (zamknięte świece H1 i H4)
    for iv in ("H1", "H4"):
        bar = last_closed(frames[iv], iv)
        if bar is None or pd.isna(bar["rsi"]):
            continue  # brak zamkniętej świecy albo za krótka historia
        rsi_now = float(bar["rsi"])
        rsi_prev = state.get("rsi", {}).get(iv)
        new_state["rsi"][iv] = rsi_now
        if rsi_prev is None or first_run:
            continue
        if rsi_prev < 30 <= rsi_now:
            title = "RSI odbił z niskiego poziomu"
        elif rsi_prev > 70 >= rsi_now:
            title = "RSI spadł z wysokiego poziomu"
        elif rsi_prev >= 30 > rsi_now:
            title = "RSI spadł poniżej 30"
        else:
            continue
        close = _fmt(float(bar["close"]) + off, instrument)
        events.append(make_event(f"{title} ({iv})", f"RSI: wcześniej {rsi_prev:.1f}, teraz {rsi_now:.1f}. "
                                                    f"Cena na zamknięciu: {close}. Zobacz wykres {iv}."))

    # 4. Poziomy: ta sama definicja co w odtwarzaniu historii (signals.level_hits):
    #    dotknięcie = poziom w zakresie świecy H1, przeskok luką = poziom między zamknięciem
    #    poprzedniej świecy a otwarciem kolejnej. Luki sprawdzamy tylko dla świec, które pojawiły
    #    się od ostatniego sprawdzenia, a przerwę 4 godz. liczymy od końca świecy z trafieniem.
    #    Wszystko na świecach - zmiana przesunięcia przez kalibrację nie jest ruchem rynku.
    h1 = frames["H1"]
    since = state.get("ostatni_bar")
    pos = int(np.count_nonzero(np.asarray(h1.index < pd.Timestamp(since)))) if since else len(h1) - 1  # pierwsza świeca >= since
    window = h1.iloc[pos:]
    prev_close = h1["close"].shift(1).iloc[pos:]
    is_new = (window.index > pd.Timestamp(since)) if since else np.zeros(len(window), dtype=bool)
    raw = float(h1["close"].iloc[-1])
    price = raw + off
    now = datetime.now()
    for lv in cfg.get("poziomy", []):
        lvl = float(lv.get("poziom") or 0)
        if not lvl:
            continue
        t_bars, g_bars = level_hits(window["low"] + off, window["high"] + off, window["open"] + off, prev_close + off, lvl)
        g_bars = g_bars & np.asarray(is_new)
        # świeca po świecy, jak w odtwarzaniu: przerwa 4 godz. od końca świecy z ostatnim zgłoszeniem
        for i in np.flatnonzero(t_bars | g_bars):
            hit_time = bar_end("H1", window.index[i])
            last_hit = new_state["dotknięcia"].get(str(lvl))
            if last_hit:
                lh = pd.Timestamp(last_hit)
                lh = lh.tz_localize("Europe/Warsaw") if lh.tzinfo is None else lh  # starszy zapis bez strefy
                if hit_time - lh < pd.Timedelta(LEVEL_COOLDOWN):
                    continue
            new_state["dotknięcia"][str(lvl)] = hit_time.isoformat()
            if first_run:
                continue
            level = _fmt(lvl, instrument)
            latest = _fmt(price, instrument)
            if t_bars[i]:
                title = f"Poziom {level}: dotknięty"
                body = f"Cena dotknęła zapisanego poziomu {level}. Ostatnia cena w panelu: {latest}."
            else:
                previous_close = _fmt(float(prev_close.iloc[i]) + off, instrument)
                opening = _fmt(float(window["open"].iloc[i]) + off, instrument)
                title = f"Poziom {level}: przeskoczony"
                body = (f"Cena przeskoczyła poziom {level} luką. Zamknięcie poprzedniej świecy: {previous_close}. "
                        f"Następna świeca otworzyła się na {opening}. Ostatnia cena w panelu: {latest}.")
            if lv.get("nazwa"):
                body += f"\nNazwa poziomu: {lv['nazwa']}."
            events.append(make_event(title, body))

    # 5. Otwarte plany pozycji: SL, TP, blisko SL, naruszenie warunku wejścia, limit czasu
    inst = ALL.get(instrument)
    plan_events, new_state["plany"] = plan.check_plan_events(
        cfg.get("plany", []), frames, off, window, state.get("plany", {}),
        decimals=inst.decimals if inst is not None else 2,
    )
    if not first_run:
        events.extend(make_event(t, b) for t, b in plan_events)

    watches = cfg.get("obserwacje wejścia", {})
    entry_events, new_state["obserwacje wejścia"] = plan.check_entry_watches(
        watches, frames, off, state.get("obserwacje wejścia", {}),
    )
    for direction, signal in entry_events.items():
        watch = watches[direction]
        title = f"Warunki wejścia {direction.upper()}"
        ratio = f"Zysk do ryzyka przy tej cenie: {signal['rr']:.2f} do 1 po kosztach."
        if signal["rr"] < watch["min rr"]:
            title = f"{direction.upper()}: zysk do ryzyka poniżej minimum"
            ratio += f" Twoje minimum to {watch['min rr']:g} do 1."
        context = (
            f"Świeca H1 zamknęła się o {pd.Timestamp(signal['świeca']).tz_convert('Europe/Warsaw'):%d.%m %H:%M} "
            f"na {_fmt(signal['cena'], instrument)}. Próg z planu: {_fmt(watch['próg'], instrument)}."
        )
        if "ekstremum odrzucenia" in signal:
            title = f"H1 · OTWÓRZ {direction.upper()}"
            context = plan.watch_rejection_text(
                direction, watch["próg"], signal["ekstremum odrzucenia"], signal["cena"], inst.decimals if inst else 2,
            )
        text = (
            f"{context}\n"
            f"Stop loss (SL): {_fmt(watch['sl'], instrument)}. Cel zysku (TP): {_fmt(watch['tp'], instrument)}.\n"
            f"{ratio} Sprawdź „Plan pozycji”."
        )
        event = make_event(title, text, instrument)
        event["id"] = f"wejscie-{instrument}-{watch['id']}"
        events.append(event)

    if cfg.get(harmonics.ALERT_SETTING):
        checked_at = pd.Timestamp.now(tz="UTC")
        patterns = harmonics.projections(frames["H4"], checked_at)
        new_state["AB=CD"] = None
        if patterns:
            projection = patterns[-1]
            status = harmonics.track(projection, h1, checked_at)
            signature = {"id": projection.id, "etap": status.stage}
            new_state["AB=CD"] = signature
            if "AB=CD" in state and state["AB=CD"] != signature:
                title, body = harmonics.describe(projection, status, off, inst.decimals if inst else 2)
                event = make_event(title, body, instrument)
                event["id"] = f"abcd-{instrument}-{projection.id}-{status.stage}"
                events.append(event)

    new_state["ostatni_bar"] = str(h1.index[-1])
    new_state["cena"] = price
    new_state["cena_surowa"] = raw
    new_state["sprawdzono"] = now.isoformat(timespec="seconds")
    for e in events:
        e["instrument"] = instrument
    return events, new_state



# ---------------------------------------------------------------- wysyłka


def _title(e: dict) -> str:
    inst = ALL.get(e.get("instrument", "US100"))
    return f"{inst.name if inst else e.get('instrument', '')} – {e['tytuł']}"


def dispatch(events: list[dict]) -> list[dict]:
    """Zapisuje zdarzenia w trwałej kolejce i próbuje je wysłać (z ponawianiem w kolejnych cyklach)."""
    delivery.enqueue(events)
    return flush()


def flush() -> list[dict]:
    return delivery.flush(title_for=_title)
