"""Monitor sygnałów (US100, złoto) działający w tle.

    python monitor.py            # sprawdza co 5 minut, aż do zamknięcia
    python monitor.py --raz      # jedno sprawdzenie
    python monitor.py --test     # wysyła powiadomienie testowe
    python monitor.py --co 120   # sprawdza co 120 sekund

Dziennik działania: dane/monitor.log. Naraz może działać tylko jedna kopia monitora.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from datetime import datetime

import pandas as pd

from us100 import data, delivery, engine, notify, paper, storage
from us100.instruments import ALL, Instrument, price_text

LOG_FILE = notify.DATA_DIR / "monitor.log"
LOCK_PORT = 47613  # zajęty port = monitor już działa
CLOSE_DELAY_S = 10
RETRY_OFFSETS_S = (60, 120)


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    if sys.stdout is not None:  # pythonw.exe nie ma konsoli
        print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 1_000_000:
        LOG_FILE.write_text("", encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


CACHE = engine.FrameCache()
STATUS_PATH = notify.DATA_DIR / "monitor_status.json"


def check_instrument(inst: Instrument, snapshot: dict) -> None:
    cfg = storage.load_settings(inst.key)
    frames = CACHE.frames(inst.ticker)
    snapshot["H1"] = CACHE.hourly_snapshot[inst.ticker]
    observed_at = pd.Timestamp(snapshot["H1"]["pobrano"])
    state = notify.load_state(inst.key)
    events, new_state = notify.check_events(frames, cfg, state, inst.key, observed_at=observed_at)
    if not state:
        events.insert(0, notify.make_event("Monitor działa",
                                           f"Ostatnia cena w panelu: {price_text(new_state['cena'], inst.decimals)}. "
                                           "Kolejna wiadomość pojawi się po wykryciu nowego zdarzenia.", inst.key))
    # Kolejność ma znaczenie: najpierw zdarzenia trafiają do trwałej kolejki, dopiero potem
    # zapisujemy nowy stan. Awaria wysyłki nie zgubi więc alertu - zostanie ponowiony.
    delivery.enqueue(events)
    notify.save_state(new_state, inst.key)
    log(f"{inst.name}: cena {new_state['cena']:.1f}, nowe zdarzenia: {len(events)}")
    decision = new_state.get("ocena doradcy")
    if decision and decision != state.get("ocena doradcy"):
        close = f" · H1 {pd.Timestamp(decision['świeca']).tz_convert(data.TZ_LOCAL):%d.%m %H:%M}" if decision["świeca"] else ""
        log(f"    Doradca{close}: {decision['powód']}")
    for e in events:
        log(f"    {e['tytuł']}: {e['treść']}")
    paper.update(inst.key, cfg, frames, observed_at=observed_at)


def next_close(now: pd.Timestamp) -> pd.Timestamp:
    return now.floor("5min") + pd.Timedelta(minutes=5)


def missing_closes(snapshot: dict, expected: pd.Timestamp, checked: set[str]) -> set[str]:
    """Ponów oczekiwaną świecę tylko przy niedawnych notowaniach lub po pierwszym błędzie."""
    due = {"H1", "M5"} if expected.minute == 0 else {"M5"}
    missing = set()
    for interval in checked & due:
        value = snapshot.get(interval, {}).get("zamknięta do")
        seen = pd.Timestamp(value) if value else None
        if seen is None or expected - 2 * data.BAR_LENGTH[interval] <= seen < expected:
            missing.add(interval)
    return missing


def run_once(expected: pd.Timestamp | None = None, only: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    previous = delivery.read_json(STATUS_PATH, {})
    errors: dict[str, str] = {} if only is None else dict(previous.get("błędy", {}))
    errors.pop("wysyłka", None)
    snapshots = previous.get("notowania", {})
    pending = {}
    for inst in ALL.values():
        intervals = {"H1", "M5"} if only is None else only.get(inst.key, set())
        snapshot = snapshots.setdefault(inst.key, {})
        if "H1" in intervals:
            try:
                check_instrument(inst, snapshot)
                errors.pop(inst.key, None)
            except Exception as exc:
                errors[inst.key] = str(exc)
                log(f"{inst.name}: błąd: {exc}")
        if "M5" in intervals:
            try:
                snapshot["M5"] = CACHE.short_snapshot(inst.ticker)
                errors.pop(f"{inst.key} M5", None)
            except Exception as exc:
                errors[f"{inst.key} M5"] = str(exc)
                log(f"{inst.name} M5: błąd: {exc}")
        if expected is not None and (missing := missing_closes(snapshot, expected, intervals)):
            pending[inst.key] = missing
    try:
        for e in notify.flush():
            failed = [ch for ch, s in e["kanały"].items() if s == "porzucono"]
            if failed:
                log(f"    porzucono wysyłkę ({', '.join(failed)}): {e['tytuł']}")
        waiting = len(notify.pending())
        if waiting:
            log(f"w kolejce do ponowienia: {waiting}")
    except Exception as exc:
        errors["wysyłka"] = str(exc)
        log(f"wysyłka: błąd: {exc}")
    delivery.write_json(
        STATUS_PATH,
        {
            "cykl": datetime.now().isoformat(timespec="seconds"),
            "błędy": errors,
            "pobrania": CACHE.stats,
            "notowania": snapshots,
            "w_kolejce": len(notify.pending()),
        },
    )
    return pending


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor sygnałów US100")
    p.add_argument("--raz", action="store_true", help="jedno sprawdzenie i koniec")
    p.add_argument("--test", action="store_true", help="wyślij powiadomienie testowe")
    p.add_argument("--co", type=int, help="własny odstęp w sekundach; domyślnie synchronizacja z zamknięciem M5 i H1")
    a = p.parse_args()

    if a.test:
        ev = notify.make_event("Test powiadomień", "Jeśli to widzisz, powiadomienia działają.")
        notify.dispatch([ev])
        log(f"test: Windows {ev['kanały']['windows']}, Telegram {ev['kanały']['telegram']}")
        return

    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        log("monitor już działa w innym oknie lub w tle - kończę")
        return

    def check(expected=None, only=None):
        try:
            return run_once(expected, only)
        except Exception as exc:  # brak sieci, limit Yahoo - spróbujemy za chwilę
            log(f"błąd: {exc}")
            return {}

    if a.raz:
        check()
        return
    if a.co is not None:
        delay = max(60, a.co)
        log(f"start, sprawdzanie co {delay} s")
        while True:
            check()
            time.sleep(delay)

    log("start, synchronizacja z zamknięciem świec M5 i H1")
    check()
    while True:
        # Jeśli poprzedni cykl skończył się tuż po granicy, nadal można pobrać
        # właśnie zamkniętą świecę zgodnie z opóźnieniem startowym.
        expected = next_close(pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=CLOSE_DELAY_S))
        time.sleep(max(0, expected.timestamp() + CLOSE_DELAY_S - time.time()))
        pending = check(expected)
        for offset in RETRY_OFFSETS_S:
            if not pending or time.time() >= expected.timestamp() + 300:
                break
            time.sleep(max(0, expected.timestamp() + offset - time.time()))
            pending = check(expected, pending)


if __name__ == "__main__":
    main()
