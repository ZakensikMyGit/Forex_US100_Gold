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

from us100 import delivery, engine, notify, paper, storage
from us100.instruments import ALL, Instrument, price_text

LOG_FILE = notify.DATA_DIR / "monitor.log"
LOCK_PORT = 47613  # zajęty port = monitor już działa


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


def check_instrument(inst: Instrument) -> None:
    cfg = storage.load_settings(inst.key)
    frames = CACHE.frames(inst.ticker)
    state = notify.load_state(inst.key)
    events, new_state = notify.check_events(frames, cfg, state, inst.key)
    if not state:
        events.insert(0, notify.make_event("Monitor działa",
                                           f"Ostatnia cena w panelu: {price_text(new_state['cena'], inst.decimals)}. "
                                           "Kolejna wiadomość pojawi się po wykryciu nowego zdarzenia.", inst.key))
    # Kolejność ma znaczenie: najpierw zdarzenia trafiają do trwałej kolejki, dopiero potem
    # zapisujemy nowy stan. Awaria wysyłki nie zgubi więc alertu - zostanie ponowiony.
    delivery.enqueue(events)
    notify.save_state(new_state, inst.key)
    log(f"{inst.name}: cena {new_state['cena']:.1f}, nowe zdarzenia: {len(events)}")
    for e in events:
        log(f"    {e['tytuł']}: {e['treść']}")
    paper.update(inst.key, cfg, frames)


def run_once() -> None:
    errors: dict[str, str] = {}
    for inst in ALL.values():
        try:
            check_instrument(inst)
        except Exception as exc:  # brak sieci, limit Yahoo - kolejny instrument i tak sprawdzamy
            errors[inst.key] = str(exc)
            log(f"{inst.name}: błąd: {exc}")
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
            "w_kolejce": len(notify.pending()),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor sygnałów US100")
    p.add_argument("--raz", action="store_true", help="jedno sprawdzenie i koniec")
    p.add_argument("--test", action="store_true", help="wyślij powiadomienie testowe")
    p.add_argument("--co", type=int, default=300, help="odstęp między sprawdzeniami w sekundach")
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

    log(f"start, sprawdzanie co {max(60, a.co)} s")
    while True:
        try:
            run_once()
        except Exception as exc:  # brak sieci, limit Yahoo - spróbujemy za chwilę
            log(f"błąd: {exc}")
        if a.raz:
            break
        time.sleep(max(60, a.co))


if __name__ == "__main__":
    main()
