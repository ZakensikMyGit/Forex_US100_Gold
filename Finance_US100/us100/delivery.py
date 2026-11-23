"""Trwała kolejka powiadomień z ponawianiem wysyłki osobno dla każdego kanału.

Przepływ: monitor wykrywa zdarzenia -> enqueue() zapisuje je na dysku (zanim zapisze się
nowy stan monitora) -> flush() próbuje wysłać. Kanał, który zwrócił błąd, jest ponawiany
w kolejnych cyklach, aż do MAX_ATTEMPTS prób albo MAX_AGE od powstania zdarzenia.
Zdarzenie trafia do historii (powiadomienia.jsonl) dopiero z ostatecznym statusem.

Statusy kanału: oczekuje, wysłano, błąd (do ponowienia), porzucono, wyłączony.
Token Telegrama jest trzymany w Menedżerze poświadczeń Windows (keyring).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "dane"
QUEUE_PATH = DATA_DIR / "kolejka_powiadomien.json"
LOG_PATH = DATA_DIR / "powiadomienia.jsonl"
LOCK_PATH = DATA_DIR / "kolejka.lock"
LEGACY_TELEGRAM_PATH = DATA_DIR / "telegram.json"

MAX_ATTEMPTS = 12
MAX_AGE = timedelta(hours=6)
CHANNELS = ("windows", "telegram")
KEYRING_SERVICE = "Finance_us100"


# ---------------------------------------------------------------- pliki


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, obj) -> None:
    """Zapis atomowy przez plik tymczasowy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def file_lock(path: Path, timeout: float = 15.0):
    """Blokada między procesami z odzyskiwaniem po przerwanym zapisie."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:  # blokada po przerwanym procesie
                if time.time() - path.stat().st_mtime > 120:
                    path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Plik {path.name} jest zablokowany przez inny proces.")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def queue_lock(timeout: float = 15.0):
    """Blokada kolejki - panel i monitor mogą wysyłać w tym samym czasie."""
    return file_lock(LOCK_PATH, timeout)


# ---------------------------------------------------------------- Telegram w Menedżerze poświadczeń


def _keyring():
    try:
        import keyring

        return keyring
    except Exception:  # brak biblioteki lub backendu
        return None


def load_telegram() -> dict:
    kr = _keyring()
    if kr is not None:
        try:
            raw = kr.get_password(KEYRING_SERVICE, "telegram")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        legacy = read_json(LEGACY_TELEGRAM_PATH, None)
        if legacy:  # migracja ze zwykłego pliku
            save_telegram(legacy)
            return legacy
        return {"token": "", "chat_id": ""}
    return read_json(LEGACY_TELEGRAM_PATH, {"token": "", "chat_id": ""})


def save_telegram(cfg: dict) -> str:
    """Zwraca miejsce zapisu: 'menedżer poświadczeń' albo 'plik'."""
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, "telegram", json.dumps(cfg))
            LEGACY_TELEGRAM_PATH.unlink(missing_ok=True)
            return "menedżer poświadczeń"
        except Exception:
            pass
    write_json(LEGACY_TELEGRAM_PATH, cfg)
    return "plik"


def telegram_storage() -> str:
    kr = _keyring()
    return "Menedżer poświadczeń Windows" if kr is not None else "plik dane/telegram.json"


# ---------------------------------------------------------------- kanały

_TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$x = $t.GetElementsByTagName('text')
$x.Item(0).AppendChild($t.CreateTextNode($env:US100_TITLE)) > $null
$x.Item(1).AppendChild($t.CreateTextNode($env:US100_BODY)) > $null
$n = [Windows.UI.Notifications.ToastNotification]::new($t)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe').Show($n)
"""


def send_windows(title: str, body: str) -> bool:
    env = {**os.environ, "US100_TITLE": title, "US100_BODY": body}
    try:
        res = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _TOAST_PS],
            env=env, capture_output=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def send_telegram(title: str, body: str, tg: dict) -> bool | None:
    """None = Telegram nie jest skonfigurowany."""
    token, chat = str(tg.get("token", "")).strip(), str(tg.get("chat_id", "")).strip()
    if not token or not chat:
        return None
    payload = urllib.parse.urlencode({"chat_id": chat, "text": f"{title}\n{body}"}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", payload, timeout=15) as r:
            return r.status == 200
    except OSError:
        return False


# ---------------------------------------------------------------- kolejka


def load_queue() -> list[dict]:
    q = read_json(QUEUE_PATH, [])
    return q if isinstance(q, list) else []


def enqueue(events: list[dict]) -> None:
    if not events:
        return
    with queue_lock():
        q = load_queue()
        known = {e.get("id") for e in q}
        for e in events:
            e.setdefault("id", uuid.uuid4().hex)
            if e["id"] in known:
                continue
            e.setdefault("utworzono", datetime.now().isoformat(timespec="seconds"))
            e["kanały"] = {ch: "oczekuje" for ch in CHANNELS}
            e["próby"] = 0
            q.append(e)
        write_json(QUEUE_PATH, q)


def _append_log(event: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in event.items() if k not in ("kanały", "próby", "utworzono")}
    row.update({ch: event["kanały"].get(ch) for ch in CHANNELS})
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush(title_for=lambda e: e.get("tytuł", ""), sender_windows=send_windows, sender_telegram=send_telegram) -> list[dict]:
    """Wysyła oczekujące zdarzenia. Zwraca zdarzenia zakończone w tym wywołaniu (trafiają do historii)."""
    """Postęp jest zapisywany po każdym kanale każdego zdarzenia, a wyjątek nadawcy liczy się
    jako błąd kanału. Przerwanie w połowie kolejki nie powoduje więc ponownej wysyłki tego,
    co już wyszło, ani podwójnego wpisu w historii."""
    finished: list[dict] = []
    with queue_lock():
        q = load_queue()
        if not q:
            return finished
        tg = load_telegram()
        now = datetime.now()

        def save() -> None:
            write_json(QUEUE_PATH, q)

        def attempt(send) -> bool | None:
            try:
                return send()
            except Exception:
                return False

        for e in list(q):
            title = title_for(e)
            st = e.setdefault("kanały", {ch: "oczekuje" for ch in CHANNELS})
            if st.get("windows") in ("oczekuje", "błąd"):
                st["windows"] = "wysłano" if attempt(lambda: sender_windows(title, e["treść"])) else "błąd"
                save()
            if st.get("telegram") in ("oczekuje", "błąd"):
                res = attempt(lambda: sender_telegram(title, e["treść"], tg))
                st["telegram"] = "wyłączony" if res is None else ("wysłano" if res else "błąd")
                save()
            e["próby"] = int(e.get("próby", 0)) + 1
            created = datetime.fromisoformat(e.get("utworzono", now.isoformat()))
            pending = [ch for ch in CHANNELS if st.get(ch) == "błąd"]
            expired = e["próby"] >= MAX_ATTEMPTS or now - created > MAX_AGE
            if pending and not expired:
                save()
                continue
            for ch in pending:
                st[ch] = "porzucono"
            # najpierw usuwamy z kolejki, potem dopisujemy do historii: w najgorszym razie
            # brakuje wpisu w historii, nigdy nie ma podwójnej wysyłki
            q.remove(e)
            save()
            _append_log(e)
            finished.append(e)
    return finished


# ---------------------------------------------------------------- historia


_OLD_STATUS = {True: "wysłano", False: "błąd", None: "wyłączony"}


def read_log(limit: int = 200, instrument: str | None = None) -> tuple[list[dict], int]:
    """Historia powiadomień od najnowszych. Uszkodzone wiersze są pomijane i liczone."""
    if not LOG_PATH.exists():
        return [], 0
    rows, bad = [], 0
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if not isinstance(r, dict):
                raise ValueError
        except ValueError:
            bad += 1
            continue
        r.setdefault("instrument", "US100")
        for ch in CHANNELS:  # stary format: True / False / None
            if not isinstance(r.get(ch), str):
                r[ch] = _OLD_STATUS.get(r.get(ch), "wyłączony")
        rows.append(r)
    if instrument:
        rows = [r for r in rows if r["instrument"] == instrument]
    return rows[::-1][:limit], bad
