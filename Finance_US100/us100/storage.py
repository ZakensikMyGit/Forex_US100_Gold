"""Ustawienia użytkownika zapisywane lokalnie w folderze dane/.

US100: dane/ustawienia.json, pozostałe instrumenty: dane/ustawienia_<KLUCZ>.json.
"""

from __future__ import annotations

import copy
import json
from math import isfinite
from pathlib import Path

from .delivery import write_json

DATA_DIR = Path(__file__).resolve().parent.parent / "dane"

DEFAULTS_US100: dict = {
    "przesunięcie": 0.0,
    "usd_za_punkt": 20.0,
    "poziomy": [
        {"nazwa": "Dołek z kwietnia 2026", "poziom": 23000},
        {"nazwa": "Fibo 0,5 fali 16 468 → 31 674", "poziom": 24071},
        {"nazwa": "Fibo 1,618 fali 27 181 → 31 673", "poziom": 24405},
        {"nazwa": "Fibo 0,786 fali 22 948 → 31 652", "poziom": 24810},
        {"nazwa": "Dołek tygodnia 20.11", "poziom": 24975},
        {"nazwa": "Pasmo EMA na H1", "poziom": 25740},
        {"nazwa": "Opór 26 000 / EMA 26 H1", "poziom": 26000},
        {"nazwa": "Przebity dołek 6.11", "poziom": 26375},
        {"nazwa": "Unieważnienie spadków (FxPro)", "poziom": 27250},
    ],
    "fibo": [
        {"nazwa": "Fala główna", "dołek": 16468, "szczyt": 31674},
        {"nazwa": "Fala od kwietnia 2026", "dołek": 22948, "szczyt": 31652},
        {"nazwa": "Ostatnia fala", "dołek": 27181, "szczyt": 31673},
    ],
    "reguły": [
        {
            "nazwa": "Odbicie: zamknięcie H1 nad pasmem średnich",
            "poziom": 25740,
            "warunek": "zamknięcie powyżej",
            "interwał": "H1",
            "znaczenie": "pierwszy krótkoterminowy sygnał odbicia, cele 26 000–26 120",
        },
        {
            "nazwa": "Powrót nad 26 000",
            "poziom": 26000,
            "warunek": "zamknięcie powyżej",
            "interwał": "H4",
            "znaczenie": "odzyskanie przebitego wsparcia",
        },
        {
            "nazwa": "Test dołka tygodnia",
            "poziom": 24975,
            "warunek": "zamknięcie poniżej",
            "interwał": "H1",
            "znaczenie": "presja na strefę 24 800",
        },
        {
            "nazwa": "Kontynuacja spadków",
            "poziom": 24400,
            "warunek": "zamknięcie poniżej",
            "interwał": "D1",
            "znaczenie": "droga do 24 070 i 23 000",
        },
        {
            "nazwa": "Unieważnienie scenariusza spadkowego",
            "poziom": 27250,
            "warunek": "zamknięcie powyżej",
            "interwał": "D1",
            "znaczenie": "według FxPro koniec scenariusza spadkowego",
        },
    ],
    "pozycje": [
        {"nazwa": "Pozycja 1", "kierunek": "long", "wolumen": 0.003, "cena otwarcia": 25980.0},
        {"nazwa": "Pozycja 2", "kierunek": "long", "wolumen": 0.003, "cena otwarcia": 25485.0},
    ],
}


DEFAULTS_GOLD: dict = {
    "przesunięcie": 0.0,
    "usd_za_punkt": 100.0,
    "poziomy": [],
    "fibo": [],
    "reguły": [],
    "pozycje": [],
}

DEFAULTS = {"US100": DEFAULTS_US100, "GOLD": DEFAULTS_GOLD}

# Kolumny tabel w edytorze - potrzebne, gdy lista jest pusta
COLUMNS = {
    "poziomy": ["nazwa", "poziom"],
    "fibo": ["nazwa", "dołek", "szczyt"],
    "reguły": ["nazwa", "poziom", "warunek", "interwał", "znaczenie"],
    "pozycje": ["nazwa", "kierunek", "wolumen", "cena otwarcia"],
}


for _d in DEFAULTS.values():
    _d.setdefault("kalibracja", {})  # czas i ceny ostatniej kalibracji przesunięcia
    _d.setdefault("plany", [])  # otwarte i zamknięte plany pozycji
    _d.setdefault("obserwacje wejścia", {})  # po jednej dla long i short
    _d.setdefault("powiadomienia AB=CD", False)
    _d.setdefault("powiadomienia doradcy", True)
    _d.setdefault("symulacja doradcy", True)
    _d.setdefault("symulacja", {})  # parametry automatycznych pozycji wirtualnych
    _d.setdefault("plan_ustawienia", {"koszt_pkt": 0.0, "min_rr": 1.5, "wolumen": 0.01, "limit_h": 24.0})

CONDITIONS = ("zamknięcie powyżej", "zamknięcie poniżej")
INTERVALS = ("D1", "H4", "H1")


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if isfinite(x) else None


def validate(tables: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], list[str]]:
    """Sprawdza wiersze z edytora. Puste wiersze są pomijane, błędne zgłaszane z numerem."""
    clean: dict[str, list[dict]] = {}
    errors: list[str] = []

    def blank(row: dict) -> bool:
        return all(v is None or v == "" or (isinstance(v, float) and v != v) for v in row.values())

    for name, rows in tables.items():
        out = []
        for i, row in enumerate(rows, start=1):
            if blank(row):
                continue
            where = f"{name}, wiersz {i}"
            r = {kk: (None if isinstance(v, float) and v != v else v) for kk, v in row.items()}
            if name == "poziomy":
                lvl = _num(r.get("poziom"))
                if not lvl or lvl <= 0:
                    errors.append(f"{where}: poziom musi być liczbą większą od zera")
                    continue
                out.append({"nazwa": str(r.get("nazwa") or f"Poziom {lvl:g}"), "poziom": lvl})
            elif name == "reguły":
                lvl = _num(r.get("poziom"))
                if not lvl or lvl <= 0:
                    errors.append(f"{where}: poziom musi być liczbą większą od zera")
                    continue
                if r.get("warunek") not in CONDITIONS:
                    errors.append(f"{where}: wybierz warunek")
                    continue
                if r.get("interwał") not in INTERVALS:
                    errors.append(f"{where}: wybierz interwał")
                    continue
                out.append({**r, "poziom": lvl, "nazwa": str(r.get("nazwa") or f"Reguła {i}"), "znaczenie": str(r.get("znaczenie") or "")})
            elif name == "fibo":
                lo, hi = _num(r.get("dołek")), _num(r.get("szczyt"))
                if not lo or not hi or lo <= 0 or hi <= lo:
                    errors.append(f"{where}: dołek i szczyt muszą być liczbami, a szczyt wyższy od dołka")
                    continue
                out.append({"nazwa": str(r.get("nazwa") or f"Fala {i}"), "dołek": lo, "szczyt": hi})
            elif name == "pozycje":
                vol, entry = _num(r.get("wolumen")), _num(r.get("cena otwarcia"))
                if r.get("kierunek") not in ("long", "short"):
                    errors.append(f"{where}: kierunek musi być long albo short")
                    continue
                if not vol or vol <= 0:
                    errors.append(f"{where}: wolumen musi być większy od zera")
                    continue
                if not entry or entry <= 0:
                    errors.append(f"{where}: cena otwarcia musi być większa od zera")
                    continue
                out.append({"nazwa": str(r.get("nazwa") or f"Pozycja {i}"), "kierunek": r["kierunek"], "wolumen": vol, "cena otwarcia": entry})
        clean[name] = out
    return clean, errors


def settings_path(key: str = "US100") -> Path:
    return DATA_DIR / ("ustawienia.json" if key == "US100" else f"ustawienia_{key}.json")


def load_settings(key: str = "US100") -> dict:
    defaults = DEFAULTS.get(key, DEFAULTS_GOLD)
    data = copy.deepcopy(defaults)
    path = settings_path(key)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update({k: v for k, v in saved.items() if k in defaults})
        except (OSError, json.JSONDecodeError):
            pass
    plan = defaults["plan_ustawienia"].copy()
    saved_plan = data["plan_ustawienia"]
    if isinstance(saved_plan, dict):
        for key in plan:
            value = _num(saved_plan.get(key))
            if value is not None and (value > 0 or (key == "koszt_pkt" and value == 0)):
                plan[key] = value
    data["plan_ustawienia"] = plan
    return data


def save_settings(data: dict, key: str = "US100") -> None:
    """Zapis atomowy: monitor nigdy nie odczyta w połowie zapisanego pliku."""
    write_json(settings_path(key), data)
