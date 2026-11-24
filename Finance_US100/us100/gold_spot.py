"""Cena spot złota i lokalne pomiary różnicy względem xStation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from math import isfinite
from urllib.request import urlopen

from . import delivery

URL = "https://api.gold-api.com/price/XAU"
REFRESH_SECONDS = 30
MAX_AGE = timedelta(seconds=2 * REFRESH_SECONDS)
COMPARISONS_PATH = delivery.DATA_DIR / "porownanie_zlota_xstation.json"


def load_quote() -> tuple[float, datetime]:
    with urlopen(URL, timeout=4) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get("symbol") != "XAU" or payload.get("currency", "USD") != "USD":
        raise ValueError("Gold API nie zwróciło notowania XAU/USD.")
    price = float(payload["price"])
    updated = datetime.fromisoformat(payload["updatedAt"])
    if not isfinite(price) or price <= 0 or updated.tzinfo is None:
        raise ValueError("Nieprawidłowa cena lub czas notowania XAU/USD.")
    return price, updated.astimezone(timezone.utc)


def load_comparisons() -> list[dict]:
    rows = delivery.read_json(COMPARISONS_PATH, [])
    return rows if isinstance(rows, list) else []


def save_comparison(bid: float, price: float, updated: datetime, captured: datetime) -> None:
    if not isfinite(bid) or bid <= 0:
        raise ValueError("Podaj dodatnią cenę BID z xStation.")
    if not timedelta(0) <= captured - updated <= MAX_AGE:
        raise ValueError("Zaczekaj na aktualną cenę spot i spróbuj ponownie.")
    row = {
        "pomiar": captured.isoformat(),
        "aktualizacja_spot": updated.isoformat(),
        "xstation_bid": bid,
        "spot": price,
    }
    with delivery.file_lock(COMPARISONS_PATH.with_suffix(".lock")):
        delivery.write_json(COMPARISONS_PATH, [*load_comparisons(), row])
