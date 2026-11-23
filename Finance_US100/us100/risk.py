"""Wynik pozycji przy różnych poziomach ceny.

W XTB 1 lot US100 to 20 USD za punkt, przeliczane na PLN po bieżącym kursie USD/PLN.
"""

from __future__ import annotations

import pandas as pd


def point_value_pln(volume: float, usdpln: float, usd_per_point: float = 20.0) -> float:
    return volume * usd_per_point * usdpln


def position_pnl(position: dict, price: float, usdpln: float, usd_per_point: float = 20.0) -> float:
    direction = 1 if str(position.get("kierunek", "long")).lower() == "long" else -1
    entry = float(position.get("cena otwarcia", 0) or 0)
    vol = float(position.get("wolumen", 0) or 0)
    return (price - entry) * direction * point_value_pln(vol, usdpln, usd_per_point)


def position_labels(positions: list[dict]) -> list[str]:
    """Unikalne etykiety kolumn: powtórzona lub pusta nazwa dostaje numer, żeby nie nadpisać kolumny."""
    used: set[str] = {"Poziom", "Cena", "Ruch [pkt]", "Razem [PLN]"}  # kolumny tabeli scenariuszy
    out = []
    for i, p in enumerate(positions, start=1):
        base = str(p.get("nazwa") or "").strip() or f"Pozycja {i}"
        label, n = base, 1
        while label in used:  # sprawdzamy z już nadanymi etykietami, także "A (2)" wpisanym ręcznie
            n += 1
            label = f"{base} ({n})"
        used.add(label)
        out.append(label)
    return out


def scenario_table(
    positions: list[dict],
    levels: list[tuple[str, float]],
    current: float,
    usdpln: float,
    usd_per_point: float = 20.0,
) -> pd.DataFrame:
    """Łączny wynik wszystkich pozycji przy każdym poziomie (bez spreadu i swapów)."""
    active = [p for p in positions if float(p.get("wolumen", 0) or 0) > 0]
    labels = position_labels(active)
    rows = []
    for name, lvl in sorted(levels + [("Teraz", current)], key=lambda x: -x[1]):
        row = {"Poziom": name, "Cena": round(lvl, 1), "Ruch [pkt]": round(lvl - current, 1)}
        total = 0.0
        for label, p in zip(labels, active):
            v = position_pnl(p, lvl, usdpln, usd_per_point)
            row[label] = round(v, 2)
            total += v
        row["Razem [PLN]"] = round(total, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def atr_stops(atr_value: float, current: float, multiples=(1.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    """Gdzie wypadłby stop-loss pozycji długiej / krótkiej oddalony o wielokrotność ATR."""
    rows = []
    for m in multiples:
        dist = m * atr_value
        rows.append(
            {
                "Wielokrotność ATR": f"{m:g}×",
                "Odległość [pkt]": round(dist, 1),
                "SL dla long": round(current - dist, 1),
                "SL dla short": round(current + dist, 1),
            }
        )
    return pd.DataFrame(rows)
