"""Dziennik faktycznych transakcji.

Wpis powstaje przy zamknięciu planu (cena i czas wyjścia z xStation, powód wyjścia) albo ręcznie
dla transakcji spoza planu. Dla każdej transakcji z planem liczymy też "plan vs rzeczywistość":
jak skończyłaby się pozycja, gdyby trzymać się SL, TP i limitu czasu z planu, oraz bez limitu.
Dzięki temu widać, czy ręczne decyzje i ustawiony limit czasu pomagają.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pandas as pd

REASONS = ("TP", "SL", "ręcznie – limit czasu", "ręcznie – zmiana warunków", "ręcznie – inny powód")
MANUAL_LIMIT = "ręcznie – limit czasu"
NO_LIMIT_H = 30 * 24  # "bez limitu" = do SL/TP, maks. 30 dni


def _local(iso: str) -> Any:
    ts = pd.Timestamp(iso)
    return ts.tz_localize("Europe/Warsaw") if ts.tzinfo is None else ts


def _hours(a: str, b: str) -> float:
    return (_local(b) - _local(a)).total_seconds() / 3600


# ---------------------------------------------------------------- zapis


def validate_close(p: dict, exit_price: float, closed_iso: str, reason: str) -> list[str]:
    errs = []
    if exit_price <= 0:
        errs.append("cena wyjścia musi być większa od zera")
    if reason not in REASONS:
        errs.append("wybierz powód wyjścia")
    try:
        if _local(closed_iso) < _local(p["otwarto"]):
            errs.append("czas wyjścia nie może być wcześniejszy niż otwarcie")
    except (ValueError, TypeError):
        errs.append("niepoprawny czas wyjścia")
    return errs


def close_plan(p: dict, exit_price: float, closed_iso: str, reason: str, pln_actual: float | None, note: str = "") -> dict:
    p.update(
        {
            "status": "zamknięty",
            "cena wyjścia": float(exit_price),
            "zamknięto": closed_iso,
            "powód wyjścia": reason,
            "wynik PLN (xStation)": pln_actual,
            "notatka wyjścia": note,
        }
    )
    return p


def new_offplan(
    direction: str, entry: float, exit_price: float, volume: float, opened_iso: str, closed_iso: str,
    reason: str, pln_actual: float | None, note: str = "",
) -> dict:
    """Transakcja spoza planu - bez SL/TP, trafia tylko do dziennika."""
    return {
        "id": uuid.uuid4().hex[:8], "kierunek": direction, "cena wejścia": float(entry), "sl": None, "tp": None,
        "wolumen": float(volume), "limit godzin": None, "warunek wejścia": None, "otwarto": opened_iso,
        "notatka": "", "status": "zamknięty", "bez planu": True, "cena wyjścia": float(exit_price),
        "zamknięto": closed_iso, "powód wyjścia": reason, "wynik PLN (xStation)": pln_actual, "notatka wyjścia": note,
    }


# ---------------------------------------------------------------- plan vs rzeczywistość


def counterfactual(p: dict, h1: pd.DataFrame, off: float, limit_h: float | None) -> dict:
    """Porównanie na ciągłej historii H1. Przy SL i TP w jednej pełnej świecy przyjmujemy SL."""
    sl, tp = p.get("sl"), p.get("tp")
    if sl is None or tp is None:
        return {"wynik": None, "pkt": None, "godziny": None}
    sign = 1 if p["kierunek"] == "long" else -1
    entry = float(p["cena wejścia"])
    opened = _local(p["otwarto"]).tz_convert("UTC")
    hour = pd.Timedelta(hours=1)
    expected = opened.floor("h")
    bars = h1[h1.index >= expected]
    deadline = opened + pd.Timedelta(hours=limit_h) if limit_h else None
    now = pd.Timestamp.now(tz="UTC")
    last_close = None
    for ts, row in bars.iterrows():
        end = ts + hour
        if end > now:
            break
        if ts != expected:
            return {"wynik": "brak ciągłości H1", "pkt": None, "godziny": None}
        expected = end
        lo, hi = row["low"] + off, row["high"] + off
        sl_hit = lo <= sl if sign > 0 else hi >= sl
        tp_hit = hi >= tp if sign > 0 else lo <= tp
        if deadline is not None and end > deadline:
            if sl_hit or tp_hit or last_close is None:
                return {"wynik": "limit wewnątrz H1 – wynik nieustalony", "pkt": None, "godziny": None}
            return {"wynik": "limit", "pkt": (last_close - entry) * sign, "godziny": limit_h}
        elapsed = (end - opened).total_seconds() / 3600
        # Ekstremum świecy rozpoczętej przed wejściem mogło wystąpić przed transakcją.
        if ts >= opened and sl_hit:
            return {"wynik": "SL", "pkt": (sl - entry) * sign, "godziny": elapsed}
        if ts >= opened and tp_hit:
            return {"wynik": "TP", "pkt": (tp - entry) * sign, "godziny": elapsed}
        last_close = float(row["close"]) + off
        if deadline is not None and end == deadline:
            return {"wynik": "limit", "pkt": (last_close - entry) * sign, "godziny": limit_h}
    return {"wynik": "w toku / brak danych", "pkt": None, "godziny": None}


# ---------------------------------------------------------------- tabela i podsumowanie


def entries(plans: list[dict], h1: pd.DataFrame, off: float, usd_pp: float, usdpln: float | None) -> pd.DataFrame:
    rows = []
    for p in plans:
        if p.get("status") != "zamknięty" or p.get("cena wyjścia") is None:
            continue
        sign = 1 if p["kierunek"] == "long" else -1
        entry, exit_px, vol = float(p["cena wejścia"]), float(p["cena wyjścia"]), float(p["wolumen"])
        pts = (exit_px - entry) * sign
        risk = abs(entry - p["sl"]) if p.get("sl") else None
        pln_actual = p.get("wynik PLN (xStation)")
        pln_est = pts * vol * usd_pp * usdpln if usdpln else None
        held = _hours(p["otwarto"], p["zamknięto"])
        limit = p.get("limit godzin")
        cf = counterfactual(p, h1, off, limit)
        cf_free = counterfactual(p, h1, off, NO_LIMIT_H)
        rows.append(
            {
                "id": p["id"],
                "kierunek": p["kierunek"],
                "plan": "nie" if p.get("bez planu") else "tak",
                "otwarto": p["otwarto"].replace("T", " "),
                "zamknięto": p["zamknięto"].replace("T", " "),
                "godziny": round(held, 1),
                "powód wyjścia": p.get("powód wyjścia", ""),
                "wejście": entry,
                "wyjście": exit_px,
                "wynik [pkt]": round(pts, 2),
                "R": round(pts / risk, 2) if risk else None,
                "wynik [PLN]": round(float(pln_actual), 2) if pln_actual is not None else (round(pln_est, 2) if pln_est is not None else None),
                "PLN z": "xStation" if pln_actual is not None else "szacunek",
                "po limicie": held > float(limit) if limit else None,
                "plan: wynik": cf["wynik"],
                "plan [pkt]": round(cf["pkt"], 2) if cf["pkt"] is not None else None,
                "bez limitu: wynik": cf_free["wynik"],
                "bez limitu [pkt]": round(cf_free["pkt"], 2) if cf_free["pkt"] is not None else None,
                "notatka": p.get("notatka wyjścia", ""),
            }
        )
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    """Wskaźniki zbiorcze i porównania: ręczne wyjścia vs plan, wyjścia z powodu limitu vs trzymanie dalej."""
    if df.empty:
        return {"transakcje": 0}
    pts = df["wynik [pkt]"]
    pln = df["wynik [PLN]"].dropna()
    broker_pln = df.loc[df["PLN z"].eq("xStation"), "wynik [PLN]"].dropna()
    out: dict = {
        "transakcje": len(df),
        "zyskowne": int((pts > 0).sum()),
        "skuteczność": float((pts > 0).mean()),
        "xStation: n": len(broker_pln),
        "xStation: zyskowne": int((broker_pln > 0).sum()),
        "xStation: skuteczność": float((broker_pln > 0).mean()) if len(broker_pln) else None,
        "suma PLN": float(pln.sum()) if len(pln) else None,
        "średnie R": float(df["R"].dropna().mean()) if df["R"].notna().any() else None,
        "mediana godzin": float(df["godziny"].median()),
    }
    manual = df[df["powód wyjścia"].str.startswith("ręcznie") & df["plan [pkt]"].notna()]
    if len(manual):
        diff = manual["wynik [pkt]"] - manual["plan [pkt]"]
        out["ręczne: n"] = len(manual)
        out["ręczne: lepiej niż plan"] = int((diff > 0).sum())
        out["ręczne: różnica do planu [pkt]"] = float(diff.sum())
    lim = df[(df["powód wyjścia"] == MANUAL_LIMIT) & df["bez limitu [pkt]"].notna()]
    if len(lim):
        diff = lim["wynik [pkt]"] - lim["bez limitu [pkt]"]
        out["limit: n"] = len(lim)
        out["limit: lepiej niż trzymanie dalej"] = int((diff > 0).sum())
        out["limit: różnica do trzymania dalej [pkt]"] = float(diff.sum())
    over = df[df["po limicie"].eq(True)]
    if len(over):
        out["trzymane po limicie: n"] = len(over)
        out["trzymane po limicie: średni wynik [pkt]"] = float(over["wynik [pkt]"].mean())
        inside = df[df["po limicie"].eq(False)]
        out["w limicie: n"] = len(inside)
        out["w limicie: średni wynik [pkt]"] = float(inside["wynik [pkt]"].mean()) if len(inside) else None
    return out


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M")
