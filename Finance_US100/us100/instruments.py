"""Instrumenty obsługiwane przez panel i monitor.

Każdy instrument XTB jest przybliżany kontraktem terminowym z CME (dane z Yahoo Finance).
Różnicę względem kursu w xStation koryguje przesunięcie ustawiane w panelu.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Instrument:
    key: str  # identyfikator w plikach ustawień i stanu
    name: str  # nazwa wyświetlana
    xtb_symbol: str  # symbol w xStation
    ticker: str  # Yahoo Finance
    source: str  # opis źródła danych
    usd_per_point: float  # wartość 1 punktu dla 1 lota w XTB (domyślna, edytowalna)
    lot_note: str
    decimals: int
    icon: str
    context: tuple[tuple[str, str, bool], ...] = field(default_factory=tuple)  # (etykieta, ticker, odwrotny kolor)


US100 = Instrument(
    key="US100",
    name="US100",
    xtb_symbol="US100",
    ticker="NQ=F",
    source="kontrakt Nasdaq 100 (NQ, CME)",
    usd_per_point=20.0,
    lot_note="1 lot = 20 USD za punkt",
    decimals=1,
    icon=":material/candlestick_chart:",
    context=(("VIX", "^VIX", True),),
)

GOLD = Instrument(
    key="GOLD",
    name="Złoto",
    xtb_symbol="GOLD",
    ticker="GC=F",
    source="kontrakt na złoto (GC, COMEX)",
    usd_per_point=100.0,
    lot_note="1 lot = 100 uncji, czyli 100 USD za każdy 1 USD ruchu ceny",
    decimals=2,
    icon=":material/diamond:",
    context=(("Indeks dolara DXY", "DX-Y.NYB", True), ("Rentowność 10-l. USA [%]", "^TNX", True)),
)

ALL: dict[str, Instrument] = {i.key: i for i in (US100, GOLD)}


def price_text(value: float, decimals: int) -> str:
    return f"{value:,.{decimals}f}".replace(",", " ")
