"""Kontrakty terminowe za tickerami Yahoo: kontrola przesunięcia i daty zmiany kontraktu.

Przesunięcie względem xStation to głównie różnica między ceną kontraktu terminowego a ceną
kasową (koszt utrzymania pozycji do wygaśnięcia). Maleje ona z dnia na dzień, a przy zmianie
kontraktu na następny (NQ=F, GC=F) skacze - US100 o ok. 200 pkt, złoto o ok. 20 USD.
Daty zmiany są przybliżone (bez świąt giełdowych), dlatego komunikaty mówią „ok.”.
"""

from __future__ import annotations

from datetime import date, timedelta

# Różnica kontrakt-kasa nie przekracza ok. 1% ceny nawet przy 3 miesiącach do wygaśnięcia.
# Większe przesunięcie oznacza prawie na pewno pomyłkę przy wpisywaniu ceny z xStation.
MAX_OFFSET_SHARE = 0.02


def offset_problem(offset: float, source_price: float) -> str | None:
    """Opis problemu, gdy przesunięcie jest nierealnie duże względem ceny; inaczej None."""
    if source_price <= 0:
        return None
    share = abs(offset) / source_price
    if share <= MAX_OFFSET_SHARE:
        return None
    return (f"Przesunięcie {offset:+g} to {share:.0%} ceny ({source_price:g}). Różnica między kontraktem a "
            f"xStation nie przekracza zwykle 1%. Sprawdź wpisaną cenę – przesunięcie nie zostało zapisane.")


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def _last_business_day(year: int, month: int) -> date:
    day = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _minus_business_days(day: date, n: int) -> date:
    while n:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            n -= 1
    return day


def _rolls(key: str, year: int) -> list[date]:
    """Przybliżone dni, w których obroty (i ticker Yahoo) przechodzą na następny kontrakt."""
    if key == "US100":
        # NQ: wygaśnięcie w 3. piątek III/VI/IX/XII, rolowanie 8 dni wcześniej (czwartek)
        return [_third_friday(year, m) - timedelta(days=8) for m in (3, 6, 9, 12)]
    if key == "GOLD":
        # GC: miesiące II/IV/VI/VIII/X/XII; pierwszy dzień zawiadomień to ostatni dzień roboczy
        # poprzedniego miesiąca, obroty przechodzą ok. 2 dni robocze wcześniej
        return [_minus_business_days(_last_business_day(year, m - 1), 2) for m in (2, 4, 6, 8, 10, 12)]
    return []


def rolls_around(key: str, day: date) -> tuple[date | None, date | None]:
    """(ostatnia zmiana kontraktu <= day, następna zmiana > day)."""
    days = sorted(d for y in (day.year - 1, day.year, day.year + 1) for d in _rolls(key, y))
    past = [d for d in days if d <= day]
    future = [d for d in days if d > day]
    return (past[-1] if past else None), (future[0] if future else None)


def roll_warning(key: str, ticker: str, calibrated: date | None, today: date, days_ahead: int = 7) -> str | None:
    """Ostrzeżenie o zmianie kontraktu po ostatniej kalibracji albo w najbliższych dniach."""
    last, upcoming = rolls_around(key, today)
    if last is None and upcoming is None:
        return None
    if calibrated is not None and last is not None and calibrated < last:
        return (f"Ok. {last:%d.%m} {ticker} przeszedł na następny kontrakt, a ostatnia kalibracja jest z "
                f"{calibrated:%d.%m}. Przesunięcie jest nieaktualne – skalibruj ponownie z ceną z xStation.")
    if upcoming is not None and (upcoming - today).days <= days_ahead:
        return (f"Ok. {upcoming:%d.%m} {ticker} przejdzie na następny kontrakt i różnica względem xStation "
                f"skoczy. Po tej dacie skalibruj przesunięcie ponownie.")
    return None
