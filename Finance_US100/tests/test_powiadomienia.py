"""Regresje czytelności i deduplikacji komunikatów monitora."""

from datetime import datetime

import pandas as pd

from us100 import indicators, notify, plan


def test_wstepne_odbicie_podaje_dolki_i_wartosci_rsi():
    index = pd.date_range("2026-11-16 10:00", periods=2, freq="4h", tz="Europe/Warsaw")
    frame = pd.DataFrame({"rsi": [22.4, 29.1]}, index=index)
    divergence = {
        "typ": "bycza", "wskaźnik": "RSI", "status": "w budowie",
        "od": index[0], "do": index[1], "cena od": 3835.2, "cena do": 3566.3,
    }
    event = notify._divergence_event(divergence, frame, "H4", 0.0, "GOLD")
    assert event["tytuł"] == "Możliwe odbicie (RSI, H4)"
    assert "niższy dołek: z 3 835.20 na 3 566.30" in event["treść"]
    assert "RSI wzrósł z 22.4 do 29.1" in event["treść"]
    assert "Sygnał wstępny. Zobacz wykres H4." in event["treść"]


def test_alert_o_luce_podaje_zamkniecie_i_otwarcie():
    index = pd.date_range("2026-11-16 10:00", periods=2, freq="h", tz="Europe/Warsaw")
    h1 = indicators.add_indicators(pd.DataFrame({
        "open": [100.0, 110.0], "close": [100.0, 111.0],
        "low": [99.0, 109.0], "high": [101.0, 112.0], "volume": [1.0, 1.0],
    }, index=index))
    cfg = {"poziomy": [{"nazwa": "Poziom testowy", "poziom": 105.0}]}
    previous = {"ostatni_bar": str(index[0])}
    events, _ = notify.check_events({"H1": h1, "H4": h1, "D1": h1}, cfg, previous)
    [alert] = [event for event in events if event["tytuł"].startswith("Poziom")]
    assert "przeskoczyła poziom 105.0 luką" in alert["treść"]
    assert "Zamknięcie poprzedniej świecy: 100.0" in alert["treść"]
    assert "Następna świeca otworzyła się na 110.0" in alert["treść"]


def test_alert_stop_loss_zacheca_do_sprawdzenia_wykonania_i_nie_powtarza_sie():
    index = pd.date_range("2026-11-16 10:00", periods=2, freq="h", tz="Europe/Warsaw")
    h1 = pd.DataFrame({"low": [99.0, 94.0], "high": [101.0, 100.0], "close": [100.0, 96.0]}, index=index)
    position = plan.new_plan("long", 100.0, 95.0, 110.0, 0.004, 24, None)
    position["otwarto"] = "2026-11-16T10:00"
    now = datetime(2026, 11, 16, 12)
    args = ([position], {"H1": h1}, 0.0, h1)
    [(title, body)], flags = plan.check_plan_events(*args, {}, now=now)
    assert title == "LONG: cena dotarła do stop loss (SL)"
    assert "Sprawdź wykonanie zlecenia w xStation." in body
    assert plan.check_plan_events(*args, flags, now=now)[0] == []
