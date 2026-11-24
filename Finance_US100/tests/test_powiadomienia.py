"""Regresje czytelności i deduplikacji komunikatów monitora."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import FunctionTestCase
from unittest.mock import patch

import pandas as pd

from us100 import advisor, delivery, indicators, notify, paper, plan


def load_tests(_loader, suite, _pattern):
    """Dołącz testy funkcyjne przy uruchomieniu przez standardowy unittest."""
    for name, test in globals().items():
        if name.startswith("test_") and callable(test):
            suite.addTest(FunctionTestCase(test))
    return suite


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
    assert alert["czas rynkowy"] == "2026-11-16T11:00:00+01:00"
    assert alert["opis czasu"] == "Otwarcie H1"


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


def test_alert_planu_po_przerwie_podaje_pierwsza_swiece_ktora_dotknela_sl():
    index = pd.date_range("2026-11-16 10:00", periods=3, freq="h", tz="Europe/Warsaw")
    h1 = pd.DataFrame({"low": [99.0, 94.0, 93.0], "high": [101.0, 100.0, 101.0],
                       "close": [100.0, 96.0, 99.0]}, index=index)
    position = plan.new_plan("long", 100.0, 95.0, 110.0, 0.004, 24, None)
    position["otwarto"] = "2026-11-16T10:00"
    [alert], _ = plan.timed_plan_events([position], {"H1": h1}, 0.0, h1, {}, now=datetime(2026, 11, 16, 13))
    assert alert.at == index[1]
    assert alert.time_label == "Świeca H1"


def test_wysylka_i_ponowienie_pokazuja_zapisany_czas():
    historical = notify.make_event("Wejście", "Treść sygnału", "GOLD",
                                   at=pd.Timestamp("2026-07-02 08:00Z"), time_label="Zamknięcie H1")
    legacy = {"tytuł": "Starszy alert", "treść": "Starsza treść", "czas": "2026-07-01 17:42"}
    with TemporaryDirectory() as temp:
        path = Path(temp)
        with (
            patch.multiple(delivery, DATA_DIR=path, QUEUE_PATH=path / "queue.json",
                           LOG_PATH=path / "log.jsonl", LOCK_PATH=path / "queue.lock"),
            patch.object(delivery, "load_telegram", return_value={}),
        ):
            delivery.enqueue([historical, legacy])
            windows, telegram = [], []

            def send_windows(title, body):
                windows.append((title, body))
                return len(windows) > 2

            def send_telegram(title, body, _):
                telegram.append((title, body))
                return True

            assert not delivery.flush(sender_windows=send_windows, sender_telegram=send_telegram)
            assert len(delivery.flush(sender_windows=send_windows, sender_telegram=send_telegram)) == 2

    assert windows[:2] == windows[2:] == telegram == [
        ("Wejście", "Zamknięcie H1 · 02.07.2026 10:00\nTreść sygnału"),
        ("Starszy alert", "Wykryto · 01.07.2026 17:42\nStarsza treść"),
    ]


def test_kolejka_nie_wysyla_po_terminie_i_deduplikuje_w_jednym_zapisie():
    with TemporaryDirectory() as temp:
        path = Path(temp)
        expired = (datetime.now(timezone.utc) - delivery.MAX_AGE - timedelta(minutes=1)).isoformat()
        event = {"id": "identyczny", "tytuł": "Sygnał", "treść": "Zaległy", "utworzono": expired}
        with (
            patch.multiple(delivery, DATA_DIR=path, QUEUE_PATH=path / "queue.json",
                           LOG_PATH=path / "log.jsonl", LOCK_PATH=path / "queue.lock"),
            patch.object(delivery, "load_telegram", return_value={}),
            patch.object(delivery, "send_windows") as windows,
            patch.object(delivery, "send_telegram") as telegram,
        ):
            delivery.enqueue([event, event.copy()])
            [pending] = delivery.load_queue()
            assert pending["id"] == "identyczny"
            [finished] = delivery.flush(sender_windows=windows, sender_telegram=telegram)
            assert list(finished["kanały"].values()) == ["porzucono", "porzucono"]
            windows.assert_not_called()
            telegram.assert_not_called()


def test_alert_reguly_uzywa_czasu_notowania_przed_zamknieciem():
    index = pd.date_range("2026-11-16 10:00", periods=2, freq="h", tz="Europe/Warsaw")
    h1 = indicators.add_indicators(pd.DataFrame({
        "open": [100.0, 108.0], "close": [100.0, 110.0],
        "low": [99.0, 107.0], "high": [101.0, 111.0], "volume": [1.0, 1.0],
    }, index=index))
    frames = {interval: h1 for interval in ("H1", "H4", "D1")}
    cfg = {"reguły": [{"nazwa": "Test", "poziom": 105, "warunek": "zamknięcie powyżej"}]}
    observed = index[-1] + pd.Timedelta(minutes=59)
    prev = {"ostatni_bar": "2026-11-16 10:00"}
    before, state = notify.check_events(frames, cfg, prev, observed_at=observed)
    assert not before
    after, _ = notify.check_events(frames, cfg, state, observed_at=index[-1] + pd.Timedelta(hours=1))
    [trigger] = after
    assert trigger["tytuł"] == "Spełniono regułę: Test"


def test_plan_ignoruje_stara_cene_i_liczy_limit_przez_zmiane_czasu():
    stamp = pd.Timestamp("2026-10-25 00:00Z").tz_convert("Europe/Warsaw")
    h1 = pd.DataFrame({"low": [89.0], "high": [95.0], "close": [90.0]}, index=pd.DatetimeIndex([stamp]))
    position = plan.new_plan("long", 100.0, 95.0, 110.0, 0.004, 1, None)
    position["otwarto"] = "2026-10-25T02:45+01:00"
    now = pd.Timestamp("2026-10-25T03:46+01:00")
    [(title, _)], flags = plan.check_plan_events([position], {"H1": h1}, 0.0, h1, {}, now=now)
    assert title == "LONG: minął limit czasu"
    assert flags[position["id"]] == {"limit": True}
    position["otwarto"] = "2026-10-25T02:30+02:00"
    assert plan.plan_status(position, 100, 1, None, now=pd.Timestamp("2026-10-25T02:30+01:00"))["Zostało godzin"] == 0


def test_doradca_wysyla_pierwsza_poprawna_propozycje_tylko_raz():
    index = pd.date_range("2026-11-16 10:00", periods=2, freq="h", tz="Europe/Warsaw")
    frame = indicators.add_indicators(pd.DataFrame({
        "open": [100.0, 110.0], "close": [100.0, 111.0],
        "low": [99.0, 109.0], "high": [101.0, 112.0], "volume": [1.0, 1.0],
    }, index=index))
    frames = {interval: frame for interval in ("H1", "H4", "D1")}
    proposal = advisor.Proposal(advisor.Level("Fibo H4", "Fibo 61,8%", "long", 105.0),
                                index[-1], 108.0, 100.0, 124.0, 2.0, (("Trend", 1), ("Momentum", 1)))
    cfg = {"plan_ustawienia": {"koszt_pkt": 0.0, "min_rr": 1.5, "limit_h": 24}, advisor.ALERT_SETTING: True}
    with (
        patch.object(advisor, "assess", return_value=advisor.Assessment("", proposal)),
        patch.object(advisor, "replay", return_value={}) as replay,
    ):
        events, state = notify.check_events(frames, cfg, {"ostatni_bar": str(index[0])}, "GOLD", observed_at=proposal.at)
        [event] = events
        assert event["tytuł"] == "Doradca · Propozycja LONG"
        assert event["czas rynkowy"] == proposal.at.isoformat()
        decision = state["ocena doradcy"]
        assert decision["świeca"] == proposal.at.isoformat()
        assert "trafiła do kolejki" in decision["powód"]
        disabled_events, state = notify.check_events(frames, {**cfg, advisor.ALERT_SETTING: False}, state, "GOLD")
        assert not disabled_events
        repeated, state = notify.check_events(frames, cfg, state, "GOLD", observed_at=proposal.at)
        assert not repeated
        assert state["ocena doradcy"] == decision
        replay.assert_called_once()


def test_symulacja_doradcy_pomija_zakres_swiecy_w_trakcie_ktorej_weszla():
    starts = pd.date_range("2026-11-16 09:00", periods=3, freq="h", tz="Europe/Warsaw")
    h1 = pd.DataFrame({
        "open": [100.0, 100.0, 101.0], "close": [100.0, 101.0, 112.0],
        "low": [98.0, 90.0, 100.0], "high": [102.0, 102.0, 113.0],
    }, index=starts)
    proposal = advisor.Proposal(advisor.Level("Sesja D1", "Minimum", "long", 100.0),
                                starts[1], 100.0, 95.0, 112.0, 2.4, ())
    cfg = {"plan_ustawienia": {"koszt_pkt": 0.2, "min_rr": 1.5, "limit_h": 24}}
    now = starts[1] + pd.Timedelta(minutes=20)
    with (
        patch.object(advisor, "assess", return_value=advisor.Assessment("", proposal)),
        patch.object(advisor, "replay", return_value={}),
    ):
        frames = {"H1": h1.iloc[:2]}
        state = paper._advisor_entry("GOLD", cfg, frames, {}, now)
        assert state["otwarte"]["doradca"]["cena wejścia"] == 101.0
        assert paper._advisor_entry("GOLD", cfg, frames, state, now) == state
    after = paper.advance("GOLD", {}, {"H1": h1}, state, starts[2] + paper.HOUR)
    assert not after["otwarte"]
    [trade] = after["zamknięte"]
    assert trade["powód"] == "TP"
    assert trade["wynik pkt"] == 10.8


def test_symulacja_recznego_planu_sprawdza_rr_przy_cenie_wejscia():
    start = pd.Timestamp("2026-11-16 10:00Z")
    h1 = pd.DataFrame({"close": [110.0]}, index=pd.DatetimeIndex([start]))
    scenario = {"id": "long", "aktywacja": start.isoformat(), "przesunięcie": 0.0,
                "sl": 95.0, "tp": 112.0, "min rr": 1.5}
    with patch.object(plan, "entry_signal", return_value={"rr": 0.13}):
        state = paper.advance("US100", {"long": scenario}, {"H1": h1}, {}, start + paper.HOUR)
    assert not state["otwarte"]
    assert state["warunki"] == {"long": False}
