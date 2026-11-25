"""Powiadomienia rodzin: wirtualne wejścia i zamknięcia, alert OTWÓRZ, brak duplikatów."""

import unittest

import pandas as pd

from us100 import advisor, families, family_alerts
from us100.data import TZ_LOCAL

INDEX = pd.date_range("2026-12-01 08:00", periods=6, freq="h", tz=TZ_LOCAL)
FRAMES = {"H1": pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=INDEX)}
START = pd.Timestamp("2026-11-01", tz="UTC")
CFG = {"symulacja rodzin": {"start": START.isoformat(), "wersje": {}}, "przesunięcie": -1.5,
       "plan_ustawienia": {"koszt_pkt": 0.5, "min_rr": 1.5, "limit_h": 24.0},
       family_alerts.ALERT_SETTING: True}
EMPTY = advisor.History(0, 0, 0, 0, None)


def _candidate(hours_after_start: int) -> families.Candidate:
    return families.Candidate("wybicie_zapowiedz", "short", INDEX[hours_after_start] + pd.Timedelta(hours=1),
                              3600.0, 10.0, 15.0, 8.0, "Zamknięcie H1 za minimum")


def _evaluation(latest=None, trades=(), live=None, status="brak przewagi", open_position=None):
    return families.Evaluation("wybicie_zapowiedz", "A+", "opis", (), EMPTY, EMPTY, EMPTY, START, status,
                               ("info", ""), latest, START, live, tuple(trades), open_position)


def _run(state, evaluation, observed):
    return family_alerts.check("GOLD", CFG, FRAMES, state, observed, evaluate=lambda *a: [evaluation])


class FamilyAlertTests(unittest.TestCase):
    def test_pierwsze_sprawdzenie_tylko_zapisuje_stan(self):
        observed = INDEX[-1] + pd.Timedelta(hours=1, minutes=5)
        events, state, _ = _run({}, _evaluation(latest=_candidate(5)), observed)
        self.assertEqual(events, [])
        self.assertEqual(len(state["sygnały"]), 1)

    def test_wirtualne_wejscie_bez_duplikatu_i_tylko_po_nowej_swiecy(self):
        observed = INDEX[-1] + pd.Timedelta(hours=1, minutes=5)
        state = {"świeca": "stara", "sygnały": [], "zamknięte": []}
        events, state, _ = _run(state, _evaluation(latest=_candidate(5)), observed)
        [event] = events
        self.assertIn("wirtualne wejście SHORT", event["tytuł"])
        self.assertIn("NIE jest sygnał do handlu", event["treść"])
        self.assertIn("3 598.50", event["treść"])  # cena xStation = cena źródła + przesunięcie
        again, same_state, _ = _run(state, _evaluation(latest=_candidate(5)), observed)
        self.assertEqual(again, [])
        self.assertIs(same_state, state)  # ta sama świeca H1 - bez ponownej oceny

    def test_zamkniecie_i_alert_otworz_tylko_dla_gotowej_rodziny(self):
        observed = INDEX[-1] + pd.Timedelta(hours=1, minutes=5)
        trade = families.Trade(_candidate(0), INDEX[1], INDEX[3], "TP", 14.5)
        live = advisor.History(25, 13, 10, 2, 1.8, 6.0)
        state = {"świeca": "stara", "sygnały": [], "zamknięte": []}

        checking = _evaluation(latest=_candidate(5), trades=[trade], live=live, status="brak przewagi")
        titles = [e["tytuł"] for e in _run(state, checking, observed)[0]]
        self.assertEqual(titles, ["Symulacja · A+ · zamknięto TP", "Symulacja · A+ · wirtualne wejście SHORT"])

        history = advisor.History(186, 73, 90, 23, 2.24, 15.0)
        ready = families.Evaluation("wybicie_zapowiedz", "A+", "opis", (), history, EMPTY, EMPTY, START, "przewaga",
                                    ("success", ""), _candidate(5), START, live, (trade,), None)
        events = _run(state, ready, observed)[0]
        [otworz] = [e for e in events if e["tytuł"].startswith("OTWÓRZ")]
        self.assertEqual(otworz["tytuł"], "OTWÓRZ SHORT · A+")
        self.assertIn("Sygnał aktualny do", otworz["treść"])
        self.assertFalse(any("wirtualne wejście" in e["tytuł"] for e in events))

        stale = _run(state, ready, observed + pd.Timedelta(days=1))[0]  # sygnał starszy niż 2 godz.
        self.assertFalse(any(e["tytuł"].startswith("OTWÓRZ") for e in stale))

    def test_pomija_kandydata_gdy_trwa_pozycja_i_gdy_symulacja_wylaczona(self):
        observed = INDEX[-1] + pd.Timedelta(hours=1, minutes=5)
        busy = families.OpenPosition(_candidate(0), INDEX[1], 3601.0)
        state = {"świeca": "stara", "sygnały": [], "zamknięte": []}
        events, _, _ = _run(state, _evaluation(latest=_candidate(5), open_position=busy), observed)
        self.assertEqual(events, [])
        events, state, changed = family_alerts.check("GOLD", {**CFG, "symulacja rodzin": {}}, FRAMES, state, observed)
        self.assertEqual((events, changed), ([], False))


if __name__ == "__main__":
    unittest.main()
