"""Kontrola przesunięcia względem xStation i dat zmiany kontraktu."""

import unittest
from datetime import date

import pandas as pd

from us100 import contracts, indicators, notify


class OffsetTests(unittest.TestCase):
    def test_odrzuca_literowke_w_cenie_zlota(self):
        # 14:04 24.11.2026: wpisano 2635 zamiast ok. 3635 przy GC=F 3642,5
        self.assertIsNotNone(contracts.offset_problem(2635.0 - 3642.5, 3642.5))

    def test_przepuszcza_zwykla_roznice_kontrakt_kasa(self):
        self.assertIsNone(contracts.offset_problem(-58.5, 26388.5))
        self.assertIsNone(contracts.offset_problem(-25.0, 3650.0))
        self.assertIsNone(contracts.offset_problem(-260.0, 26300.0))  # po zmianie NQ na kontrakt marcowy


class RollTests(unittest.TestCase):
    def test_daty_zmiany_kontraktu(self):
        # NQ grudzień 2026 wygasa 18.12, obroty przechodzą 10.12
        self.assertEqual(contracts.rolls_around("US100", date(2026, 11, 24)), (date(2026, 9, 10), date(2026, 12, 10)))
        # GC grudzień 2026: pierwszy dzień zawiadomień 30.11, obroty przechodzą 26.11
        self.assertEqual(contracts.rolls_around("GOLD", date(2026, 11, 24))[1], date(2026, 11, 26))
        # przełom roku: następny NQ to marzec 2027 (wygaśnięcie 19.03)
        self.assertEqual(contracts.rolls_around("US100", date(2026, 12, 20))[1], date(2027, 3, 11))

    def test_ostrzega_przed_i_po_zmianie(self):
        today = date(2026, 12, 5)
        self.assertIn("przejdzie", contracts.roll_warning("US100", "NQ=F", today, today))
        self.assertIsNone(contracts.roll_warning("US100", "NQ=F", date(2026, 11, 24), date(2026, 11, 24)))
        after = contracts.roll_warning("US100", "NQ=F", date(2026, 12, 1), date(2026, 12, 14))
        self.assertIn("skalibruj ponownie", after)
        self.assertIsNone(contracts.roll_warning("US100", "NQ=F", date(2026, 12, 11), date(2026, 12, 14)))


class RecalibrationAlertTests(unittest.TestCase):
    def test_zmiana_przesuniecia_nie_wysyla_alertu_reguly(self):
        index = pd.date_range("2026-11-24 10:00", periods=3, freq="h", tz="Europe/Warsaw")
        h1 = indicators.add_indicators(pd.DataFrame({
            "open": [3640.0] * 3, "close": [3642.0] * 3,
            "low": [3638.0] * 3, "high": [3644.0] * 3, "volume": [1.0] * 3,
        }, index=index))
        frames = {interval: h1 for interval in ("H1", "H4", "D1")}
        cfg = {"przesunięcie": 0.0, "reguły": [{"nazwa": "Test dołka", "poziom": 3500, "warunek": "zamknięcie poniżej"}]}
        observed = index[-1] + pd.Timedelta(hours=1, minutes=5)
        _, state = notify.check_events(frames, cfg, {"ostatni_bar": str(index[-1])}, "GOLD", observed_at=observed)
        self.assertEqual(state["przesunięcie"], 0.0)

        cfg["przesunięcie"] = -1007.5  # błędna kalibracja: cena spada „o 1000 USD”
        events, state = notify.check_events(frames, cfg, state, "GOLD", observed_at=observed)
        self.assertEqual(events, [])
        events, _ = notify.check_events(frames, cfg, state, "GOLD", observed_at=observed)
        self.assertEqual(events, [])  # stan reguły zapisany po kalibracji - bez spóźnionego alertu


if __name__ == "__main__":
    unittest.main()
