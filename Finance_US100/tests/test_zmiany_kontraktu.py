"""Zmiany kontraktu: wybór kontraktów, wykrycie chwili zmiany i wyrównanie historii."""

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from us100 import indicators, notify, rolls
from us100.data import TZ_LOCAL

INDEX = pd.date_range("2026-11-25 00:00", periods=10, freq="h", tz=TZ_LOCAL)


def _bars(closes, index=INDEX):
    closes = list(closes)
    opens = [closes[0], *closes[:-1]]
    return pd.DataFrame({"open": opens, "close": closes, "high": [max(o, c) + 1 for o, c in zip(opens, closes)],
                         "low": [min(o, c) - 1 for o, c in zip(opens, closes)], "volume": 1.0}, index=index)


FRONT = _bars([3500 + i for i in range(10)])
NEXT = _bars([3530 + i for i in range(10)])
CONTINUOUS = _bars([3500 + i for i in range(5)] + [3535 + i for i in range(5)])


class ContractPairTests(unittest.TestCase):
    def test_kontrakty_biezacy_i_nastepny(self):
        self.assertEqual(rolls.contract_pair("GOLD", date(2026, 11, 18)), ("GCZ26.CMX", "GCG27.CMX"))
        self.assertEqual(rolls.contract_pair("GOLD", date(2027, 1, 20)), ("GCG27.CMX", "GCJ27.CMX"))
        self.assertEqual(rolls.contract_pair("US100", date(2026, 12, 10)), ("NQZ26.CME", "NQH27.CME"))
        self.assertEqual(rolls.contract_pair("US100", date(2026, 12, 20)), ("NQH27.CME", "NQM27.CME"))
        self.assertTrue(rolls.near_roll("GOLD", date(2026, 11, 25)))
        self.assertFalse(rolls.near_roll("US100", date(2026, 11, 25)))


class DetectTests(unittest.TestCase):
    def test_chwila_i_wielkosc_zmiany(self):
        roll = rolls.detect("GC=F", CONTINUOUS, FRONT, NEXT, "GCZ26.CMX", "GCG27.CMX")
        self.assertEqual(roll.at, INDEX[5].tz_convert("UTC"))
        self.assertEqual(roll.gap, 30.0)
        self.assertEqual(roll.bar["close"], 3535.0)

    def test_brak_zmiany_przed_przelaczeniem(self):
        self.assertIsNone(rolls.detect("GC=F", FRONT, FRONT, NEXT, "a", "b"))
        self.assertIsNone(rolls.detect("GC=F", NEXT, FRONT, NEXT, "a", "b"))  # brak świec starego kontraktu

    def test_rejestr_bez_duplikatow(self):
        roll = rolls.detect("GC=F", CONTINUOUS, FRONT, NEXT, "GCZ26.CMX", "GCG27.CMX")
        with TemporaryDirectory() as tmp, patch.object(rolls, "REGISTRY_PATH", Path(tmp) / "zmiany.json"):
            self.assertTrue(rolls.save(roll))
            self.assertFalse(rolls.save(roll))
            self.assertEqual(rolls.load("GC=F"), [roll])
            self.assertEqual(rolls.load("NQ=F"), [])


class AdjustTests(unittest.TestCase):
    def test_historia_bez_luki_i_kumulacja_zmian(self):
        roll = rolls.detect("GC=F", CONTINUOUS, FRONT, NEXT, "GCZ26.CMX", "GCG27.CMX")
        adjusted = rolls.adjust(CONTINUOUS, "GC=F", "H1", [roll])
        self.assertEqual(list(adjusted["close"]), [3530 + i for i in range(5)] + [3535 + i for i in range(5)])
        self.assertEqual(adjusted.loc[INDEX[5], "open"], NEXT.loc[INDEX[5], "open"])  # świeca nowego kontraktu
        self.assertEqual(list(CONTINUOUS["close"])[:2], [3500, 3501])  # oryginał bez zmian
        earlier = rolls.Roll("GC=F", INDEX[2].tz_convert("UTC"), 10.0, "x", "y")
        both = rolls.adjust(CONTINUOUS, "GC=F", "H1", [earlier, roll])
        self.assertEqual(both.loc[INDEX[0], "close"], 3540.0)  # +10 i +30
        self.assertEqual(both.loc[INDEX[3], "close"], 3533.0)  # tylko +30

    def test_sesja_d1_ze_zmiana_zlozona_z_wyrownanych_h1(self):
        # Sesja 25.11 kończy się o 23:00 czasu polskiego, poprzednia (24.11) o 23:00 dnia 24.11.
        hourly_index = pd.date_range("2026-11-24 23:00", periods=10, freq="h", tz=TZ_LOCAL)
        continuous = _bars([3500 + i for i in range(5)] + [3535 + i for i in range(5)], hourly_index)
        roll = rolls.Roll("GC=F", hourly_index[5].tz_convert("UTC"), 30.0, "x", "y")
        hourly = rolls.adjust(continuous, "GC=F", "H1", [roll])
        daily = pd.DataFrame({"open": [3490.0, 3500.0], "high": [3495.0, 3545.0], "low": [3480.0, 3499.0],
                              "close": [3494.0, 3539.0], "volume": 1.0},
                             index=pd.DatetimeIndex(["2026-11-24", "2026-11-25"]))
        adjusted = rolls.adjust_daily(daily, "GC=F", hourly, [roll])
        self.assertEqual(adjusted.iloc[0]["close"], 3524.0)  # sesja przed zmianą: +30
        self.assertEqual(adjusted.iloc[1]["open"], hourly["open"].iloc[0])
        self.assertEqual(adjusted.iloc[1]["low"], hourly["low"].min())


class QuietAfterRollTests(unittest.TestCase):
    def test_wyrownanie_historii_nie_wysyla_alertu_reguly(self):
        h1 = indicators.add_indicators(_bars([3500.0, 3501.0, 3502.0], INDEX[:3]))
        frames = {interval: h1 for interval in ("H1", "H4", "D1")}
        cfg = {"przesunięcie": 0.0, "reguły": [{"nazwa": "Test", "poziom": 3501.5, "warunek": "zamknięcie powyżej"}]}
        observed = h1.index[-1] + pd.Timedelta(hours=1, minutes=5)
        state = {"ostatni_bar": str(h1.index[-1]), "reguły": {}, "przesunięcie": float("nan")}
        events, new_state = notify.check_events(frames, cfg, state, "GOLD", observed_at=observed)
        self.assertEqual(events, [])
        self.assertEqual(new_state["przesunięcie"], 0.0)


if __name__ == "__main__":
    unittest.main()
