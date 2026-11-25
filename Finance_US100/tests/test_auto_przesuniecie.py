"""Automatyczne przesunięcie: pary cen z tej samej chwili, mediana, próg zmiany."""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from us100 import auto_offset
from us100.data import TZ_LOCAL


def _m5(closes, start="2026-11-25 14:00"):
    index = pd.date_range(start, periods=len(closes), freq="5min", tz=TZ_LOCAL)
    return pd.DataFrame({"close": closes}, index=index)


class GoldTests(unittest.TestCase):
    def test_pary_spot_i_zamkniecia_m5_z_tej_samej_chwili(self):
        m5 = _m5([3600.0, 3602.0, 3604.0, 3606.0])  # świece 14:00-14:20, końce 14:05-14:20
        ends = m5.index + pd.Timedelta(minutes=5)
        spots = [{"czas": (ends[i] + pd.Timedelta(seconds=10)).isoformat(), "cena": close - 8.0 + i * 0.1}
                 for i, close in enumerate(m5["close"])]
        spots.append({"czas": (ends[1] + pd.Timedelta(minutes=2)).isoformat(), "cena": 1.0})  # bez pary (2 min)
        estimate = auto_offset.gold_estimate(spots, m5)
        self.assertEqual(estimate.samples, 4)
        self.assertAlmostEqual(estimate.value, -7.85)
        self.assertAlmostEqual(estimate.spread, 0.3)
        self.assertEqual(estimate.price, 3606.0)
        self.assertIsNone(auto_offset.gold_estimate(spots[:2], m5))  # za mało par

    def test_zapis_spot_bez_duplikatow_i_starych_wpisow(self):
        with TemporaryDirectory() as tmp, patch.object(auto_offset, "SPOT_PATH", Path(tmp) / "spot.json"):
            now = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
            auto_offset.record_spot(3600.0, now - timedelta(days=4), now - timedelta(days=4))
            auto_offset.record_spot(3601.0, now, now)
            auto_offset.record_spot(3601.0, now, now)
            self.assertEqual([r["cena"] for r in auto_offset.load_spots()], [3601.0])


class Us100Tests(unittest.TestCase):
    def test_roznica_ndx_i_nq_na_wspolnych_swiecach(self):
        nq = _m5([26400.0, 26410.0, 26420.0, 26430.0, 26440.0])
        ndx = _m5([26355.0, 26364.0, 26376.0], start="2026-11-25 14:10")  # różnice -65, -66, -64
        estimate = auto_offset.us100_estimate(ndx, nq)
        self.assertEqual((estimate.samples, estimate.value, estimate.price), (3, -65.0, 26440.0))


def _estimate(value, minutes=0, price=26400.0):
    return auto_offset.Estimate(value, 12, 3.0, pd.Timestamp("2026-11-25 15:00", tz="UTC") + pd.Timedelta(minutes=minutes),
                                "test", price)


class ApplyTests(unittest.TestCase):
    def test_prog_zmiany_korekta_i_powtorzony_pomiar(self):
        cfg = {"przesunięcie": -64.0, auto_offset.SETTING: True, auto_offset.RECORD: {}, auto_offset.CORRECTION: 0.0}
        self.assertIsNone(auto_offset.apply(cfg, "US100", _estimate(-65.0)))  # 1 pkt < koszt 2 pkt
        self.assertEqual(cfg["przesunięcie"], -64.0)
        self.assertEqual(cfg[auto_offset.RECORD]["pomiar"], -65.0)
        self.assertIsNone(auto_offset.apply(cfg, "US100", _estimate(-80.0)))  # ten sam czas pomiaru
        self.assertIsNotNone(auto_offset.apply(cfg, "US100", _estimate(-70.0, minutes=5)))
        self.assertEqual(cfg["przesunięcie"], -70.0)
        cfg[auto_offset.CORRECTION] = 6.5
        auto_offset.apply(cfg, "US100", _estimate(-67.0, minutes=10))
        self.assertEqual(cfg["przesunięcie"], -60.5)

    def test_odrzuca_nierealny_pomiar_i_szanuje_wylaczenie(self):
        cfg = {"przesunięcie": -8.0, auto_offset.SETTING: True, auto_offset.RECORD: {}}
        self.assertIsNone(auto_offset.apply(cfg, "GOLD", _estimate(-1000.0, price=3640.0)))
        self.assertEqual(cfg["przesunięcie"], -8.0)
        self.assertIn("odrzucono", cfg[auto_offset.RECORD])
        off = {"przesunięcie": -8.0, auto_offset.SETTING: False}
        self.assertIsNone(auto_offset.apply(off, "GOLD", _estimate(-2.0, price=3640.0)))
        self.assertNotIn(auto_offset.RECORD, off)


if __name__ == "__main__":
    unittest.main()
