"""Rodziny sygnałów: wybicie poziomu sesji, zapowiedź na H1, rozliczenie i bramka."""

import unittest

import pandas as pd

from us100 import families
from us100.data import TZ_LOCAL


def _frames(closes: list[float]) -> dict[str, pd.DataFrame]:
    # Sesja 02.11 kończy się o 23:00 czasu polskiego (17:00 w Nowym Jorku): maksimum 110, minimum 90.
    # Sesja 03.11 (świece H1 od 00:00 do 22:00) ma maksimum 120, więc od 04.11 poziomem LONG jest 120.
    d1 = pd.DataFrame({"open": [100.0, 100.0], "high": [110.0, 120.0], "low": [90.0, 95.0], "close": [100.0, 100.0]},
                      index=pd.DatetimeIndex(["2026-11-02", "2026-11-03"]))
    index = pd.date_range("2026-11-03 00:00", periods=len(closes), freq="h", tz=TZ_LOCAL)
    opens = [closes[0], *closes[:-1]]
    h1 = pd.DataFrame({"open": opens, "close": closes,
                       "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
                       "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
                       "atr": 1.0}, index=index)
    return {"D1": d1, "H1": h1}


PARAMS = families.Params(0.0, 1.5, 24.0)


class SessionBreakTests(unittest.TestCase):
    def test_tylko_pierwsze_wybicie_w_sesji_i_nowy_poziom_po_sesji(self):
        closes = [100, 105, 111, 108, 112] + [109] * 19 + [118, 121]  # 24.: 04.11 00:00
        frames = _frames(closes)
        now = frames["H1"].index[-1] + pd.Timedelta(hours=1)
        found = families._session_breaks(frames, now, precursor=False)
        self.assertEqual([(c.direction, c.at.hour, c.reference) for c in found], [("long", 3, 111.0), ("long", 2, 121.0)])
        self.assertEqual((found[0].risk, found[0].reward), (1.0, 1.5))

    def test_brak_zagladania_w_przyszlosc(self):
        closes = [100, 105, 111, 108, 112] + [109] * 19 + [118, 121]
        frames = _frames(closes)
        full = families._session_breaks(frames, frames["H1"].index[-1] + pd.Timedelta(hours=1), precursor=False)
        cut = frames["H1"].index[10]
        early = families._session_breaks(frames, cut, precursor=False)
        self.assertEqual(early, [c for c in full if c.at <= cut])

    def test_zapowiedz_wymaga_sciskania_pod_poziomem(self):
        squeeze = [100, 106, 107, 107.5, 108.2, 109.0, 111.0]
        far = [100, 106, 105, 104, 105, 105, 111.0]
        for closes, expected in ((squeeze, 1), (far, 0)):
            frames = _frames(closes)
            now = frames["H1"].index[-1] + pd.Timedelta(hours=1)
            with self.subTest(closes=closes):
                self.assertEqual(len(families._session_breaks(frames, now, precursor=False)), 1)
                self.assertEqual(len(families._session_breaks(frames, now, precursor=True)), expected)

    def test_rozliczenie_wchodzi_na_otwarciu_i_trzyma_jedna_pozycje(self):
        index = pd.date_range("2026-11-03 00:00", periods=6, freq="h", tz=TZ_LOCAL)
        h1 = pd.DataFrame({"open": [100, 101, 101, 102, 103, 104.0], "close": [101, 101, 102, 103, 104, 105.0],
                           "high": [101, 101.5, 102.5, 103.5, 104.5, 105.5], "low": [99.5, 100.5, 100.8, 101.5, 102.5, 103.5]},
                          index=index)
        ends = index + pd.Timedelta(hours=1)
        first = families.Candidate("x", "long", ends[0], 101.0, 1.0, 1.5, 8, "")
        overlapping = families.Candidate("x", "long", ends[1], 101.0, 1.0, 1.5, 8, "")
        [trade] = families.simulate(h1, ends, [overlapping, first], 0.25)
        self.assertEqual((trade.opened, trade.outcome, trade.points), (index[1], "TP", 1.25))


class GateTests(unittest.TestCase):
    def test_prog_z_korekta_na_liczbe_rodzin(self):
        self.assertAlmostEqual(families.margin_z(1), 1.96, places=2)
        self.assertAlmostEqual(families.margin_z(3), 2.39, places=2)

    def test_mala_proba_nie_dostaje_przewagi(self):
        frames = _frames([100, 105, 111, 108] + [109] * 400)
        family = families.Family("wybicie", "A", "", lambda f, now, _p: families._session_breaks(f, now, precursor=False))
        ev = families.evaluate(family, frames, PARAMS, frames["H1"].index[-1] + pd.Timedelta(hours=1), holdout_days=1)
        self.assertEqual(ev.status, "za mało danych")
        self.assertEqual(ev.verdict[0], "info")


if __name__ == "__main__":
    unittest.main()
