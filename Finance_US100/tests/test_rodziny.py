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


class RegimeTests(unittest.TestCase):
    def _h4(self, closes):
        from us100 import indicators
        index = pd.date_range("2026-01-05 00:00", periods=len(closes), freq="4h", tz=TZ_LOCAL)
        frame = pd.DataFrame({"open": closes, "close": closes, "high": [c + 1 for c in closes],
                              "low": [c - 1 for c in closes]}, index=index)
        return indicators.add_indicators(frame)

    def test_adx_odroznia_trend_od_konsolidacji(self):
        from us100 import indicators
        trend = self._h4([100 + 2 * i for i in range(300)])
        flat = self._h4([100 + (i % 2) for i in range(300)])
        self.assertGreater(indicators.adx(trend).iloc[-1], 40)
        self.assertLess(indicators.adx(flat).iloc[-1], 20)

    def test_reżim_trendu_i_tylko_zamkniete_h4(self):
        from us100 import regime
        h4 = self._h4([100 + 2 * i for i in range(300)])
        last_end = h4.index[-1] + pd.Timedelta(hours=4)
        self.assertEqual(regime.current(h4, last_end).label, regime.UP)
        # w trakcie ostatniej H4 znany jest stan z przedostatniej, a nie z tej trwającej
        during = regime.at(h4, pd.DatetimeIndex([h4.index[-1] + pd.Timedelta(hours=2)]))
        before = regime.classify(h4).iloc[-2]
        self.assertEqual(during["adx"].iloc[0], before["adx"])
        self.assertEqual(regime.at(h4, pd.DatetimeIndex([h4.index[0]]))["reżim"].iloc[0], "brak danych")


class PullbackTests(unittest.TestCase):
    def _bars(self, closes, rsi):
        index = pd.date_range("2026-11-03 00:00", periods=len(closes), freq="h", tz=TZ_LOCAL)
        frame = pd.DataFrame({"close": closes, "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
                              "rsi": rsi, "atr": 3.0}, index=index)
        return frame, index + pd.Timedelta(hours=1)

    def test_long_po_korekcie_tylko_w_trendzie_zgodnym_z_d1(self):
        from us100 import regime
        closes = [100.0 + 0.5 * i for i in range(30)] + [112.0, 111.0, 110.5, 110.0, 113.5]
        rsi = [60.0] * 31 + [44.0, 42.0, 43.0, 55.0]
        h1, ends = self._bars(closes, rsi)
        n = len(h1)
        up, flat = [regime.UP] * n, [regime.RANGE] * n
        calm, wild = [False] * n, [True] * n
        [signal] = families.pullback_rule(h1, ends, up, calm, [1] * n)
        self.assertEqual((signal.direction, signal.at, signal.reference), ("long", ends[-1], 113.5))
        # SL za dołkiem korekty (109.5) + 0,1 ATR (ATR 3), TP 2× ryzyko
        self.assertAlmostEqual(signal.risk, 113.5 - 109.5 + 0.3)
        self.assertAlmostEqual(signal.reward, 2 * signal.risk)
        self.assertFalse(families.pullback_rule(h1, ends, up, calm, [-1] * n))  # D1 przeciwny
        self.assertFalse(families.pullback_rule(h1, ends, flat, calm, [1] * n))  # konsolidacja
        self.assertFalse(families.pullback_rule(h1, ends, up, wild, [1] * n))  # wysoka zmienność
        shallow = [60.0] * n  # bez korekty RSI
        h1_shallow, _ = self._bars(closes, shallow)
        self.assertFalse(families.pullback_rule(h1_shallow, ends, up, calm, [1] * n))


class OpeningRangeTests(unittest.TestCase):
    def _day(self, closes_by_hour, atr=2.0, day="2026-11-03"):
        hours = sorted(closes_by_hour)
        index = pd.DatetimeIndex([pd.Timestamp(f"{day} {h:02d}:00", tz="America/New_York") for h in hours]).tz_convert(TZ_LOCAL)
        closes = [closes_by_hour[h] for h in hours]
        frame = pd.DataFrame({"close": closes, "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
                              "atr": atr}, index=index)
        if 9 in hours:  # zakres otwarcia 98-102
            frame.loc[index[hours.index(9)], ["high", "low"]] = [102.0, 98.0]
        return frame, index + pd.Timedelta(hours=1)

    def test_pierwsze_wybicie_zakresu_z_wyjsciem_na_zamknieciu_sesji(self):
        h1, ends = self._day({8: 100, 9: 100, 10: 101, 11: 103, 12: 97, 13: 104})
        [signal] = families.opening_range_rule(h1, ends)
        self.assertEqual(signal.direction, "long")
        self.assertEqual(signal.at.tz_convert("America/New_York").hour, 12)
        self.assertEqual((signal.risk, signal.reward, signal.limit_h), (3.0, 4.5, 4.0))  # SL w połowie zakresu (100)

    def test_ryzyko_poza_zakresem_atr_zuzywa_dzien_bez_sygnalu(self):
        h1, ends = self._day({9: 100, 10: 103, 11: 97}, atr=1.0)
        self.assertFalse(families.opening_range_rule(h1, ends))

    def test_bez_swiecy_otwarcia_brak_sygnalu_i_tylko_dla_us100(self):
        h1, ends = self._day({10: 101, 11: 103})
        self.assertFalse(families.opening_range_rule(h1, ends))
        self.assertIn("otwarcie", [f.key for f in families.for_instrument("US100")])
        self.assertNotIn("otwarcie", [f.key for f in families.for_instrument("GOLD")])


class MeanReversionTests(unittest.TestCase):
    def _bar(self, **values):
        base = {"close": 100.0, "high": 101.0, "low": 99.0, "rsi": 50.0, "atr": 2.0,
                "bb_low": 95.0, "bb_mid": 100.0, "bb_up": 105.0}
        base.update(values)
        index = pd.DatetimeIndex([pd.Timestamp("2026-11-03 10:00", tz=TZ_LOCAL)])
        return pd.DataFrame([base], index=index), index + pd.Timedelta(hours=1)

    def test_powrot_do_wstegi_w_konsolidacji(self):
        from us100 import regime
        h1, ends = self._bar(close=95.5, low=94.0, high=96.0, rsi=35.0)
        [signal] = families.mean_reversion_rule(h1, ends, [regime.RANGE], [False])
        self.assertEqual(signal.direction, "long")
        self.assertAlmostEqual(signal.risk, 2.0)  # 1,7 za knotem < 1 ATR
        self.assertAlmostEqual(signal.reward, 4.5)  # do średniej 100
        self.assertFalse(families.mean_reversion_rule(h1, ends, [regime.UP], [False]))
        self.assertFalse(families.mean_reversion_rule(h1, ends, [regime.RANGE], [True]))

        short, ends = self._bar(close=104.5, high=106.0, low=104.0, rsi=65.0)
        self.assertEqual(families.mean_reversion_rule(short, ends, [regime.RANGE], [False])[0].direction, "short")

    def test_pomija_slabe_rsi_i_cel_blizszy_niz_ryzyko(self):
        from us100 import regime
        h1, ends = self._bar(close=95.5, low=94.0, high=96.0, rsi=45.0)
        self.assertFalse(families.mean_reversion_rule(h1, ends, [regime.RANGE], [False]))
        near, ends = self._bar(close=95.5, low=94.0, high=96.0, rsi=35.0, bb_mid=97.0)
        self.assertFalse(families.mean_reversion_rule(near, ends, [regime.RANGE], [False]))


class LiveSimulationTests(unittest.TestCase):
    def test_tylko_sygnaly_po_starcie_i_otwarta_pozycja(self):
        index = pd.date_range("2026-11-03 00:00", periods=8, freq="h", tz=TZ_LOCAL)
        h1 = pd.DataFrame({"open": [100, 101, 101, 102, 103, 104, 104, 104.0],
                           "close": [101, 101, 102, 103, 104, 104, 104, 104.0],
                           "high": [101, 101.5, 102.5, 103.5, 104.5, 104.4, 104.4, 104.4],
                           "low": [99.5, 100.5, 100.8, 101.5, 102.5, 103.8, 103.8, 103.8]}, index=index)
        ends = index + pd.Timedelta(hours=1)
        before = families.Candidate("x", "long", ends[0], 101.0, 1.0, 1.5, 24, "")
        after = families.Candidate("x", "long", ends[4], 104.0, 1.0, 1.5, 24, "")
        trades, position = families.simulate_open(h1, ends, [c for c in (before, after) if c.at >= ends[3]], 0.0)
        self.assertEqual(trades, [])
        self.assertEqual((position.candidate, position.opened, position.entry), (after, index[5], 104.0))
        self.assertEqual((position.stop, position.target), (103.0, 105.5))

    def test_nowa_wersja_regul_zaczyna_symulacje_od_nowa(self):
        start = pd.Timestamp("2026-11-24 15:00", tz="UTC")
        state = {"start": start.isoformat(), "wersje": {}}
        starts, changed = families.live_starts(state, "GOLD", start)
        self.assertTrue(changed)
        self.assertEqual(set(starts), {f.key for f in families.for_instrument("GOLD")})
        later = start + pd.Timedelta(days=3)
        starts, changed = families.live_starts(state, "GOLD", later)
        self.assertFalse(changed)
        self.assertEqual(starts["wybicie"], start)
        state["wersje"]["wybicie"]["opis"] = "stare reguły"
        starts, changed = families.live_starts(state, "GOLD", later)
        self.assertTrue(changed)
        self.assertEqual(starts["wybicie"], later)
        self.assertEqual(families.live_starts({}, "GOLD", later), ({}, False))

    def test_gotowosc_wymaga_wszystkich_warunkow(self):
        from us100 import advisor
        empty = advisor.History(0, 0, 0, 0, None)
        start = pd.Timestamp("2026-11-01", tz="UTC")
        base = families.Evaluation("x", "X", "", (), empty, empty, empty, start, "przewaga", ("success", ""), None,
                                   start, advisor.History(25, 12, 10, 3, 1.5, 4.0))
        self.assertTrue(all(ok for _, ok in families.readiness(base, start + pd.Timedelta(days=30))))
        self.assertFalse(all(ok for _, ok in families.readiness(base, start + pd.Timedelta(days=10))))
        losing = families.Evaluation("x", "X", "", (), empty, empty, empty, start, "przewaga", ("success", ""), None,
                                     start, advisor.History(25, 5, 18, 2, -0.5, 4.0))
        self.assertFalse(families.readiness(losing, start + pd.Timedelta(days=30))[3][1])
