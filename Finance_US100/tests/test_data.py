import unittest
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import monitor
from us100 import advisor, archive, chart_annotations, engine, harmonics, indicators, price_structures, signals, storage
from us100.data import TZ_LOCAL, bar_end, bars_between


class BarsBetweenTests(unittest.TestCase):
    def test_inclusive_bounds_with_utc_and_warsaw_across_clock_change(self):
        hours = pd.date_range("2026-10-25 00:00", periods=5, freq="h", tz="UTC").tz_convert(TZ_LOCAL)
        frame = pd.DataFrame({"close": range(5)}, index=hours)
        self.assertNotEqual(hours[0].utcoffset(), hours[2].utcoffset())

        for end in (hours[2], hours[2].tz_convert("UTC")):
            with self.subTest(end=end):
                self.assertTrue(bars_between(frame, hours[0], end).index.equals(hours[:3]))


class RefreshTests(unittest.TestCase):
    def test_retries_only_missing_relevant_closes_and_skips_old_market_data(self):
        hour = pd.Timestamp("2026-11-24 12:00", tz=TZ_LOCAL)
        self.assertEqual(monitor.next_close(hour - pd.Timedelta(seconds=1)), hour)
        self.assertEqual(monitor.next_close(hour), hour + pd.Timedelta(minutes=5))
        snapshot = {
            "H1": {"zamknięta do": (hour - pd.Timedelta(hours=1)).isoformat()},
            "M5": {"zamknięta do": hour.isoformat()},
        }
        self.assertEqual(monitor.missing_closes(snapshot, hour, {"H1", "M5"}), {"H1"})
        self.assertEqual(monitor.missing_closes(snapshot, hour + pd.Timedelta(minutes=5), {"H1", "M5"}), {"M5"})
        self.assertFalse(monitor.missing_closes(snapshot, hour + pd.Timedelta(days=1), {"H1", "M5"}))

    def test_m5_archive_excludes_open_bar_and_snapshot_survives_read_for_chart(self):
        index = pd.date_range("2026-11-24 12:00", periods=4, freq="5min", tz=TZ_LOCAL)
        bars = pd.DataFrame({name: [10., 11., 12., 13.] for name in archive.COLUMNS}, index=index)
        now = index[-1] + pd.Timedelta(seconds=10)
        bars.attrs["pobrano"] = now
        with TemporaryDirectory() as temp, patch.object(archive, "ARCHIVE_DIR", Path(temp)):
            archive.save("GC=F", "M5", bars, now)
            self.assertTrue(engine.load_short_chart("GC=F", "M5").index.equals(index[:3]))
            live = engine.load_short_chart("GC=F", "M5", {"M5": engine._latest_bar(bars, "M5", now)})
            self.assertTrue(live.index.equals(index))
            self.assertEqual(len(engine.load_short_chart("GC=F", "M15")), 1)

    def test_new_download_does_not_turn_previous_partial_bar_into_a_close(self):
        index = pd.date_range("2026-11-24 08:00", periods=4, freq="h", tz=TZ_LOCAL)
        columns = {key: [10., 11., 12., 13.] for key in archive.COLUMNS}
        old = pd.DataFrame(columns, index=index)
        old.attrs["pobrano"] = index[-1] + pd.Timedelta(minutes=40)
        fresh = old.iloc[:-1].copy()
        fresh.attrs["pobrano"] = index[-1] + pd.Timedelta(hours=1, minutes=10)
        self.assertEqual(signals.last_closed(old, "H1", fresh.attrs["pobrano"]).name, index[-2])
        daily = pd.DataFrame({key: [10.0] for key in archive.COLUMNS}, index=pd.DatetimeIndex(["2026-11-23"]))
        daily.attrs["pobrano"] = old.attrs["pobrano"]
        cache = engine.FrameCache()
        cache._raw["GC=F"] = (daily, old, time.time())
        cache._daily_hour["GC=F"] = pd.Timestamp.now(tz="UTC").floor("h")
        with patch.object(engine, "_download", return_value=fresh):
            frames = cache.frames("GC=F")
        self.assertEqual(frames["H1"].index[-1], index[-2])
        self.assertTrue(frames["H4"].empty)
        self.assertEqual(cache.hourly_snapshot["GC=F"]["zamknięta do"], index[-1].isoformat())

    def test_partial_legacy_settings_are_filled_without_discarding_saved_cost(self):
        with TemporaryDirectory() as temp, patch.object(storage, "DATA_DIR", Path(temp)):
            storage.save_settings({"plan_ustawienia": {"koszt_pkt": 2.5, "limit_h": -1}}, "GOLD")
            self.assertEqual(storage.load_settings("GOLD")["plan_ustawienia"],
                             {"koszt_pkt": 2.5, "min_rr": 1.5, "wolumen": 0.01, "limit_h": 24.0})


class AdvisorReplayTests(unittest.TestCase):
    def test_time_limit_in_a_gap_uses_first_available_open_and_actual_time(self):
        times = pd.DatetimeIndex(["2026-11-20 20:00", "2026-11-20 21:00", "2026-11-23 00:00"], tz=TZ_LOCAL)
        bars = pd.DataFrame({"open": [100., 100., 97.], "close": [100., 101., 99.],
                             "high": [101., 102., 101.], "low": [99., 99., 96.]}, index=times)
        proposal = advisor.Proposal(advisor.Level("Sesja D1", "Minimum", "long", 100),
                                    times[1], 100, 95, 113, 2.6, ())
        market = SimpleNamespace(bars=bars, ends=times + pd.Timedelta(hours=1),
                                 assess_bar=lambda i, *_: advisor.Assessment("", proposal if i == 0 else None))
        [trade] = list(advisor._trades(market, 0.0, 1.5, 1.5))
        self.assertEqual((trade.outcome, trade.closed, trade.points), ("limit", times[-1], -3.0))


class ChartAnnotationsTests(unittest.TestCase):
    def test_entry_at_exact_close_marks_that_candle_and_respects_toggle(self):
        starts = pd.date_range("2026-11-24 09:00", periods=2, freq="h", tz=TZ_LOCAL).as_unit("us")
        view = pd.DataFrame({"open": [100.0, 103.0], "close": [102.0, 104.0],
                             "high": [106.0, 108.0], "low": [96.0, 100.0]}, index=starts)
        proposal = advisor.Proposal(advisor.Level("Sesja D1", "Minimum", "long", 100.0),
                                    starts[1], 102.0, 98.0, 108.0, 1.5, ())
        args = ([], {}, starts[1], [], proposal)
        overlays, _, arrow = chart_annotations.build(view, "H1", 3.0, {"entry"}, *args)
        self.assertEqual(arrow["value"], [0, 105.0])
        self.assertFalse(overlays["markLine"]["data"])

        selected, _, arrow = chart_annotations.build(view, "H1", 3.0, {"entry"}, *args, show_exit_levels=True)
        self.assertIsNotNone(arrow)
        self.assertEqual(
            [(line["name"], line["yAxis"]) for line in selected["markLine"]["data"]],
            [("SL", 101.0), ("TP", 111.0)],
        )

        disabled, extent, arrow = chart_annotations.build(view, "H1", 3.0, set(), *args, show_exit_levels=True)
        self.assertIsNone(arrow)
        self.assertFalse(extent)
        self.assertTrue(all(not layer["data"] for layer in disabled.values()))

    def test_harmonic_projection_uses_closed_candles_on_each_chart_interval(self):
        prices = [110, 106, 100, 104, 114, 120, 116, 111, 108, 110, 112, 116, 120, 126, 126, 127]
        for interval, freq in (("M5", "5min"), ("M15", "15min"), ("H1", "h"), ("H4", "4h"), ("D1", "D")):
            with self.subTest(interval=interval):
                times = pd.date_range("2026-11-10", periods=len(prices), freq=freq,
                                      tz=None if interval == "D1" else TZ_LOCAL)
                bars = pd.DataFrame({"open": prices, "close": prices, "high": [p + 1 for p in prices],
                                     "low": [p - 1 for p in prices]}, index=times)
                bars.loc[times[-1], "high"] = 131
                before = bar_end(interval, times[9])
                after = bar_end(interval, times[10])
                self.assertFalse(harmonics.projections(bars, before, interval=interval))
                [projection] = harmonics.projections(bars, after, interval=interval)
                self.assertEqual(projection.c.time, times[8])
                self.assertEqual(projection.c.confirmed_at, after)
                self.assertEqual(harmonics.track(projection, bars, after, interval=interval).stage, "building")
                final = bar_end(interval, times[-1])
                state = harmonics.track(projection, bars, final, interval=interval)
                self.assertEqual((state.stage, state.at), ("rejected", final))
                overlays, _, _ = chart_annotations.build(
                    bars, interval, 0, {"harmonics"}, [], {}, final, [(projection, state)], None,
                )
                self.assertTrue(any(point.get("label", {}).get("formatter") == "D" for point in overlays["markPoint"]["data"]))


class AdvisorVolumeTests(unittest.TestCase):
    def test_candidate_volume_only_uses_preceding_positive_observations(self):
        volumes = pd.Series([100.0] * 12 + [0.0] * 8 + [200.0, 0.0, 1000.0])
        ratios = indicators.relative_volume(volumes)
        self.assertEqual(ratios.iloc[20], 2.0)
        self.assertTrue(pd.isna(ratios.iloc[21]))
        self.assertTrue(pd.isna(ratios.iloc[22]))

    def test_strong_volume_opposing_long_holds_the_entry(self):
        market = advisor.Market.__new__(advisor.Market)
        market.trigger = "H1"
        market.bars = pd.DataFrame([{"open": 101, "close": 102, "low": 99, "high": 112, "atr": 2}])
        market.ends = [pd.Timestamp("2026-11-24 10:00", tz=TZ_LOCAL)]
        market.relative_volume = [2.0]
        market.close_position = indicators.close_position(market.bars)
        with (
            patch.object(market, "levels", return_value=[advisor.Level("Sesja D1", "Minimum", "long", 100)]),
            patch.object(market, "evidence", return_value=(("Trend", 1), ("Momentum", 1), ("Reakcja ceny", 0))),
        ):
            blocked = market.assess_bar(0, 0, 1.5)
            self.assertIn("Podwyższony wolumen", blocked.reason)
            self.assertEqual(blocked.proposal.evidence[-1][1], -1)

            market.relative_volume[0] = float("nan")
            neutral = market.assess_bar(0, 0, 1.5)
            self.assertFalse(neutral.reason)
            self.assertEqual(neutral.proposal.evidence[-1][1], 0)
            self.assertIn("Koszt 100 pkt ceny", market.assess_bar(0, 100, 1.5).reason)
            self.assertIn("Nawet przy zerowym koszcie", market.assess_bar(0, 0, 3).reason)

            market.relative_volume[0] = 2.0
            market.bars.at[0, "close"] = 112.0
            market.close_position = indicators.close_position(market.bars)
            confirmed = market.assess_bar(0, 0, 1.5)
            self.assertFalse(confirmed.reason)
            self.assertEqual(confirmed.proposal.evidence[-1][1], 1)


class PriceStructuresTests(unittest.TestCase):
    def test_zzb_requires_break_close_and_later_retest_in_both_directions(self):
        times = pd.date_range("2026-11-20 09:00", periods=22, freq="5min", tz=TZ_LOCAL)
        price = [80 + i * .6 for i in range(16)] + [98, 95, 96, 102, 101, 96]
        bars = pd.DataFrame({"open": price, "close": price, "high": [p + 1 for p in price],
                             "low": [p - 1 for p in price]}, index=times)
        bars.loc[times[16], "high"] = 100
        bars.loc[times[19:21], "low"] = 99.95
        mirrored = bars.assign(open=200 - bars["open"], close=200 - bars["close"],
                               high=200 - bars["low"], low=200 - bars["high"])
        for source, role in ((bars, "Wsparcie"), (mirrored, "Opór")):
            with self.subTest(role=role):
                def zone_at(i):
                    as_of = bar_end("M5", times[i])
                    return next((z for z in price_structures.polarity_zones(source, "M5", as_of)
                                 if z.pivot.price == 100 and z.role == role), None)

                self.assertIsNone(zone_at(18))
                self.assertEqual(zone_at(19).broken_at, bar_end("M5", times[19]))
                self.assertIsNone(zone_at(19).retested_at)
                self.assertEqual(zone_at(20).retested_at, bar_end("M5", times[20]))
                self.assertIsNone(zone_at(21))

    def test_pof_measures_from_wave_end_toward_start(self):
        time = pd.Timestamp("2026-11-20 12:00", tz=TZ_LOCAL)
        low = harmonics.Pivot(time, time, 100, False)
        high = harmonics.Pivot(time + pd.Timedelta(hours=1), time + pd.Timedelta(hours=3), 120, True)
        up = dict(price_structures.WaveMeasurement(low, high).levels())
        down = dict(price_structures.WaveMeasurement(high, low).levels())
        self.assertEqual(up[0.5], 110)
        self.assertAlmostEqual(up[0.618], 107.64)
        self.assertAlmostEqual(down[0.618], 112.36)
