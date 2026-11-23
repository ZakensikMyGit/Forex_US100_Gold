import unittest

import pandas as pd

from us100.data import TZ_LOCAL, bars_between


class BarsBetweenTests(unittest.TestCase):
    def test_inclusive_bounds_with_utc_and_warsaw_across_clock_change(self):
        hours = pd.date_range("2026-10-25 00:00", periods=5, freq="h", tz="UTC").tz_convert(TZ_LOCAL)
        frame = pd.DataFrame({"close": range(5)}, index=hours)
        self.assertNotEqual(hours[0].utcoffset(), hours[2].utcoffset())

        for end in (hours[2], hours[2].tz_convert("UTC")):
            with self.subTest(end=end):
                self.assertTrue(bars_between(frame, hours[0], end).index.equals(hours[:3]))
