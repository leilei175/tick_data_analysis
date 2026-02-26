import unittest
from datetime import datetime

from mylib.date_utils import parse_date, date_to_str, get_quarter_from_date
from update_data import parse_date as update_parse_date
from update_data import date_to_str as update_date_to_str


class TestSharedUtils(unittest.TestCase):
    def test_parse_date_supports_multiple_formats(self):
        self.assertEqual(parse_date('20260102'), datetime(2026, 1, 2))
        self.assertEqual(parse_date('2026-01-02'), datetime(2026, 1, 2))
        self.assertEqual(parse_date('2026/01/02'), datetime(2026, 1, 2))

    def test_date_to_str_consistency_with_update_data_wrappers(self):
        dt = datetime(2026, 2, 14)
        self.assertEqual(date_to_str(dt), '20260214')
        self.assertEqual(update_date_to_str(dt), '20260214')
        self.assertEqual(update_parse_date('2026-02-14'), dt)

    def test_get_quarter_from_date(self):
        self.assertEqual(get_quarter_from_date('2026-01-01'), '20260331')
        self.assertEqual(get_quarter_from_date('2026-06-20'), '20260630')
        self.assertEqual(get_quarter_from_date('2026-09-30'), '20260930')
        self.assertEqual(get_quarter_from_date('2026-12-10'), '20261231')


if __name__ == "__main__":
    unittest.main()
