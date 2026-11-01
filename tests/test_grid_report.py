"""GridRunReport / GridPointResult 的小回归，防止 Optional 排序键回退。"""
from __future__ import annotations

import unittest

from greenwave.grid.runner import GridPointResult, GridRunReport


def _result(score: float | None) -> GridPointResult:
    return GridPointResult(90.0, 1.0, "optimal", score, None, None)


class TestGridReport(unittest.TestCase):
    def test_best_sorts_numeric_scores_and_skips_none(self):
        report = GridRunReport((_result(10.0), _result(None), _result(30.0)))
        self.assertEqual([r.composite_best for r in report.best(2)], [30.0, 10.0])
        self.assertEqual(report.best(0), [])
        self.assertEqual(len(report.best(10)), 2)


if __name__ == "__main__":
    unittest.main()
