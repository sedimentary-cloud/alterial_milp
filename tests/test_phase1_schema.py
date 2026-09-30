"""Phase 1 tests: v2 boundary schema and derived green/red intervals."""
from __future__ import annotations

import unittest

from greenwave.schema.models import (BoundarySpec, DirectionPlan, PlanSpec,
                                     ProblemInput, AtomRef)


def _problem(plans):
    return ProblemInput.from_obj({
        "corridor": {"corridor_id": "t", "intersections": ["I1", "I2"],
                     "segments": [{"from": "I1", "to": "I2", "distance_m": 200.0,
                                   "speed_up_mps": 10.0, "speed_down_mps": 10.0}]},
        "plans": plans,
        "objective": {"band_demands": [
            {"id": "GW_up", "type": "green", "direction": "up",
             "nodes": ["I1", "I2"], "weight": 1.0,
             "margin": {"delta_min_s": 0.0, "coef": 1.0}}]},
        "grid": {"cycles_s": [90.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 10, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
    })


def _plan(iid, up_bounds, up_mask, down_bounds=None, down_mask=None, double=False):
    if down_bounds is None:
        down_bounds = [{"pos_r": 0.4}] if double else [{"pos_r": 0.5}]
    if down_mask is None:
        down_mask = [False, True] if double else [True, False]
    return {"id": f"{iid}_P1", "intersection": iid, "double_cycle": double,
            "directions": {
                "up": {"boundaries": up_bounds, "green_mask": up_mask},
                "down": {"boundaries": down_bounds, "green_mask": down_mask},
            }}


class TestBoundarySchema(unittest.TestCase):
    def test_hidden_boundary_inside_green_window(self):
        p = PlanSpec.from_obj(_plan(
            "I1",
            up_bounds=[{"pos_r": 0.2}, {"pos_r": 0.5}],
            up_mask=[True, True, False],
        ))
        windows = p.derived_windows("up")
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0].start_r, 0.0)
        self.assertAlmostEqual(windows[0].end_r, 0.5)
        # hidden boundary 0 is still addressable/adjustable
        self.assertEqual(p.boundary_count("up"), 2)
        self.assertAlmostEqual(p.nominal_boundary_r("up", 0), 0.2)

    def test_red_interval_wraps_and_after_green(self):
        p = PlanSpec.from_obj(_plan(
            "I1",
            up_bounds=[{"pos_r": 0.2}, {"pos_r": 0.6}],
            up_mask=[False, True, False],
        ))
        reds = p.derived_red_intervals("up")
        self.assertEqual(len(reds), 1)
        r = reds[0]
        self.assertEqual(r.rid, 1)
        self.assertEqual(r.kind, "wrap")
        self.assertEqual(r.after_green, 1)
        self.assertAlmostEqual(r.start_r, 0.6)
        self.assertAlmostEqual(r.end_r, 1.2)
        self.assertEqual(r.wrap, 1)

    def test_gap_then_wrap_ids(self):
        p = PlanSpec.from_obj(_plan(
            "I1",
            up_bounds=[{"pos_r": 0.2}, {"pos_r": 0.75}],
            up_mask=[False, True, False],
        ))
        # green [0.2, 0.75); exactly one green window -> one wrap red
        self.assertEqual(len(p.derived_windows("up")), 1)
        reds = p.derived_red_intervals("up")
        self.assertEqual([r.rid for r in reds], [1])
        self.assertAlmostEqual(reds[0].start_r, 0.75)
        self.assertAlmostEqual(reds[0].end_r, 1.2)

    def test_boundary_endpoint_bounds_include_adjust(self):
        p = PlanSpec.from_obj(_plan(
            "I1",
            up_bounds=[{"pos_r": 0.5, "adjust": [-0.05, 0.05]}],
            up_mask=[True, False],
        ))
        lo, hi = p.window_endpoint_bound_r("up", 0, "start")
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0)
        lo, hi = p.window_endpoint_bound_r("up", 0, "end")
        self.assertAlmostEqual(lo, 0.45)
        self.assertAlmostEqual(hi, 0.55)

    def test_double_cycle_green_per_half_red_full_cycle(self):
        p = PlanSpec.from_obj(_plan(
            "I1",
            up_bounds=[{"pos_r": 0.2}, {"pos_r": 0.4}],
            up_mask=[False, True, False],
            double=True,
        ))
        greens = p.derived_windows("up")
        self.assertEqual(len(greens), 2)
        self.assertAlmostEqual(greens[0].start_r, 0.2)
        self.assertAlmostEqual(greens[0].end_r, 0.4)
        self.assertAlmostEqual(greens[1].start_r, 0.7)
        self.assertAlmostEqual(greens[1].end_r, 0.9)
        reds = p.derived_red_intervals("up")
        self.assertEqual([r.rid for r in reds], [1, 2])
        self.assertEqual(reds[0].kind, "gap")
        self.assertAlmostEqual(reds[0].start_r, 0.4)
        self.assertAlmostEqual(reds[0].end_r, 0.7)
        self.assertEqual(reds[1].kind, "wrap")
        self.assertAlmostEqual(reds[1].start_r, 0.9)
        self.assertAlmostEqual(reds[1].end_r, 1.2)

    def test_first_last_green_rejected_by_validation(self):
        p = _problem([
            _plan("I1", up_bounds=[{"pos_r": 0.5}], up_mask=[True, True]),
            _plan("I2", up_bounds=[{"pos_r": 0.5}], up_mask=[True, False]),
        ])
        issues = p.validate_all()
        self.assertTrue(any(i.code == "V2" and "first and last" in i.message for i in issues))

    def test_all_green_and_all_red_rejected(self):
        p_all_green = _problem([
            _plan("I1", up_bounds=[{"pos_r": 0.5}], up_mask=[True, True]),
            _plan("I2", up_bounds=[{"pos_r": 0.5}], up_mask=[True, False]),
        ])
        self.assertTrue(any(i.code == "V2" and "at least one red" in i.message
                            for i in p_all_green.validate_all()))
        p_all_red = _problem([
            _plan("I1", up_bounds=[{"pos_r": 0.5}], up_mask=[False, False]),
            _plan("I2", up_bounds=[{"pos_r": 0.5}], up_mask=[False, False]),
        ])
        self.assertTrue(any(i.code == "V2" and "at least one green" in i.message
                            for i in p_all_red.validate_all()))

    def test_min_phase_must_be_positive(self):
        d = _plan("I1", up_bounds=[{"pos_r": 0.5}], up_mask=[True, False])
        d["min_phase_s"] = 0.0
        p = _problem([d, _plan("I2", up_bounds=[{"pos_r": 0.5}], up_mask=[True, False])])
        self.assertTrue(any(i.code == "V3" and "min_phase_s" in i.message
                            for i in p.validate_all()))

    def test_boundary_atom_parse_and_validation(self):
        a = AtomRef.from_obj({"kind": "boundary", "intersection": "I1",
                              "plan": "I1_P1", "direction": "up", "index": 1})
        self.assertEqual(a.index, 1)
        p = _problem([
            _plan("I1", up_bounds=[{"pos_r": 0.5}], up_mask=[True, False]),
            _plan("I2", up_bounds=[{"pos_r": 0.5}], up_mask=[True, False]),
        ])
        self.assertEqual(p.validate_all(), [])

    def test_old_windows_input_rejected(self):
        with self.assertRaises(ValueError):
            PlanSpec.from_obj({"id": "I1_P1", "intersection": "I1",
                               "windows": {"up": [[0.0, 0.4]], "down": [[0.0, 0.4]]}})


if __name__ == "__main__":
    unittest.main()
