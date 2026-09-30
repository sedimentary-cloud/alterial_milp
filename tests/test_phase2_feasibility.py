"""Phase 2 tests: boundary feasibility filtering and witness keys."""
from __future__ import annotations

import unittest

from greenwave.errors import GWInfeasibleGridPoint
from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.schema.models import ProblemInput

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _legacy_plan import convert_problem  # noqa: E402


def _base_old(plans):
    return {
        "corridor": {"corridor_id": "t", "intersections": ["I1", "I2", "I3"],
                     "segments": [
                         {"from": "I1", "to": "I2", "distance_m": 500.0,
                          "speed_up_mps": 11.1111111111, "speed_down_mps": 11.1111111111},
                         {"from": "I2", "to": "I3", "distance_m": 500.0,
                          "speed_up_mps": 11.1111111111, "speed_down_mps": 11.1111111111}]},
        "plans": plans,
        "objective": {"band_demands": [
            {"id": "GW_up", "type": "green", "direction": "up",
             "nodes": ["I1", "I2", "I3"], "weight": 1.0,
             "margin": {"delta_min_s": 0.0, "coef": 1.0}}],
            "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
            "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [90.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 10, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
    }


def _old_plan(iid, up, down, double=False, adjustable=None):
    return {"id": f"{iid}_P1", "intersection": iid, "double_cycle": double,
            "windows": {"up": up, "down": down},
            "adjustable": adjustable or {"up": {}, "down": {}},
            "hard_constraints": [], "soft_constraints": []}


class TestBoundaryFeasibility(unittest.TestCase):
    def test_legacy_simple_plan_feasible(self):
        old = _base_old([_old_plan(i, [[0.0, 0.45]], [[0.0, 0.45]]) for i in ("I1", "I2", "I3")])
        problem = ProblemInput.from_obj(convert_problem(old))
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        self.assertEqual(len(ctx.feasible_plans["I1"]), 1)
        self.assertEqual(ctx.witness_for("I1_P1"), {})

    def test_double_cycle_witness_key_is_boundary(self):
        old = _base_old([
            _old_plan("I1", [[0.0, 0.2]], [[0.0, 0.2]], double=True,
                      adjustable={"up": {"0": {"start": [-0.02, 0.02],
                                               "end": [-0.02, 0.02]}},
                                  "down": {}}),
            _old_plan("I2", [[0.0, 0.45]], [[0.0, 0.45]]),
            _old_plan("I3", [[0.0, 0.45]], [[0.0, 0.45]]),
        ])
        problem = ProblemInput.from_obj(convert_problem(old))
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        witness = ctx.witness_for("I1_P1")
        # old window [0.0, 0.2] -> one declared boundary 0.2; both endpoints
        # map to the same boundary, so the witness is keyed by boundary index.
        self.assertIn(("up", 0), witness)
        self.assertEqual(len(witness), 1)

    def test_min_phase_violation_drops_plan(self):
        d = {
            "id": "I1_P1", "intersection": "I1", "double_cycle": False,
            "min_phase_s": 5.0,
            "directions": {
                "up": {"boundaries": [{"pos_r": 0.001}, {"pos_r": 0.002}],
                       "green_mask": [True, True, False]},
                "down": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]},
            },
        }
        problem = ProblemInput.from_obj({
            "corridor": {"corridor_id": "t", "intersections": ["I1"],
                         "segments": []},
            "plans": [d],
            "objective": {"band_demands": []},
            "grid": {"cycles_s": [90.0], "speed_ratios": [1.0]},
            "solver": {"backend": "scipy", "time_limit_s": 10, "mip_rel_gap": 0.0,
                       "num_workers": 1,
                       "heuristic": {"enabled": False, "refine_rounds": 0}},
        })
        with self.assertRaises(GWInfeasibleGridPoint):
            FeasibilityFilter(problem).filter(90.0, 1.0)
        status = FeasibilityFilter(problem).check_plan(problem.plan("I1_P1"), 90.0)
        self.assertFalse(status.feasible)
        self.assertTrue(any("min_phase" in v for v in status.violations))

    def test_adjustable_boundary_lp_hard_constraint(self):
        def problem_with_rhs(rhs_ratio):
            return ProblemInput.from_obj({
                "corridor": {"corridor_id": "t", "intersections": ["I1"],
                             "segments": []},
                "plans": [{
                    "id": "I1_P1", "intersection": "I1",
                    "directions": {
                        "up": {"boundaries": [{"pos_r": 0.5, "adjust": [-0.05, 0.05]}],
                               "green_mask": [True, False]},
                        "down": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]},
                    },
                    "hard_constraints": [{
                        "id": "b_le", "sense": "<=", "rhs_ratio": rhs_ratio, "hard": True,
                        "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                            "plan": "I1_P1", "direction": "up", "index": 0},
                                   "coef": 1.0}],
                    }],
                }],
                "objective": {"band_demands": []},
                "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
                "solver": {"backend": "scipy", "time_limit_s": 10, "mip_rel_gap": 0.0,
                           "num_workers": 1,
                           "heuristic": {"enabled": False, "refine_rounds": 0}},
            })
        feasible = problem_with_rhs(0.47)
        status = FeasibilityFilter(feasible).check_plan(feasible.plan("I1_P1"), 100.0)
        self.assertTrue(status.feasible)
        # nominal 0.5C = 50s; rhs 0.44C = 44s, but min adjusted boundary is 45s
        infeasible = problem_with_rhs(0.44)
        status2 = FeasibilityFilter(infeasible).check_plan(infeasible.plan("I1_P1"), 100.0)
        self.assertFalse(status2.feasible)


if __name__ == "__main__":
    unittest.main()
