"""Phase 3 tests: v2 MILP construction with boundary-derived window expressions."""
from __future__ import annotations

import os
import sys
import unittest

from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.model.builder import ModelBuilder
from greenwave.schema.models import PlanSpec, ProblemInput
from greenwave.solve import make_backend
from greenwave.solve.backend_base import SolveOptions

sys.path.insert(0, os.path.dirname(__file__))
from _legacy_plan import convert_problem  # noqa: E402


def _segment(a, b, dist=500.0, v=11.1111111111):
    return {"from": a, "to": b, "distance_m": dist, "speed_up_mps": v, "speed_down_mps": v}


def _old_plan(iid, up, down, double=False, adjustable=None, hard=(), soft=()):
    return {"id": f"{iid}_P1", "intersection": iid, "double_cycle": double,
            "windows": {"up": up, "down": down},
            "adjustable": adjustable or {"up": {}, "down": {}},
            "hard_constraints": list(hard), "soft_constraints": list(soft)}


def _base(plans, demands, intersections=("I1", "I2", "I3"), segments=None, cycle=90.0):
    return {
        "corridor": {"corridor_id": "t", "intersections": list(intersections),
                     "segments": segments or [_segment(intersections[0], intersections[1]),
                                              _segment(intersections[1], intersections[2])]},
        "plans": plans,
        "objective": {"band_demands": demands,
                      "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
                      "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [cycle], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
    }


def _solve(old_problem):
    problem = ProblemInput.from_obj(convert_problem(old_problem))
    ctx = FeasibilityFilter(problem).filter(old_problem["grid"]["cycles_s"][0], 1.0)
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend("scipy")
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    result = backend.solve(session, SolveOptions(20.0, 0.0))
    return problem, ctx, model, result


class TestBoundaryModel(unittest.TestCase):
    def test_t1_green_wave_objective(self):
        plans = [_old_plan(i, [[0.0, 0.45]], [[0.0, 0.45]]) for i in ("I1", "I2", "I3")]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "weight": 1.0,
                    "margin": {"delta_min_s": 0.0, "coef": 1.0}}]
        _, _, model, result = _solve(_base(plans, demands))
        self.assertEqual(result.status.value, "optimal")
        self.assertAlmostEqual(model.expr_composite().evaluate(result.values), 40.5, places=4)

    def test_t4_two_windows_objective(self):
        plans = [_old_plan(i, [[0.0, 0.2], [0.5, 0.7]], [[0.0, 0.2], [0.5, 0.7]])
                 for i in ("I1", "I2", "I3")]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "max_bands": 2, "weight": 1.0,
                    "margin": {"delta_min_s": 0.0, "coef": 1.0}}]
        _, _, model, result = _solve(_base(plans, demands))
        self.assertAlmostEqual(model.expr_composite().evaluate(result.values), 36.0, places=3)

    def test_t5_red_band_objective(self):
        plans = [_old_plan(i, [[0.0, 0.45]], [[0.0, 0.45]]) for i in ("I1", "I2")]
        demands = [{"id": "RW_up", "type": "red", "direction": "up",
                    "nodes": ["I1", "I2"], "max_bands": 3, "weight": 1.0}]
        _, _, model, result = _solve(_base(
            plans, demands, intersections=("I1", "I2"),
            segments=[_segment("I1", "I2", dist=200.0, v=10.0)],
        ))
        self.assertAlmostEqual(model.expr_composite().evaluate(result.values), 40.5, places=3)

    def test_double_cycle_shares_one_boundary_delta(self):
        plans = [
            _old_plan("I1", [[0.0, 0.2]], [[0.0, 0.2]], double=True,
                      adjustable={"up": {"0": {"start": [-0.02, 0.02],
                                               "end": [-0.02, 0.02]}}, "down": {}}),
            _old_plan("I2", [[0.0, 0.45]], [[0.0, 0.45]]),
            _old_plan("I3", [[0.0, 0.45]], [[0.0, 0.45]]),
        ]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "weight": 1.0,
                    "margin": {"delta_min_s": 0.0, "coef": 1.0}}]
        problem, ctx, model, result = _solve(_base(plans, demands))
        self.assertEqual(result.status.value, "optimal")
        # A single declared boundary delta is shared by both half-cycle copies.
        self.assertEqual(sum(1 for v in model.vars if v.name.startswith("bd[I1_P1,up")), 1)
        self.assertGreater(model.expr_composite().evaluate(result.values), 0.0)

    def test_double_cycle_window_touching_half_seam(self):
        # Half-cycle green [0, 0.3): the copy starts exactly at the 0.5 seam.
        # The seam is a const edge with value 0.5, not 1.0.
        def dc_plan(iid):
            return {
                "id": f"{iid}_P1", "intersection": iid, "double_cycle": True,
                "min_phase_s": 1e-3,
                "directions": {
                    "up": {"boundaries": [{"pos_r": 0.3, "adjust": [0.0, 0.2]}],
                           "green_mask": [True, False]},
                    "down": {"boundaries": [{"pos_r": 0.3}],
                             "green_mask": [True, False]},
                },
            }
        problem = ProblemInput.from_obj({
            "corridor": {"corridor_id": "seam", "intersections": ["I1", "I2"],
                         "segments": [_segment("I1", "I2", dist=200.0, v=10.0)]},
            "plans": [dc_plan("I1"), dc_plan("I2")],
            "objective": {"band_demands": [{"id": "GW_up", "type": "green",
                                            "direction": "up", "nodes": ["I1", "I2"],
                                            "weight": 1.0,
                                            "margin": {"delta_min_s": 0.0, "coef": 1.0}}],
                          "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
            "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
            "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                       "num_workers": 1,
                       "heuristic": {"enabled": False, "refine_rounds": 0}},
        })
        plan = problem.plan("I1_P1")
        wins = plan.derived_windows("up")
        self.assertAlmostEqual(wins[1].start_r, 0.5)
        self.assertAlmostEqual(wins[1].end_r, 0.8)
        # seam edge constants are 0.5, not 1.0
        lo, hi = plan.window_endpoint_bound_r("up", 1, "start")
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)
        # And the model sees an effective start of 0.5*C for the copy.
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        expr = model.meta.effective[("I1", "up")].s[2]
        self.assertTrue(expr.is_constant())
        self.assertAlmostEqual(expr.const, 50.0)

        # Half-cycle green [0.3, 0.5): the declared window ends at the seam.
        plan2 = PlanSpec.from_obj({
            "id": "I1_P2", "intersection": "I1", "double_cycle": True,
            "directions": {
                "up": {"boundaries": [{"pos_r": 0.3}], "green_mask": [False, True]},
                "down": {"boundaries": [{"pos_r": 0.3}], "green_mask": [False, True]},
            },
        })
        wins2 = plan2.derived_windows("up")
        self.assertAlmostEqual(wins2[0].end_r, 0.5)
        lo, hi = plan2.window_endpoint_bound_r("up", 0, "end")
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_hidden_boundary_constraint_is_respected(self):
        # Declare [0.2, 0.5), mask [T,T,F]: derived green [0,0.5).
        # Add a hard constraint on the hidden boundary b1: b1 >= 0.25C.
        plan = {
            "id": "I1_P1", "intersection": "I1", "double_cycle": False,
            "min_phase_s": 1e-3,
            "directions": {
                "up": {"boundaries": [{"pos_r": 0.2, "adjust": [0.0, 0.1]},
                                       {"pos_r": 0.5}],
                       "green_mask": [True, True, False]},
                "down": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]},
            },
            "hard_constraints": [{
                "id": "hidden_min", "sense": ">=", "rhs_ratio": 0.25, "hard": True,
                "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                    "plan": "I1_P1", "direction": "up", "index": 0},
                           "coef": 1.0}],
            }],
        }
        import copy
        p2 = copy.deepcopy(plan); p2["id"] = "I2_P1"; p2["intersection"] = "I2"
        p3 = copy.deepcopy(plan); p3["id"] = "I3_P1"; p3["intersection"] = "I3"
        problem = ProblemInput.from_obj({
            "corridor": {"corridor_id": "t", "intersections": ["I1", "I2", "I3"],
                         "segments": [_segment("I1", "I2"), _segment("I2", "I3")]},
            "plans": [plan, p2, p3],
            "objective": {"band_demands": [{"id": "GW_up", "type": "green",
                                            "direction": "up",
                                            "nodes": ["I1", "I2", "I3"], "weight": 1.0,
                                            "margin": {"delta_min_s": 0.0, "coef": 1.0}}],
                          "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
                          "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
            "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
            "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                       "num_workers": 1,
                       "heuristic": {"enabled": False, "refine_rounds": 0}},
        })
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        backend = make_backend("scipy")
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        result = backend.solve(session, SolveOptions(20.0, 0.0))
        self.assertEqual(result.status.value, "optimal")
        b = model.meta.plan_boundary[("I1_P1", "up", 0)].evaluate(result.values)
        self.assertGreaterEqual(b, 0.25 * 100.0 - 1e-7)


if __name__ == "__main__":
    unittest.main()
