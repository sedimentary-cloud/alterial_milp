"""方向级 uncontrolled（全周期绿、无相位边界）的 schema/模型回归。"""
from __future__ import annotations

import unittest

from greenwave import run
from greenwave.errors import GWInputError, GWValidationError
from greenwave.model.builder import ModelBuilder
from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.schema.models import DirectionPlan, PlanSpec, ProblemInput


def _raw(*, red: bool = False, both_uncontrolled: bool = False) -> dict:
    up_i1 = {"uncontrolled": True}
    down_i1 = {"boundaries": [{"pos_r": 0.5}],
               "green_mask": [{"green": True}, {"green": False}]}
    if both_uncontrolled:
        down_i1 = {"uncontrolled": True}
    demands = [{"id": "GW_up", "type": "green", "direction": "up",
                "nodes": ["I1", "I2"], "weight": 1.0}]
    if red:
        demands.append({"id": "RW_up", "type": "red", "direction": "up",
                        "nodes": ["I1", "I2"], "weight": 1.0})
    return {
        "corridor": {"corridor_id": "uncontrolled", "intersections": ["I1", "I2"],
                     "segments": [{"from": "I1", "to": "I2", "distance_m": 300.0,
                                   "speed_up_mps": 10.0, "speed_down_mps": 10.0}]},
        "plans": [
            {"id": "I1_P1", "intersection": "I1",
             "directions": {"up": up_i1, "down": down_i1}},
            {"id": "I2_P1", "intersection": "I2",
             "directions": {
                 "up": {"boundaries": [{"pos_r": 0.5}],
                        "green_mask": [{"green": True}, {"green": False}]},
                 "down": {"boundaries": [{"pos_r": 0.5}],
                          "green_mask": [{"green": True}, {"green": False}]},
             }},
        ],
        "objective": {"band_demands": demands,
                      "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [90.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False}},
    }


class TestUncontrolledDirection(unittest.TestCase):
    def test_schema_allows_omitted_boundaries_and_mask(self):
        p = PlanSpec.from_obj(_raw()["plans"][0])
        self.assertTrue(p.is_uncontrolled("up"))
        self.assertFalse(p.is_uncontrolled("down"))
        self.assertEqual(len(p.direction_plan("up").boundaries), 0)
        self.assertEqual(p.direction_plan("up").mask_bool, (True,))
        self.assertEqual(len(p.derived_windows("up")), 1)
        w = p.derived_windows("up")[0]
        self.assertAlmostEqual(w.start_r, 0.0)
        self.assertAlmostEqual(w.end_r, 1.0)
        self.assertEqual(p.derived_red_intervals("up"), ())
        self.assertEqual(p.derived_windows("down")[0].end_r, 0.5)

    def test_roundtrip_keeps_uncontrolled_flag(self):
        p = ProblemInput.from_obj(_raw())
        p2 = ProblemInput.from_obj(p.model_dump())
        self.assertTrue(p2.plan("I1_P1").is_uncontrolled("up"))
        self.assertEqual(p2.plan("I1_P1").direction_plan("up").boundaries, ())
        self.assertFalse(p2.plan("I1_P1").is_uncontrolled("down"))

    def test_both_uncontrolled_is_rejected(self):
        with self.assertRaises(GWInputError):
            PlanSpec.from_obj(_raw(both_uncontrolled=True)["plans"][0])

    def test_uncontrolled_cannot_carry_boundaries_or_margins(self):
        with self.assertRaises(GWInputError):
            DirectionPlan.from_obj({"uncontrolled": True,
                                    "boundaries": [{"pos_r": 0.5}]})
        with self.assertRaises(GWInputError):
            DirectionPlan.from_obj({
                "uncontrolled": True,
                "green_mask": [{"green": True,
                                "margin": {"start": {"delta_min_s": 3, "coef": 1}}}],
            })
        with self.assertRaises(GWInputError):
            DirectionPlan.from_obj({"uncontrolled": "true"})

    def test_red_demand_on_uncontrolled_direction_is_static_error(self):
        p = ProblemInput.from_obj(_raw(red=True))
        issues = p.validate_all()
        v14 = [i for i in issues if i.code == "V14"]
        self.assertEqual(len(v14), 1)
        self.assertEqual(v14[0].level, "error")
        self.assertIn("RW_up", v14[0].message)
        with self.assertRaises(GWValidationError):
            run(_raw(red=True))

    def test_model_builder_rejects_red_demand_if_validation_bypassed(self):
        problem = ProblemInput.from_obj(_raw(red=True))
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        with self.assertRaises(GWInputError) as ctx_err:
            ModelBuilder(problem, ctx).build()
        self.assertIn("uncontrolled", str(ctx_err.exception))

    def test_green_demand_can_use_uncontrolled_direction(self):
        out = run(_raw())
        grid = out["grid_results"][0]
        self.assertEqual(grid["status"], "optimal")
        self.assertIsNotNone(grid["composite_best"])
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(sol["plan_selection"]["I1"], "I1_P1")
        # 全绿方向不作为带宽上界；I2 的绿窗宽度限制带宽 ≤ 45s。
        self.assertLessEqual(sol["score"]["band"]["reward_breakdown"]["bandwidth"], 45.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
