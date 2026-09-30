"""Phase 4 tests: plan-native semantic compiler on boundary plans."""
from __future__ import annotations

import unittest

from greenwave.errors import GWSemanticError
from greenwave.semantic import compile_semantics


def _problem(plans):
    return {
        "corridor": {"corridor_id": "t", "intersections": ["I1", "I2"],
                     "segments": [{"from": "I1", "to": "I2", "distance_m": 200.0,
                                   "speed_up_mps": 10.0, "speed_down_mps": 10.0}]},
        "plans": plans,
        "objective": {"band_demands": [
            {"id": "GW_up", "type": "green", "direction": "up",
             "nodes": ["I1", "I2"], "weight": 1.0,
             "margin": {"delta_min_s": 0.0, "coef": 1.0}}]},
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 10, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
    }


def _plan(iid, up_bounds, up_mask, down_bounds=None, down_mask=None):
    if down_bounds is None:
        down_bounds = [{"pos_r": 0.5}]
    if down_mask is None:
        down_mask = [True, False]
    return {"id": f"{iid}_P1", "intersection": iid, "double_cycle": False,
            "directions": {
                "up": {"boundaries": up_bounds, "green_mask": up_mask},
                "down": {"boundaries": down_bounds, "green_mask": down_mask},
            }}


def _green(index=0, position=None, iid="I1", plan=None, d="up"):
    ref = {"kind": "green"}
    if position is not None:
        ref["position"] = position
    else:
        ref["index"] = index
    return {"intersection": iid, "plan": plan or f"{iid}_P1", "direction": d, "window": ref}


class TestSemanticV2(unittest.TestCase):
    def test_green_at_cycle_start_validates_mask(self):
        good = _problem([
            _plan("I1", [{"pos_r": 0.5}], [True, False]),
            _plan("I2", [{"pos_r": 0.5}], [True, False]),
        ])
        res = compile_semantics(good, {"semantic_rules": [
            {"id": "R1", "kind": "green_at_cycle_start", "target": _green()} ]})
        self.assertEqual(res.report.to_dict()["generated_constraints"][0]["path"],
                         "validated_mask")
        bad = _problem([
            _plan("I1", [{"pos_r": 0.5}], [False, True]),
            _plan("I2", [{"pos_r": 0.5}], [True, False]),
        ])
        with self.assertRaises(GWSemanticError):
            compile_semantics(bad, {"semantic_rules": [
                {"id": "R1", "kind": "green_at_cycle_start", "target": _green()} ]})

    def test_sync_boundary_equality(self):
        plans = [
            _plan("I1", [{"pos_r": 0.2}, {"pos_r": 0.5}], [False, True, False],
                  down_bounds=[{"pos_r": 0.3}, {"pos_r": 0.6}],
                  down_mask=[False, True, False]),
            _plan("I2", [{"pos_r": 0.5}], [True, False]),
        ]
        res = compile_semantics(_problem(plans), {"semantic_rules": [{
            "id": "S1", "kind": "sync", "relation": "start_start",
            "targets": [_green(), _green(d="down")],
            "enforcement": "hard",
        }]})
        cs = res.problem_json["plans"][0]["hard_constraints"]
        self.assertEqual(len(cs), 1)
        self.assertEqual(cs[0]["id"], "sem:S1")
        self.assertEqual(cs[0]["sense"], "==")
        self.assertEqual(len(cs[0]["terms"]), 2)
        self.assertTrue(all(t["atom"]["kind"] == "boundary" for t in cs[0]["terms"]))

    def test_green_duration_limit(self):
        plans = [
            _plan("I1", [{"pos_r": 0.2}, {"pos_r": 0.5}], [False, True, False]),
            _plan("I2", [{"pos_r": 0.5}], [True, False]),
        ]
        res = compile_semantics(_problem(plans), {"semantic_rules": [{
            "id": "G1", "kind": "duration_limit", "target": _green(),
            "sense": ">=", "value": 20.0, "unit": "s", "enforcement": "hard",
        }]})
        cs = res.problem_json["plans"][0]["hard_constraints"]
        self.assertEqual(cs[0]["terms"][0]["atom"]["kind"], "window_duration")

    def test_red_duration_limit_is_plan_scoped(self):
        plans = [
            _plan("I1", [{"pos_r": 0.2}, {"pos_r": 0.5}], [False, True, False]),
            _plan("I2", [{"pos_r": 0.5}], [True, False]),
        ]
        res = compile_semantics(_problem(plans), {"semantic_rules": [{
            "id": "RD1", "kind": "duration_limit",
            "target": {"intersection": "I1", "plan": "I1_P1", "direction": "up",
                       "window": {"kind": "red", "after_green": 1}},
            "sense": ">=", "value": 30.0, "unit": "s", "enforcement": "hard",
        }]})
        cs = res.problem_json["plans"][0]["hard_constraints"]
        self.assertEqual(cs[0]["terms"][0]["atom"]["kind"], "red_duration")
        self.assertEqual(cs[0]["terms"][0]["atom"]["window"], 1)

    def test_red_semantics_allows_multi_plan_intersection(self):
        plans = [
            _plan("I1", [{"pos_r": 0.2}, {"pos_r": 0.5}], [False, True, False]),
            {"id": "I1_P2", "intersection": "I1", "double_cycle": False,
             "directions": {
                 "up": {"boundaries": [{"pos_r": 0.3}, {"pos_r": 0.6}],
                        "green_mask": [False, True, False]},
                 "down": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]}}},
            _plan("I2", [{"pos_r": 0.5}], [True, False]),
        ]
        res = compile_semantics(_problem(plans), {"semantic_rules": [{
            "id": "RD1", "kind": "duration_limit",
            "target": {"intersection": "I1", "plan": "I1_P2", "direction": "up",
                       "window": {"kind": "red", "after_green": 1}},
            "sense": "<=", "value": 40.0, "unit": "s", "enforcement": "hard",
        }]})
        cs = res.problem_json["plans"][1]["hard_constraints"]
        self.assertEqual(cs[0]["terms"][0]["atom"]["kind"], "red_duration")


if __name__ == "__main__":
    unittest.main()
