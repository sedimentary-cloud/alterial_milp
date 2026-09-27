"""Regression tests for the semantic compiler layer."""
from __future__ import annotations

import unittest

from greenwave import run
from greenwave.errors import GWSemanticError
from greenwave.semantic import compile_semantics
from test_document_cases import _base_problem, _plan


class TestSemanticCompiler(unittest.TestCase):
    def _green_target(self, intersection="I1", plan="I1_P1", direction="up", index=0):
        return {"intersection": intersection, "plan": plan, "direction": direction,
                "window": {"kind": "green", "index": index}}

    def _red_target(self, intersection="I1", plan="I1_P1", direction="up", **window):
        return {"intersection": intersection, "plan": plan, "direction": direction,
                "window": {"kind": "red", **window}}

    def test_green_at_cycle_start_rewrites_fixed_endpoint(self):
        d = _base_problem(margin=0.0)
        for p in d["plans"]:
            p["windows"] = {"up": [[0.1, 0.45]], "down": [[0.1, 0.45]]}
        sem = {"semantic_rules": [{
            "id": "R1", "kind": "green_at_cycle_start",
            "target": self._green_target(),
        }]}
        res = compile_semantics(d, sem)
        self.assertAlmostEqual(
            res.problem_json["plans"][0]["windows"]["up"][0][0], 0.0, places=9)
        self.assertEqual(res.report.to_dict()["generated_constraints"][0]["path"],
                         "rewrite_window")

    def test_green_at_cycle_end_locks_adjustable_endpoint_with_constraint(self):
        d = _base_problem(margin=0.0)
        d["plans"][0]["adjustable"] = {"up": {"0": {"end": [-0.05, 0.05]}},
                                       "down": {}}
        sem = {"semantic_rules": [{
            "id": "R2", "kind": "green_at_cycle_end",
            "target": self._green_target(),
        }]}
        res = compile_semantics(d, sem)
        cs = res.problem_json["plans"][0]["hard_constraints"]
        self.assertEqual(len(cs), 1)
        self.assertEqual(cs[0]["id"], "sem:R2")
        self.assertEqual(cs[0]["rhs_ratio"], 1.0)
        self.assertEqual(cs[0]["terms"][0]["atom"]["kind"], "endpoint")
        self.assertEqual(cs[0]["terms"][0]["atom"]["side"], "end")

    def test_sync_same_plan(self):
        d = _base_problem(margin=0.0)
        sem = {"semantic_rules": [{
            "id": "S1", "kind": "sync", "relation": "start_start",
            "targets": [self._green_target(direction="up"),
                        self._green_target(direction="down")],
            "enforcement": "hard",
        }]}
        res = compile_semantics(d, sem)
        cs = res.problem_json["plans"][0]["hard_constraints"]
        self.assertEqual(len(cs), 1)
        self.assertEqual(cs[0]["id"], "sem:S1")
        self.assertEqual(cs[0]["sense"], "==")
        self.assertEqual(len(cs[0]["terms"]), 2)

    def test_duration_limit_green_and_red(self):
        d = _base_problem(margin=0.0)
        sem = {"semantic_rules": [
            {"id": "G1", "kind": "duration_limit",
             "target": self._green_target(),
             "sense": ">=", "value": 25.0, "unit": "s", "enforcement": "hard"},
            {"id": "RD1", "kind": "duration_limit",
             "target": self._red_target(after_green=1),
             "sense": ">=", "value": 10.0, "unit": "s", "enforcement": "hard"},
        ]}
        res = compile_semantics(d, sem)
        green_constraints = res.problem_json["plans"][0]["hard_constraints"]
        self.assertEqual(green_constraints[0]["terms"][0]["atom"]["kind"],
                         "window_duration")
        templates = res.problem_json["constraint_templates"]
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["id"], "sem:RD1")
        self.assertEqual(templates[0]["scope"]["kind"], "all_plan_red_windows")
        self.assertEqual(templates[0]["scope"]["window_indices"], [1])

    def test_red_index_out_of_range_reports_ledger(self):
        d = _base_problem(margin=0.0)
        sem = {"semantic_rules": [{
            "id": "BAD", "kind": "duration_limit",
            "target": self._red_target(index=5),
            "sense": ">=", "value": 10.0, "unit": "s", "enforcement": "hard",
        }]}
        with self.assertRaises(GWSemanticError) as ctx:
            compile_semantics(d, sem)
        text = str(ctx.exception)
        self.assertIn("rid=1", text)
        self.assertIn("does not exist", text)

    def test_red_rule_rejects_multi_plan_intersection(self):
        d = _base_problem(margin=0.0)
        # 给 I1 再加一个方案，使红窗寻址 v1 报错。
        d["plans"].append(_plan("I1", [[0.2, 0.5]], [[0.2, 0.5]]))
        d["plans"][-1]["id"] = "I1_P2"
        sem = {"semantic_rules": [{
            "id": "RD", "kind": "duration_limit",
            "target": self._red_target(after_green=1),
            "sense": ">=", "value": 10.0, "unit": "s", "enforcement": "hard",
        }]}
        with self.assertRaises(GWSemanticError) as ctx:
            compile_semantics(d, sem)
        self.assertIn("single-plan", str(ctx.exception))

    def test_run_accepts_semantics_keyword(self):
        d = _base_problem(margin=0.0)
        sem = {"semantic_rules": [{
            "id": "G1", "kind": "duration_limit",
            "target": self._green_target(),
            "sense": ">=", "value": 25.0, "unit": "s", "enforcement": "hard",
        }]}
        out = run(d, semantics=sem)
        self.assertEqual(out["grid_results"][0]["status"], "optimal")


if __name__ == "__main__":
    unittest.main()
