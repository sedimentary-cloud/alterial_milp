"""Regression tests for the design-document synthetic cases T1--T8."""
from __future__ import annotations

import copy
import math
import os
import unittest

from greenwave import run
from greenwave.schema.models import ProblemInput
from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.model.builder import ModelBuilder
from greenwave.solve import make_backend
from greenwave.solve.backend_base import SolveOptions
from greenwave.report.solution import SolutionDecoder


def _seg(from_, to, dist=500.0, v=11.1111111111, dist_up=None, dist_down=None):
    obj = {"from": from_, "to": to, "speed_up_mps": v, "speed_down_mps": v}
    if dist is not None:
        obj["distance_m"] = dist
    if dist_up is not None:
        obj["distance_up_m"] = dist_up
    if dist_down is not None:
        obj["distance_down_m"] = dist_down
    return obj


def _plan(iid, up, down, double=False, adjustable=None, hard=(), soft=()):
    return {"id": f"{iid}_P1", "intersection": iid, "double_cycle": double,
            "windows": {"up": up, "down": down},
            "adjustable": adjustable or {"up": {}, "down": {}},
            "hard_constraints": list(hard), "soft_constraints": list(soft)}


def _base_problem(cycle=90.0, plans=None, demands=None, margin=0.0,
                  solver=None, global_constraints=(), corridor=None, pareto=None):
    if corridor is None:
        corridor = {"corridor_id": "test", "intersections": ["I1", "I2", "I3"],
                    "segments": [_seg("I1", "I2"), _seg("I2", "I3")]}
    if plans is None:
        plans = [_plan(i, [[0.0, 0.45]], [[0.0, 0.45]]) for i in ("I1", "I2", "I3")]
    if demands is None:
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "max_bands": None,
                    "weight": 1.0, "margin": {"delta_min_s": margin, "coef": 1.0}}]
    return {
        "corridor": corridor,
        "plans": plans,
        "objective": {"band_demands": demands,
                      "margin_default": {"delta_min_s": margin, "coef": 1.0},
                      "pareto": pareto or {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [cycle], "speed_ratios": [1.0]},
        "solver": solver or {"backend": "scipy", "time_limit_s": 30, "mip_rel_gap": 0.0,
                             "num_workers": 1, "heuristic": {"enabled": False, "refine_rounds": 0}},
        "global_constraints": list(global_constraints),
    }


class TestDocumentCases(unittest.TestCase):
    def _composite(self, out):
        self.assertEqual(out["grid_results"][0]["status"], "optimal")
        return out["grid_results"][0]["composite_best"]

    def _bands(self, out):
        return out["pareto"]["points"][0]["solution"]["bands"]

    # T1: perfect green wave at C=90
    def test_t1_perfect_green(self):
        out = run(_base_problem(cycle=90.0))
        self.assertAlmostEqual(self._composite(out), 40.5, places=4)
        bands = self._bands(out)
        self.assertAlmostEqual(bands[0]["width_s"], 40.5, places=4)

    # T2: same as T1 but C=80; travel time 45s mod 80 still permits 36s
    def test_t2_cycle_80(self):
        out = run(_base_problem(cycle=80.0))
        self.assertAlmostEqual(self._composite(out), 36.0, places=4)

    # T3: bidirectional green wave
    def test_t3_bidirectional(self):
        plans = [_plan(i, [[0.0, 0.45]], [[0.0, 0.45]]) for i in ("I1", "I2", "I3")]
        demands = [
            {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
             "weight": 1.0, "margin": {"delta_min_s": 0.0}},
            {"id": "GW_down", "type": "green", "direction": "down", "nodes": ["I3", "I2", "I1"],
             "weight": 1.0, "margin": {"delta_min_s": 0.0}},
        ]
        out = run(_base_problem(plans=plans, demands=demands, margin=0.0))
        self.assertAlmostEqual(self._composite(out), 81.0, places=3)
        self.assertEqual(len(self._bands(out)), 2)

    # T4: two windows and two band slots
    def test_t4_multi_window_multi_band(self):
        plans = [_plan(i, [[0.0, 0.2], [0.5, 0.7]], [[0.0, 0.2], [0.5, 0.7]])
                 for i in ("I1", "I2", "I3")]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "max_bands": 2, "weight": 1.0,
                    "margin": {"delta_min_s": 0.0}}]
        out = run(_base_problem(plans=plans, demands=demands, margin=0.0))
        self.assertAlmostEqual(self._composite(out), 36.0, places=3)
        bands = self._bands(out)
        self.assertEqual(len(bands), 2)

    # T5: red band overlap across adjacent pair
    def test_t5_red_band(self):
        corridor = {"corridor_id": "red", "intersections": ["I1", "I2"],
                    "segments": [_seg("I1", "I2", dist=200.0, v=10.0)]}
        plans = [_plan("I1", [[0.0, 0.45]], [[0.0, 0.45]]),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]])]
        demands = [{"id": "RW_up", "type": "red", "direction": "up",
                    "nodes": ["I1", "I2"], "max_bands": 3, "weight": 1.0}]
        out = run(_base_problem(plans=plans, demands=demands, margin=0.0, corridor=corridor))
        # With one 40.5 s green and a 90 s red complement, the offset freedom allows
        # the shifted upstream green to sit entirely inside a red interval.
        self.assertAlmostEqual(self._composite(out), 40.5, places=3)
        band = self._bands(out)[0]
        self.assertGreater(band["width_s"], 0.0)
        self.assertIsNotNone(band["red_instance"])

    # constraint_templates: absolute duration thresholds keep seconds across grid search
    def test_constraint_template_absolute_duration(self):
        d = _base_problem(margin=0.0)
        d["grid"]["cycles_s"] = [90.0, 180.0]
        d["constraint_templates"] = [{
            "id": "all_green_min_50s",
            "scope": {"kind": "all_plan_windows", "directions": ["up", "down"]},
            "terms": [{"atom": {"kind": "window_duration"}, "coef": 1.0}],
            "sense": ">=",
            "rhs": 50.0,
            "rhs_unit": "s",
            "hard": True,
        }]
        out = run(d)
        statuses = {g["cycle_s"]: g["status"] for g in out["grid_results"]}
        # C=90 时窗口只有 40.5s，低于绝对 50s 硬下限；C=180 时窗口 81s，可行。
        self.assertEqual(statuses[90.0], "infeasible")
        self.assertEqual(statuses[180.0], "optimal")

    # constraint_templates: red_duration is the complement of selected green windows
    def test_constraint_template_red_duration_absolute(self):
        d = _base_problem(margin=0.0)
        d["grid"]["cycles_s"] = [90.0, 180.0]
        d["constraint_templates"] = [{
            "id": "all_red_min_50s",
            "scope": {"kind": "all_plan_red_windows", "directions": ["up", "down"]},
            "terms": [{"atom": {"kind": "red_duration"}, "coef": 1.0}],
            "sense": ">=",
            "rhs": 50.0,
            "rhs_unit": "s",
            "hard": True,
        }]
        out = run(d)
        statuses = {g["cycle_s"]: g["status"] for g in out["grid_results"]}
        # C=90 时绿灯 40.5s，红灯补集 49.5s，低于绝对 50s；C=180 时红灯 99s。
        self.assertEqual(statuses[90.0], "infeasible")
        self.assertEqual(statuses[180.0], "optimal")

    # constraint_templates: soft duration shortfall is penalized in seconds
    def test_constraint_template_soft_duration_loss(self):
        d = _base_problem(margin=0.0)
        for p in d["plans"]:
            p["windows"] = {"up": [[0.0, 0.1]], "down": [[0.0, 0.1]]}
        d["constraint_templates"] = [{
            "id": "all_green_loss_12s",
            "scope": {"kind": "all_plan_windows", "directions": ["up", "down"]},
            "terms": [{"atom": {"kind": "window_duration"}, "coef": 1.0}],
            "sense": ">=",
            "rhs": 12.0,
            "rhs_unit": "s",
            "hard": False,
            "soft": {"coef": 1.0, "owner": "intersection_loss"},
        }]
        out = run(d)
        grid = out["grid_results"][0]
        self.assertEqual(grid["status"], "optimal")
        # 3 个路口 × 2 个方向 × (12 - 9)=3s = 18s 短失误。
        self.assertAlmostEqual(grid["negative_intersection_loss_s"], -18.0, places=4)
        sol = out["pareto"]["points"][0]["solution"]
        self.assertAlmostEqual(sol["intersection_loss_s"], 18.0, places=4)

    # T6: adjustable endpoint + intersection-loss soft constraint + epsilon scan
    def test_t6_adjustable_soft_pareto(self):
        soft = [{"id": "endsoft", "terms": [{
                    "atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1",
                             "direction": "up", "window": 0, "side": "end"}, "coef": 1.0}],
                 "sense": "<=", "rhs_ratio": 0.40, "hard": False,
                 "soft": {"coef": 10.0, "owner": "intersection_loss"}}]
        plans = [_plan("I1", [[0.0, 0.45]], [[0.0, 0.45]],
                       adjustable={"up": {"0": {"end": [-0.05, 0.05]}}, "down": {}}, soft=soft),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]]),
                 _plan("I3", [[0.0, 0.45]], [[0.0, 0.45]])]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "weight": 1.0,
                    "margin": {"delta_min_s": 0.0}}]
        out = run(_base_problem(plans=plans, demands=demands, margin=0.0,
                                pareto={"num_points": 2, "relax_max": 0.5, "topk_grids": 1}))
        self.assertAlmostEqual(self._composite(out), 40.5, places=3)
        pts = out["pareto"]["points"]
        self.assertTrue(any(abs(p["intersection_loss"]) <= 45.0 + 1e-4 for p in pts))
        # l=0 point must respect B* exactly at its relaxation
        self.assertAlmostEqual(pts[0]["composite_actual"], 40.5, places=3)

    # T7: cycle-dependent min-green filtering
    def test_t7_plan_filtering(self):
        hard = [{"id": "mg", "terms": [
                    {"atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1",
                              "direction": "up", "window": 0, "side": "end"}, "coef": 1.0},
                    {"atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1",
                              "direction": "up", "window": 0, "side": "start"}, "coef": -1.0}],
                 "sense": ">=", "rhs": 15.0, "hard": True}]
        plans = [_plan("I1", [[0.0, 0.2]], [[0.0, 0.2]], hard=hard),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]]),
                 _plan("I3", [[0.0, 0.45]], [[0.0, 0.45]])]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "weight": 1.0,
                    "margin": {"delta_min_s": 0.0}}]
        d = _base_problem(cycle=60.0, plans=plans, demands=demands, margin=0.0)
        d["grid"]["cycles_s"] = [60.0, 90.0]
        out = run(d)
        statuses = {g["cycle_s"]: g["status"] for g in out["grid_results"]}
        self.assertEqual(statuses[60.0], "skipped_infeasible")
        self.assertEqual(statuses[90.0], "optimal")
        filtered = [g for g in out["grid_results"] if g["cycle_s"] == 60.0][0]
        self.assertTrue(filtered["dropped_plans"])

    # T8: double-cycle copy consistency
    def test_t8_double_cycle_consistency(self):
        plans = [_plan("I1", [[0.0, 0.2]], [[0.0, 0.2]], double=True,
                       adjustable={"up": {"0": {"start": [-0.02, 0.02], "end": [-0.02, 0.02]}},
                                   "down": {}}),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]]),
                 _plan("I3", [[0.0, 0.45]], [[0.0, 0.45]])]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "weight": 1.0,
                    "margin": {"delta_min_s": 0.0}}]
        d = _base_problem(plans=plans, demands=demands, margin=0.0)
        # solve model directly so we can inspect adjustments
        problem = ProblemInput.from_obj(d)
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        c11 = [r for r in model.cons.rows if r.meta.group == "C11"]
        self.assertGreaterEqual(len(c11), 2)
        backend = make_backend("scipy")
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        res = backend.solve(session, SolveOptions(30.0, 0.0))
        self.assertEqual(res.status.value, "optimal")
        sol = SolutionDecoder(model).decode(res.values)
        adj = sol.adjustments_r
        self.assertAlmostEqual(adj[("I1_P1", "up", 0, "start")],
                               adj[("I1_P1", "up", 1, "start")], places=6)
        self.assertAlmostEqual(adj[("I1_P1", "up", 0, "end")],
                               adj[("I1_P1", "up", 1, "end")], places=6)


    def test_highspy_backend_smoke(self):
        d = _base_problem(cycle=90.0)
        d["solver"]["backend"] = "highspy"
        out = run(d)
        self.assertEqual(out["grid_results"][0]["status"], "optimal")
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 40.5, places=3)

    def test_process_pool_stage1(self):
        d = _base_problem(cycle=90.0)
        d["grid"]["cycles_s"] = [80.0, 90.0, 100.0]
        d["solver"]["num_workers"] = 2
        d["solver"]["heuristic"]["enabled"] = True
        out = run(d)
        statuses = {g["cycle_s"]: g["status"] for g in out["grid_results"]}
        self.assertEqual(set(statuses.values()), {"optimal"})
        self.assertEqual(len(out["grid_results"]), 3)


    def test_global_constraints_hard_and_soft(self):
        hard = [{"id": "g_hard", "terms": [{"atom": {"kind": "bandwidth", "demand": "GW_up"},
                                              "coef": 1.0}],
                 "sense": ">=", "rhs": 35.0, "hard": True}]
        out = run(_base_problem(cycle=90.0, margin=0.0, global_constraints=hard))
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 40.5, places=3)

        soft = [{"id": "g_soft", "terms": [{"atom": {"kind": "bandwidth", "demand": "GW_up"},
                                             "coef": 1.0}],
                 "sense": ">=", "rhs": 50.0, "hard": False,
                 "soft": {"coef": 2.0, "owner": "composite"}}]
        out2 = run(_base_problem(cycle=90.0, margin=0.0, global_constraints=soft))
        # 40.5 - 2 * (50 - 40.5) = 21.5
        self.assertAlmostEqual(out2["grid_results"][0]["composite_best"], 21.5, places=3)

    def test_allow_window_sharing_switch(self):
        plans = [_plan(i, [[0.0, 0.2], [0.5, 0.7]], [[0.0, 0.2], [0.5, 0.7]])
                 for i in ("I1", "I2", "I3")]
        demands = [{"id": "GW_up", "type": "green", "direction": "up",
                    "nodes": ["I1", "I2", "I3"], "max_bands": 2, "weight": 1.0,
                    "margin": {"delta_min_s": 0.0}}]
        d = _base_problem(plans=plans, demands=demands, margin=0.0)
        d["objective"]["allow_window_sharing"] = True
        out = run(d)
        self.assertEqual(out["grid_results"][0]["status"], "optimal")
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 36.0, places=3)


    def test_time_space_diagram_render(self):
        import os
        import tempfile
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
        from greenwave.report import TimeSpaceDiagram
        d = _base_problem(cycle=90.0, margin=0.0)
        problem = ProblemInput.from_obj(d)
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        backend = make_backend("scipy")
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        res = backend.solve(session, SolveOptions(30.0, 0.0))
        sol = SolutionDecoder(model).decode(res.values)
        path = os.path.join(tempfile.mkdtemp(), "time_space.png")
        TimeSpaceDiagram.render(sol, problem, ctx, path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_infeasible_grid_reports_diagnosis(self):
        d = _base_problem(cycle=90.0, margin=0.0)
        d["global_constraints"] = [{
            "id": "impossible", "terms": [{"atom": {"kind": "bandwidth", "demand": "GW_up"},
                                           "coef": 1.0}],
            "sense": ">=", "rhs": 1000.0, "hard": True}]
        out = run(d)
        grid = out["grid_results"][0]
        self.assertEqual(grid["status"], "infeasible")
        self.assertIn("diagnosis groups", grid["message"])


    def test_directional_segment_lengths(self):
        corridor = {
            "corridor_id": "dir_len",
            "intersections": ["I1", "I2"],
            "segments": [_seg("I1", "I2", dist=None, v=10.0, dist_up=250.0, dist_down=400.0)],
        }
        plans = [_plan("I1", [[0.0, 0.45]], [[0.0, 0.45]]),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.45]])]
        demands = [
            {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2"],
             "weight": 1.0, "margin": {"delta_min_s": 0.0}},
            {"id": "GW_down", "type": "green", "direction": "down", "nodes": ["I2", "I1"],
             "weight": 1.0, "margin": {"delta_min_s": 0.0}},
        ]
        d = _base_problem(plans=plans, demands=demands, margin=0.0, corridor=corridor)
        problem = ProblemInput.from_obj(d)
        self.assertAlmostEqual(problem.corridor.segments[0].distance_for("up"), 250.0)
        self.assertAlmostEqual(problem.corridor.segments[0].distance_for("down"), 400.0)
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        self.assertAlmostEqual(ctx.tau_s[("I1", "up")], 25.0, places=6)
        self.assertAlmostEqual(ctx.tau_s[("I2", "down")], 40.0, places=6)
        # Isolate each direction: a single demand with one segment can always use
        # the full 40.5 s green, but the two directions may trade off when combined.
        out_up = run(_base_problem(plans=plans, demands=[demands[0]], margin=0.0, corridor=corridor))
        out_down = run(_base_problem(plans=plans, demands=[demands[1]], margin=0.0, corridor=corridor))
        self.assertAlmostEqual(out_up["grid_results"][0]["composite_best"], 40.5, places=3)
        self.assertAlmostEqual(out_down["grid_results"][0]["composite_best"], 40.5, places=3)
        out_both = run(d)
        self.assertEqual(out_both["grid_results"][0]["status"], "optimal")


    def test_balanced_group_targets_and_missing_policy(self):
        corridor = {"corridor_id": "balanced_test", "intersections": ["I1", "I2", "I3"],
                    "segments": [_seg("I1", "I2"), _seg("I2", "I3")]}
        plans = [_plan("I1", [[0.0, 0.45]], [[0.0, 0.2]]),
                 _plan("I2", [[0.0, 0.45]], [[0.0, 0.2]]),
                 _plan("I3", [[0.0, 0.45]], [[0.0, 0.2]])]
        demands = [
            {"id": "GW_up", "type": "green", "direction": "up",
             "nodes": ["I1", "I2", "I3"], "max_bands": 1, "weight": 1.0,
             "margin": {"delta_min_s": 0.0}},
            {"id": "GW_down", "type": "green", "direction": "down",
             "nodes": ["I3", "I2", "I1"], "max_bands": 1, "weight": 1.0,
             "margin": {"delta_min_s": 0.0}},
        ]
        d = _base_problem(plans=plans, demands=demands, margin=0.0, corridor=corridor)
        d["objective"]["balanced_groups"] = [{
            "id": "balance_up_down",
            "demands": ["GW_up", "GW_down"],
            "weight": 5.0,
            "on_infeasible": "zero",
            "targets_s": {"GW_up": 30.0, "GW_down": 20.0},
        }]
        out = run(d)
        self.assertEqual(out["grid_results"][0]["status"], "optimal")
        # up=40.5, down=18 -> normalized min = min(1.35, 0.9) = 0.9
        # composite = 40.5 + 18 + 5 * 0.9 = 63.0
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 63.0, places=3)
        sol = out["pareto"]["points"][0]["solution"]
        self.assertAlmostEqual(sol["balanced_groups"]["balance_up_down"], 0.9, places=4)

    # heuristic lower bound and analytic containment invariant
    def test_invariants_greedy_bound_and_containment(self):
        d = _base_problem(cycle=90.0, margin=0.0)
        d["solver"]["heuristic"]["enabled"] = True
        out = run(d)
        hb = out["grid_results"][0]["heuristic_bound"]
        self.assertIsNotNone(hb)
        self.assertLessEqual(hb, out["grid_results"][0]["composite_best"] + 1e-6)
        problem = ProblemInput.from_obj(d)
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        # decode fresh solution and verify each green band is contained in selected windows
        model = ModelBuilder(problem, ctx).build()
        backend = make_backend("scipy")
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        res = backend.solve(session, SolveOptions(30.0, 0.0))
        sol = SolutionDecoder(model).decode(res.values)
        for band in sol.bands:
            b = problem.demand(band.demand)
            if b.type != "green":
                continue
            table = ctx.travel[b.id]
            for j, node in enumerate(b.nodes):
                slot = band.windows[node]
                plan = problem.plan(sol.plan_selection[node])
                w = plan.effective_windows(b.direction)[slot - 1]
                adj_s = sol.adjustments_r.get((plan.id, b.direction, slot - 1, "start"), 0.0)
                adj_e = sol.adjustments_r.get((plan.id, b.direction, slot - 1, "end"), 0.0)
                ws = (w.start_r + adj_s) * 90.0 + sol.offsets_s[node]
                we = (w.end_r + adj_e) * 90.0 + sol.offsets_s[node]
                lo = band.start_s + table.cum_T_s[j]
                hi = lo + band.width_s
                n = round((lo - ws) / 90.0)
                self.assertTrue(any(lo >= ws + m * 90.0 - 1e-6 and
                                    hi <= we + m * 90.0 + 1e-6
                                    for m in (n - 1, n, n + 1)),
                                f"band {band.demand} not contained at {node}")


if __name__ == "__main__":
    unittest.main()
