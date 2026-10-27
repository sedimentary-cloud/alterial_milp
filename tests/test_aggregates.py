"""具名聚合（boundary 线性组合取 max/min）的 schema / 建模 / 解码测试。"""
from __future__ import annotations

import copy
import unittest

from greenwave import run
from greenwave.schema.models import AtomRef, ProblemInput

CYCLE = 90.0


def _seg(a, b, dist=300.0, v=15.0):
    return {"from": a, "to": b, "distance_m": dist,
            "speed_up_mps": v, "speed_down_mps": v}


def _plan(pid, iid, boundaries, mask, *, soft=(), min_phase=0.001, double=False):
    return {"id": pid, "intersection": iid, "double_cycle": double,
            "min_phase_s": min_phase,
            "directions": {d: {"boundaries": [{"pos_r": x} for x in boundaries],
                               "green_mask": [{"green": bool(g)} for g in mask]}
                           for d in ("up", "down")},
            "hard_constraints": [], "soft_constraints": list(soft)}


def _phase_min_agg(atoms, *, plan, contributions, op="min"):
    """三条表达式的 min：b0、b1−b0、C−b1（用 ratio 常数）。"""
    b0, b1 = atoms
    return {
        "id": "min_phase", "op": op, "plan": plan,
        "expressions": [
            [{"atom": {"kind": "boundary", "intersection": "I1", "plan": plan,
                       "direction": "up", "index": b0}, "coef": 1.0}],
            [{"atom": {"kind": "boundary", "intersection": "I1", "plan": plan,
                       "direction": "up", "index": b1}, "coef": 1.0},
             {"atom": {"kind": "boundary", "intersection": "I1", "plan": plan,
                       "direction": "up", "index": b0}, "coef": -1.0}],
            [{"atom": {"kind": "const", "const": 1.0, "unit": "ratio"}, "coef": 1.0},
             {"atom": {"kind": "boundary", "intersection": "I1", "plan": plan,
                       "direction": "up", "index": b1}, "coef": -1.0}],
        ],
        "contributions": list(contributions),
    }


def _problem(plans, aggregates, *, soft=None, demand_nodes=("I1", "I2"), cycle=CYCLE):
    return {
        "corridor": {"corridor_id": "agg", "intersections": ["I1", "I2"],
                     "segments": [_seg("I1", "I2")]},
        "plans": plans,
        "objective": {"band_demands": [{
            "id": "GW_up", "type": "green", "direction": "up",
            "nodes": list(demand_nodes), "weight": 1.0}],
            "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [cycle], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                   "num_workers": 1,
                   "heuristic": {"enabled": False, "refine_rounds": 0}},
        "aggregates": aggregates,
    }


def _solve(problem):
    from greenwave.model.builder import ModelBuilder
    from greenwave.preprocess.feasibility import FeasibilityFilter
    from greenwave.report.solution import SolutionDecoder
    from greenwave.solve import make_backend
    from greenwave.solve.backend_base import SolveOptions
    ctx = FeasibilityFilter(problem).filter(CYCLE, 1.0)
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend("scipy")
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(20.0, 0.0))
    return model, ctx, res, SolutionDecoder(model).decode(res.values)


def _solve_min_loss(problem):
    from greenwave.model.builder import ModelBuilder
    from greenwave.preprocess.feasibility import FeasibilityFilter
    from greenwave.report.solution import SolutionDecoder
    from greenwave.solve import make_backend
    from greenwave.solve.backend_base import SolveOptions
    ctx = FeasibilityFilter(problem).filter(CYCLE, 1.0)
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend("scipy")
    session = backend.create(model)
    backend.set_objective(session, model.expr_loss(), "min")
    res = backend.solve(session, SolveOptions(20.0, 0.0))
    return model, ctx, SolutionDecoder(model).decode(res.values), float(res.objective)


class TestAggregateSchema(unittest.TestCase):
    def test_parse_roundtrip_and_validation(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "weight": 2.0}])
        raw = _problem(plans, [agg])
        p = ProblemInput.from_obj(raw)
        self.assertEqual(p.aggregates[0].op, "min")
        self.assertEqual(p.aggregates[0].plan, "I1_P1")
        self.assertEqual(p.validate_all(), [])
        p2 = ProblemInput.from_obj(p.model_dump())
        self.assertEqual(len(p2.aggregates), 1)
        self.assertEqual(p2.aggregates[0].contributions[0].weight, 2.0)

    def test_const_unit_only_on_const_atom(self):
        with self.assertRaisesRegex(ValueError, "unit"):
            AtomRef.from_obj({"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                              "direction": "up", "index": 0, "unit": "ratio"})

    def test_unsupported_atom_in_aggregate(self):
        from greenwave.schema.models import AggregateSpec
        with self.assertRaisesRegex(ValueError, "boundary/const"):
            AggregateSpec.from_obj({
                "id": "bad", "op": "min", "plan": "I1_P1",
                "expressions": [[{"atom": {"kind": "bandwidth", "demand": "GW_up"},
                                  "coef": 1.0}]],
                "contributions": [{"target": "band_objective", "weight": 1.0}],
            })

    def test_validation_errors(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        cases = []
        # 未知 plan
        cases.append(_phase_min_agg((0, 1), plan="NOPE",
                                    contributions=[{"target": "band_objective", "weight": 1.0}]))
        # boundary 下标越界
        bad = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "weight": 1.0}])
        bad["expressions"][0][0]["atom"]["index"] = 9
        cases.append(bad)
        for agg in cases:
            p = ProblemInput.from_obj(_problem(plans, [agg]))
            codes = [i.code for i in p.validate_all()]
            self.assertIn("V13", codes, agg["id"])


class TestAggregateMilp(unittest.TestCase):
    def test_min_phase_reward_in_band_objective(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "weight": 2.0}])
        out = run(_problem(plans, [agg]))
        grid = out["grid_results"][0]
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(grid["status"], "optimal")
        # β = 45（I1 窗 54、I2 窗 45）；min 相位 = 18；composite = 45 + 2×18 = 81。
        self.assertAlmostEqual(grid["bandwidth_objective_s"], 45.0, places=3)
        self.assertAlmostEqual(sol["aggregates"]["min_phase"], 18.0, places=3)
        self.assertAlmostEqual(grid["composite_best"], 81.0, places=3)
        self.assertAlmostEqual(out["pareto"]["points"][0]["composite_actual"], 81.0, places=3)

    def test_min_aggregate_offsets_intersection_loss(self):
        # 可调边界 → 聚合不是常数，验证 L 最小化把 t 推到 min(相位)。
        soft = [{"id": "b0_cap",
                 "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                     "plan": "I1_P1", "direction": "up", "index": 0},
                            "coef": 1.0}],
                 "sense": "<=", "rhs_ratio": 0.1, "hard": False,
                 "soft": {"coef": 1.0, "owner": "intersection_loss"}}]
        plans = [{"id": "I1_P1", "intersection": "I1", "double_cycle": False,
                  "min_phase_s": 0.001,
                  "directions": {d: {
                      "boundaries": [{"pos_r": 0.2, "adjust": [-0.05, 0.05]},
                                     {"pos_r": 0.6, "adjust": [-0.05, 0.05]}],
                      "green_mask": [{"green": True}, {"green": True}, {"green": False}]}
                      for d in ("up", "down")},
                  "hard_constraints": [], "soft_constraints": soft},
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "intersection_loss", "weight": 1.0}])
        problem = ProblemInput.from_obj(_problem(plans, [agg]))
        model, _, sol, objective = _solve_min_loss(problem)
        # 解码的净损失 = 模型 expr_loss 的最优值，且 = raw − min(相位)。
        self.assertAlmostEqual(objective, sol.loss_s, places=4)
        self.assertAlmostEqual(sol.loss_s, sol.raw_loss_s - sol.aggregates["min_phase"],
                               places=4)
        self.assertLessEqual(sol.loss_s, sol.raw_loss_s + 1e-9)

    def test_multi_plan_gating(self):
        plans = [_plan("I1_A", "I1", [0.2], [True, False]),
                 _plan("I1_B", "I1", [0.6], [True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        # A 的窗口只有 18s，B 有 54s → B 被选中；A 的聚合应为 0。
        agg = _phase_min_agg((0, 0), plan="I1_A",
                             contributions=[{"target": "band_objective", "weight": 3.0}])
        out = run(_problem(plans, [agg]))
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(sol["plan_selection"]["I1"], "I1_B")
        self.assertAlmostEqual(sol["aggregates"]["min_phase"], 0.0, places=6)
        # composite 只含带宽，不含未选中聚合的奖励。
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 45.0, places=3)

    def test_unavailable_plan_contributes_zero(self):
        plans = [_plan("I1_A", "I1", [0.2], [True, False], min_phase=100.0),
                 _plan("I1_B", "I1", [0.6], [True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 0), plan="I1_A",
                             contributions=[{"target": "band_objective", "weight": 3.0}])
        out = run(_problem(plans, [agg]))
        self.assertEqual(out["grid_results"][0]["status"], "optimal")
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(sol["plan_selection"]["I1"], "I1_B")
        self.assertAlmostEqual(sol["aggregates"]["min_phase"], 0.0, places=6)

    def test_double_cycle_seam_and_ratio_constant(self):
        # 双周期：b0=18，接缝 45−18=27；min = 18。
        plans = [_plan("I1_P1", "I1", [0.2], [True, False], double=True),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = {
            "id": "min_phase", "op": "min", "plan": "I1_P1",
            "expressions": [
                [{"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                           "direction": "up", "index": 0}, "coef": 1.0}],
                [{"atom": {"kind": "const", "const": 0.5, "unit": "ratio"}, "coef": 1.0},
                 {"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                           "direction": "up", "index": 0}, "coef": -1.0}],
            ],
            "contributions": [{"target": "band_objective", "weight": 1.0}],
        }
        out = run(_problem(plans, [agg]))
        sol = out["pareto"]["points"][0]["solution"]
        self.assertAlmostEqual(sol["aggregates"]["min_phase"], 18.0, places=3)
        # β = 18（双周期窗口 18s），composite = 18 + 18 = 36。
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 36.0, places=3)

    def test_ratio_const_in_plan_hard_constraint_feasibility(self):
        # 硬约束：boundary0 <= 0.2C + 1s（比例 + 秒）。C=90 时上限 19s；
        # 标称 0.25C=22.5s，adjust [-0.05,0.05] → δ 最小 -0.0389 可行。
        plan = {
            "id": "I1_P1", "intersection": "I1", "double_cycle": False,
            "min_phase_s": 0.001,
            "directions": {d: {
                "boundaries": [{"pos_r": 0.25, "adjust": [-0.05, 0.05]}],
                "green_mask": [{"green": True}, {"green": False}]}
                for d in ("up", "down")},
            "hard_constraints": [{
                "id": "b0_cap", "terms": [{
                    "atom": {"kind": "boundary", "intersection": "I1",
                             "plan": "I1_P1", "direction": "up", "index": 0},
                    "coef": 1.0}],
                "sense": "<=", "rhs_ratio": 0.2, "rhs_s": 1.0, "hard": True,
            }],
            "soft_constraints": [],
        }
        problem = ProblemInput.from_obj(_problem([plan, _plan("I2_P1", "I2", [0.5], [True, False])], []))
        from greenwave.preprocess.feasibility import FeasibilityFilter
        ctx = FeasibilityFilter(problem).filter(CYCLE, 1.0)
        self.assertIn("I1_P1", [p.id for p in ctx.feasible_plans["I1"]])
        w = ctx.witness_for("I1_P1")[("up", 0)]
        self.assertLessEqual((0.25 + w) * CYCLE, 19.0 + 1e-6)

    def test_multistage_falls_back_with_aggregates(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "weight": 2.0}])
        raw = _problem(plans, [agg])
        raw["solver"]["multistage"] = {"enabled": True, "fallback_to_full_milp": True}
        out = run(raw)
        # 多阶段不理解聚合，必须回退完整 MILP 才能得到 81。
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 81.0, places=3)

    def test_greedy_runs_with_aggregates(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "weight": 2.0}])
        raw = _problem(plans, [agg])
        raw["solver"]["heuristic"]["enabled"] = True
        raw["solver"]["heuristic"]["refine_rounds"] = 1
        out = run(raw)
        hb = out["grid_results"][0]["heuristic_bound"]
        self.assertIsNotNone(hb)
        self.assertLessEqual(hb, out["grid_results"][0]["composite_best"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
