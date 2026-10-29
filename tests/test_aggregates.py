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


def _three_node_problem(aggregates):
    plans = [_plan("I1_P1", "I1", [0.2], [True, False]),
             _plan("I2_P1", "I2", [0.5], [True, False]),
             _plan("I3_P1", "I3", [0.3], [True, False])]
    raw = {
        "corridor": {"corridor_id": "agg3", "intersections": ["I1", "I2", "I3"],
                     "segments": [_seg("I1", "I2"), _seg("I2", "I3")]},
        "plans": plans,
        "objective": {"band_demands": [
            {"id": "GW12", "type": "green", "direction": "up",
             "nodes": ["I1", "I2"], "weight": 1.0},
            {"id": "GW23", "type": "green", "direction": "up",
             "nodes": ["I2", "I3"], "weight": 1.0}],
            "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [CYCLE], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 30, "mip_rel_gap": 0.0,
                   "num_workers": 1,
                   "heuristic": {"enabled": False, "refine_rounds": 0},
                   "multistage": {"enabled": True, "max_rounds": 0, "restarts": 1}},
        "aggregates": aggregates,
    }
    return raw


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
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 2.0, "direction": "maximize"}])
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
                "contributions": [{"target": "band_objective", "sign": 1, "weight": 1.0, "direction": "maximize"}],
            })

    def test_validation_errors(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        cases = []
        # 未知 plan
        cases.append(_phase_min_agg((0, 1), plan="NOPE",
                                    contributions=[{"target": "band_objective", "sign": 1, "weight": 1.0, "direction": "maximize"}]))
        # boundary 下标越界
        bad = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 1.0, "direction": "maximize"}])
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
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 2.0, "direction": "maximize"}])
        out = run(_problem(plans, [agg]))
        grid = out["grid_results"][0]
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(grid["status"], "optimal")
        # β = 45（I1 窗 54、I2 窗 45）；min 相位 = 18；composite = 45 + 2×18 = 81。
        self.assertAlmostEqual(grid["bandwidth_objective_s"], 45.0, places=3)
        self.assertAlmostEqual(sol["aggregates"]["min_phase"], 18.0, places=3)
        self.assertAlmostEqual(grid["composite_best"], 81.0, places=3)
        self.assertAlmostEqual(out["pareto"]["points"][0]["composite_actual"], 81.0, places=3)

    def test_min_aggregate_reward_offsets_intersection_loss(self):
        # 可调边界 → 聚合不是常数，验证路口分数 R−L 把 t 推到 min(相位)。
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
        contribution = {"target": "intersection_reward", "sign": 1,
                        "weight": 1.0, "direction": "maximize"}
        agg = _phase_min_agg((0, 1), plan="I1_P1", contributions=[contribution])
        problem = ProblemInput.from_obj(_problem(plans, [agg]))
        model, _, sol, objective = _solve_min_loss(problem)
        # 模型最小化 net cost = L − R；解码值与目标一致，且 R = min(相位)。
        self.assertAlmostEqual(objective, sol.loss_s, places=4)
        self.assertAlmostEqual(sol.intersection_reward_s, sol.aggregates["min_phase"],
                               places=4)
        self.assertAlmostEqual(sol.intersection_loss_s, sol.raw_loss_s, places=4)
        self.assertAlmostEqual(sol.intersection_composite_s,
                               sol.intersection_reward_s - sol.intersection_loss_s,
                               places=4)
        self.assertAlmostEqual(sol.intersection_cost_s, -sol.intersection_composite_s,
                               places=4)

    def test_multi_plan_gating(self):
        plans = [_plan("I1_A", "I1", [0.2], [True, False]),
                 _plan("I1_B", "I1", [0.6], [True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        # A 的窗口只有 18s，B 有 54s → B 被选中；A 的聚合应为 0。
        agg = _phase_min_agg((0, 0), plan="I1_A",
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 3.0, "direction": "maximize"}])
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
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 3.0, "direction": "maximize"}])
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
            "contributions": [{"target": "band_objective", "sign": 1, "weight": 1.0, "direction": "maximize"}],
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
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 2.0, "direction": "maximize"}])
        raw = _problem(plans, [agg])
        raw["solver"]["multistage"] = {"enabled": True, "fallback_to_full_milp": True}
        out = run(raw)
        # 多阶段不理解聚合，必须回退完整 MILP 才能得到 81。
        self.assertAlmostEqual(out["grid_results"][0]["composite_best"], 81.0, places=3)

    def test_multistage_keeps_decomposition_for_loss_only_aggregate(self):
        from dataclasses import replace
        from greenwave.multistage import solve_multistage
        from greenwave.preprocess.feasibility import FeasibilityFilter

        agg = {
            "id": "min_phase", "op": "min", "plan": "I1_P1",
            "expressions": [
                [{"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                           "direction": "up", "index": 0}, "coef": 1.0}],
                [{"atom": {"kind": "const", "const": 1.0, "unit": "ratio"}, "coef": 1.0},
                 {"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                           "direction": "up", "index": 0}, "coef": -1.0}],
            ],
            "contributions": [{"target": "intersection_reward", "sign": 1,
                               "weight": 1.0, "direction": "maximize"}],
        }
        problem = ProblemInput.from_obj(_three_node_problem([agg]))
        ctx = FeasibilityFilter(problem).filter(CYCLE, 1.0)
        cfg = replace(problem.solver.multistage, enabled=True, max_rounds=0, restarts=1)
        ms = solve_multistage(problem, ctx, cfg)
        self.assertIsNotNone(ms.solution)
        self.assertIn("语义分解", ms.stage1_paths)
        self.assertAlmostEqual(ms.solution.aggregates["min_phase"], 18.0, places=3)
        self.assertAlmostEqual(ms.solution.intersection_reward_s, 18.0, places=3)
        self.assertAlmostEqual(ms.solution.intersection_cost_s, -18.0, places=3)

    def test_greedy_runs_with_aggregates(self):
        plans = [_plan("I1_P1", "I1", [0.2, 0.6], [True, True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        agg = _phase_min_agg((0, 1), plan="I1_P1",
                             contributions=[{"target": "band_objective", "sign": 1, "weight": 2.0, "direction": "maximize"}])
        raw = _problem(plans, [agg])
        raw["solver"]["heuristic"]["enabled"] = True
        raw["solver"]["heuristic"]["refine_rounds"] = 1
        out = run(raw)
        hb = out["grid_results"][0]["heuristic_bound"]
        self.assertIsNotNone(hb)
        self.assertLessEqual(hb, out["grid_results"][0]["composite_best"] + 1e-6)


if __name__ == "__main__":
    unittest.main()


class TestIntersectionScore(unittest.TestCase):
    def test_plan_reward_and_loss_scores(self):
        soft_r = [{"id": "b0_reward",
                   "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                       "plan": "I1_P1", "direction": "up", "index": 0},
                              "coef": 1.0}],
                   "sense": ">=", "rhs_ratio": 0.1, "hard": False,
                   "soft": {"coef": 1.0, "owner": "intersection_reward"}}]
        soft_l = [{"id": "b0_loss",
                   "terms": [{"atom": {"kind": "boundary", "intersection": "I2",
                                       "plan": "I2_P1", "direction": "up", "index": 0},
                              "coef": 1.0}],
                   "sense": "<=", "rhs_ratio": 0.3, "hard": False,
                   "soft": {"coef": 1.0, "owner": "intersection_loss"}}]
        plans = [_plan("I1_P1", "I1", [0.2], [True, False], soft=soft_r),
                 _plan("I2_P1", "I2", [0.5], [True, False], soft=soft_l)]
        out = run(_problem(plans, []))
        sol = out["pareto"]["points"][0]["solution"]
        # I1 b0=18 → reward 18−9=9；I2 b0=45 → loss 45−27=18。
        self.assertAlmostEqual(sol["intersection_reward_s"], 9.0, places=4)
        self.assertAlmostEqual(sol["intersection_loss_s"], 18.0, places=4)
        self.assertAlmostEqual(sol["intersection_composite_s"], -9.0, places=4)
        self.assertAlmostEqual(sol["intersection_cost_s"], 9.0, places=4)
        self.assertAlmostEqual(sol["loss_s"], 9.0, places=4)
        point = out["pareto"]["points"][0]
        self.assertAlmostEqual(point["intersection_composite"], -9.0, places=4)
        self.assertAlmostEqual(point["intersection_cost"], 9.0, places=4)

    def test_reward_gated_by_plan_selection(self):
        soft_a = [{"id": "a_reward",
                   "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                       "plan": "I1_A", "direction": "up", "index": 0},
                              "coef": 1.0}],
                   "sense": ">=", "rhs_ratio": 0.1, "hard": False,
                   "soft": {"coef": 1.0, "owner": "intersection_reward"}}]
        plans = [_plan("I1_A", "I1", [0.2], [True, False], soft=soft_a),
                 _plan("I1_B", "I1", [0.6], [True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        out = run(_problem(plans, []))
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(sol["plan_selection"]["I1"], "I1_B")
        self.assertAlmostEqual(sol["intersection_reward_s"], 0.0, places=6)
        self.assertAlmostEqual(sol["intersection_composite_s"], 0.0, places=6)

    def test_global_reward_constraint(self):
        plan_i1 = _plan("I1_P1", "I1", [0.2], [True, False])
        plan_i2 = _plan("I2_P1", "I2", [0.5], [True, False])
        raw = _problem([plan_i1, plan_i2], [])
        raw["global_constraints"] = [{
            "id": "bw_reward",
            "terms": [{"atom": {"kind": "bandwidth", "demand": "GW_up"}, "coef": 1.0}],
            "sense": ">=", "rhs_s": 0.0, "hard": False,
            "soft": {"coef": 1.0, "owner": "intersection_reward"},
        }]
        out = run(raw)
        sol = out["pareto"]["points"][0]["solution"]
        self.assertAlmostEqual(sol["intersection_reward_s"], sol["band_reward_s"], places=4)

    def test_reward_owner_validation(self):
        from greenwave.schema.models import SoftPenalty
        with self.assertRaisesRegex(ValueError, "owner"):
            SoftPenalty.from_obj({"coef": 1.0, "owner": "composite"})
        # equality reward 校验报 V6
        soft = [{"id": "eq_reward",
                 "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                     "plan": "I1_P1", "direction": "up", "index": 0},
                            "coef": 1.0}],
                 "sense": "==", "rhs_s": 10.0, "hard": False,
                 "soft": {"coef": 1.0, "owner": "intersection_reward"}}]
        p = ProblemInput.from_obj(_problem(
            [_plan("I1_P1", "I1", [0.2], [True, False], soft=soft),
             _plan("I2_P1", "I2", [0.5], [True, False])], []))
        self.assertIn("V6", [i.code for i in p.validate_all()])

    def test_aggregate_contribution_legality(self):
        from greenwave.schema.models import AggregateSpec

        def agg(op, contribution):
            return {
                "id": "x", "op": op, "plan": "I1_P1",
                "expressions": [[{"atom": {"kind": "const", "const": 1.0}, "coef": 1.0}]],
                "contributions": [contribution],
            }

        legal = [
            ("min", {"target": "band_objective", "sign": 1, "weight": 1.0,
                     "direction": "maximize"}),
            ("min", {"target": "intersection_reward", "sign": 1, "weight": 1.0,
                     "direction": "maximize"}),
            ("max", {"target": "band_objective", "sign": -1, "weight": 1.0,
                     "direction": "minimize"}),
            ("max", {"target": "intersection_loss", "sign": 1, "weight": 1.0,
                     "direction": "minimize"}),
        ]
        for op, c in legal:
            self.assertEqual(AggregateSpec.from_obj(agg(op, c)).op, op)
        illegal = [
            ("max", {"target": "intersection_loss", "sign": 1, "weight": 1.0,
                     "direction": "maximize"}),
            ("min", {"target": "intersection_loss", "sign": 1, "weight": 1.0,
                     "direction": "minimize"}),
            ("max", {"target": "band_objective", "sign": 1, "weight": 1.0,
                     "direction": "minimize"}),
            ("min", {"target": "intersection_reward", "sign": -1, "weight": 1.0,
                     "direction": "maximize"}),
        ]
        for op, c in illegal:
            with self.assertRaises(ValueError):
                AggregateSpec.from_obj(agg(op, c))
        with self.assertRaisesRegex(ValueError, "sign"):
            AggregateSpec.from_obj(agg("min", {"target": "band_objective",
                                               "weight": 1.0,
                                               "direction": "maximize"}))
