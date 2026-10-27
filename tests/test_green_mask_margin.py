"""Per-green-segment margin: schema, window lookup, MILP behaviour, warnings."""
from __future__ import annotations

import unittest

from greenwave import run
from greenwave.schema.models import (GreenSegmentSpec, ProblemInput)

CYCLE = 90.0


def _seg(from_, to, dist=300.0, v=15.0):
    return {"from": from_, "to": to, "distance_m": dist,
            "speed_up_mps": v, "speed_down_mps": v}


def _margin(delta=3.0, coef=0.5, hard=0.0):
    return {"delta_min_s": delta, "coef": coef, "hard_min_s": hard}


def _plan(iid, mask, margins=None, *, double=False, boundaries=(0.5,), pid=None):
    """`mask` 是对象列表；`margins` = (segment_index, side, spec) 的列表。"""
    segs = [dict(seg) for seg in mask]
    for si, side, spec in (margins or ()):
        segs[si].setdefault("margin", {})[side] = spec
    bounds = [{"pos_r": b} for b in boundaries]
    return {"id": pid or f"{iid}_P1", "intersection": iid, "double_cycle": double,
            "min_phase_s": 0.001,
            "directions": {
                "up": {"boundaries": [dict(b) for b in bounds],
                       "green_mask": [dict(x) for x in segs]},
                "down": {"boundaries": [dict(b) for b in bounds],
                         "green_mask": [dict(x) for x in segs]},
            }}


def _problem(plans, demand_nodes=("I1", "I2"), hard_constraints=()):
    return {
        "corridor": {"corridor_id": "mask_margin", "intersections": ["I1", "I2"],
                     "segments": [_seg("I1", "I2")]},
        "plans": plans,
        "objective": {"band_demands": [{
            "id": "GW_up", "type": "green", "direction": "up",
            "nodes": list(demand_nodes), "weight": 1.0}],
            "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [CYCLE], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 30, "mip_rel_gap": 0.0,
                   "num_workers": 1, "heuristic": {"enabled": False,
                                                   "refine_rounds": 0}},
        "global_constraints": list(hard_constraints),
    }


class TestGreenMaskMarginSchema(unittest.TestCase):
    def test_window_margins_use_first_and_last_green_segment(self):
        mask = [{"green": True}, {"green": True}, {"green": False}]
        p = _plan("I1", mask, margins=[
            (0, "start", _margin(5.0)),
            (0, "end", _margin(99.0)),      # 同色内部边界：不生效
            (1, "start", _margin(88.0)),    # 同色内部边界：不生效
            (1, "end", _margin(2.5, coef=2.0)),
            (2, "start", _margin(77.0)),    # 红段：不生效
        ], boundaries=(0.3, 0.6))
        problem = ProblemInput.from_obj(_problem([p, _plan("I2", mask, boundaries=(0.3, 0.6))]))
        plan = problem.plan("I1_P1")
        ms, me = plan.window_margins("up", 0)
        self.assertAlmostEqual(ms.delta_min_s, 5.0)
        self.assertAlmostEqual(me.delta_min_s, 2.5)
        self.assertAlmostEqual(me.coef, 2.0)

    def test_double_cycle_copies_share_margin(self):
        p = _plan("I1", [{"green": True}, {"green": False}],
                  margins=[(0, "start", _margin(4.0)), (0, "end", _margin(6.0))],
                  double=True, boundaries=(0.2,))
        problem = ProblemInput.from_obj(_problem([p, _plan("I2", [{"green": True}, {"green": False}])]))
        plan = problem.plan("I1_P1")
        self.assertEqual(plan.window_count("up"), 2)
        for k in (0, 1):
            ms, me = plan.window_margins("up", k)
            self.assertAlmostEqual(ms.delta_min_s, 4.0)
            self.assertAlmostEqual(me.delta_min_s, 6.0)

    def test_demand_level_margin_is_rejected(self):
        raw = _problem([_plan("I1", [{"green": True}, {"green": False}]),
                        _plan("I2", [{"green": True}, {"green": False}])])
        raw["objective"]["band_demands"][0]["margin"] = {"delta_min_s": 1.0}
        with self.assertRaisesRegex(ValueError, "green segment"):
            ProblemInput.from_obj(raw)
        raw = _problem([_plan("I1", [{"green": True}, {"green": False}]),
                        _plan("I2", [{"green": True}, {"green": False}])])
        raw["objective"]["margin_default"] = {"delta_min_s": 1.0}
        with self.assertRaisesRegex(ValueError, "margin_default"):
            ProblemInput.from_obj(raw)

    def test_ignored_margins_are_warned(self):
        mask = [{"green": True}, {"green": True}, {"green": False}]
        p = _plan("I1", mask, margins=[
            (0, "end", _margin(99.0)),
            (1, "start", _margin(88.0)),
            (2, "start", _margin(77.0)),
        ], boundaries=(0.3, 0.6))
        q = _plan("I2", [{"green": True}, {"green": False}])
        problem = ProblemInput.from_obj(_problem([p, q]))
        msgs = [i.message for i in problem.validate_all()]
        self.assertTrue(any("same-color boundary" in m for m in msgs))
        self.assertTrue(any("is red; its margin is ignored" in m for m in msgs))
        # 附带：未配置 margin 的窗口就是没有约束/损失。
        self.assertEqual(problem.plan("I1_P1").window_margins("up", 0), (None, None))


class TestGreenMaskMarginMilp(unittest.TestCase):
    def _solve(self, plans):
        return run(_problem(plans))

    def test_soft_margin_is_paid_when_band_fills_window(self):
        plans = [_plan("I1", [{"green": True}, {"green": False}],
                       margins=[(0, "start", _margin(3.0, coef=0.2)),
                                (0, "end", _margin(3.0, coef=0.2))]),
                 _plan("I2", [{"green": True}, {"green": False}],
                       margins=[(0, "start", _margin(3.0, coef=0.2)),
                                (0, "end", _margin(3.0, coef=0.2))])]
        out = self._solve(plans)
        grid = out["grid_results"][0]
        sol = out["pareto"]["points"][0]["solution"]
        self.assertEqual(grid["status"], "optimal")
        self.assertAlmostEqual(grid["bandwidth_objective_s"], 45.0, places=3)
        # 带宽满窗 → 两个节点 × 两侧各缺 3s，罚 4 × 3 × 0.2 = 2.4。
        self.assertAlmostEqual(sol["margin_loss_s"], 2.4, places=3)
        self.assertAlmostEqual(grid["composite_best"], 42.6, places=3)

    def test_hard_margin_caps_bandwidth(self):
        plans = [_plan("I1", [{"green": True}, {"green": False}],
                       margins=[(0, "start", _margin(3.0, coef=0.5, hard=3.0)),
                                (0, "end", _margin(3.0, coef=0.5, hard=3.0))]),
                 _plan("I2", [{"green": True}, {"green": False}],
                       margins=[(0, "start", _margin(3.0, coef=0.5, hard=3.0)),
                                (0, "end", _margin(3.0, coef=0.5, hard=3.0))])]
        out = self._solve(plans)
        grid = out["grid_results"][0]
        sol = out["pareto"]["points"][0]["solution"]
        self.assertAlmostEqual(grid["bandwidth_objective_s"], 39.0, places=3)
        self.assertAlmostEqual(sol["margin_loss_s"], 0.0, places=4)

    def test_hard_margin_makes_case_infeasible(self):
        w = (20.0 / CYCLE,)
        plans = [_plan("I1", [{"green": True}, {"green": False}],
                       margins=[(0, "start", _margin(0.0, coef=0.0, hard=12.0)),
                                (0, "end", _margin(0.0, coef=0.0, hard=12.0))],
                       boundaries=w),
                 _plan("I2", [{"green": True}, {"green": False}],
                       margins=[(0, "start", _margin(0.0, coef=0.0, hard=12.0)),
                                (0, "end", _margin(0.0, coef=0.0, hard=12.0))],
                       boundaries=w)]
        # 窗宽 20s，两方向各留 12s 不可能；再强制带宽 >= 1s → 不可行。
        raw = _problem(plans)
        raw["global_constraints"] = [{
            "id": "force_band", "terms": [{"atom": {"kind": "bandwidth",
                                                    "demand": "GW_up"}, "coef": 1.0}],
            "sense": ">=", "rhs": 1.0, "hard": True}]
        out = run(raw)
        self.assertEqual(out["grid_results"][0]["status"], "infeasible")

    def test_multi_plan_margins_are_plan_conditioned(self):
        """两个方案同路口、margin 系数不同：slack 变量必须按方案分开。"""
        from greenwave.model.builder import ModelBuilder
        from greenwave.preprocess.feasibility import FeasibilityFilter
        raw = _problem([_plan("I1", [{"green": True}, {"green": False}],
                              margins=[(0, "start", _margin(1.0, coef=1.0)),
                                       (0, "end", _margin(1.0, coef=1.0))], pid="I1_A"),
                        _plan("I1", [{"green": True}, {"green": False}],
                              margins=[(0, "start", _margin(2.0, coef=2.0)),
                                       (0, "end", _margin(2.0, coef=2.0))], pid="I1_B"),
                        _plan("I2", [{"green": True}, {"green": False}])])
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(CYCLE, 1.0)
        model = ModelBuilder(problem, ctx).build()
        margin_vars = [n for n in model.vars.names
                       if n.startswith(("gw_margin_lo", "gw_margin_hi"))]
        self.assertTrue(any("I1_A" in n for n in margin_vars))
        self.assertTrue(any("I1_B" in n for n in margin_vars))
        self.assertEqual(len(margin_vars), 4)   # 1 槽 × 2 侧 × 2 方案（I2 未配 margin）
        # 目标里两项系数不同（1.0 / 2.0），确认没有把两个方案的 coef 混在一起。
        ref = model.meta.margins[0]
        self.assertEqual(sorted(c for _, c in ref.slacks), [1.0, 1.0, 2.0, 2.0])
        # 求解一次，确认解码复合目标 == 求解器目标（方案条件化没有写错）。
        from greenwave.report.solution import SolutionDecoder
        from greenwave.solve import make_backend
        from greenwave.solve.backend_base import SolveOptions
        backend = make_backend("scipy")
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        res = backend.solve(session, SolveOptions(time_limit_s=20.0, mip_rel_gap=0.0))
        self.assertEqual(res.status.value, "optimal")
        sol = SolutionDecoder(model).decode(res.values)
        self.assertAlmostEqual(res.objective, sol.composite_s, places=5)


if __name__ == "__main__":
    unittest.main()
