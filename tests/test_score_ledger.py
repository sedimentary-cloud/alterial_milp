"""四账本（band/intersection × reward/loss）与报告 JSON 的回归测试。"""
from __future__ import annotations

import unittest

from greenwave import run
from greenwave.model.builder import ModelBuilder
from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.report.solution import SolutionDecoder
from greenwave.schema.models import ProblemInput
from greenwave.solve import make_backend
from greenwave.solve.backend_base import SolveOptions

CYCLE = 90.0


def _seg(a, b, dist=300.0, v=15.0):
    return {"from": a, "to": b, "distance_m": dist,
            "speed_up_mps": v, "speed_down_mps": v}


def _plan(pid, iid, boundaries, mask, *, soft=(), double=False):
    return {"id": pid, "intersection": iid, "double_cycle": double,
            "min_phase_s": 0.001,
            "directions": {d: {"boundaries": [{"pos_r": x} for x in boundaries],
                               "green_mask": [{"green": bool(g)} for g in mask]}
                           for d in ("up", "down")},
            "hard_constraints": [], "soft_constraints": list(soft)}


def _problem(plans, *, demands, corridor=None, cycle=CYCLE, aggregates=()):
    return {
        "corridor": corridor or {"corridor_id": "ledger", "intersections": ["I1", "I2"],
                                 "segments": [_seg("I1", "I2")]},
        "plans": plans,
        "objective": {"band_demands": list(demands),
                      "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [cycle], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                   "num_workers": 1,
                   "heuristic": {"enabled": False, "refine_rounds": 0}},
        "aggregates": list(aggregates),
    }


def _rich_band_case():
    soft_band = [{"id": "b0_cap",
                  "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                      "plan": "I1_P1", "direction": "up", "index": 0},
                             "coef": 1.0}],
                  "sense": "<=", "rhs_ratio": 0.1, "hard": False,
                  "soft": {"coef": 1.0, "owner": "band_loss"}}]
    plan_i1 = {
        "id": "I1_P1", "intersection": "I1", "double_cycle": False,
        "min_phase_s": 0.001,
        "directions": {d: {
            "boundaries": [{"pos_r": 0.2}, {"pos_r": 0.6}],
            "green_mask": [
                {"green": True, "margin": {"start": {"delta_min_s": 3.0, "coef": 0.2}}},
                {"green": True, "margin": {"end": {"delta_min_s": 3.0, "coef": 0.2}}},
                {"green": False},
            ]} for d in ("up", "down")},
        "hard_constraints": [], "soft_constraints": soft_band,
    }
    exprs = [
        [{"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                   "direction": "up", "index": 0}, "coef": 1.0}],
        [{"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                   "direction": "up", "index": 1}, "coef": 1.0},
         {"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                   "direction": "up", "index": 0}, "coef": -1.0}],
        [{"atom": {"kind": "const", "const": 1.0, "unit": "ratio"}, "coef": 1.0},
         {"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                   "direction": "up", "index": 1}, "coef": -1.0}],
    ]
    agg = [
        {"id": "amin", "op": "min", "plan": "I1_P1", "expressions": exprs,
         "contributions": [{"target": "band_objective", "sign": 1,
                            "weight": 1.0, "direction": "maximize"}]},
        {"id": "amax", "op": "max", "plan": "I1_P1", "expressions": exprs,
         "contributions": [{"target": "band_objective", "sign": -1,
                            "weight": 0.5, "direction": "minimize"}]},
    ]
    raw = _problem(
        [plan_i1, _plan("I2_P1", "I2", [0.5], [True, False])],
        demands=[{"id": "GW_up", "type": "green", "direction": "up",
                  "nodes": ["I1", "I2"], "weight": 1.0}],
        aggregates=agg,
    )
    raw["objective"]["band_demands"][0]["green_boundary"] = {
        "start": {"free_s": 0.0, "coef": 0.1},
        "end": {"free_s": 0.0, "coef": 0.1},
    }
    return raw


def _red_edge_case():
    plans = [_plan("I1_P1", "I1", [0.45], [True, False]),
             _plan("I2_P1", "I2", [0.8], [True, False])]
    raw = _problem(
        plans,
        demands=[{"id": "RW_up", "type": "red", "direction": "up",
                  "nodes": ["I1", "I2"], "max_bands": 1, "weight": 3.0}],
        corridor={"corridor_id": "red_ledger", "intersections": ["I1", "I2"],
                  "segments": [_seg("I1", "I2", dist=200.0, v=10.0)]},
    )
    raw["objective"]["red_boundary_default"] = {
        "start": {"free_s": 0.0, "coef": 1.0},
        "end": {"free_s": 0.0, "coef": 1.0},
    }
    return raw


def _decode_stage1(raw):
    problem = ProblemInput.from_obj(raw)
    ctx = FeasibilityFilter(problem).filter(CYCLE, 1.0)
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend("scipy")
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(20.0, 0.0))
    return problem, model, SolutionDecoder(model).decode(res.values), float(res.objective)


class TestScoreLedger(unittest.TestCase):
    def _assert_ledger(self, ledger):
        self.assertAlmostEqual(ledger["reward_s"],
                               sum(ledger["reward_breakdown"].values()), places=6)
        self.assertAlmostEqual(ledger["loss_s"],
                               sum(ledger["loss_breakdown"].values()), places=6)
        self.assertAlmostEqual(ledger["composite_s"],
                               ledger["reward_s"] - ledger["loss_s"], places=6)

    def test_band_ledger_breakdown_and_json_shape(self):
        out = run(_rich_band_case())
        grid = out["grid_results"][0]
        point = out["pareto"]["points"][0] if out["pareto"]["points"] else None
        self.assertIsNotNone(point)
        sol = point["solution"]
        band = sol["score"]["band"]
        inter = sol["score"]["intersection"]

        self.assertEqual(set(band["reward_breakdown"]),
                         {"bandwidth", "balanced", "aggregate"})
        self.assertEqual(set(band["loss_breakdown"]),
                         {"margin", "green_edge", "red_edge", "soft",
                          "balanced_min_existing", "aggregate"})
        self.assertEqual(set(inter["reward_breakdown"]), {"soft", "aggregate"})
        self.assertEqual(set(inter["loss_breakdown"]), {"soft", "aggregate"})
        for ledger in (band, inter):
            self._assert_ledger(ledger)

        # 带宽 + aggregate 回报 = 45 + 18；带损失 = 软约束 9 + 绿边 0.9 + max 聚合 18。
        self.assertAlmostEqual(band["reward_breakdown"]["bandwidth"], 45.0, places=4)
        self.assertAlmostEqual(band["reward_breakdown"]["aggregate"], 18.0, places=4)
        self.assertAlmostEqual(band["loss_breakdown"]["soft"], 9.0, places=4)
        self.assertAlmostEqual(band["loss_breakdown"]["green_edge"], 0.9, places=4)
        self.assertAlmostEqual(band["loss_breakdown"]["aggregate"], 18.0, places=4)
        self.assertAlmostEqual(band["reward_s"], 63.0, places=4)
        self.assertAlmostEqual(band["loss_s"], 27.9, places=4)
        self.assertAlmostEqual(band["composite_s"], 35.1, places=4)
        self.assertAlmostEqual(grid["score"]["band"]["composite_s"],
                               band["composite_s"], places=4)

        # 顶层只保留两个别名；旧的扁平字段不再出现。
        self.assertAlmostEqual(sol["composite_s"], band["composite_s"], places=4)
        self.assertAlmostEqual(sol["loss_s"], inter["cost_s"], places=4)
        for old in ("band_reward_s", "band_loss_s", "margin_loss_s",
                    "green_edge_loss_s", "intersection_reward_s",
                    "intersection_loss_s", "raw_intersection_loss_s",
                    "aggregate_band_reward_s", "aggregate_loss_s"):
            self.assertNotIn(old, sol)
        self.assertIn("score", grid)
        self.assertNotIn("bandwidth_objective_s", grid)
        self.assertNotIn("negative_intersection_loss_s", grid)
        self.assertIn("score", point)
        self.assertNotIn("intersection_loss", point)

    def test_heatmap_reader_understands_nested_score(self):
        from greenwave.report.heatmap import _extract_rows
        out = run(_rich_band_case())
        rows = _extract_rows(out)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["bandwidth_objective_s"], 45.0, places=4)
        self.assertAlmostEqual(row["negative_margin_loss_s"], 0.0, places=4)
        self.assertAlmostEqual(row["composite_objective_s"], 35.1, places=4)

    def test_legacy_properties_and_green_edge_end_travel(self):
        raw = _rich_band_case()
        problem, model, sol, objective = _decode_stage1(raw)
        self.assertAlmostEqual(sol.composite_s, objective, places=4)
        self.assertAlmostEqual(sol.band_reward_s, sol.band.reward_s, places=6)
        self.assertAlmostEqual(sol.bandwidth_reward_s, 45.0, places=4)
        self.assertAlmostEqual(sol.band_loss_s, sol.band.loss_s, places=6)
        self.assertAlmostEqual(sol.red_edge_loss_s, 0.0, places=6)
        # C8_EDGE end 必须带上下游行 T；否则 I2 会出现 20s 的假罚。
        self.assertAlmostEqual(sol.green_edge_loss_s, 0.9, places=4)

    def test_balanced_reward_and_min_existing_loss_split(self):
        plans = [_plan("I1_P1", "I1", [0.2], [True, False]),
                 _plan("I2_P1", "I2", [0.5], [True, False])]
        raw = _problem(
            plans,
            demands=[
                {"id": "GW_up", "type": "green", "direction": "up",
                 "nodes": ["I1", "I2"], "weight": 1.0},
                {"id": "GW_down", "type": "green", "direction": "down",
                 "nodes": ["I2", "I1"], "weight": 0.0},
            ],
        )
        raw["objective"]["balanced_groups"] = [{
            "id": "BG", "demands": ["GW_up", "GW_down"], "weight": 1.0,
            "on_infeasible": "skip", "min_existing": 2,
            "min_existing_penalty_s": 5.0,
        }]
        _, _, sol, objective = _decode_stage1(raw)
        self.assertAlmostEqual(sol.composite_s, objective, places=4)
        # 只有 up 带存在；均衡组 skip + min_existing=2 → 缺 1 个，罚 5s。
        self.assertAlmostEqual(sol.band.reward_breakdown["bandwidth"], 18.0, places=3)
        self.assertAlmostEqual(sol.band.reward_breakdown["balanced"], 18.0, places=3)
        self.assertAlmostEqual(sol.balanced_min_existing_loss_s, 5.0, places=3)
        self.assertAlmostEqual(sol.band.reward_s, 36.0, places=3)
        self.assertAlmostEqual(sol.band.loss_s, 5.0, places=3)
        self.assertAlmostEqual(sol.band.composite_s, 31.0, places=3)
        self.assertAlmostEqual(sol.band.reward_s,
                               sum(sol.band.reward_breakdown.values()), places=6)
        self.assertAlmostEqual(sol.band.loss_s,
                               sum(sol.band.loss_breakdown.values()), places=6)

    def test_intersection_ledger_breakdown(self):
        soft_r = [{"id": "i1_reward",
                   "terms": [{"atom": {"kind": "boundary", "intersection": "I1",
                                       "plan": "I1_P1", "direction": "up", "index": 0},
                              "coef": 1.0}],
                   "sense": ">=", "rhs_ratio": 0.1, "hard": False,
                   "soft": {"coef": 1.0, "owner": "intersection_reward"}}]
        soft_l = [{"id": "i2_loss",
                   "terms": [{"atom": {"kind": "boundary", "intersection": "I2",
                                       "plan": "I2_P1", "direction": "up", "index": 0},
                              "coef": 1.0}],
                   "sense": "<=", "rhs_ratio": 0.3, "hard": False,
                   "soft": {"coef": 1.0, "owner": "intersection_loss"}}]
        plans = [_plan("I1_P1", "I1", [0.2], [True, False], soft=soft_r),
                 _plan("I2_P1", "I2", [0.5], [True, False], soft=soft_l)]
        exprs_i1 = [
            [{"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                       "direction": "up", "index": 0}, "coef": 1.0}],
            [{"atom": {"kind": "const", "const": 1.0, "unit": "ratio"}, "coef": 1.0},
             {"atom": {"kind": "boundary", "intersection": "I1", "plan": "I1_P1",
                       "direction": "up", "index": 0}, "coef": -1.0}],
        ]
        exprs_i2 = [
            [{"atom": {"kind": "boundary", "intersection": "I2", "plan": "I2_P1",
                       "direction": "up", "index": 0}, "coef": 1.0}],
            [{"atom": {"kind": "const", "const": 1.0, "unit": "ratio"}, "coef": 1.0},
             {"atom": {"kind": "boundary", "intersection": "I2", "plan": "I2_P1",
                       "direction": "up", "index": 0}, "coef": -1.0}],
        ]
        aggs = [
            {"id": "i1_min", "op": "min", "plan": "I1_P1", "expressions": exprs_i1,
             "contributions": [{"target": "intersection_reward", "sign": 1,
                                "weight": 1.0, "direction": "maximize"}]},
            {"id": "i2_max", "op": "max", "plan": "I2_P1", "expressions": exprs_i2,
             "contributions": [{"target": "intersection_loss", "sign": 1,
                                "weight": 0.5, "direction": "minimize"}]},
        ]
        raw = _problem(
            plans,
            demands=[{"id": "GW_up", "type": "green", "direction": "up",
                      "nodes": ["I1", "I2"], "weight": 1.0}],
            aggregates=aggs,
        )
        _, _, sol, _ = _decode_stage1(raw)
        inter = sol.intersection
        self.assertAlmostEqual(inter.reward_breakdown["soft"], 9.0, places=4)
        self.assertAlmostEqual(inter.reward_breakdown["aggregate"], 18.0, places=4)
        self.assertAlmostEqual(inter.loss_breakdown["soft"], 18.0, places=4)
        self.assertAlmostEqual(inter.loss_breakdown["aggregate"], 22.5, places=4)
        self.assertAlmostEqual(inter.reward_s, 27.0, places=4)
        self.assertAlmostEqual(inter.loss_s, 40.5, places=4)
        self.assertAlmostEqual(inter.composite_s, -13.5, places=4)
        self.assertAlmostEqual(inter.cost_s, 13.5, places=4)
        self.assertAlmostEqual(inter.reward_s, sum(inter.reward_breakdown.values()), places=6)
        self.assertAlmostEqual(inter.loss_s, sum(inter.loss_breakdown.values()), places=6)
        self.assertAlmostEqual(inter.composite_s, inter.reward_s - inter.loss_s, places=6)
        self.assertAlmostEqual(sol.intersection_reward_s, 27.0, places=4)
        self.assertAlmostEqual(sol.intersection_loss_s, 40.5, places=4)
        self.assertAlmostEqual(sol.loss_s, 13.5, places=4)

    def test_red_edge_geometric_recompute(self):
        raw = _red_edge_case()
        problem, model, sol, objective = _decode_stage1(raw)
        self.assertAlmostEqual(sol.composite_s, objective, places=4)
        band = sol.bands[0]
        up_node = "I1"
        tau = 20.0
        plan_id = sol.plan_selection[up_node]
        s_r, e_r = sol.derived_windows[(plan_id, "up")][0]
        phi = sol.offsets_s[up_node]
        m = band.red_up_wrap
        gap_start = band.start_s - (phi + s_r * CYCLE + tau) + m * CYCLE
        gap_end = (phi + e_r * CYCLE + tau) - (band.start_s + band.width_s) + m * CYCLE
        expected = max(0.0, gap_start) + max(0.0, gap_end)
        self.assertAlmostEqual(sol.red_edge_loss_s, expected, places=4)
        self.assertAlmostEqual(sol.band.loss_breakdown["red_edge"], expected, places=4)
        self.assertGreater(expected, 0.0)
        self.assertAlmostEqual(
            sol.band.loss_s, sum(sol.band.loss_breakdown.values()), places=4)
        self.assertAlmostEqual(
            sol.band.composite_s, sol.band.reward_s - sol.band.loss_s, places=4)


if __name__ == "__main__":
    unittest.main()
