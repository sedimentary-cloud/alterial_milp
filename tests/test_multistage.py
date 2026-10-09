"""多阶段快速优化：配置、先验提取、固定组合 LP、pipeline 集成测试。"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenwave import ProblemInput, run, run_multistage  # noqa: E402
from greenwave.heuristic.greedy import GreedyConstructor  # noqa: E402
from greenwave.model.builder import ArterialModel, ModelBuilder  # noqa: E402
from greenwave.model.constraints import ConstraintMeta, ConstraintStore  # noqa: E402
from greenwave.model.expressions import LinearExpr  # noqa: E402
from greenwave.model.variables import GVarType, VariableRegistry  # noqa: E402
from greenwave.multistage import (extract_prior, find_stage1_components, freeze_combo,  # noqa: E402
                                  validate_prior, unfixed_integrality,
                                  solve_fixed_boundaries, solve_fixed_combo_lp,
                                  solve_multistage)
from greenwave.preprocess.feasibility import FeasibilityFilter  # noqa: E402
from greenwave.report.solution import SolutionDecoder  # noqa: E402
from greenwave.schema.models import (BandDemandSpec, MultistageConfig,  # noqa: E402
                                     PlanSpec, SolverConfig)
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402
from greenwave.stage1_decompose import (DemandGroup, plan_dp_candidate,  # noqa: E402
                                        semantic_demand_groups)


def _segment(a: str, b: str, distance_m: float = 500.0,
             speed_mps: float = 10.0) -> dict:
    return {"from": a, "to": b, "distance_m": distance_m,
            "speed_up_mps": speed_mps, "speed_down_mps": speed_mps}


def _two_window_problem(*, adjustable: bool = False, multistage: bool = False) -> dict:
    """3 路口、上行链式绿波、每个路口两个绿窗的小案例。"""
    nodes = ["I1", "I2", "I3"]
    adj = [0.0, 0.1] if adjustable else None
    plans = []
    for iid in nodes:
        b0 = {"pos_r": 0.2}
        b1 = {"pos_r": 0.4}
        b2 = {"pos_r": 0.8}
        if adj is not None:
            for b in (b0, b1, b2):
                b["adjust"] = list(adj)
        plans.append({
            "id": f"{iid}_P1", "intersection": iid, "double_cycle": False,
            "min_phase_s": 1e-3,
            "directions": {
                "up": {"boundaries": [b0, b1, b2],
                       "green_mask": [True, False, True, False]},
                "down": {"boundaries": [{"pos_r": 0.5}],
                         "green_mask": [True, False]},
            },
        })
    return {
        "corridor": {
            "corridor_id": "ms_test",
            "intersections": nodes,
            "segments": [_segment("I1", "I2"), _segment("I2", "I3")],
        },
        "plans": plans,
        "objective": {
            "band_demands": [{
                "id": "GW_up", "type": "green", "direction": "up",
                "nodes": nodes, "max_bands": 2, "weight": 1.0,
                "margin": {"delta_min_s": 0.0, "coef": 1.0},
            }],
            "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
            "allow_window_sharing": True,
            "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1},
        },
        "global_constraints": [],
        "constraint_templates": [],
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {
            "backend": "scipy", "time_limit_s": 20.0, "mip_rel_gap": 0.0,
            "num_workers": 1,
            "heuristic": {"enabled": False, "refine_rounds": 0},
            "multistage": {"enabled": bool(multistage), "max_rounds": 2,
                           "restarts": 1},
        },
    }


def _independent_red_problem() -> dict:
    """两个互不相邻的红波对，用于验证 Stage-1 自动分解。"""
    nodes = ["I1", "I2", "I3", "I4", "I5"]
    plans = []
    for iid in nodes:
        plans.append({
            "id": f"{iid}_P1", "intersection": iid, "double_cycle": False,
            "min_phase_s": 1e-3,
            "directions": {
                "up": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]},
                "down": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]},
            },
        })
    return {
        "corridor": {
            "corridor_id": "indep_red",
            "intersections": nodes,
            "segments": [_segment(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)],
        },
        "plans": plans,
        "objective": {
            "band_demands": [
                {"id": "R12", "type": "red", "direction": "up",
                 "nodes": ["I1", "I2"], "max_bands": 1, "weight": 1.0},
                {"id": "R45", "type": "red", "direction": "up",
                 "nodes": ["I4", "I5"], "max_bands": 1, "weight": 1.0},
            ],
            "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
            "allow_window_sharing": False,
            "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1},
        },
        "global_constraints": [],
        "constraint_templates": [],
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {
            "backend": "scipy", "time_limit_s": 20.0, "mip_rel_gap": 0.0,
            "num_workers": 1,
            "heuristic": {"enabled": False, "refine_rounds": 0},
            "multistage": {"enabled": True, "decompose_stage1": True},
        },
    }


def _witness_required_problem() -> dict:
    """可调边界必须取非零 witness 才能满足方案硬约束的案例。"""
    nodes = ["I1", "I2"]
    plans = []
    for iid in nodes:
        pid = f"{iid}_P1"
        plans.append({
            "id": pid, "intersection": iid, "double_cycle": False,
            "min_phase_s": 1e-3,
            "directions": {
                "up": {"boundaries": [{"pos_r": 0.4, "adjust": [0.0, 0.2]}],
                       "green_mask": [True, False]},
                "down": {"boundaries": [{"pos_r": 0.5}],
                         "green_mask": [True, False]},
            },
            "hard_constraints": [{
                "id": f"{pid}_up_min",
                "terms": [{"atom": {"kind": "boundary", "intersection": iid,
                                    "plan": pid, "direction": "up", "index": 0},
                           "coef": 1.0}],
                "sense": ">=", "rhs_ratio": 0.5, "hard": True,
            }],
        })
    return {
        "corridor": {"corridor_id": "witness_req", "intersections": nodes,
                     "segments": [_segment("I1", "I2")]},
        "plans": plans,
        "objective": {
            "band_demands": [{"id": "GW_up", "type": "green", "direction": "up",
                              "nodes": nodes, "max_bands": 1, "weight": 1.0,
                              "margin": {"delta_min_s": 0.0, "coef": 1.0}}],
            "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
            "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1},
        },
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {
            "backend": "scipy", "time_limit_s": 20.0, "mip_rel_gap": 0.0,
            "num_workers": 1,
            "heuristic": {"enabled": False, "refine_rounds": 0},
            "multistage": {"enabled": True, "max_rounds": 2, "restarts": 1},
        },
    }


def _solve_full(problem, ctx):
    model = ModelBuilder(problem, ctx).build()
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(20.0, 0.0))
    if res.values is None:
        raise AssertionError(f"full MILP failed: {res.status} {res.message}")
    return model, res, SolutionDecoder(model).decode(res.values)


class TestMultistageConfig(unittest.TestCase):
    def test_default_is_disabled(self):
        cfg = SolverConfig.from_obj({})
        self.assertFalse(cfg.multistage.enabled)
        self.assertEqual(cfg.multistage.max_rounds, 3)

    def test_parse_enabled_config(self):
        cfg = SolverConfig.from_obj({
            "multistage": {"enabled": True, "max_rounds": 5,
                           "restarts": 2, "improvement_tol_s": 0.01},
        })
        self.assertTrue(cfg.multistage.enabled)
        self.assertEqual(cfg.multistage.max_rounds, 5)
        self.assertEqual(cfg.multistage.restarts, 2)
        self.assertAlmostEqual(cfg.multistage.improvement_tol_s, 0.01)
        self.assertTrue(cfg.multistage.decompose_stage1)
        self.assertEqual(cfg.multistage.decompose_min_component_size, 2)


class TestPriorExtraction(unittest.TestCase):
    def test_greedy_and_decoder_slots_are_one_based(self):
        raw = _two_window_problem(multistage=False)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)

        # Decoder 输出必须带 wraps，且窗口槽是 1-based。
        _model, _res, sol = _solve_full(problem, ctx)
        self.assertTrue(sol.bands)
        for band in sol.bands:
            demand = problem.demand(band.demand)
            if demand.type != "green":
                continue
            for node in demand.nodes:
                self.assertGreaterEqual(band.windows[node], 1)
                self.assertIn(node, band.wraps)

        # 贪心构造器也必须输出 1-based 槽位和 wraps。
        greedy = GreedyConstructor(problem, ctx).run()
        self.assertTrue(greedy.bands)
        for band in greedy.bands:
            demand = problem.demand(band.demand)
            if demand.type != "green":
                continue
            for node in demand.nodes:
                self.assertGreaterEqual(band.windows[node], 1)
                self.assertIn(node, band.wraps)

    def test_multi_plan_selection_is_frozen(self):
        import copy
        raw = _two_window_problem(multistage=False)
        extra = []
        for plan in raw["plans"]:
            clone = copy.deepcopy(plan)
            clone["id"] = plan["id"] + "_B"
            extra.append(clone)
        raw["plans"].extend(extra)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model, _res, sol = _solve_full(problem, ctx)
        self.assertTrue(model.meta.plan_vars)
        prior = extract_prior(sol, problem, ctx)
        fixed_model = ModelBuilder(problem, ctx).build()
        freeze_combo(fixed_model, prior)
        self.assertEqual(unfixed_integrality(fixed_model), ())

    def test_extract_prior_and_freeze_combo_removes_integrality(self):
        raw = _two_window_problem(multistage=False)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model, _res, sol = _solve_full(problem, ctx)
        prior = extract_prior(sol, problem, ctx)
        self.assertGreater(len(prior.bands), 0)
        self.assertEqual(validate_prior(problem, ctx, prior), ())

        fixed_model = ModelBuilder(problem, ctx).build()
        freeze_combo(fixed_model, prior)
        self.assertEqual(unfixed_integrality(fixed_model), ())


class TestStage1Decomposition(unittest.TestCase):
    def test_fixed_variables_are_constants_in_connectivity_graph(self):
        reg = VariableRegistry()
        a = reg.add("a", 0.0, 1.0, GVarType.CONTINUOUS, {})
        fixed = reg.add("fixed", 0.0, 0.0, GVarType.CONTINUOUS, {})
        b = reg.add("b", 0.0, 1.0, GVarType.CONTINUOUS, {})
        cons = ConstraintStore()
        cons.add(LinearExpr.of(a, 1.0).add(fixed, 1.0), "<=", 1.0,
                 ConstraintMeta("TEST", "a + fixed", {}))
        cons.add(LinearExpr.of(b, 1.0).add(fixed, 1.0), "<=", 1.0,
                 ConstraintMeta("TEST", "b + fixed", {}))
        model = ArterialModel(reg, cons, LinearExpr(), "max", None)
        comps = find_stage1_components(model)
        self.assertEqual(len(comps), 2)
        self.assertEqual({tuple(c) for c in comps}, {(a.index,), (b.index,)})

    def test_stage1_symmetry_constraints_are_added(self):
        raw = _two_window_problem(multistage=False)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx, stage1_symmetry=True,
                             min_band_width_s=1.0).build()
        groups = [row.meta.group for row in model.cons.rows]
        self.assertIn("SYM", groups)
        labels = [row.meta.label for row in model.cons.rows if row.meta.group == "SYM"]
        self.assertTrue(any("e slot monotonic" in x for x in labels))
        self.assertTrue(any("positive width" in x for x in labels))

    def test_green_wrap_bounds_are_tight_and_present(self):
        raw = _two_window_problem(multistage=False)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        n_vars = [v for v in model.vars if v.name.startswith("gw_cycle[")]
        self.assertTrue(n_vars)
        for h in n_vars:
            self.assertLessEqual(model.vars.ubs[h.index] - model.vars.lbs[h.index], 3.0)
        self.assertTrue(any(row.meta.group == "NRANGE" for row in model.cons.rows))

    def test_stage1_symmetry_preserves_objective_on_small_case(self):
        raw = _two_window_problem(multistage=False)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        plain = solve_fixed_boundaries(problem, ctx, {}, decompose=False,
                                       symmetry_breaking=False,
                                       min_band_width_s=0.0)
        sym = solve_fixed_boundaries(problem, ctx, {}, decompose=False,
                                     symmetry_breaking=True,
                                     min_band_width_s=1.0)
        self.assertIsNotNone(plain)
        self.assertIsNotNone(sym)
        self.assertAlmostEqual(float(plain[0].composite_s), float(sym[0].composite_s),
                               places=5)

    def test_global_constraint_merges_demand_components(self):
        raw = _independent_red_problem()
        raw["global_constraints"] = [{
            "id": "sum_bands",
            "terms": [
                {"atom": {"kind": "bandwidth", "demand": "R12"}, "coef": 1.0},
                {"atom": {"kind": "bandwidth", "demand": "R45"}, "coef": 1.0},
            ],
            "sense": ">=", "rhs_s": 0.0, "hard": True,
        }]
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        comps = find_stage1_components(model)
        name_to_idx = {name: i for i, name in enumerate(model.vars.names)}
        a = name_to_idx["rw_width[R12,1]"]
        b = name_to_idx["rw_width[R45,1]"]
        same = any(a in comp and b in comp for comp in comps)
        self.assertTrue(same)

    def test_independent_red_pairs_decompose_exactly(self):
        raw = _independent_red_problem()
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        comps = find_stage1_components(model)
        self.assertGreaterEqual(len(comps), 2)

        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertIsNotNone(mono)
        self.assertIsNotNone(dec)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s), places=6)
        self.assertIn("decomposed", dec[3].message)


# ---------------------------------------------------------------------------
# 语义子问题分解（greenwave.stage1_decompose）
# ---------------------------------------------------------------------------

def _chain_problem(*, n_nodes: int = 5, spanning: bool = False,
                   plans_at_middle: int = 1,
                   link_global: bool = False,
                   adjustable: bool = False) -> dict:
    """一段干线：需求全是相邻对窗口绿波（spanning=True 时再加一条跨全线的带）。"""
    nodes = [f"I{i + 1}" for i in range(n_nodes)]
    boundary = {"pos_r": 0.5, "adjust": [0.0, 0.2]} if adjustable else {"pos_r": 0.5}
    plans = []
    for k, iid in enumerate(nodes):
        count = plans_at_middle if k == 1 else 1
        for c in range(count):
            plans.append({
                "id": f"{iid}_P{c + 1}", "intersection": iid, "double_cycle": False,
                "directions": {
                    "up": {"boundaries": [dict(boundary)], "green_mask": [True, False]},
                    "down": {"boundaries": [dict(boundary)], "green_mask": [True, False]},
                },
            })
    demands = []
    for k in range(n_nodes - 1):
        a, b = nodes[k], nodes[k + 1]
        demands.append({"id": f"GW_up_{a}_{b}", "type": "green", "direction": "up",
                        "nodes": [a, b], "max_bands": 1, "weight": 1.0})
        demands.append({"id": f"GW_down_{b}_{a}", "type": "green", "direction": "down",
                        "nodes": [b, a], "max_bands": 1, "weight": 1.0})
    if spanning:
        demands.append({"id": "GW_up_full", "type": "green", "direction": "up",
                        "nodes": list(nodes), "max_bands": 1, "weight": 1.0})
    globals_ = []
    if link_global:
        globals_.append({
            "id": "link_first_last", "terms": [
                {"atom": {"kind": "bandwidth", "demand": demands[0]["id"]}, "coef": 1.0},
                {"atom": {"kind": "bandwidth", "demand": demands[-1]["id"]}, "coef": 1.0},
            ], "sense": ">=", "rhs_s": 0.0, "hard": True,
        })
    return {
        "corridor": {
            "corridor_id": "chain",
            "intersections": nodes,
            "segments": [_segment(nodes[i], nodes[i + 1], distance_m=300.0)
                         for i in range(n_nodes - 1)],
        },
        "plans": plans,
        "objective": {
            "band_demands": demands,
            "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
            "allow_window_sharing": False,
            "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1},
        },
        "global_constraints": globals_,
        "constraint_templates": [],
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {
            "backend": "scipy", "time_limit_s": 20.0, "mip_rel_gap": 0.0,
            "num_workers": 1,
            "heuristic": {"enabled": False, "refine_rounds": 0},
            "multistage": {"enabled": True, "decompose_stage1": True,
                           "decompose_min_component_size": 2},
        },
    }


class TestSemanticStage1Decomposition(unittest.TestCase):
    """相邻对窗口绿波 -> 逐对求解再拼装；跨带的约束必须把组并回去。"""

    def _ctx(self, raw):
        problem = ProblemInput.from_obj(raw)
        return problem, FeasibilityFilter(problem).filter(100.0, 1.0)

    def test_chain_of_local_bands_splits_into_pairs(self):
        problem, ctx = self._ctx(_chain_problem(n_nodes=5))
        groups = semantic_demand_groups(problem, ctx)
        self.assertEqual([(g.label, len(g.demand_ids)) for g in groups],
                         [("I1-I2", 2), ("I2-I3", 2), ("I3-I4", 2), ("I4-I5", 2)])
        self.assertEqual(groups[0].node_ids, ("I1", "I2"))
        self.assertEqual(groups[1].node_ids, ("I2", "I3"))
        self.assertEqual((groups[0].lo, groups[-1].hi), (0, 4))

    def test_decomposed_matches_monolith_objective(self):
        problem, ctx = self._ctx(_chain_problem(n_nodes=5))
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertIsNotNone(mono)
        self.assertIsNotNone(dec)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s), delta=1e-4)
        self.assertAlmostEqual(float(mono[0].band_reward_s), float(dec[0].band_reward_s),
                               delta=1e-4)
        self.assertIn("decomposed", dec[3].message)
        self.assertIn("semantic groups", dec[3].message)

    def test_decomposed_matches_monolith_with_nonzero_delta(self):
        """回归：共享路口的 φ 必须在**建模前**固定，否则回绕整数 n 的区间失配，
        分解会给出低于整模的最优值。"""
        problem, ctx = self._ctx(_chain_problem(n_nodes=4, adjustable=True))
        delta = {}
        for p in problem.plans:
            for d in ("up", "down"):
                for j in range(p.boundary_count(d)):
                    delta[(p.id, d, j)] = 0.15
        mono = solve_fixed_boundaries(problem, ctx, delta, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, delta, decompose=True)
        self.assertIsNotNone(mono)
        self.assertIsNotNone(dec)
        self.assertIn("decomposed", dec[3].message)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s),
                               delta=1e-4)

    def test_red_window_pairs_also_decompose(self):
        raw = _chain_problem(n_nodes=5)
        nodes = raw["corridor"]["intersections"]
        raw["objective"]["band_demands"] = []
        for k in range(len(nodes) - 1):
            a, b = nodes[k], nodes[k + 1]
            raw["objective"]["band_demands"].append(
                {"id": f"RW_{b}_{a}", "type": "red", "direction": "down",
                 "nodes": [b, a], "max_bands": 1, "weight": 1.0})
            raw["objective"]["band_demands"].append(
                {"id": f"RW_{a}_{b}", "type": "red", "direction": "up",
                 "nodes": [a, b], "max_bands": 1, "weight": 1.0})
        problem, ctx = self._ctx(raw)
        self.assertEqual(len(semantic_demand_groups(problem, ctx)), 4)
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertIn("decomposed", dec[3].message)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s),
                               delta=1e-4)

    def test_spanning_demand_blocks_decomposition(self):
        problem, ctx = self._ctx(_chain_problem(n_nodes=5, spanning=True))
        self.assertEqual(len(semantic_demand_groups(problem, ctx)), 1)
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertIsNotNone(dec)
        self.assertNotIn("decomposed", dec[3].message)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s), delta=1e-4)

    def test_global_constraint_linking_two_groups_merges_them(self):
        problem, ctx = self._ctx(_chain_problem(n_nodes=5, link_global=True))
        self.assertEqual(len(semantic_demand_groups(problem, ctx)), 1)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertNotIn("decomposed", dec[3].message)

    def test_offset_link_constraint_keeps_linked_nodes_in_one_group(self):
        """跨组约束即使只引用 offset（φ），被连起来的路口也必须落在同一组。"""
        raw = _chain_problem(n_nodes=5)
        raw["global_constraints"] = [{
            "id": "tie_offsets", "terms": [
                {"atom": {"kind": "offset", "intersection": "I1"}, "coef": 1.0},
                {"atom": {"kind": "offset", "intersection": "I4"}, "coef": -1.0},
            ], "sense": "==", "rhs_s": 0.0, "hard": True,
        }]
        problem, ctx = self._ctx(raw)
        groups = semantic_demand_groups(problem, ctx)
        same = [g for g in groups if "I1" in g.node_ids and "I4" in g.node_ids]
        self.assertEqual(len(same), 1)
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s),
                               delta=1e-4)

    def test_demands_sharing_two_nodes_stay_in_one_group(self):
        raw = _chain_problem(n_nodes=4)
        raw["objective"]["band_demands"] = [
            {"id": "D13", "type": "green", "direction": "up",
             "nodes": ["I1", "I3"], "max_bands": 1, "weight": 1.0},
            {"id": "D24", "type": "green", "direction": "up",
             "nodes": ["I2", "I4"], "max_bands": 1, "weight": 1.0},
        ]
        problem, ctx = self._ctx(raw)
        groups = semantic_demand_groups(problem, ctx)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0].demand_ids), {"D13", "D24"})

    def test_multi_plan_shared_intersection_blocks_split(self):
        problem, ctx = self._ctx(_chain_problem(n_nodes=3, plans_at_middle=2))
        self.assertEqual(len(semantic_demand_groups(problem, ctx)), 1)
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertNotIn("decomposed", dec[3].message)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s), delta=1e-4)

    def test_local_bands_only_but_single_group_no_split(self):
        raw = _chain_problem(n_nodes=3)
        raw["objective"]["band_demands"] = [
            {"id": "GW_up_1_2", "type": "green", "direction": "up",
             "nodes": ["I1", "I2"], "max_bands": 1, "weight": 1.0},
        ]
        problem, ctx = self._ctx(raw)
        groups = semantic_demand_groups(problem, ctx)
        # 只有一条需求 -> 只有一组；组区间覆盖整条干线（尾部空闲路口并进该组）。
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0].demand_ids), {"GW_up_1_2"})
        self.assertEqual((groups[0].lo, groups[0].hi), (0, 2))
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True)
        self.assertNotIn("decomposed", dec[3].message)


def _multi_plan_chain_problem(n_nodes: int = 4, n_plans: int = 2,
                              soft_at: str | None = None) -> dict:
    """多方案路口 + 全相邻对窗口绿波（固定边界）：DP+枚举的适用场景。"""
    nodes = [f"I{i + 1}" for i in range(n_nodes)]
    plans = []
    for iid in nodes:
        for c in range(n_plans):
            plan = {
                "id": f"{iid}_P{c + 1}", "intersection": iid, "double_cycle": False,
                "directions": {
                    "up": {"boundaries": [{"pos_r": 0.45 + 0.06 * c}],
                           "green_mask": [True, False]},
                    "down": {"boundaries": [{"pos_r": 0.50 + 0.06 * c}],
                             "green_mask": [True, False]},
                },
            }
            if soft_at == iid:
                # owner=composite 的方案软约束：接口路口会被两个组共享，
                # 这类常数罚必须只计一次，DP 才能和整模同口径。
                plan["soft_constraints"] = [{
                    "id": f"{iid}_up_soft", "terms": [{
                        "atom": {"kind": "boundary", "intersection": iid,
                                 "plan": f"{iid}_P{c + 1}", "direction": "up",
                                 "index": 0}, "coef": 1.0}],
                    "sense": "<=", "rhs_ratio": 0.47, "hard": False,
                    "soft": {"coef": 4.0, "owner": "composite"},
                }]
            plans.append(plan)
    demands = []
    for k in range(n_nodes - 1):
        a, b = nodes[k], nodes[k + 1]
        demands.append({"id": f"GW_up_{a}_{b}", "type": "green", "direction": "up",
                        "nodes": [a, b], "max_bands": 1, "weight": 1.0,
                        "margin": {"delta_min_s": 0.0, "coef": 1.0}})
        demands.append({"id": f"GW_down_{b}_{a}", "type": "green", "direction": "down",
                        "nodes": [b, a], "max_bands": 1, "weight": 1.0,
                        "margin": {"delta_min_s": 0.0, "coef": 1.0}})
    return {
        "corridor": {"corridor_id": "multiplan",
                     "intersections": nodes,
                     "segments": [_segment(nodes[i], nodes[i + 1], distance_m=300.0)
                                  for i in range(n_nodes - 1)]},
        "plans": plans,
        "objective": {"band_demands": demands,
                      "margin_default": {"delta_min_s": 0.0, "coef": 1.0},
                      "allow_window_sharing": True,
                      "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}},
        "global_constraints": [],
        "constraint_templates": [],
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 30.0, "mip_rel_gap": 0.0,
                   "num_workers": 1,
                   "heuristic": {"enabled": False, "refine_rounds": 0},
                   "multistage": {"enabled": True, "decompose_stage1": True,
                                  "decompose_dp": True,
                                  "decompose_dp_max_solves": 400,
                                  "decompose_dp_probe_s": 0.0}},
    }


class TestPlanEnumerationDP(unittest.TestCase):
    """方案枚举 + 链式 DP：多方案路口下固定方案即可分解，DP 在方案维上搜索。"""

    def _ctx(self, raw):
        problem = ProblemInput.from_obj(raw)
        return problem, FeasibilityFilter(problem).filter(100.0, 1.0)

    def test_candidate_detected_only_with_multiple_plans(self):
        problem, ctx = self._ctx(_multi_plan_chain_problem(n_nodes=4, n_plans=2))
        # 普通语义分解会因为"共享路口多方案"并成 1 组
        self.assertEqual(len(semantic_demand_groups(problem, ctx)), 1)
        cand = plan_dp_candidate(problem, ctx, max_solves=400)
        self.assertIsNotNone(cand)
        self.assertEqual(len(cand.groups), 3)
        self.assertEqual(cand.n_solves, 3 * 2 * 2)
        self.assertTrue(cand.is_chain)
        # 单方案时不需要 DP，交给普通分解
        single = _multi_plan_chain_problem(n_nodes=4, n_plans=1)
        p1, c1 = self._ctx(single)
        self.assertIsNone(plan_dp_candidate(p1, c1, max_solves=400))
        # 求解次数上限
        self.assertIsNone(plan_dp_candidate(problem, ctx, max_solves=5))

    def test_dp_matches_monolith(self):
        problem, ctx = self._ctx(_multi_plan_chain_problem(n_nodes=4, n_plans=2))
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False,
                                      symmetry_breaking=False, min_band_width_s=0.0)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True,
                                     symmetry_breaking=False, min_band_width_s=0.0)
        self.assertIsNotNone(mono)
        self.assertIsNotNone(dec)
        self.assertIn("plan-enumeration DP", dec[3].message)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s),
                               delta=1e-4)

    def test_interface_node_penalty_counted_once(self):
        """接口路口带 owner=composite 的方案软约束时，DP 仍须与整模一致。"""
        raw = _multi_plan_chain_problem(n_nodes=4, n_plans=2, soft_at="I2")
        problem, ctx = self._ctx(raw)
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False,
                                      symmetry_breaking=False, min_band_width_s=0.0)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True,
                                     symmetry_breaking=False, min_band_width_s=0.0)
        self.assertIn("plan-enumeration DP", dec[3].message)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s),
                               delta=1e-4)

    def test_dp_can_be_disabled(self):
        raw = _multi_plan_chain_problem(n_nodes=4, n_plans=2)
        raw["solver"]["multistage"]["decompose_dp"] = False
        problem, ctx = self._ctx(raw)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True,
                                     symmetry_breaking=False, min_band_width_s=0.0)
        self.assertIsNotNone(dec)
        self.assertNotIn("plan-enumeration DP", dec[3].message)

    def test_dp_records_planned_assignment(self):
        problem, ctx = self._ctx(_multi_plan_chain_problem(n_nodes=4, n_plans=2))
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True,
                                     symmetry_breaking=False, min_band_width_s=0.0)
        for iid in problem.corridor.intersections:
            self.assertIn(iid, dec[0].plan_selection)


def _two_node_red_problem():
    """两点走廊 I4->I5 + 一条下行红带 I5->I4（用于红带上游周期下标 m 的回归）。"""
    from greenwave.schema.models import (CorridorSpec, GridConfig, MarginSpec,
                                         ObjectiveConfig, ParetoConfig, SegmentSpec,
                                         SolverConfig)
    nodes = ["I4", "I5"]
    plans = tuple(PlanSpec.from_obj({
        "id": f"{i}_P1", "intersection": i, "double_cycle": False,
        "directions": {"up": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]},
                       "down": {"boundaries": [{"pos_r": 0.5}], "green_mask": [True, False]}},
    }) for i in nodes)
    red = BandDemandSpec.from_obj({
        "id": "RW_down_I5_I4", "type": "red", "direction": "down",
        "nodes": ["I5", "I4"], "max_bands": 1, "weight": 1.0})
    return ProblemInput(
        corridor=CorridorSpec(corridor_id="t", intersections=tuple(nodes),
                              segments=(SegmentSpec(from_="I4", to="I5", distance_m=300.0,
                                                    speed_up_mps=10.0, speed_down_mps=10.0),)),
        plans=plans,
        objective=ObjectiveConfig(band_demands=(red,), margin_default=MarginSpec(0.0, 1.0),
                                  allow_window_sharing=True,
                                  pareto=ParetoConfig(0, 0.0, 1)),
        grid=GridConfig(cycles_s=(100.0,), speed_ratios=(1.0,)),
        solver=SolverConfig(time_limit_s=30.0, mip_rel_gap=0.0, num_workers=1))


class TestRedUpstreamWrap(unittest.TestCase):
    """红带上游周期下标 m：没有它时，把整组做时间平移（钉锚点 φ）会掉最优值。

    物理上整个系统对时间平移不变，所以"钉锚点 φ"后最优值必须和自由解相同。
    """

    def test_red_band_optimum_is_shift_invariant(self):
        from greenwave.stage1_decompose import _solve_group
        problem = _two_node_red_problem()
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        group = DemandGroup(demand_ids=("RW_down_I5_I4",), lo=0, hi=1,
                            node_ids=("I4", "I5"))
        kw = dict(adjustments_r={}, symmetry=False, min_band_width_s=0.0,
                  time_limit_s=30.0, mip_rel_gap=0.0)
        free = _solve_group(problem, ctx, group, fixed_phi={}, **kw)
        self.assertIsNotNone(free)
        self.assertGreater(float(free.objective), 0.0)
        for delta in range(0, 100, 10):
            got = _solve_group(problem, ctx, group, fixed_phi={"I4": float(delta)}, **kw)
            self.assertAlmostEqual(float(got.objective), float(free.objective),
                                   delta=1e-4,
                                   msg=f"钉住锚点 φ[I4]={delta} 后最优值掉了，说明 m 没生效")

    def test_red_band_has_wrap_variable(self):
        problem = _two_node_red_problem()
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        model = ModelBuilder(problem, ctx).build()
        rm = model.meta.red_demands["RW_down_I5_I4"]
        self.assertTrue(rm.m_shift)
        lo = model.vars.lbs[rm.m_shift[1].index]
        hi = model.vars.ubs[rm.m_shift[1].index]
        self.assertLess(lo, hi)          # m 必须有多个取值，否则补的这一维是死的
        names = set(model.vars.names)
        self.assertTrue(any(n.startswith("rw_up_cycle[RW_down_I5_I4,") for n in names))

    def test_red_multi_plan_case_decomposes_and_matches_monolith(self):
        """红波 + 多方案：m 修好后各组可同时达到上界，证书通过 -> 分解生效。"""
        raw = _multi_plan_chain_problem(n_nodes=4, n_plans=2)
        nodes = raw["corridor"]["intersections"]
        demands = []
        for k in range(len(nodes) - 1):
            a, b = nodes[k], nodes[k + 1]
            demands.append({"id": f"GW_up_{a}_{b}", "type": "green", "direction": "up",
                            "nodes": [a, b], "max_bands": 1, "weight": 1.0,
                            "margin": {"delta_min_s": 0.0, "coef": 1.0}})
            demands.append({"id": f"RW_down_{b}_{a}", "type": "red", "direction": "down",
                            "nodes": [b, a], "max_bands": 1, "weight": 1.0})
        raw["objective"]["band_demands"] = demands
        problem, ctx = self._ctx(raw)
        mono = solve_fixed_boundaries(problem, ctx, {}, decompose=False,
                                      symmetry_breaking=False, min_band_width_s=0.0)
        dec = solve_fixed_boundaries(problem, ctx, {}, decompose=True,
                                     symmetry_breaking=False, min_band_width_s=0.0)
        self.assertIsNotNone(mono)
        self.assertIsNotNone(dec)
        self.assertAlmostEqual(float(mono[0].composite_s), float(dec[0].composite_s),
                               delta=1e-4)

    def _ctx(self, raw):
        problem = ProblemInput.from_obj(raw)
        return problem, FeasibilityFilter(problem).filter(100.0, 1.0)


class TestMultistageSolver(unittest.TestCase):
    def test_fixed_combo_lp_matches_or_exceeds_stage1(self):
        raw = _two_window_problem(multistage=False)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)

        s1 = solve_fixed_boundaries(problem, ctx, {})
        self.assertIsNotNone(s1)
        sol1, prior1, _m1, _r1 = s1
        s2 = solve_fixed_combo_lp(problem, ctx, prior1)
        self.assertIsNotNone(s2)
        sol2, _prior2, _m2, _r2 = s2
        self.assertGreaterEqual(float(sol2.composite_s) + 1e-7,
                                float(sol1.composite_s))

    def test_solve_multistage_smoke(self):
        raw = _two_window_problem(multistage=True)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        result = solve_multistage(problem, ctx, problem.solver.multistage)
        self.assertIn(result.status, ("feasible", "optimal"))
        self.assertIsNotNone(result.solution)
        self.assertIsNotNone(result.objective)
        self.assertGreaterEqual(float(result.objective), 0.0)
        self.assertGreaterEqual(result.stage1_time_s, 0.0)
        self.assertGreaterEqual(result.stage2_time_s, 0.0)
        self.assertTrue(result.stage_records)
        self.assertTrue(any(r.stage == "stage1" for r in result.stage_records))
        self.assertTrue(any(r.stage == "stage2" for r in result.stage_records))

    def test_multistage_uses_plan_witness_start(self):
        raw = _witness_required_problem()
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        from greenwave.multistage import _witness_values  # noqa: E402
        witness = _witness_values(ctx)
        self.assertTrue(witness)
        self.assertTrue(all(abs(v) > 1e-12 for v in witness.values()))

        # δ=0 时方案自己的硬约束不满足，固定边界 Stage-1 应失败。
        zero = solve_fixed_boundaries(problem, ctx, {}, decompose=False,
                                      symmetry_breaking=False, min_band_width_s=0.0)
        self.assertIsNone(zero)

        # 多阶段默认应优先使用 witness start，因此能正常求解。
        result = solve_multistage(problem, ctx, problem.solver.multistage)
        self.assertIsNotNone(result.solution)
        self.assertIsNotNone(result.objective)
        self.assertGreater(result.stage1_time_s, 0.0)

    def test_multistage_improves_fixed_boundary_start(self):
        raw = _two_window_problem(adjustable=True, multistage=True)
        problem = ProblemInput.from_obj(raw)
        ctx = FeasibilityFilter(problem).filter(100.0, 1.0)
        s1 = solve_fixed_boundaries(problem, ctx, {})
        self.assertIsNotNone(s1)
        sol1, _prior1, _m1, _r1 = s1
        result = solve_multistage(problem, ctx, problem.solver.multistage)
        self.assertIsNotNone(result.solution)
        self.assertGreaterEqual(float(result.objective) + 1e-7,
                                float(sol1.composite_s))

    def test_run_multistage_pipeline(self):
        raw = _two_window_problem(multistage=False)
        out = run_multistage(raw, max_rounds=1, restarts=1)
        self.assertIn("grid_results", out)
        self.assertEqual(len(out["grid_results"]), 1)
        self.assertIsNotNone(out["grid_results"][0]["composite_best"])
        self.assertIn("pareto", out)

    def test_original_run_still_works(self):
        raw = _two_window_problem(multistage=False)
        out = run(raw)
        self.assertIn("grid_results", out)
        self.assertEqual(len(out["grid_results"]), 1)
        self.assertIsNotNone(out["grid_results"][0]["composite_best"])


if __name__ == "__main__":
    unittest.main()
