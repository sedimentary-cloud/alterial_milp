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
from greenwave.schema.models import MultistageConfig, SolverConfig  # noqa: E402
from greenwave.solve import make_backend  # noqa: E402
from greenwave.solve.backend_base import SolveOptions  # noqa: E402


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
        a = name_to_idx["red[R12,1].beta"]
        b = name_to_idx["red[R45,1].beta"]
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
