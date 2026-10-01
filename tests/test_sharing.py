"""Tests for green-band window sharing and circular time separation."""
from __future__ import annotations

import unittest

from greenwave.model.builder import ModelBuilder
from greenwave.model.constraints import ConstraintMeta
from greenwave.model.expressions import LinearExpr
from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.report import SolutionDecoder
from greenwave.schema.models import ProblemInput
from greenwave.solve import make_backend
from greenwave.solve.backend_base import SolveOptions


def _problem(allow_sharing: bool):
    return {
        "corridor": {
            "corridor_id": "share",
            "intersections": ["I1", "I2"],
            "segments": [{"from": "I1", "to": "I2", "distance_m": 200.0,
                          "speed_up_mps": 10.0, "speed_down_mps": 10.0}],
        },
        "plans": [
            {"id": f"{iid}_P1", "intersection": iid, "double_cycle": False,
             "directions": {
                 "up": {"boundaries": [{"pos_r": 0.6}], "green_mask": [True, False]},
                 "down": {"boundaries": [{"pos_r": 0.6}], "green_mask": [True, False]},
             }}
            for iid in ("I1", "I2")
        ],
        "objective": {
            "band_demands": [{
                "id": "GW_up", "type": "green", "direction": "up",
                "nodes": ["I1", "I2"], "max_bands": 2, "weight": 1.0,
                "margin": {"delta_min_s": 0.0, "coef": 1.0},
            }],
            "allow_window_sharing": allow_sharing,
            "pareto": {"num_points": 0, "relax_max": 0.0, "topk_grids": 1},
        },
        "grid": {"cycles_s": [100.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                   "num_workers": 1,
                   "heuristic": {"enabled": False, "refine_rounds": 0}},
    }


class TestGreenSharing(unittest.TestCase):
    def test_qmax_uncapped_only_when_sharing(self):
        p = ProblemInput.from_obj(_problem(True))
        ctx = FeasibilityFilter(p).filter(100.0, 1.0)
        demand = p.demand("GW_up")
        self.assertEqual(ctx.demand_max_bands(demand, allow_sharing=False), 1)
        self.assertEqual(ctx.demand_max_bands(demand, allow_sharing=True), 2)

    def test_two_bands_can_share_one_window(self):
        p = ProblemInput.from_obj(_problem(True))
        ctx = FeasibilityFilter(p).filter(100.0, 1.0)
        model = ModelBuilder(p, ctx).build()
        gm = model.meta.green["GW_up"]
        self.assertEqual(gm.qmax, 2)

        # Circular-separation constraints must exist.
        labels = [row.meta.label for row in model.cons.rows if row.meta.group == "C7"]
        self.assertTrue(any("band disjoint" in label for label in labels))
        self.assertTrue(any("band wrap disjoint" in label for label in labels))
        # No window-exclusive constraints when sharing is enabled.
        self.assertFalse(any("window exclusive" in label for label in labels))

        # Force both slots to exist and both widths to be positive.
        for q in (1, 2):
            model.cons.add(LinearExpr.of(gm.e[q], 1.0), ">=", 1.0,
                           ConstraintMeta("TEST", f"force band {q}"))
            model.cons.add(LinearExpr.of(gm.beta[q], 1.0), ">=", 1.0,
                           ConstraintMeta("TEST", f"force width {q}"))

        backend = make_backend("scipy")
        session = backend.create(model)
        backend.set_objective(session, model.expr_composite(), "max")
        result = backend.solve(session, SolveOptions(20.0, 0.0))
        self.assertEqual(result.status.value, "optimal")
        sol = SolutionDecoder(model).decode(result.values)
        bands = [b for b in sol.bands if b.demand == "GW_up"]
        self.assertEqual(len(bands), 2)
        for b in bands:
            self.assertGreater(b.width_s, 0.0)
        # On the circle, the two bands must not overlap.
        C = 100.0
        intervals = []
        for b in bands:
            start = b.start_s % C
            intervals.append((start, start + b.width_s))
        intervals.sort()
        (s1, e1), (s2, e2) = intervals
        self.assertLessEqual(e1, s2 + 1e-6)
        # And they must not be duplicates one period apart.
        self.assertGreater(abs((s2 - s1) % C), 1e-6)


if __name__ == "__main__":
    unittest.main()
