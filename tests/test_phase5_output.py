"""Phase 5 tests: end-to-end v2 output, greedy and diagram consumers."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

from greenwave import run

sys.path.insert(0, os.path.dirname(__file__))
from _legacy_plan import convert_problem  # noqa: E402


def _old_problem(heuristic=True):
    return {
        "corridor": {"corridor_id": "t", "intersections": ["I1", "I2", "I3"],
                     "segments": [
                         {"from": "I1", "to": "I2", "distance_m": 500.0,
                          "speed_up_mps": 11.1111111111, "speed_down_mps": 11.1111111111},
                         {"from": "I2", "to": "I3", "distance_m": 500.0,
                          "speed_up_mps": 11.1111111111, "speed_down_mps": 11.1111111111}]},
        "plans": [{"id": f"{i}_P1", "intersection": i, "double_cycle": False,
                   "windows": {"up": [[0.0, 0.45]], "down": [[0.0, 0.45]]},
                   "adjustable": {"up": {}, "down": {}},
                   "hard_constraints": [], "soft_constraints": []}
                  for i in ("I1", "I2", "I3")],
        "objective": {"band_demands": [
            {"id": "GW_up", "type": "green", "direction": "up",
             "nodes": ["I1", "I2", "I3"], "weight": 1.0}],
            "pareto": {"num_points": 1, "relax_max": 0.0, "topk_grids": 1}},
        "grid": {"cycles_s": [90.0], "speed_ratios": [1.0]},
        "solver": {"backend": "scipy", "time_limit_s": 20, "mip_rel_gap": 0.0,
                   "num_workers": 1,
                   "heuristic": {"enabled": heuristic, "refine_rounds": 1}},
    }


class TestOutputV2(unittest.TestCase):
    def test_run_has_boundaries_and_derived_windows(self):
        out = run(convert_problem(_old_problem(heuristic=True)))
        self.assertEqual(out["grid_results"][0]["status"], "optimal")
        sol = out["pareto"]["points"][0]["solution"]
        self.assertIn("boundaries", sol)
        self.assertIn("derived_windows", sol)
        self.assertIn("adjustments", sol)
        self.assertAlmostEqual(sol["band_reward_s"], 40.5, places=4)
        # window for I1 up should be [0, 0.45]
        arr = sol["derived_windows"]["I1_P1.up"]
        self.assertEqual(len(arr), 1)
        self.assertAlmostEqual(arr[0][0], 0.0)
        self.assertAlmostEqual(arr[0][1], 0.45)

    def test_diagram_render_v2(self):
        out = run(convert_problem(_old_problem(heuristic=False)))
        sol = out["pareto"]["points"][0]["solution"]
        # Build a solution view compatible with the diagram renderer by
        # reusing the public JSON path through a tiny adapter.
        from greenwave.schema.models import ProblemInput
        from greenwave.preprocess.feasibility import FeasibilityFilter
        from greenwave.report import TimeSpaceDiagram
        problem = ProblemInput.from_obj(convert_problem(_old_problem(heuristic=False)))
        ctx = FeasibilityFilter(problem).filter(90.0, 1.0)
        from greenwave.__main__ import _solution_view
        path = os.path.join(tempfile.gettempdir(), "v2_phase5_ts.png")
        if os.path.exists(path):
            os.remove(path)
        TimeSpaceDiagram.render(_solution_view(sol), problem, ctx, path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 1000)
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
