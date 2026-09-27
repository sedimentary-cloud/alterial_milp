"""Minimal CLI: python -m greenwave problem.json --out result.json --diagram-dir out/"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from .heuristic.greedy import ConstructedBand
from .pipeline import run
from .preprocess.feasibility import FeasibilityFilter
from .report.diagram import TimeSpaceDiagram
from .schema.models import ProblemInput
from .semantic import compile_semantics


def _solution_view(sol_dict):
    adjustments = {}
    for key, value in (sol_dict.get("adjustments") or {}).items():
        parts = key.split(".")
        if len(parts) == 4:
            pid, d, k, side = parts
            adjustments[(pid, d, int(k), side)] = float(value)
    bands = []
    for b in sol_dict.get("bands", []):
        bands.append(ConstructedBand(
            demand=b["demand"], slot=int(b.get("slot", 1)), exists=bool(b.get("exists", True)),
            width_s=float(b.get("width_s", 0.0)), start_s=float(b.get("start_at_anchor_s", 0.0)),
            windows={k: int(v) for k, v in (b.get("windows") or {}).items()},
            red_instance=tuple(b["red_instance"]) if b.get("red_instance") is not None else None,
        ))
    return SimpleNamespace(plan_selection=dict(sol_dict.get("plan_selection", {})),
                           offsets_s=dict(sol_dict.get("offsets_s", {})),
                           adjustments_r=adjustments, bands=tuple(bands))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Arterial green/red wave MILP")
    parser.add_argument("input", help="input problem JSON")
    parser.add_argument("--out", default=None, help="write result JSON to this path")
    parser.add_argument("--diagram-dir", default=None, help="directory for time-space PNGs")
    parser.add_argument("--semantics", default=None, help="high-level semantic rules JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="only compile semantics, do not solve")
    args = parser.parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as f:
        problem_json = json.load(f)
    if args.semantics:
        with open(args.semantics, "r", encoding="utf-8") as f:
            semantics_json = json.load(f)
        compiled = compile_semantics(problem_json, semantics_json)
        problem_json = compiled.problem_json
        if args.dry_run:
            report_text = json.dumps(compiled.report.to_dict(), ensure_ascii=False, indent=2)
            print(report_text)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(problem_json, f, ensure_ascii=False, indent=2)
            return 0
    result = run(problem_json)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)

    if args.diagram_dir:
        problem = ProblemInput.from_obj(problem_json)
        os.makedirs(args.diagram_dir, exist_ok=True)
        pareto = result.get("pareto")
        groups = pareto if isinstance(pareto, list) else [pareto]
        for gi, group in enumerate(groups):
            if not group:
                continue
            grid = group.get("grid") or {}
            cyc = grid.get("cycle_s")
            kap = grid.get("speed_ratio")
            if cyc is None or kap is None:
                continue
            ctx = FeasibilityFilter(problem).filter(cyc, kap)
            for pi, point in enumerate(group.get("points", [])):
                sol = point.get("solution")
                if sol is None:
                    continue
                out_path = os.path.join(args.diagram_dir,
                                        f"grid{gi}_point{pi}_C{cyc:g}_k{kap:g}.png")
                TimeSpaceDiagram.render(_solution_view(sol), problem, ctx, out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
