"""Test-only helpers: convert old v1 window plans into v2 boundary plans.

This is deliberately not a production compatibility layer.  It exists so the
legacy T1--T8 regression cases can be replayed against the v2 model.
"""
from __future__ import annotations

from typing import Any


def _bounds_and_mask(windows, hi: float, double: bool):
    interior: set[float] = set()
    for s, e in windows:
        for v in (float(s), float(e)):
            if 1e-12 < v < hi - 1e-12:
                interior.add(round(v, 12))
    bounds = sorted(interior)
    edges = [0.0] + bounds + [hi]
    mask = []
    for a, b in zip(edges, edges[1:]):
        mid = (a + b) / 2.0
        mask.append(any(s - 1e-12 <= mid <= e + 1e-12 for s, e in windows))
    if len(mask) >= 2 and mask[0] and mask[-1]:
        raise ValueError("cannot convert wrapping green windows to v2")
    return bounds, mask


def convert_direction(windows, adjustable_spec, double: bool):
    hi = 0.5 if double else 1.0
    bounds, mask = _bounds_and_mask(windows, hi, double)
    # Map old window endpoints to boundary indices.
    endpoint_to_idx: dict[tuple[int, str], int] = {}
    for wi, (s, e) in enumerate(windows):
        if 1e-12 < s < hi - 1e-12:
            endpoint_to_idx[(wi, "start")] = bounds.index(round(float(s), 12))
        if 1e-12 < e < hi - 1e-12:
            endpoint_to_idx[(wi, "end")] = bounds.index(round(float(e), 12))
    adjust: dict[int, list[float] | None] = {i: None for i in range(len(bounds))}
    for wi, spec in (adjustable_spec or {}).items():
        wi = int(wi)
        for side in ("start", "end"):
            rng = spec.get(side)
            if rng is None:
                continue
            idx = endpoint_to_idx.get((wi, side))
            if idx is None:
                continue
            cur = adjust[idx]
            if cur is None:
                adjust[idx] = [float(rng[0]), float(rng[1])]
            else:
                cur[0] = max(cur[0], float(rng[0]))
                cur[1] = min(cur[1], float(rng[1]))
                if cur[0] > cur[1] + 1e-15:
                    raise ValueError("incompatible adjustable ranges on shared boundary")
    out_bounds = []
    for i, pos in enumerate(bounds):
        out = {"pos_r": pos}
        if adjust[i] is not None:
            out["adjust"] = adjust[i]
        out_bounds.append(out)
    return {"boundaries": out_bounds, "green_mask": mask}, endpoint_to_idx, bounds


def convert_constraint(c: dict, endpoint_to_idx: dict, bounds: list[float], hi: float) -> dict:
    terms = []
    for t in c.get("terms", ()):
        a = dict(t["atom"])
        if a.get("kind") == "endpoint":
            wi = int(a["window"])
            side = a["side"]
            idx = endpoint_to_idx.get((wi, side))
            if idx is None:
                ratio = 0.0 if side == "start" else hi
                if abs(ratio) <= 1e-12:
                    terms.append({"atom": {"kind": "const", "const": 0.0}, "coef": t["coef"]})
                    continue
                raise ValueError("conversion does not support constraints on cycle-end edge")
            terms.append({"atom": {
                "kind": "boundary",
                "intersection": a.get("intersection"),
                "plan": a.get("plan"),
                "direction": a.get("direction"),
                "index": idx,
            }, "coef": t["coef"]})
        else:
            terms.append(t)
    out = dict(c)
    out["terms"] = terms
    return out


def convert_plan(old: dict) -> dict:
    double = bool(old.get("double_cycle", False))
    hi = 0.5 if double else 1.0
    windows_in = old.get("windows", {})
    adj_in = old.get("adjustable", {}) or {}
    directions = {}
    maps = {}
    for d in ("up", "down"):
        dir_plan, mapping, bounds = convert_direction(windows_in.get(d, ()), adj_in.get(d), double)
        directions[d] = dir_plan
        maps[d] = (mapping, bounds, hi)
    hard = []
    soft = []
    for c in old.get("hard_constraints", ()):
        # determine direction from first endpoint atom
        d = next((t["atom"].get("direction") for t in c.get("terms", ())
                  if t.get("atom", {}).get("kind") == "endpoint"), "up")
        hard.append(convert_constraint(c, maps[d][0], maps[d][1], maps[d][2]))
    for c in old.get("soft_constraints", ()):
        d = next((t["atom"].get("direction") for t in c.get("terms", ())
                  if t.get("atom", {}).get("kind") == "endpoint"), "up")
        soft.append(convert_constraint(c, maps[d][0], maps[d][1], maps[d][2]))
    return {
        "id": old["id"],
        "intersection": old["intersection"],
        "double_cycle": double,
        "min_phase_s": float(old.get("min_phase_s", 1e-3)),
        "directions": directions,
        "hard_constraints": hard,
        "soft_constraints": soft,
    }


def convert_problem(old: dict) -> dict:
    import copy
    out = copy.deepcopy(old)
    out["plans"] = [convert_plan(p) for p in old["plans"]]
    return out
