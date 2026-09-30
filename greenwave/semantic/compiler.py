"""语义规则编译器：高层规则 -> ProblemInput 增量补丁（v2：相位边界）。"""
from __future__ import annotations

from dataclasses import replace

from ..errors import GWSemanticError
from ..schema.models import (AtomRef, ConstraintSpec, LinearTerm, PlanSpec,
                             ProblemInput, SoftPenalty)
from .ledger import describe_red_ledger, red_ledger
from .models import SemanticInput, SemanticRule, Target
from .resolver import resolve_green_slot, resolve_red_interval
from .report import CompileReport


def _desc(target: Target | None) -> str:
    if target is None:
        return ""
    w = target.window
    if w.kind == "green":
        where = f"index={w.index}" if w.index is not None else f"position={w.position}"
    else:
        where = f"index={w.index}" if w.index is not None else f"after_green={w.after_green}"
    return f"{target.intersection}/{target.plan}/{target.direction}/{w.kind}({where})"


def _fail(rule: SemanticRule, reason: str, target: Target | None = None, candidates=()) -> GWSemanticError:
    return GWSemanticError(rule.id, _desc(target or rule.target),
                           reason, tuple(candidates))


def _penalty(rule: SemanticRule) -> SoftPenalty | None:
    if rule.enforcement == "soft":
        if rule.penalty is None:
            raise _fail(rule, "soft enforcement requires a penalty")
        return rule.penalty
    if rule.penalty is not None:
        raise _fail(rule, "hard enforcement must not carry penalty")
    return None


def _make_constraint(rule: SemanticRule, terms: tuple[LinearTerm, ...], *,
                     sense: str, rhs_s: float | None = None,
                     rhs_ratio: float | None = None) -> ConstraintSpec:
    soft = _penalty(rule)
    return ConstraintSpec(
        id=f"sem:{rule.id}",
        terms=terms,
        sense=sense,
        hard=(soft is None),
        rhs_s=rhs_s,
        rhs_ratio=rhs_ratio,
        soft=soft,
    )


def _append_plan_constraint(plan: PlanSpec, c: ConstraintSpec) -> PlanSpec:
    if c.hard:
        return replace(plan, hard_constraints=plan.hard_constraints + (c,))
    return replace(plan, soft_constraints=plan.soft_constraints + (c,))


def _boundary_term(intersection: str, plan_id: str, d: str, index: int,
                   coef: float) -> LinearTerm:
    return LinearTerm(
        atom=AtomRef(kind="boundary", intersection=intersection, plan=plan_id,
                     direction=d, index=index),
        coef=coef,
    )


def _green_duration_term(target: Target, plan_id: str, d: str, k_eff: int,
                         coef: float = 1.0) -> LinearTerm:
    return LinearTerm(
        atom=AtomRef(kind="window_duration", intersection=target.intersection,
                     plan=plan_id, direction=d, window=k_eff),
        coef=coef,
    )


def _endpoint_linear_terms(plan: PlanSpec, d: str, k_eff: int, side: str,
                           coef: float) -> tuple[list[LinearTerm], float]:
    """Return (boundary terms, rhs_ratio) for ``coef * endpoint == 0``.

    The result is used only for sync equalities.  Cycle-edge constants and
    double-cycle 0.5 offsets are moved to ``rhs_ratio``.
    """
    w = plan.window_edge_span(d, k_eff)
    edge = w.start_edge if side == "start" else w.end_edge
    wrap = 0 if side == "start" else w.wrap
    kind, src, offset = plan.expanded_edge_sources(d)[edge]
    if kind == "const":
        return [], -coef * (float(offset) + wrap)
    terms = [_boundary_term(plan.intersection, plan.id, d, src, coef)]
    return terms, -coef * (offset + wrap)


def _compile_cycle_lock(rule: SemanticRule, problem: ProblemInput,
                        plans: dict[str, PlanSpec], report: dict) -> None:
    assert rule.target is not None
    target = rule.target
    if target.window.kind != "green":
        raise _fail(rule, "green_at_cycle_start/end requires a green window", target)
    side = "start" if rule.kind == "green_at_cycle_start" else "end"
    gs = resolve_green_slot(problem, target, rule.id)
    plan = plans[gs.plan.id]
    d = gs.direction
    w = plan.derived_windows(d)[gs.effective_index]
    n_edges = len(plan.expanded_edge_ratios(d))
    ok = (w.start_edge == 0) if side == "start" else (w.end_edge == n_edges - 1)
    if not ok:
        bit = 0 if side == "start" else -1
        raise _fail(rule, (
            f"green window does not {'start' if side == 'start' else 'end'} at cycle "
            f"{'start' if side == 'start' else 'end'}; set green_mask[{bit}] explicitly "
            f"so the complete green interval lies inside one cycle"), target)
    report["generated_constraints"].append(
        {"rule": rule.id, "plan": plan.id, "kind": rule.kind,
         "constraint_id": None, "path": "validated_mask"})


def _compile_sync(rule: SemanticRule, problem: ProblemInput,
                  plans: dict[str, PlanSpec], report: dict) -> None:
    if len(rule.targets) != 2:
        raise _fail(rule, "sync requires exactly two targets")
    a, b = rule.targets
    if a.window.kind != "green" or b.window.kind != "green":
        raise _fail(rule, "sync requires two green windows")
    if a.plan != b.plan:
        raise _fail(rule, "sync targets must belong to the same plan")
    if rule.relation not in ("start_start", "end_end", "start_end", "end_start"):
        raise _fail(rule, f"unsupported sync relation {rule.relation!r}")
    ga = resolve_green_slot(problem, a, rule.id)
    gb = resolve_green_slot(problem, b, rule.id)
    if ga.plan.id != gb.plan.id:
        raise _fail(rule, "sync targets must belong to the same plan")
    side_a, side_b = {
        "start_start": ("start", "start"),
        "end_end": ("end", "end"),
        "start_end": ("start", "end"),
        "end_start": ("end", "start"),
    }[rule.relation]
    plan = plans[ga.plan.id]
    terms_a, rhs_a = _endpoint_linear_terms(plan, ga.direction, ga.effective_index, side_a, 1.0)
    terms_b, rhs_b = _endpoint_linear_terms(plan, gb.direction, gb.effective_index, side_b, -1.0)
    terms = tuple(terms_a + terms_b)
    rhs_ratio = rhs_a + rhs_b
    if not terms:
        if abs(rhs_ratio) <= 1e-12:
            report["generated_constraints"].append(
                {"rule": rule.id, "plan": plan.id, "kind": rule.kind,
                 "constraint_id": None, "path": "validated_noop"})
            return
        raise _fail(rule, "sync endpoints are cycle edges/offsets that cannot be equal", a)
    c = _make_constraint(rule, terms, sense="==", rhs_ratio=rhs_ratio)
    plans[plan.id] = _append_plan_constraint(plans[plan.id], c)
    report["generated_constraints"].append(
        {"rule": rule.id, "plan": plan.id, "kind": rule.kind,
         "constraint_id": c.id, "path": "plan_constraint"})


def _compile_duration_limit(rule: SemanticRule, problem: ProblemInput,
                            plans: dict[str, PlanSpec], report: dict) -> None:
    assert rule.target is not None
    target = rule.target
    if rule.sense not in ("<=", ">="):
        raise _fail(rule, "duration_limit sense must be '<=' or '>='", target)
    if rule.value is None or rule.unit not in ("s", "ratio"):
        raise _fail(rule, "duration_limit requires value and unit in {'s','ratio'}", target)
    rhs_s = rule.value if rule.unit == "s" else None
    rhs_ratio = rule.value if rule.unit == "ratio" else None
    if target.window.kind == "green":
        gs = resolve_green_slot(problem, target, rule.id)
        term = _green_duration_term(target, gs.plan.id, gs.direction, gs.effective_index)
        c = _make_constraint(rule, (term,), sense=rule.sense,
                             rhs_s=rhs_s, rhs_ratio=rhs_ratio)
        plans[gs.plan.id] = _append_plan_constraint(plans[gs.plan.id], c)
        report["generated_constraints"].append(
            {"rule": rule.id, "plan": gs.plan.id, "kind": rule.kind,
             "constraint_id": c.id, "path": "plan_constraint"})
        return
    rr = resolve_red_interval(problem, target, rule.id)
    term = LinearTerm(atom=AtomRef(kind="red_duration", intersection=target.intersection,
                                   plan=rr.plan.id, direction=rr.direction, window=rr.rid),
                      coef=1.0)
    c = _make_constraint(rule, (term,), sense=rule.sense,
                         rhs_s=rhs_s, rhs_ratio=rhs_ratio)
    plans[rr.plan.id] = _append_plan_constraint(plans[rr.plan.id], c)
    report["generated_constraints"].append(
        {"rule": rule.id, "plan": rr.plan.id, "kind": rule.kind,
         "constraint_id": c.id, "path": "plan_constraint", "red_rid": rr.rid})


def compile_semantics(problem_json: dict, semantics_json: dict) -> tuple[ProblemInput, CompileReport, dict]:
    """Compile semantic rules into a ProblemInput and a report."""
    problem = ProblemInput.from_obj(problem_json)
    sem = SemanticInput.from_obj(semantics_json)
    seen_ids = set()
    for rule in sem.rules:
        if rule.id in seen_ids:
            raise GWSemanticError(rule.id, "", "duplicate semantic rule id")
        seen_ids.add(rule.id)
    plans = {p.id: p for p in problem.plans}
    report: dict = {"warnings": [], "generated_constraints": [],
                    "generated_templates": [], "red_ledgers": []}

    for rule in sem.rules:
        targets = ([rule.target] if rule.target is not None else []) + list(rule.targets)
        for t in targets:
            if t is not None and t.window.kind == "red":
                plan = problem.plan(t.plan)
                report["red_ledgers"].append({
                    "rule": rule.id,
                    "intersection": t.intersection,
                    "plan": t.plan,
                    "direction": t.direction,
                    "lines": list(describe_red_ledger(plan, t.direction)),
                })

    for rule in sem.rules:
        if rule.kind in ("green_at_cycle_start", "green_at_cycle_end"):
            _compile_cycle_lock(rule, problem, plans, report)
        elif rule.kind == "sync":
            _compile_sync(rule, problem, plans, report)
        elif rule.kind == "duration_limit":
            _compile_duration_limit(rule, problem, plans, report)
        else:
            raise _fail(rule, f"unsupported semantic rule kind {rule.kind!r}")

    new_plans = tuple(plans[p.id] for p in problem.plans)
    new_problem = replace(problem, plans=new_plans)
    report_model = CompileReport(
        red_ledgers=tuple(report["red_ledgers"]),
        locked_endpoints=(),
        generated_constraints=tuple(report["generated_constraints"]),
        generated_templates=(),
        warnings=tuple(report["warnings"]),
    )
    return new_problem, report_model, new_problem.model_dump()


__all__ = ["compile_semantics"]
