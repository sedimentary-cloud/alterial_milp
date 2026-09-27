"""语义规则编译器：高层规则 -> ProblemInput 增量补丁。"""
from __future__ import annotations

from dataclasses import replace

from ..errors import GWSemanticError
from ..schema.models import (AtomRef, ConstraintScope, ConstraintSpec,
                             ConstraintTemplate, LinearTerm, PlanSpec,
                             ProblemInput, SoftPenalty, WindowSpec)
from .ledger import describe_red_ledger, red_ledger
from .models import SemanticInput, SemanticRule, Target
from .report import CompileReport
from .resolver import resolve_green_slot, resolve_red_interval


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


def _set_window_side(plan: PlanSpec, d: str, orig_index: int, side: str,
                     value: float, rule: SemanticRule) -> PlanSpec:
    wins = list(plan.windows.get(d, ()))
    if orig_index < 0 or orig_index >= len(wins):
        raise _fail(rule, f"declared green window index {orig_index} does not exist")
    w = wins[orig_index]
    wins[orig_index] = replace(w, **{f"{side}_r": float(value)})
    new_windows = dict(plan.windows)
    new_windows[d] = tuple(wins)
    return replace(plan, windows=new_windows)


def _endpoint_term(target: Target, plan_id: str, d: str, k_eff: int, side: str,
                   coef: float) -> LinearTerm:
    return LinearTerm(
        atom=AtomRef(kind="endpoint", intersection=target.intersection,
                     plan=plan_id, direction=d, window=k_eff, side=side),
        coef=coef,
    )


def _green_duration_term(target: Target, plan_id: str, d: str, k_eff: int,
                         coef: float = 1.0) -> LinearTerm:
    return LinearTerm(
        atom=AtomRef(kind="window_duration", intersection=target.intersection,
                     plan=plan_id, direction=d, window=k_eff),
        coef=coef,
    )


def _compile_cycle_lock(rule: SemanticRule, problem: ProblemInput, plans: dict[str, PlanSpec],
                        locks: dict[tuple, tuple[str, str]], report: dict) -> None:
    assert rule.target is not None
    target = rule.target
    if target.window.kind != "green":
        raise _fail(rule, "green_at_cycle_start/end requires a green window", target)
    side = "start" if rule.kind == "green_at_cycle_start" else "end"
    value = 0.0 if side == "start" else 1.0
    gs = resolve_green_slot(problem, target, rule.id)
    plan = plans[gs.plan.id]
    key = (gs.plan.id, gs.direction, gs.effective_index, side)
    old = locks.get(key)
    if old is not None and old[0] != rule.id:
        if old[1] != value:
            raise _fail(rule, f"endpoint already locked to {old[1]} by rule {old[0]!r}",
                        target)
        report["warnings"].append(
            f"rule {rule.id!r}: endpoint already locked to {value} by {old[0]!r}; merged")
    locks[key] = (rule.id, str(value))

    if plan.endpoint_is_adjustable(gs.direction, gs.effective_index, side):
        term = _endpoint_term(target, plan.id, gs.direction, gs.effective_index, side, 1.0)
        if side == "start":
            c = _make_constraint(rule, (term,), sense="==", rhs_s=0.0)
        else:
            c = _make_constraint(rule, (term,), sense="==", rhs_ratio=1.0)
        plans[plan.id] = _append_plan_constraint(plan, c)
        report["generated_constraints"].append(
            {"rule": rule.id, "plan": plan.id, "kind": rule.kind,
             "constraint_id": c.id, "path": "plan_constraint"})
    else:
        plans[plan.id] = _set_window_side(plan, gs.direction, gs.original_index, side, value, rule)
        report["generated_constraints"].append(
            {"rule": rule.id, "plan": plan.id, "kind": rule.kind,
             "constraint_id": None, "path": "rewrite_window",
             "window_index": gs.original_index, "side": side, "value": value})


def _compile_sync(rule: SemanticRule, problem: ProblemInput, plans: dict[str, PlanSpec],
                  report: dict) -> None:
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
    terms = (
        _endpoint_term(a, ga.plan.id, ga.direction, ga.effective_index, side_a, 1.0),
        _endpoint_term(b, gb.plan.id, gb.direction, gb.effective_index, side_b, -1.0),
    )
    c = _make_constraint(rule, terms, sense="==", rhs_s=0.0)
    plans[ga.plan.id] = _append_plan_constraint(plans[ga.plan.id], c)
    report["generated_constraints"].append(
        {"rule": rule.id, "plan": ga.plan.id, "kind": rule.kind,
         "constraint_id": c.id, "path": "plan_constraint"})


def _compile_duration_limit(rule: SemanticRule, problem: ProblemInput,
                            plans: dict[str, PlanSpec], templates: list,
                            report: dict) -> None:
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
    # red
    rr = resolve_red_interval(problem, target, rule.id)
    term = LinearTerm(atom=AtomRef(kind="red_duration"), coef=1.0)
    scope = ConstraintScope(kind="all_plan_red_windows",
                            directions=(rr.direction,),
                            plan_ids=(rr.plan.id,),
                            window_indices=(rr.rid,),
                            skip_double_cycle_copies=True)
    soft = _penalty(rule)
    tmpl = ConstraintTemplate(
        id=f"sem:{rule.id}", scope=scope, terms=(term,), sense=rule.sense,
        hard=(soft is None), rhs_s=rhs_s, rhs_ratio=rhs_ratio, soft=soft,
    )
    templates.append(tmpl)
    report["generated_templates"].append(
        {"rule": rule.id, "plan": rr.plan.id, "kind": rule.kind,
         "template_id": tmpl.id, "red_rid": rr.rid})


def compile_semantics(problem_json: dict, semantics_json: dict) -> tuple[ProblemInput, CompileReport, dict]:
    """Compile semantic rules into a ProblemInput and a report.

    Returns ``(problem, report, problem_json_after_patch)``.
    """
    problem = ProblemInput.from_obj(problem_json)
    sem = SemanticInput.from_obj(semantics_json)
    seen_ids = set()
    for rule in sem.rules:
        if rule.id in seen_ids:
            raise GWSemanticError(rule.id, "", "duplicate semantic rule id")
        seen_ids.add(rule.id)
    plans = {p.id: p for p in problem.plans}
    templates = list(problem.constraint_templates)
    locks: dict[tuple, tuple[str, str]] = {}
    report: dict = {"warnings": [], "generated_constraints": [],
                    "generated_templates": [], "red_ledgers": []}

    # Record red ledgers for every plan referenced by a red rule.
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
            _compile_cycle_lock(rule, problem, plans, locks, report)
        elif rule.kind == "sync":
            _compile_sync(rule, problem, plans, report)
        elif rule.kind == "duration_limit":
            _compile_duration_limit(rule, problem, plans, templates, report)
        else:
            raise _fail(rule, f"unsupported semantic rule kind {rule.kind!r}")

    new_plans = tuple(plans[p.id] for p in problem.plans)
    new_problem = replace(problem, plans=new_plans, constraint_templates=tuple(templates))
    report_model = CompileReport(
        red_ledgers=tuple(report["red_ledgers"]),
        locked_endpoints=tuple(
            {"plan": k[0], "direction": k[1], "window": k[2], "side": k[3],
             "rule": v[0], "value": v[1]}
            for k, v in locks.items()
        ),
        generated_constraints=tuple(report["generated_constraints"]),
        generated_templates=tuple(report["generated_templates"]),
        warnings=tuple(report["warnings"]),
    )
    return new_problem, report_model, new_problem.model_dump()


__all__ = ["compile_semantics"]
