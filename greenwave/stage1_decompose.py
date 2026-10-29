"""Stage-1 的语义子问题分解。

给高中生看的小课堂
------------------
Stage-1 的模型长这样：相位边界 δ 已经固定，剩下的是"组合层"——每一条绿波/红波带
选哪个窗口、绕几个周期、带宽多宽，再加每个路口的周期偏置 φ。

为什么可以按"语义"（相邻对）把它拆开？
------------------------------------
看一条带在第 j 个路口的包含约束：

    u + T_j - φ_j - s_k - n·C - M(1-a) ≥ 0

它只牵涉**这一条带自己的** u/β/e/n/a 和它经过的路口的 φ。也就是说：

* 模型里**没有**任何一行同时约束两条不同的带（除非用户写了 global_constraints /
  balanced_groups 这类跨带的东西）；
* 唯一被多条带共用的自由变量是 φ_j，而 φ_j 在约束里只以 ``u - φ_j`` 的形式出现——
  u 是每条带自己的自由变量，所以 φ_j 只是一个"公共平移"（gauge）：把某个组内
  所有 φ 和 u 同时平移，约束不变。

于是：只要没有跨带的约束行，把带按"共享路口"切成若干组分别求解，再把各组解
按"相邻组之间只有一个共享路口"串起来，就是一个**精确**分解（全局最优 =
Σ 组最优）。这正是"全部是相邻对窗口绿波/红波"这种目标的常见形态。

什么时候不能拆（本模块会保守地把组并回去，或干脆放弃分解）
------------------------------------------------------
1. 两条带共享 ≥ 2 个路口——它们各自约束同一组相对偏置，拆开会互相矛盾；
2. 用户在 global_constraints / balanced_groups / constraint_templates /
   方案自带的约束里把不同组的需求（或不同组的 φ）连在一起；
3. 共享路口是多方案路口（有方案选择变量 x 和窗口可用性 h，被两条带同时引用）；
4. 拼出来的解在完整模型上校验不通过（此时返回 None，调用方回退整模 MIP）。

因此分解不是"猜"的：本模块最后一定会把拼出来的解**代回完整模型逐行校验**，
校验不过就放弃，绝不返回一个不满足原模型的解。
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from .errors import GWModelError
from .model.builder import ArterialModel, ModelBuilder
from .model import naming
from .preprocess.context import GridPointContext
from .preprocess.feasibility import FeasibilityFilter
from .report.solution import SolutionDecoder, SolutionRecord
from .schema.models import (ConstraintSpec, ConstraintTemplate, ProblemInput)
from .solve import make_backend
from .solve.backend_base import SolveOptions, SolveResult, SolveStatus

# 拼装解的容差：行右端越大，允许的绝对误差越大。
_VERIFY_ABS_TOL = 1e-6
_VERIFY_REL_TOL = 1e-9
# 最优性证书的相对容差：拼装解必须达到"各组自由（未被钉住）最优值之和"。
_CERT_REL_TOL = 1e-5


def _certify(assembled_obj: float, upper_bound: float) -> bool:
    """最优性证书。

    各组**自由**（接口 φ 不被钉住）求解得到的是上界 ``upper_bound``：任何可行解
    都不可能超过各组上界之和。如果拼装出来的解（已逐行校验过可行性）达到了这个
    上界，那它就是全局最优，可以安全返回；否则说明"把接口 φ 钉住"损失了自由度
    （例如红带的回绕是用离散的红窗实例下标编码的，不一定能重新吸收 φ 的平移），
    此时必须放弃分解、回退整模——绝不返回一个"可行但次优"的解冒充最优。
    """
    tol = _CERT_REL_TOL * max(1.0, abs(upper_bound)) + _VERIFY_ABS_TOL
    return assembled_obj + tol >= upper_bound


@dataclass(frozen=True)
class DemandGroup:
    """一组可以独立求解的需求，以及它在干线上覆盖的路口区间。"""

    demand_ids: tuple[str, ...]
    lo: int
    hi: int
    node_ids: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.node_ids[0]}-{self.node_ids[-1]}"

    @property
    def node_set(self) -> frozenset[str]:
        return frozenset(self.node_ids)


@dataclass(frozen=True)
class DecomposeOutcome:
    solution: SolutionRecord
    result: SolveResult
    groups: tuple[DemandGroup, ...]
    group_times_s: tuple[float, ...]


# ---------------------------------------------------------------------------
# 1. 语义分组
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[object, object] = {}

    def add(self, key: object) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: object) -> object:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: object, b: object) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _merge_demands_sharing_two_nodes(problem: ProblemInput, uf: _UnionFind) -> None:
    """两条带共享 ≥2 个路口时必须同组（它们约束同一组相对偏置）。"""
    nodes = problem.corridor.intersections
    for i, di in enumerate(problem.objective.band_demands):
        si = set(di.nodes)
        for dj in problem.objective.band_demands[i + 1:]:
            if len(si & set(dj.nodes)) >= 2:
                uf.union(("d", di.id), ("d", dj.id))


def _cross_scopes(problem: ProblemInput) -> list[tuple[set[str], set[str], set[str]]]:
    """收集"可能跨越子问题"的约束作用域：(需求集, 路口集, 方案集)。

    一条约束只要牵涉到的对象落在不同组里，这些组就必须并回去。
    """
    scopes: list[tuple[set[str], set[str], set[str]]] = []

    def atoms(c: ConstraintSpec) -> tuple[set[str], set[str]]:
        dems = {t.atom.demand for t in c.terms
                if t.atom.kind == "bandwidth" and t.atom.demand}
        iids = {t.atom.intersection for t in c.terms
                if t.atom.kind == "offset" and t.atom.intersection}
        return dems, iids

    for c in problem.global_constraints:
        dems, iids = atoms(c)
        scopes.append((dems, iids, set()))
    for p in problem.plans:
        for c in list(p.hard_constraints) + list(p.soft_constraints):
            dems, iids = atoms(c)
            # 方案约束一定牵涉自己的路口；再加上 offset 原子指向的路口。
            scopes.append((dems, iids | {p.intersection}, {p.id}))
    for tmpl in problem.constraint_templates:
        scopes.append((set(), set(), set(tmpl.scope.plan_ids or ())))
    for g in problem.objective.balanced_groups:
        scopes.append((set(g.demands), set(), set()))
    return scopes


def _merge_touched_groups(problem: ProblemInput,
                          runs: Sequence[tuple[tuple[str, ...], int, int]],
                          scopes: Sequence[tuple[set[str], set[str], set[str]]],
                          multi_plan_nodes: set[str]) -> list[list[str]]:
    """把"放不进同一个子问题"的组并起来。

    注意：相邻组的覆盖区间会共享 1 个 gauge 路口，所以"某条约束引用了这个共享
    路口"**不**意味着要并组——只要存在一个组能同时容纳这条约束引用的所有对象
    （需求 + 路口）即可，否则才并组。
    """
    nodes = problem.corridor.intersections
    index = {iid: i for i, iid in enumerate(nodes)}
    plan_iid = {p.id: p.intersection for p in problem.plans}
    run_of_demand = {did: k for k, (ids, _lo, _hi) in enumerate(runs) for did in ids}

    def holders(iid: str) -> list[int]:
        i = index[iid]
        return [k for k, (_ids, lo, hi) in enumerate(runs) if lo <= i <= hi]

    parent = list(range(len(runs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for dems, iids, pids in scopes:
        req_nodes = set(iids) | {plan_iid[p] for p in pids if p in plan_iid}
        already = any(
            set(runs[k][0]) >= dems and all(k in holders(n) for n in req_nodes)
            for k in range(len(runs))
        )
        if already:
            continue
        need = {run_of_demand[d] for d in dems if d in run_of_demand}
        for node in req_nodes:
            hs = holders(node)
            if not hs or need & set(hs):
                continue
            need.add(hs[0])
        ordered = sorted(need)
        for k in ordered[1:]:
            union(ordered[0], k)

    # 多方案路口：x 与窗口可用性 h 会被该路口的所有带共用，必须并组。
    for iid in multi_plan_nodes:
        hs = holders(iid)
        for k in hs[1:]:
            union(hs[0], k)

    buckets: dict[int, list[str]] = {}
    for k, run in enumerate(runs):
        buckets.setdefault(find(k), []).extend(run[0])
    return [list(dict.fromkeys(v)) for v in buckets.values()]


def _demand_groups_from_uf(problem: ProblemInput, uf: _UnionFind) -> list[list[str]]:
    buckets: dict[object, list[str]] = {}
    for b in problem.objective.band_demands:
        buckets.setdefault(uf.find(("d", b.id)), []).append(b.id)
    return [v for v in buckets.values()]


def _runs_for_groups(problem: ProblemInput, groups: Sequence[Sequence[str]]
                     ) -> list[tuple[tuple[str, ...], int, int]]:
    """把需求组变成覆盖整条干线的连续区间（缺口挂到右边的组，尾部补到最后一组）。"""
    index = {iid: i for i, iid in enumerate(problem.corridor.intersections)}
    by_id = {b.id: b for b in problem.objective.band_demands}
    spans = []
    for g in groups:
        idxs = [index[n] for did in g for n in by_id[did].nodes]
        spans.append((tuple(g), min(idxs), max(idxs)))
    spans.sort(key=lambda x: (x[1], x[2]))

    n_nodes = len(problem.corridor.intersections)
    out: list[tuple[tuple[str, ...], int, int]] = []
    prev_hi = -1
    for demand_ids, lo, hi in spans:
        new_lo = min(prev_hi + 1, lo)
        new_hi = max(hi, new_lo)
        out.append((demand_ids, new_lo, new_hi))
        prev_hi = new_hi
    if out:
        last_ids, last_lo, _ = out[-1]
        out[-1] = (last_ids, last_lo, max(out[-1][2], n_nodes - 1))
    return out


def _merge_overlapping_runs(problem: ProblemInput,
                            runs: list[tuple[tuple[str, ...], int, int]]
                            ) -> list[tuple[tuple[str, ...], int, int]]:
    """两个组的覆盖区间重叠 ≥2 个路口时必须并组（否则相对偏置会冲突）。"""
    changed = True
    while changed and len(runs) > 1:
        changed = False
        runs = sorted(runs, key=lambda x: (x[1], x[2]))
        merged: list[tuple[tuple[str, ...], int, int]] = []
        for demand_ids, lo, hi in runs:
            if merged:
                p_ids, p_lo, p_hi = merged[-1]
                if min(p_hi, hi) - max(p_lo, lo) + 1 >= 2:
                    merged[-1] = (tuple(dict.fromkeys(p_ids + demand_ids)),
                                  min(p_lo, lo), max(p_hi, hi))
                    changed = True
                    continue
            merged.append((demand_ids, lo, hi))
        runs = merged
    return runs


def semantic_demand_groups(problem: ProblemInput,
                           ctx: GridPointContext,
                           *, merge_multi_plan: bool = True
                           ) -> tuple[DemandGroup, ...]:
    """把需求按"可独立求解"的语义关系分组。

    返回的组按路口顺序排列；相邻组的覆盖区间最多重叠 1 个路口（共享的 gauge
    路口），缺口路口会被并进相邻组，保证所有路口都被覆盖。

    规则（迭代到定点）：
    1. 两条带共享 ≥2 个路口 -> 同组（它们约束同一组相对偏置）；
    2. 两个组的覆盖区间重叠 ≥2 个路口 -> 并组；
    3. 同一条跨带约束（global / 方案约束 / 模板 / 均衡组）牵涉到多个组 -> 并组；
    4. 共享路口是多方案路口 -> 并组（x 与窗口可用性 h 被多条带同时引用）。

    ``merge_multi_plan=False`` 时跳过规则 4——这时组可能更细，但组之间只通过
    "共享路口的方案选择"耦合；只要把方案固定下来，分解依然精确。这就是
    "方案枚举 + 链式 DP" 的前提（见 :func:`plan_dp_candidate`）。
    """
    if not problem.objective.band_demands:
        return ()

    uf = _UnionFind()
    for b in problem.objective.band_demands:
        uf.add(("d", b.id))
    _merge_demands_sharing_two_nodes(problem, uf)
    groups = _demand_groups_from_uf(problem, uf)

    scopes = _cross_scopes(problem)
    multi_plan = ({iid for iid in problem.corridor.intersections
                   if len(ctx.feasible_plans.get(iid, ())) > 1}
                  if merge_multi_plan else set())

    for _ in range(len(problem.objective.band_demands) + 1):
        runs = _merge_overlapping_runs(problem, _runs_for_groups(problem, groups))
        merged = _merge_touched_groups(problem, runs, scopes, multi_plan)
        if len(merged) == len(groups):
            break
        groups = merged
    return _finalize_runs(problem, groups)


def _finalize_runs(problem: ProblemInput, groups: Sequence[Sequence[str]]
                   ) -> tuple[DemandGroup, ...]:
    nodes = problem.corridor.intersections
    runs = _runs_for_groups(problem, groups)
    out: list[DemandGroup] = []
    for demand_ids, lo, hi in runs:
        out.append(DemandGroup(demand_ids=tuple(demand_ids), lo=lo, hi=hi,
                               node_ids=tuple(nodes[lo:hi + 1])))
    return tuple(out)


# ---------------------------------------------------------------------------
# 2. 子问题构造与求解
# ---------------------------------------------------------------------------

def build_subproblem(problem: ProblemInput, group: DemandGroup) -> ProblemInput:
    """把整条干线问题裁成这个组覆盖的那一段（需求和方案都是原对象的子集）。

    具名聚合：target=band_objective 的项会进入阶段 1 的目标，无法按组拆分，
    直接拒绝；target=intersection_loss 的项在阶段 1 不可见，子问题里丢掉，
    拼装完整解时再由 :func:`_fill_aggregate_values` 补上聚合变量。
    """
    if problem.aggregates_affect_composite():
        raise GWModelError(
            "stage1 decomposition does not support aggregate contributions "
            "to band_objective; fall back to full MILP")
    index = {iid: i for i, iid in enumerate(problem.corridor.intersections)}
    nodes = problem.corridor.intersections[group.lo:group.hi + 1]
    node_set = set(nodes)

    corridor = replace(problem.corridor,
                       corridor_id=f"{problem.corridor.corridor_id}::{group.label}",
                       intersections=nodes,
                       segments=problem.corridor.segments[group.lo:group.hi])

    keep_plan_ids = {p.id for p in problem.plans if p.intersection in node_set}
    plans = tuple(p for p in problem.plans if p.id in keep_plan_ids)

    demands = tuple(b for b in problem.objective.band_demands
                    if b.id in set(group.demand_ids))
    demand_ids = {b.id for b in demands}
    balanced = tuple(g for g in problem.objective.balanced_groups
                     if set(g.demands) <= demand_ids)
    objective = replace(problem.objective, band_demands=demands,
                        balanced_groups=balanced)

    globals_kept = []
    for c in problem.global_constraints:
        touched = {a.demand for t in c.terms for a in [t.atom]
                   if a.kind == "bandwidth" and a.demand}
        touched |= {a.intersection for t in c.terms for a in [t.atom]
                    if a.kind == "offset" and a.intersection}
        if touched and touched <= (demand_ids | node_set):
            globals_kept.append(c)

    templates_kept = tuple(t for t in problem.constraint_templates
                           if _template_is_inside(t, keep_plan_ids))

    multistage = replace(problem.solver.multistage, decompose_stage1=False,
                         enabled=False)
    solver = replace(problem.solver, multistage=multistage)

    # 只保留全局/方案/模板约束；聚合在阶段 1 不可见，子问题不建。
    return replace(problem, corridor=corridor, plans=plans, objective=objective,
                   solver=solver, global_constraints=tuple(globals_kept),
                   constraint_templates=templates_kept, aggregates=())


def _template_is_inside(tmpl: ConstraintTemplate, plan_ids: set[str]) -> bool:
    scope = tmpl.scope
    if scope.plan_ids is None:
        return True          # 全体作用域：按节点展开，拆开是等价的
    return set(scope.plan_ids) <= plan_ids


def _free_var_count(model: ArterialModel) -> int:
    return sum(1 for i in range(len(model.vars))
               if abs(model.vars.lbs[i] - model.vars.ubs[i]) > 1e-9)


def _freeze_boundaries(model: ArterialModel,
                       adjustments_r: Mapping[tuple[str, str, int], float]) -> None:
    for (pid, d, j), h in model.meta.boundary_delta.items():
        val = float(adjustments_r.get((pid, d, j), 0.0))
        lb = model.vars.lbs[h.index]
        ub = model.vars.ubs[h.index]
        if val < lb - 1e-9 or val > ub + 1e-9:
            raise GWModelError(
                f"boundary {pid}/{d}/{j} value {val} outside [{lb}, {ub}]")
        model.vars.fix(h, min(max(val, lb), ub))


@dataclass(frozen=True)
class _GroupSolution:
    group: DemandGroup
    model: ArterialModel
    values: np.ndarray
    objective: float
    time_s: float


def _solve_group(problem: ProblemInput, ctx: GridPointContext, group: DemandGroup,
                 *, adjustments_r: Mapping[tuple[str, str, int], float],
                 symmetry: bool, min_band_width_s: float,
                 time_limit_s: float, mip_rel_gap: float,
                 fixed_phi: Mapping[str, float],
                 plan_assign: Mapping[str, str] | None = None) -> _GroupSolution | None:
    """求解一个语义组；``plan_assign`` 非空时把每个路口的方案固定成指定方案。"""
    sub = build_subproblem(problem, group)
    if plan_assign is not None:
        sub = replace(sub, plans=tuple(
            p for p in sub.plans if plan_assign.get(p.intersection) == p.id))
        if len(sub.plans) != len(sub.corridor.intersections):
            return None          # 有路口没有可用方案
    sub_ctx = FeasibilityFilter(sub).filter(ctx.cycle_s, ctx.kappa)
    # 共享路口的 φ 必须在建模时就固定：回绕整数 n 的可行区间由 φ 的上下界推出，
    # 建完再改 φ 会让 n 区间与真实 φ 不匹配，从而卡住子问题的最优解。
    phi_fix = {iid: float(val) for iid, val in fixed_phi.items() if iid in group.node_ids}
    model = ModelBuilder(sub, sub_ctx, stage1_symmetry=symmetry,
                         min_band_width_s=min_band_width_s,
                         phi_fix=phi_fix).build()
    _freeze_boundaries(model, adjustments_r)
    backend = make_backend(sub.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, SolveOptions(time_limit_s=time_limit_s,
                                              mip_rel_gap=mip_rel_gap))
    if res.values is None:
        return None
    return _GroupSolution(group=group, model=model, values=res.values,
                          objective=float(res.objective or 0.0),
                          time_s=float(res.time_s))


def _group_penalty_composite(problem: ProblemInput, plan_id: str,
                             adjustments_r: Mapping[tuple[str, str, int], float],
                             cycle_s: float) -> float:
    """某个方案在固定 δ 下**计入 composite** 的常数罚（owner=band_loss 的软约束）。

    链式 DP 里接口路口被两个组共享，这份常数罚会被两边各计一次；用它做一次扣除，
    保证 Σ_g f_g 与整模 composite 的口径一致。
    """
    plan = problem.plan(plan_id)
    if not plan.soft_constraints:
        return 0.0
    deltas = {(d, j): float(v) for (pid, d, j), v in adjustments_r.items()
              if pid == plan_id}
    meas = FeasibilityFilter(problem)
    total = 0.0
    for c in plan.soft_constraints:
        if c.soft is None or c.soft.owner != "band_loss":
            continue
        lhs = meas._constraint_value(plan, c, cycle_s, deltas)  # noqa: SLF001
        rhs = c.rhs_for_cycle(cycle_s)
        if c.sense == "<=":
            viol = max(0.0, lhs - rhs)
        elif c.sense == ">=":
            viol = max(0.0, rhs - lhs)
        else:
            viol = abs(lhs - rhs)
        total += float(c.soft.coef) * viol
    return total


# ---------------------------------------------------------------------------
# 3. 拼装 + 校验
# ---------------------------------------------------------------------------

def _verify_rows(model: ArterialModel, values: np.ndarray) -> str | None:
    """把解代回完整模型逐行校验；返回第一条违反的说明，全部通过则返回 None。"""
    for row in model.cons.rows:
        lhs = row.expr.evaluate(values)
        tol = _VERIFY_ABS_TOL + _VERIFY_REL_TOL * abs(row.rhs)
        if row.sense == "<=":
            bad = lhs > row.rhs + tol
        elif row.sense == ">=":
            bad = lhs < row.rhs - tol
        else:
            bad = abs(lhs - row.rhs) > tol
        if bad:
            return (f"{row.meta.group}:{row.meta.label} {row.sense} {row.rhs:.6g} "
                    f"but lhs={lhs:.6g}")
    return None


def _selected_plan_id(model: ArterialModel, values: np.ndarray,
                      intersection: str) -> str | None:
    """从拼装后的完整变量值里读出某路口选中的方案 id。"""
    meta = model.meta
    for pid, x in meta.plan_vars.items():
        if meta.problem.plan(pid).intersection == intersection and float(values[x.index]) > 0.5:
            return pid
    fixed = meta.fixed_plans.get(intersection)
    if fixed is not None:
        return fixed
    feasible = meta.ctx.feasible_plans.get(intersection, ())
    return feasible[0].id if len(feasible) == 1 else None


def _fill_aggregate_values(model: ArterialModel, values: np.ndarray) -> None:
    """按完整解补全聚合变量 t（子问题里没有建聚合）。

    t 只影响阶段 2/Pareto 的解码与整模逐行校验；阶段 1 的 composite 不含它。
    """
    meta = model.meta
    for am in meta.aggregates.values():
        if am.var is None:
            continue
        selected = am.available
        if selected and am.spec.plan is not None:
            plan = meta.problem.plan(am.spec.plan)
            selected = (_selected_plan_id(model, values, plan.intersection) == plan.id)
        if selected:
            vals = [e.evaluate(values) for e in am.expressions]
            value = max(vals) if am.spec.op == "max" else min(vals)
        else:
            value = 0.0
        values[am.var.index] = float(value)


def _assemble(problem: ProblemInput, model: ArterialModel,
              solved: Sequence[_GroupSolution], *,
              label: str,
              plan_by_node: Mapping[str, str] | None = None,
              ) -> DecomposeOutcome | None:
    """把各子问题的解按变量名拼装回完整模型，逐行校验后解码。

    ``plan_by_node`` 非空时（方案枚举+DP 路径）额外补上整模独有的
    ``x``（方案选择）与 ``bz = x·δ`` 两类变量。
    """
    full_index = {name: i for i, name in enumerate(model.vars.names)}
    values = np.zeros(len(model.vars))
    covered = np.zeros(len(model.vars), dtype=bool)

    for got in solved:
        for i, name in enumerate(got.model.vars.names):
            idx = full_index.get(name)
            if idx is None:
                return None          # 子问题变量不该在整模里找不到
            if not covered[idx]:
                values[idx] = float(got.values[i])
                covered[idx] = True
            elif (naming.is_family(name, naming.OFFSET)
                  and abs(values[idx] - float(got.values[i])) > 1e-6):
                return None          # 共享路口 φ 没能对齐

    if plan_by_node is not None:
        # 未选中方案的变量在整模里存在（bd / bz / 方案软约束的松弛变量），但固定
        # 方案的子模型里没有；它们对目标没有贡献，置 0 即可（随后逐行校验兜底）。
        def _plan_of(name: str) -> str | None:
            return naming.plan_of(name)

        unselected = {p.id for p in problem.plans
                      if plan_by_node.get(p.intersection) != p.id}
        for name, idx in full_index.items():
            if naming.is_family(name, naming.PLAN_ON):
                iid, pid = naming.keys_of(name)
                values[idx] = 1.0 if plan_by_node.get(iid) == pid else 0.0
                covered[idx] = True
            elif naming.is_family(name, naming.BND_XDELTA):
                pid, d, j = naming.keys_of(name)
                iid = problem.plan(pid).intersection
                bd_idx = full_index.get(naming.bnd_delta(pid, d, j))
                chosen = plan_by_node.get(iid) == pid
                values[idx] = float(values[bd_idx]) if (chosen and bd_idx is not None) else 0.0
                covered[idx] = True
            elif not covered[idx] and _plan_of(name) in unselected:
                values[idx] = 0.0
                covered[idx] = True

    if not covered.all():
        # 子问题是按"钉住的 φ 范围"推导可行窗口键的，可能比整模少几个键
        # （多方案时还少未选中方案的 bd/bz/viol）。少掉的键在整模里同样不可能
        # 取到有效值，置 0 即可——下面的逐行校验是兜底：只要有一行不成立就返回
        # None 回退整模，所以这里不会放过"看起来可行"的错误解。
        for idx in np.flatnonzero(~covered):
            values[idx] = 0.0
        covered[:] = True

    # 子问题里没有聚合变量；按整模的选中方案/δ 补出 t，再逐行校验。
    _fill_aggregate_values(model, values)
    bad = _verify_rows(model, values)
    if bad is not None:
        return None

    sol = SolutionDecoder(model).decode(values)
    obj = float(sol.composite_s)
    times = [g.time_s for g in solved]
    res = SolveResult(
        status=SolveStatus.OPTIMAL,
        objective=obj,
        values=values,
        bound=obj,
        time_s=float(sum(times)),
        message=f"{label}, group solve time {sum(times):.3f}s",
    )
    return DecomposeOutcome(solution=sol, result=res,
                            groups=tuple(g.group for g in solved),
                            group_times_s=tuple(times))


def solve_stage1_decomposed(
    problem: ProblemInput,
    ctx: GridPointContext,
    model: ArterialModel,
    *,
    adjustments_r: Mapping[tuple[str, str, int], float],
    symmetry: bool = False,
    min_band_width_s: float = 0.0,
    time_limit_s: float | None = None,
    mip_rel_gap: float | None = None,
    min_group_size: int = 2,
) -> DecomposeOutcome | None:
    """按语义子问题分解求解 Stage-1。

    ``model`` 必须是已经固定好 δ 的完整模型（调用方已经建好，避免重复建模）。
    返回 ``None`` 表示"不适用或拼装校验失败"，调用方应回退整模 MIP。
    """
    groups = semantic_demand_groups(problem, ctx)
    if len(groups) < 2:
        return None

    limit = float(time_limit_s if time_limit_s is not None
                  else problem.solver.time_limit_s)
    gap = float(mip_rel_gap if mip_rel_gap is not None
                else problem.solver.mip_rel_gap)

    # 逐组求解：后一组如果和前一组共享路口，就把那个 φ 固定成前一组的值，
    # 这样两组的解可以直接拼在一起（φ 是 gauge，只有相对值有意义）。
    # 同时记录"各组自由（不钉 φ）最优值之和"作为最优性证书的上界。
    phi_known: dict[str, float] = {}
    solved: list[_GroupSolution] = []
    upper_bound = 0.0
    for group in groups:
        fixed_phi = {iid: phi_known[iid] for iid in group.node_ids
                     if iid in phi_known}
        got = _solve_group(problem, ctx, group, adjustments_r=adjustments_r,
                           symmetry=symmetry, min_band_width_s=min_band_width_s,
                           time_limit_s=limit, mip_rel_gap=gap,
                           fixed_phi=fixed_phi)
        if got is None:
            return None
        solved.append(got)
        if fixed_phi:
            free = _solve_group(problem, ctx, group, adjustments_r=adjustments_r,
                                symmetry=symmetry, min_band_width_s=min_band_width_s,
                                time_limit_s=limit, mip_rel_gap=gap,
                                fixed_phi={})
            if free is None:
                return None
            upper_bound += float(free.objective)
        else:
            upper_bound += float(got.objective)
        for iid in group.node_ids:
            try:
                phi_known[iid] = float(
                    got.values[got.model.vars.by_name(naming.offset(iid)).index])
            except GWModelError:
                continue

    # 兼容原先 decompose_min_component_size 的语义：至少要有两个"规模达标"的组，
    # 否则分解没有意义，交回整模 MIP。
    if sum(1 for g in solved
           if _free_var_count(g.model) >= max(1, int(min_group_size))) < 2:
        return None

    outcome = _assemble(problem, model, solved,
                        label=(f"stage1 decomposed into {len(groups)} semantic groups "
                               f"({', '.join(g.label for g in groups)})"))
    if outcome is None:
        return None
    if not _certify(float(outcome.solution.composite_s), upper_bound):
        return None
    return outcome


# ---------------------------------------------------------------------------
# 4. 方案枚举 + 链式 DP（多方案路口）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DpPlan:
    """枚举+DP 的可行性描述：语义组 + 每路口可行方案 + 子问题求解次数。"""

    groups: tuple[DemandGroup, ...]
    plans_by_node: Mapping[str, tuple[str, ...]]
    n_solves: int

    @property
    def is_chain(self) -> bool:
        """相邻组是否只共享 1 个路口（链式结构，DP 状态才不会爆炸）。"""
        for a, b in zip(self.groups, self.groups[1:]):
            if len(set(a.node_ids) & set(b.node_ids)) != 1:
                return False
        return True


def plan_dp_candidate(problem: ProblemInput, ctx: GridPointContext, *,
                      max_solves: int = 400) -> DpPlan | None:
    """判断"方案枚举 + 链式 DP"是否适用；不适用返回 None。

    适用条件：
    1. 忽略"多方案并组"后，语义组数 ≥ 2 且构成链（相邻组只共享 1 个路口）；
    2. 至少有一个路口是多方案（否则走普通语义分解就够了）；
    3. 枚举规模 ``Σ_g Π_{i∈g}|P_i|`` 不超过 ``max_solves``。
    """
    groups = semantic_demand_groups(problem, ctx, merge_multi_plan=False)
    if len(groups) < 2:
        return None
    plans_by_node: dict[str, tuple[str, ...]] = {}
    for iid in problem.corridor.intersections:
        ids = tuple(p.id for p in ctx.feasible_plans.get(iid, ()))
        if not ids:
            return None
        plans_by_node[iid] = ids
    if all(len(v) <= 1 for v in plans_by_node.values()):
        return None
    n_solves = 0
    for g in groups:
        prod = 1
        for iid in g.node_ids:
            prod *= len(plans_by_node[iid])
        n_solves += prod
    if n_solves > max(1, int(max_solves)):
        return None
    plan = DpPlan(groups=groups, plans_by_node=plans_by_node, n_solves=n_solves)
    return plan if plan.is_chain else None


def _plan_combos(values: Sequence[tuple[str, ...]]) -> Iterable[tuple[str, ...]]:
    return itertools.product(*values) if values else [()]


def probe_first_subproblem(
    problem: ProblemInput,
    ctx: GridPointContext,
    plan: DpPlan,
    *,
    adjustments_r: Mapping[tuple[str, str, int], float],
    symmetry: bool = False,
    min_band_width_s: float = 0.0,
    time_limit_s: float | None = None,
    mip_rel_gap: float | None = None,
) -> tuple[float | None, dict[tuple[int, tuple[tuple[str, str], ...]], _GroupSolution]]:
    """跑 DP 的第一个子问题，用于预估总耗时；结果放进 cache 供 DP 复用。"""
    limit = float(time_limit_s if time_limit_s is not None
                  else problem.solver.time_limit_s)
    gap = float(mip_rel_gap if mip_rel_gap is not None
                else problem.solver.mip_rel_gap)
    cache: dict[tuple[int, tuple[tuple[str, str], ...]], _GroupSolution] = {}
    if not plan.groups:
        return None, cache
    group = plan.groups[0]
    assign = {iid: plan.plans_by_node[iid][0] for iid in group.node_ids}
    t0 = time.perf_counter()
    got = _solve_group(problem, ctx, group, adjustments_r=adjustments_r,
                       symmetry=symmetry, min_band_width_s=min_band_width_s,
                       time_limit_s=limit, mip_rel_gap=gap,
                       fixed_phi={}, plan_assign=assign)
    elapsed = time.perf_counter() - t0
    if got is None:
        return None, cache
    cache[(0, tuple(sorted(assign.items())))] = got
    return elapsed, cache


def solve_stage1_plan_dp(
    problem: ProblemInput,
    ctx: GridPointContext,
    model: ArterialModel,
    *,
    plan: DpPlan,
    adjustments_r: Mapping[tuple[str, str, int], float],
    symmetry: bool = False,
    min_band_width_s: float = 0.0,
    time_limit_s: float | None = None,
    mip_rel_gap: float | None = None,
    cache: dict[tuple[int, tuple[tuple[str, str], ...]], _GroupSolution] | None = None,
) -> DecomposeOutcome | None:
    """多方案路口：枚举各语义组的方案组合 + 沿走廊做链式 DP。

    固定方案后每个路口的窗口都是常数，``x``/``h`` 的耦合消失，于是
    ``全局最优 = max_π Σ_g f_g(π|_{nodes(g)})``，而 ``f_g`` 只依赖它自己那几个
    路口的方案；组构成链、相邻组只共享 1 个路口 -> 用 DP 精确求解，代价
    ``Σ_g Π_{i∈g}|P_i|`` 次子问题求解（而不是 ``Π_i |P_i|``）。

    接口路口的 φ 用"从左到右链式固定"的方式对齐（与普通语义分解一致），
    所以 DP 只需要决定**方案指派**，最后再按该指派重解一次各组拼装。
    """
    limit = float(time_limit_s if time_limit_s is not None
                  else problem.solver.time_limit_s)
    gap = float(mip_rel_gap if mip_rel_gap is not None
                else problem.solver.mip_rel_gap)
    cycle = ctx.cycle_s
    cache = {} if cache is None else cache

    nodes = problem.corridor.intersections
    # 每个接口路口"计入 composite 的常数罚"只算一次：归到它是 hi 端点的那个组。
    penalty_cache: dict[tuple[str, str], float] = {}

    def penalty(iid: str, pid: str) -> float:
        key = (iid, pid)
        if key not in penalty_cache:
            penalty_cache[key] = _group_penalty_composite(problem, pid,
                                                          adjustments_r, cycle)
        return penalty_cache[key]

    def solve_combo(g_idx: int, assign: dict[str, str]) -> _GroupSolution | None:
        key = (g_idx, tuple(sorted(assign.items())))
        if key not in cache:
            got = _solve_group(problem, ctx, plan.groups[g_idx],
                               adjustments_r=adjustments_r,
                               symmetry=symmetry, min_band_width_s=min_band_width_s,
                               time_limit_s=limit, mip_rel_gap=gap,
                               fixed_phi={}, plan_assign=assign)
            if got is None:
                return None
            cache[key] = got
        return cache[key]

    # --- 逐组枚举方案组合，只保留 (lo 方案, hi 方案) -> 最优值 ---
    best_per_group: list[dict[tuple[str, str], tuple[float, dict[str, str]]]] = []
    for g_idx, group in enumerate(plan.groups):
        lo_iid, hi_iid = group.node_ids[0], group.node_ids[-1]
        table: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}
        for p_lo in plan.plans_by_node[lo_iid]:
            for p_hi in plan.plans_by_node[hi_iid]:
                if len(group.node_ids) == 2:
                    internal_choices = [()]
                    internal_nodes: tuple[str, ...] = ()
                else:
                    internal_nodes = tuple(n for n in group.node_ids[1:-1])
                    internal_choices = _plan_combos(
                        [plan.plans_by_node[n] for n in internal_nodes])
                for combo in internal_choices:
                    assign = {lo_iid: p_lo, hi_iid: p_hi}
                    assign.update(zip(internal_nodes, combo))
                    got = solve_combo(g_idx, assign)
                    if got is None:
                        continue
                    val = float(got.objective)
                    if g_idx != len(plan.groups) - 1:
                        val -= penalty(hi_iid, p_hi)     # 接口罚只计一次
                    key = (p_lo, p_hi)
                    if key not in table or val > table[key][0]:
                        table[key] = (val, assign)
        if not table:
            return None
        best_per_group.append(table)

    # --- 链式 DP ---
    dp: dict[str, tuple[float, dict[str, str]]] = {}
    back: list[dict[str, tuple[str, dict[str, str]]]] = []
    first = best_per_group[0]
    layer0: dict[str, tuple[str, dict[str, str]]] = {}
    for (p_lo, p_hi), (val, assign) in first.items():
        cur = dp.get(p_hi)
        if cur is None or val > cur[0]:
            dp[p_hi] = (val, assign)
            layer0[p_hi] = (p_lo, assign)
    back.append(layer0)
    for g_idx in range(1, len(plan.groups)):
        table = best_per_group[g_idx]
        new_dp: dict[str, tuple[float, dict[str, str]]] = {}
        layer: dict[str, tuple[str, dict[str, str]]] = {}
        for (p_lo, p_hi), (val, assign) in table.items():
            prev = dp.get(p_lo)
            if prev is None:
                continue
            total = prev[0] + val
            if p_hi not in new_dp or total > new_dp[p_hi][0]:
                new_dp[p_hi] = (total, assign)
                layer[p_hi] = (p_lo, assign)
        if not new_dp:
            return None
        dp = new_dp
        back.append(layer)

    # --- 回溯出最优方案指派 ---
    best_hi = max(dp, key=lambda k: dp[k][0])
    dp_value = float(dp[best_hi][0])
    plan_by_node: dict[str, str] = {}
    cur_hi = best_hi
    for g_idx in range(len(plan.groups) - 1, -1, -1):
        p_lo, assign = back[g_idx][cur_hi]
        plan_by_node.update(assign)
        cur_hi = p_lo
    if set(plan_by_node) != set(nodes):
        return None

    # --- 按最优指派重解各组（链式固定 φ）并拼装 ---
    phi_known: dict[str, float] = {}
    solved: list[_GroupSolution] = []
    for group in plan.groups:
        fixed_phi = {iid: phi_known[iid] for iid in group.node_ids if iid in phi_known}
        got = _solve_group(problem, ctx, group, adjustments_r=adjustments_r,
                           symmetry=symmetry, min_band_width_s=min_band_width_s,
                           time_limit_s=limit, mip_rel_gap=gap,
                           fixed_phi=fixed_phi, plan_assign=plan_by_node)
        if got is None:
            return None
        solved.append(got)
        for iid in group.node_ids:
            phi_known[iid] = float(got.values[got.model.vars.by_name(naming.offset(iid)).index])

    outcome = _assemble(problem, model, solved,
                       label=(f"stage1 plan-enumeration DP: {len(plan.groups)} groups, "
                              f"{len(cache)} sub-solves, plans "
                              + ", ".join(f"{i}={plan_by_node[i]}" for i in nodes)),
                       plan_by_node=plan_by_node)
    if outcome is None:
        return None
    if not _certify(float(outcome.solution.composite_s), dp_value):
        return None
    return outcome


__all__ = ["DemandGroup", "DecomposeOutcome", "DpPlan", "semantic_demand_groups",
           "build_subproblem", "plan_dp_candidate", "probe_first_subproblem",
           "solve_stage1_decomposed",
           "solve_stage1_plan_dp"]
