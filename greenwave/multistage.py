"""多阶段快速优化：固定组合 + 固定边界枚举 + LP 交替迭代。

设计定位
--------
完整 MILP 仍然保留为 ``greenwave.run`` 的默认路径。本模块提供一条
工程化的快速启发式路径：

1. **Stage 1**：固定相位边界（可选连周期偏置 φ 一起固定），重新选择
   组合变量 ``x / e / a / n / ag / ar``。这里暂时复用完整 MILP 模型，
   把边界 δ 固定后求解一个缩小的 MIP。
2. **Stage 2**：固定 Stage 1 得到的组合变量，放开 ``φ / δ / u / β / σ``，
   求解纯 LP。
3. **交替迭代**：用 Stage 2 得到的偏置/边界再跑 Stage 1，重复直到
   目标不再提升。

核心技巧是把所有组合变量固定后，MILP 退化成 LP；固定边界后，窗口
端点都变成常数，组合搜索也大幅缩小。它不是全局最优，但适合大规模
快速 incumbent、超时 fallback 和多起点搜索。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .errors import GWModelError
from .heuristic.greedy import ConstructedBand, GreedyConstructor, GreedySolution
from .model.builder import ArterialModel, ModelBuilder
from .model.constraints import ConstraintMeta, ConstraintStore
from .model.expressions import LinearExpr
from .model.variables import GVarType, VarHandle, VariableRegistry
from .preprocess.context import GridPointContext
from .report.solution import SolutionDecoder, SolutionRecord
from .schema.models import ProblemInput
from .solve import make_backend
from .solve.backend_base import SolveOptions, SolveResult, SolveStatus


# ---------------------------------------------------------------------------
# 先验数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorBand:
    """一条带槽的先验组合信息。窗口槽统一使用 1-based 编号。"""

    demand: str
    slot: int
    exists: bool = True
    width_s: float = 0.0
    start_s: float = 0.0
    windows: Mapping[str, int] = field(default_factory=dict)
    wraps: Mapping[str, int] = field(default_factory=dict)
    red_instance: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class PriorSolution:
    """多阶段算法使用的先验组合解。

    该结构只保存“组合层”信息，不要求 u/β/φ/δ 的精确值；Stage 2 会重新
    优化连续变量。
    """

    plan_selection: Mapping[str, str]
    adjustments_r: Mapping[tuple[str, str, int], float] = field(default_factory=dict)
    bands: Mapping[tuple[str, int], PriorBand] = field(default_factory=dict)
    demand_order: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    source: str = ""

    def band(self, demand: str, slot: int) -> PriorBand | None:
        return self.bands.get((demand, int(slot)))

    def green_bands(self, demand: str):
        return [b for (d, _), b in self.bands.items()
                if d == demand and b.red_instance is None and b.exists]

    def red_bands(self, demand: str):
        return [b for (d, _), b in self.bands.items()
                if d == demand and b.red_instance is not None and b.exists]


@dataclass(frozen=True)
class StageTimeRecord:
    """一次 Stage-1 或 Stage-2 调用的耗时记录。"""

    start_idx: int
    round_idx: int
    stage: str              # "stage1" | "stage2"
    time_s: float
    objective: float | None = None


@dataclass(frozen=True)
class MultistageResult:
    """多阶段求解结果，可直接转成 GridPointResult。"""

    status: str
    objective: float | None
    solution: SolutionRecord | None
    time_s: float
    rounds: int = 0
    history: tuple[float, ...] = ()
    message: str = ""
    # 多阶段各阶段累计耗时；用于 benchmark / profile。
    stage1_time_s: float = 0.0
    stage2_time_s: float = 0.0
    stage_records: tuple[StageTimeRecord, ...] = ()


# ---------------------------------------------------------------------------
# 先验提取/校验
# ---------------------------------------------------------------------------

def _band_from_constructed(band: ConstructedBand) -> PriorBand | None:
    if band is None or band.width_s <= 1e-7:
        return None
    red_instance = None
    if band.red_instance is not None:
        if len(band.red_instance) < 3:
            return None
        g, r, k = (int(band.red_instance[0]), int(band.red_instance[1]),
                   int(band.red_instance[2]))
        red_instance = (g, r, k)
    return PriorBand(
        demand=str(band.demand),
        slot=int(band.slot),
        exists=True,
        width_s=float(band.width_s),
        start_s=float(band.start_s),
        windows={str(k): int(v) for k, v in dict(band.windows).items()},
        wraps={str(k): int(v) for k, v in dict(getattr(band, "wraps", {})).items()},
        red_instance=red_instance,
    )


def _order_from_bands(bands: Iterable[PriorBand]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for band in bands:
        grouped.setdefault(band.demand, []).append(int(band.slot))
    return {did: tuple(sorted(slots)) for did, slots in grouped.items()}


def extract_prior(solution: SolutionRecord, problem: ProblemInput,
                  ctx: GridPointContext, *, source: str = "solution") -> PriorSolution:
    """从 ``SolutionRecord`` 提取组合先验。

    解码器输出的窗口槽已经是 1-based；这里只做结构整理和宽度过滤。
    """
    if solution is None:
        raise GWModelError("cannot extract prior from None solution")
    bands: dict[tuple[str, int], PriorBand] = {}
    for band in solution.bands:
        pb = _band_from_constructed(band)
        if pb is None:
            continue
        bands[(pb.demand, pb.slot)] = pb
    return PriorSolution(
        plan_selection={str(k): str(v) for k, v in dict(solution.plan_selection).items()},
        adjustments_r={tuple(k): float(v) for k, v in dict(solution.adjustments_r).items()},
        bands=bands,
        demand_order=_order_from_bands(bands.values()),
        source=source,
    )


def extract_prior_from_greedy(greedy: GreedySolution, problem: ProblemInput,
                              ctx: GridPointContext) -> PriorSolution:
    """从贪心解提取组合先验。

    新版的 ``GreedyConstructor`` 已把窗口槽统一成 1-based；该函数主要
    负责过滤零宽带和补齐 order 信息。
    """
    if greedy is None:
        raise GWModelError("cannot extract prior from None greedy solution")
    bands: dict[tuple[str, int], PriorBand] = {}
    for band in greedy.bands:
        pb = _band_from_constructed(band)
        if pb is None:
            continue
        bands[(pb.demand, pb.slot)] = pb
    return PriorSolution(
        plan_selection={str(k): str(v) for k, v in dict(greedy.plan_selection).items()},
        adjustments_r={tuple(k): float(v) for k, v in dict(greedy.adjustments_r).items()},
        bands=bands,
        demand_order=_order_from_bands(bands.values()),
        source="greedy",
    )


def validate_prior(problem: ProblemInput, ctx: GridPointContext,
                   prior: PriorSolution) -> tuple[str, ...]:
    """检查先验组合是否结构合法。

    返回问题字符串元组；空元组表示通过。这里只做结构/范围校验，具体
    的约束可行性由 Stage 1/Stage 2 的求解器保证。
    """
    issues: list[str] = []
    for iid in problem.corridor.intersections:
        pid = prior.plan_selection.get(iid)
        if pid is None:
            issues.append(f"{iid}: missing plan selection")
            continue
        feasible_ids = {p.id for p in ctx.feasible_plans.get(iid, ())}
        if pid not in feasible_ids:
            issues.append(f"{iid}: selected plan {pid!r} is not feasible in this grid point")

    # 同一需求、同一路口、同一窗口的占用情况，用于共享关闭时的保守校验。
    occupied: set[tuple[str, str, int]] = set()
    for (did, slot), band in prior.bands.items():
        demand = problem.demand_or_none(did)
        if demand is None:
            issues.append(f"unknown demand {did!r} in prior")
            continue
        if not band.exists:
            continue
        if did in prior.demand_order and slot not in prior.demand_order[did]:
            issues.append(f"{did}/{slot}: slot not listed in demand_order")

        if demand.type == "green":
            if len(band.windows) != len(demand.nodes):
                issues.append(f"{did}/{slot}: green windows keys do not match demand nodes")
                continue
            if set(band.wraps) != set(demand.nodes):
                issues.append(f"{did}/{slot}: green wraps keys do not match demand nodes")
            for node in demand.nodes:
                pid = prior.plan_selection.get(node)
                if pid is None:
                    continue
                plan = problem.plan(pid)
                k = int(band.windows.get(node, 0))
                if not (1 <= k <= plan.window_count(demand.direction)):
                    issues.append(
                        f"{did}/{slot}/{node}: window slot {k} outside 1..{plan.window_count(demand.direction)}")
                occupied_key = (did, node, k)
                if not problem.objective.allow_window_sharing and occupied_key in occupied:
                    issues.append(f"{did}: window sharing not allowed but slot {k} reused at {node}")
                occupied.add(occupied_key)
        else:
            if len(demand.nodes) != 2 or band.red_instance is None:
                issues.append(f"{did}/{slot}: red band needs red_instance(g,r,k)")
                continue
            g, r, k = (int(band.red_instance[0]), int(band.red_instance[1]),
                       int(band.red_instance[2]))
            up_node, down_node = demand.nodes
            if up_node not in prior.plan_selection or down_node not in prior.plan_selection:
                continue
            up_plan = problem.plan(prior.plan_selection[up_node])
            down_plan = problem.plan(prior.plan_selection[down_node])
            if not (1 <= g <= up_plan.window_count(demand.direction)):
                issues.append(f"{did}/{slot}: upstream green slot {g} out of range")
            # RedWindowBuilder 为每个有效绿窗都生成一个 wrap 红窗实例，
            # 因此模型内红窗槽数是 2K-1，而不是 plan.red_window_count 的 K。
            k_green = max(1, down_plan.window_count(demand.direction))
            max_red_slot = max(1, 2 * k_green - 1)
            if not (1 <= r <= max_red_slot):
                issues.append(f"{did}/{slot}: downstream red slot {r} out of range 1..{max_red_slot}")
            if k < -10 or k > 10:
                issues.append(f"{did}/{slot}: downstream period shift k={k} looks out of range")
    return tuple(issues)


# ---------------------------------------------------------------------------
# 固定组合 / 边界
# ---------------------------------------------------------------------------

def _fix_checked(model: ArterialModel, h: VarHandle, value: float) -> None:
    lb = model.vars.lbs[h.index]
    ub = model.vars.ubs[h.index]
    v = float(value)
    tol = 1e-7
    if v < lb - tol or v > ub + tol:
        raise GWModelError(
            f"prior value {v} for {h.name} outside variable bounds [{lb}, {ub}]")
    model.vars.fix(h, min(max(v, lb), ub))


def unfixed_integrality(model: ArterialModel) -> tuple[str, ...]:
    """返回仍然未被上下界固定的二进制/整数变量名。"""
    out: list[str] = []
    for h in model.vars:
        if model.vars.types[h.index] in (GVarType.BINARY, GVarType.INTEGER):
            if abs(model.vars.lbs[h.index] - model.vars.ubs[h.index]) > 1e-9:
                out.append(h.name)
    return tuple(out)


def _is_fixed_var(model: ArterialModel, idx: int, tol: float = 1e-9) -> bool:
    return abs(model.vars.lbs[idx] - model.vars.ubs[idx]) <= tol


def find_stage1_components(model: ArterialModel, *, tol: float = 1e-9) -> tuple[tuple[int, ...], ...]:
    """按变量-约束耦合图求 Stage-1 的连通分量。

    关键约定：**固定变量视为常数，不作为图的顶点。** 如果某个固定变量出现在
    一条约束里，它只能把常数项移到右边，不能把两个自由变量错误地连在一起。
    """
    free = [idx for idx in range(len(model.vars)) if not _is_fixed_var(model, idx, tol)]
    if not free:
        return ()
    parent = {idx: idx for idx in free}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in model.cons.rows:
        # 只收集自由变量；固定变量直接忽略，绝不参与连通性传播。
        row_vars = [idx for idx, coef in row.expr.items()
                    if abs(coef) > tol and idx in parent]
        if len(row_vars) >= 2:
            first = row_vars[0]
            for idx in row_vars[1:]:
                union(first, idx)

    groups: dict[int, list[int]] = {}
    for idx in free:
        groups.setdefault(find(idx), []).append(idx)
    return tuple(tuple(sorted(v)) for v in groups.values())


def _component_model(model: ArterialModel, component: tuple[int, ...]) -> ArterialModel:
    """把完整 Stage-1 模型沿一个连通分量切片成独立子模型。

    固定变量被替换为常数项，不进入局部变量表；这与建图阶段的
    “固定变量视为常数”保持一致。
    """
    local_vars = VariableRegistry()
    local_map: dict[int, VarHandle] = {}
    for full_idx in component:
        name = model.vars.names[full_idx]
        meta = model.vars.metas[full_idx]
        local = local_vars.add(name, model.vars.lbs[full_idx], model.vars.ubs[full_idx],
                               model.vars.types[full_idx], meta)
        local_map[full_idx] = local

    def local_expr(expr: LinearExpr, context: str, *, skip_foreign: bool = False) -> LinearExpr:
        out = LinearExpr.constant(float(expr.const))
        for full_idx, coef in expr.items():
            if full_idx in local_map:
                out.add(local_map[full_idx], coef)
            elif _is_fixed_var(model, full_idx):
                # 固定变量视为常数，不进入局部变量表。
                out.add_const(coef * model.vars.lbs[full_idx])
            elif skip_foreign:
                # 目标函数按变量可加；其他分量/孤立变量在目标里单独处理。
                continue
            else:
                raise GWModelError(
                    f"{context}: cross-component variable {model.vars.names[full_idx]!r} detected")
        return out

    local_cons = ConstraintStore()
    for row in model.cons.rows:
        # 完全不涉及当前分量的约束直接跳过；涉及两个分量的约束说明建图漏了边。
        if not any(idx in local_map for idx, _coef in row.expr.items()):
            continue
        expr = local_expr(row.expr, f"constraint {row.meta.label}")
        if expr.is_constant():
            c = expr.const
            if row.sense == "<=" and c > row.rhs + 1e-7:
                raise GWModelError(f"fixed row infeasible: {row.meta.label}")
            if row.sense == ">=" and c < row.rhs - 1e-7:
                raise GWModelError(f"fixed row infeasible: {row.meta.label}")
            if row.sense == "==" and abs(c - row.rhs) > 1e-7:
                raise GWModelError(f"fixed row infeasible: {row.meta.label}")
            continue
        local_cons.add(expr, row.sense, row.rhs, row.meta)

    objective = local_expr(model.objective or LinearExpr(), "objective", skip_foreign=True)
    return ArterialModel(local_vars, local_cons, objective, model.objective_sense, model.meta)


def freeze_combo(model: ArterialModel, prior: PriorSolution) -> None:
    """把先验组合变量固定到模型上。"""
    problem = model.meta.problem
    ctx = model.meta.ctx
    meta = model.meta

    # 1) 方案选择 x。
    for iid in problem.corridor.intersections:
        pid = prior.plan_selection.get(iid)
        if pid is None:
            raise GWModelError(f"prior missing plan selection for {iid}")
        if iid in meta.fixed_plans:
            if meta.fixed_plans[iid] != pid:
                raise GWModelError(
                    f"prior selected plan {pid!r}, but model fixed plan is {meta.fixed_plans[iid]!r}")
            continue
        for p in ctx.feasible_plans.get(iid, ()):
            h = meta.plan_vars.get(p.id)
            if h is not None:
                _fix_checked(model, h, 1.0 if p.id == pid else 0.0)

    # 2) 绿带：e、a、n。
    for b in problem.objective.band_demands:
        if b.type != "green":
            continue
        gm = meta.green.get(b.id)
        if gm is None:
            continue
        for q in range(1, gm.qmax + 1):
            pb = prior.band(b.id, q)
            if pb is not None and pb.exists:
                _fix_checked(model, gm.e[q], 1.0)
                for j, node in enumerate(b.nodes):
                    k = int(pb.windows.get(node, 0))
                    if k <= 0 or (q, j, k) not in gm.a:
                        raise GWModelError(
                            f"prior {b.id}/{q}/{node}: green window slot {k} is not available")
                    for kk in range(1, ctx.max_window_slots(node, b.direction) + 1):
                        h = gm.a.get((q, j, kk))
                        if h is not None:
                            _fix_checked(model, h, 1.0 if kk == k else 0.0)
                    n = pb.wraps.get(node)
                    if n is None:
                        raise GWModelError(f"prior {b.id}/{q}/{node}: missing wrap period n")
                    nv = gm.n.get((q, j))
                    if nv is not None:
                        _fix_checked(model, nv, int(n))
            else:
                _fix_checked(model, gm.e[q], 0.0)
                if q in gm.beta:
                    _fix_checked(model, gm.beta[q], 0.0)
                if q in gm.u:
                    _fix_checked(model, gm.u[q], 0.0)
                for j, node in enumerate(b.nodes):
                    for kk in range(1, ctx.max_window_slots(node, b.direction) + 1):
                        h = gm.a.get((q, j, kk))
                        if h is not None:
                            _fix_checked(model, h, 0.0)
                    nv = gm.n.get((q, j))
                    if nv is not None:
                        _fix_checked(model, nv, 0.0)

    # 3) 红带：e、ag、ar。
    for b in problem.objective.band_demands:
        if b.type != "red":
            continue
        rm = meta.red_demands.get(b.id)
        if rm is None:
            continue
        for q in range(1, rm.qmax + 1):
            pb = prior.band(b.id, q)
            if pb is not None and pb.exists and pb.red_instance is not None:
                g, r, k = (int(pb.red_instance[0]), int(pb.red_instance[1]),
                           int(pb.red_instance[2]))
                if (q, g) not in rm.ag:
                    raise GWModelError(f"prior {b.id}/{q}: upstream green slot {g} unavailable")
                if (q, r, k) not in rm.ar:
                    raise GWModelError(
                        f"prior {b.id}/{q}: downstream red instance ({r},{k}) unavailable")
                _fix_checked(model, rm.e[q], 1.0)
                for gg in rm.ag_slots:
                    h = rm.ag.get((q, gg))
                    if h is not None:
                        _fix_checked(model, h, 1.0 if gg == g else 0.0)
                for (rr, kk) in rm.ar_keys:
                    h = rm.ar.get((q, rr, kk))
                    if h is not None:
                        _fix_checked(model, h, 1.0 if (rr, kk) == (r, k) else 0.0)
            else:
                _fix_checked(model, rm.e[q], 0.0)
                if q in rm.beta:
                    _fix_checked(model, rm.beta[q], 0.0)
                if q in rm.u:
                    _fix_checked(model, rm.u[q], 0.0)
                for g in rm.ag_slots:
                    h = rm.ag.get((q, g))
                    if h is not None:
                        _fix_checked(model, h, 0.0)
                for (r, k) in rm.ar_keys:
                    h = rm.ar.get((q, r, k))
                    if h is not None:
                        _fix_checked(model, h, 0.0)

    # 4) 均衡组的 y = OR(e)：先验组合已经决定了存在需求。
    for gid, bgm in meta.balanced.items():
        for did, y in bgm.y.items():
            exists = any(d == did and pb.exists for (d, _), pb in prior.bands.items())
            _fix_checked(model, y, 1.0 if exists else 0.0)


def _witness_values(ctx: GridPointContext) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for pid, entries in ctx.plan_witness_r.items():
        for d, j, val in entries:
            out[(pid, d, int(j))] = float(val)
    return out


def freeze_boundaries(
    model: ArterialModel,
    adjustments_r: Mapping[tuple[str, str, int], float],
    ctx: GridPointContext | None = None,
    *,
    fallback: Mapping[tuple[str, str, int], float] | None = None,
) -> None:
    """把可调相位边界 δ 固定为给定值。

    未在 ``adjustments_r`` 中出现的边界默认固定为 0（名义边界）。
    如果调用方希望用可行性 witness 作为缺失值回退，可显式传入
    ``fallback=...``。这样空字典就代表“全部固定为名义 δ”。
    """
    for (pid, d, j), h in model.meta.boundary_delta.items():
        if (pid, d, j) in adjustments_r:
            val = float(adjustments_r[(pid, d, j)])
        else:
            val = float((fallback or {}).get((pid, d, j), 0.0))
        _fix_checked(model, h, val)


# ---------------------------------------------------------------------------
# 求解阶段
# ---------------------------------------------------------------------------

def _solve_options(time_limit_s: float, mip_rel_gap: float = 0.0) -> SolveOptions:
    return SolveOptions(time_limit_s=float(time_limit_s), mip_rel_gap=float(mip_rel_gap))


def _component_is_nontrivial(model: ArterialModel, component: tuple[int, ...],
                            *, tol: float = 1e-9) -> bool:
    """分量是否含至少一个有效约束或非零目标项。"""
    comp_set = set(component)
    for row in model.cons.rows:
        if any(idx in comp_set and abs(coef) > tol for idx, coef in row.expr.items()):
            return True
    obj = model.objective or LinearExpr()
    return any(idx in comp_set and abs(coef) > tol for idx, coef in obj.items())


def _solve_decomposed_stage1(
    model: ArterialModel,
    problem: ProblemInput,
    ctx: GridPointContext,
    components: tuple[tuple[int, ...], ...],
    *,
    time_limit_s: float,
    mip_rel_gap: float,
) -> tuple[SolutionRecord, PriorSolution, ArterialModel, SolveResult] | None:
    """按变量-约束连通分量分别求解 Stage-1 子问题。

    返回 None 表示分解路径失败，调用方应回退整模 MIP。
    """
    backend = make_backend(problem.solver.backend)
    n = len(model.vars)
    full_values: list[float] = [float(model.vars.lbs[i]) for i in range(n)]
    total_time = 0.0
    parts: list[str] = []
    for comp in components:
        local_model = _component_model(model, comp)
        if len(local_model.vars) == 0:
            continue
        if len(local_model.cons.rows) == 0:
            # 无约束分量：直接取目标改善方向的边界，避免为无用变量创建子问题。
            obj_items = dict(local_model.objective.items())
            for local_idx, full_idx in enumerate(comp):
                coef = float(obj_items.get(local_idx, 0.0))
                lo = float(model.vars.lbs[full_idx])
                hi = float(model.vars.ubs[full_idx])
                if model.objective_sense == "max":
                    full_values[full_idx] = hi if coef > 0 else lo
                else:
                    full_values[full_idx] = lo if coef > 0 else hi
            parts.append(f"{len(comp)}vars/trivial")
            continue
        session = backend.create(local_model)
        backend.set_objective(session, local_model.objective or LinearExpr(),
                              model.objective_sense)
        res = backend.solve(session, _solve_options(time_limit_s, mip_rel_gap))
        total_time += float(res.time_s)
        if res.values is None:
            return None
        for local_idx, full_idx in enumerate(comp):
            full_values[full_idx] = float(res.values[local_idx])
        parts.append(f"{len(comp)}vars/{res.status.value}")

    sol = SolutionDecoder(model).decode(full_values)
    prior = extract_prior(sol, problem, ctx, source="stage1-decomposed")
    res = SolveResult(
        status=SolveStatus.OPTIMAL,
        objective=float(sol.composite_s),
        values=None,
        bound=None,
        time_s=total_time,
        message=f"stage1 decomposed into {len(parts)} components: {', '.join(parts)}",
    )
    return sol, prior, model, res


def solve_fixed_boundaries(
    problem: ProblemInput,
    ctx: GridPointContext,
    adjustments_r: Mapping[tuple[str, str, int], float],
    *,
    time_limit_s: float | None = None,
    mip_rel_gap: float | None = None,
    decompose: bool | None = None,
    symmetry_breaking: bool | None = None,
    min_band_width_s: float | None = None,
) -> tuple[SolutionRecord, PriorSolution, ArterialModel, SolveResult] | None:
    """Stage 1：固定 δ，求解组合 MIP。"""
    cfg_ms = problem.solver.multistage
    use_symmetry = (cfg_ms.stage1_symmetry_breaking
                    if symmetry_breaking is None else bool(symmetry_breaking))
    min_width = (cfg_ms.stage1_min_band_width_s
                 if min_band_width_s is None else float(min_band_width_s))
    model = ModelBuilder(problem, ctx, stage1_symmetry=use_symmetry,
                         min_band_width_s=min_width).build()
    freeze_boundaries(model, adjustments_r, ctx)

    use_decompose = (problem.solver.multistage.decompose_stage1
                     if decompose is None else bool(decompose))
    stage_limit = (time_limit_s if time_limit_s is not None
                   else problem.solver.time_limit_s)
    stage_gap = (mip_rel_gap if mip_rel_gap is not None
                 else problem.solver.mip_rel_gap)

    if use_decompose:
        components = find_stage1_components(model)
        nontrivial = [c for c in components if _component_is_nontrivial(model, c)]
        min_size = max(1, int(problem.solver.multistage.decompose_min_component_size))
        large = [c for c in nontrivial if len(c) >= min_size]
        if len(large) > 1:
            decomposed = _solve_decomposed_stage1(
                model, problem, ctx, components,
                time_limit_s=stage_limit, mip_rel_gap=stage_gap,
            )
            if decomposed is not None:
                return decomposed
            # 分解失败则继续走下面的整模 MIP。

    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    res = backend.solve(session, _solve_options(stage_limit, stage_gap))
    if res.values is None:
        return None
    sol = SolutionDecoder(model).decode(res.values)
    prior = extract_prior(sol, problem, ctx, source="stage1")
    return sol, prior, model, res


def solve_fixed_combo_lp(
    problem: ProblemInput,
    ctx: GridPointContext,
    prior: PriorSolution,
    *,
    time_limit_s: float | None = None,
) -> tuple[SolutionRecord, PriorSolution, ArterialModel, SolveResult] | None:
    """Stage 2：固定组合，放开 φ/δ/u/β，求解 LP。"""
    model = ModelBuilder(problem, ctx).build()
    freeze_combo(model, prior)
    left = unfixed_integrality(model)
    if left:
        raise GWModelError(
            "fixed-combo LP still has unfixed integrality: " + ", ".join(left[:10]))
    backend = make_backend(problem.solver.backend)
    session = backend.create(model)
    backend.set_objective(session, model.expr_composite(), "max")
    options = _solve_options(
        time_limit_s if time_limit_s is not None else problem.solver.time_limit_s,
        0.0,
    )
    res = backend.solve(session, options)
    if res.values is None:
        return None
    best_sol = SolutionDecoder(model).decode(res.values)
    best_res = res

    # 退化最优面 tie-break：在 composite 保持不变的前提下，尝试选择更大的
    # 相位边界调整量。LP 经常有多个同目标解，其中一个可能让后续 Stage 1
    # 继续改进。这里只在新解“实际 composite”不更差时才接受，避免 tie-break
    # 损害原目标。
    if res.objective is not None:
        delta_expr = LinearExpr()
        for iid, pid in prior.plan_selection.items():
            plan = problem.plan(pid)
            for d in ("up", "down"):
                for j in range(plan.boundary_count(d)):
                    h = model.meta.boundary_delta.get((pid, d, j))
                    if h is not None:
                        delta_expr.add(h, 1.0)
        if delta_expr.items():
            tol = 1e-6 * max(1.0, abs(float(res.objective)))
            backend.add_constraint(
                session, model.expr_composite(), ">=", float(res.objective) - tol,
                ConstraintMeta("MS_TIE", "keep composite for boundary tie-break", {}),
            )
            backend.set_objective(session, delta_expr, "max")
            res2 = backend.solve(session, options)
            if res2.values is not None:
                sol2 = SolutionDecoder(model).decode(res2.values)
                # 模型目标保持不变，但解码器重算的实际值可能有微小数值差异；
                # 给一个很小的接受带，让后续 Stage 1 有机会利用这个平局解。
                accept_tol = 1e-4 + 1e-6 * max(1.0, abs(float(best_sol.composite_s)))
                if sol2.composite_s >= best_sol.composite_s - accept_tol:
                    best_sol = sol2
                    best_res = res2

    new_prior = extract_prior(best_sol, problem, ctx, source="stage2")
    return best_sol, new_prior, model, best_res


# ---------------------------------------------------------------------------
# 多阶段调度
# ---------------------------------------------------------------------------

def _initial_adjustment_starts(problem: ProblemInput, ctx: GridPointContext,
                               restarts: int) -> list[dict[tuple[str, str, int], float]]:
    starts: list[dict[tuple[str, str, int], float]] = [{}]
    witness = _witness_values(ctx)
    if witness:
        starts.append(dict(witness))
    if problem.solver.heuristic.enabled:
        try:
            greedy = GreedyConstructor(problem, ctx).run()
            starts.append({tuple(k): float(v) for k, v in greedy.adjustments_r.items()})
        except Exception:
            pass

    # 收集所有可调边界范围，用于确定性的 restarts 扰动。
    ranges: list[tuple[tuple[str, str, int], float, float, float]] = []
    for p in problem.plans:
        for d in ("up", "down"):
            for j in range(p.boundary_count(d)):
                rng = p.boundary_adjust_range(d, j)
                if rng is not None:
                    ranges.append(((p.id, d, j), float(rng[0]), float(rng[1]),
                                   float(p.nominal_boundary_r(d, j))))

    base = starts[0]
    n_restarts = max(1, int(restarts))
    ridx = 0
    while len(starts) < n_restarts:
        if not ranges:
            starts.append(dict(base))
            break
        new = dict(base)
        ridx += 1
        for pos, (key, lo, hi, _nom) in enumerate(ranges):
            frac = ((ridx + pos) % 5) / 5.0
            new[key] = lo + frac * (hi - lo)
        starts.append(new)

    # 去重，避免同一个 start 重复跑。
    uniq: list[dict[tuple[str, str, int], float]] = []
    seen: set[tuple] = set()
    for s in starts:
        key = tuple(sorted((k, round(v, 12)) for k, v in s.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(dict(s))
    return uniq[:n_restarts]


def solve_multistage(
    problem: ProblemInput,
    ctx: GridPointContext,
    config=None,
) -> MultistageResult:
    """在当前格点上执行多阶段快速优化。"""
    cfg = config if config is not None else problem.solver.multistage
    t0 = time.perf_counter()
    starts = _initial_adjustment_starts(problem, ctx, cfg.restarts)
    best_sol: SolutionRecord | None = None
    best_prior: PriorSolution | None = None
    best_obj = -math.inf
    history: list[float] = []
    total_rounds = 0
    messages: list[str] = []
    stage1_time = 0.0
    stage2_time = 0.0
    stage_records: list[StageTimeRecord] = []

    stage1_limit = cfg.stage1_time_limit_s
    stage2_limit = cfg.stage2_time_limit_s

    for start_idx, adjustments in enumerate(starts):
        t_stage = time.perf_counter()
        s1 = solve_fixed_boundaries(problem, ctx, adjustments,
                                    time_limit_s=stage1_limit)
        dt_stage = time.perf_counter() - t_stage
        stage1_time += dt_stage
        stage_records.append(StageTimeRecord(
            start_idx=start_idx, round_idx=0, stage="stage1", time_s=dt_stage,
            objective=None if s1 is None else float(s1[0].composite_s)))
        if s1 is None:
            messages.append(f"start{start_idx}: stage1 no solution")
            continue
        cur_sol, cur_prior, _model, _res = s1
        cur_obj = float(cur_sol.composite_s)
        history.append(cur_obj)

        for round_idx in range(max(0, int(cfg.max_rounds))):
            total_rounds += 1
            t_stage = time.perf_counter()
            s2 = solve_fixed_combo_lp(problem, ctx, cur_prior,
                                      time_limit_s=stage2_limit)
            dt_stage = time.perf_counter() - t_stage
            stage2_time += dt_stage
            stage_records.append(StageTimeRecord(
                start_idx=start_idx, round_idx=round_idx, stage="stage2",
                time_s=dt_stage,
                objective=None if s2 is None else float(s2[0].composite_s)))
            if s2 is None:
                messages.append(f"start{start_idx}/round{round_idx}: stage2 no solution")
                break
            sol2, prior2, _m2, _r2 = s2
            obj2 = float(sol2.composite_s)
            if obj2 > cur_obj + float(cfg.improvement_tol_s):
                cur_sol, cur_prior, cur_obj = sol2, prior2, obj2
            else:
                history.append(obj2)
                break
            history.append(obj2)

            # 用 Stage 2 的 φ/δ 再跑一次 Stage 1，继续更新组合。
            t_stage = time.perf_counter()
            s1b = solve_fixed_boundaries(problem, ctx, cur_prior.adjustments_r,
                                         time_limit_s=stage1_limit)
            dt_stage = time.perf_counter() - t_stage
            stage1_time += dt_stage
            stage_records.append(StageTimeRecord(
                start_idx=start_idx, round_idx=round_idx, stage="stage1",
                time_s=dt_stage,
                objective=None if s1b is None else float(s1b[0].composite_s)))
            if s1b is None:
                break
            sol1b, prior1b, _m1b, _r1b = s1b
            obj1b = float(sol1b.composite_s)
            if obj1b > cur_obj + float(cfg.improvement_tol_s):
                cur_sol, cur_prior, cur_obj = sol1b, prior1b, obj1b
                history.append(obj1b)
            else:
                break

        if cur_obj > best_obj:
            best_obj = cur_obj
            best_sol = cur_sol
            best_prior = cur_prior

    elapsed = time.perf_counter() - t0
    if best_sol is None:
        return MultistageResult(
            status="infeasible", objective=None, solution=None, time_s=elapsed,
            rounds=total_rounds, history=tuple(history),
            message="; ".join(messages) or "multistage found no feasible solution",
            stage1_time_s=stage1_time, stage2_time_s=stage2_time,
            stage_records=tuple(stage_records),
        )
    msg = (f"multistage heuristic: rounds={total_rounds}, starts={len(starts)}, "
           f"stage1={stage1_time:.3f}s, stage2={stage2_time:.3f}s")
    if messages:
        msg += " | " + "; ".join(messages)
    return MultistageResult(
        status="feasible",
        objective=float(best_obj),
        solution=best_sol,
        time_s=elapsed,
        rounds=total_rounds,
        history=tuple(history),
        message=msg,
        stage1_time_s=stage1_time,
        stage2_time_s=stage2_time,
        stage_records=tuple(stage_records),
    )


__all__ = [
    "PriorBand", "PriorSolution", "MultistageResult", "StageTimeRecord",
    "extract_prior", "extract_prior_from_greedy", "validate_prior",
    "freeze_combo", "freeze_boundaries", "unfixed_integrality",
    "find_stage1_components",
    "solve_fixed_boundaries", "solve_fixed_combo_lp", "solve_multistage",
]
