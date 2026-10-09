"""标准案例的可复用配置改造函数（性能测试矩阵专用）。

这个模块只做一件事：把 ``examples/standard_problem.json`` 改造成测试矩阵里的
任意一个配置，并保证改造是**可复现、可解释**的。所有改造都作用在原始 JSON
（raw dict）上，最后再交给 ``ProblemInput.from_obj``。

矩阵维度
--------
objective    : 6 种目标（见 ``OBJECTIVES``）
double_cycle : False / True
               True  → 保留 I3 的 double_cycle 与 red1/red2 约束，涉及 I3（或跨
                       全线）的需求给 max_bands=2，并打开 allow_window_sharing；
               False → I3 降级成单周期，删掉只对双周期有意义的 I3 红窗约束，
                       所有需求 max_bands=1，关闭窗口共享。
n_plans      : 每个路口的候选方案数 1..5；>1 时按"整体相位平移"克隆出同一路口
               的邻近方案（保持边界个数/绿掩码结构一致，只挪位置）。
boundaries   : "fixed"      → 删掉所有 adjust 区间，δ≡0；
               "adjustable" → 保留 adjust 区间。

固定口径（与历史 benchmark 保持一致）
------------------------------------
* C=140s、κ=1.0（标准案例自带）；
* backend=scipy(HiGHS)、mip_rel_gap=0、heuristic 关闭、单线程；
* 所有需求 margin 关掉（delta_min_s=0），红波边界亲和项保留（标准案例自带）；
* global_constraints / constraint_templates 清空：它们引用了本矩阵里不存在的
  需求 id（如 GW_up_full），保留会直接校验失败；
* 方案自身 hard_constraints 保留（最小相位、上下行同步、绿窗/红窗下限）。

方案克隆为什么要"整体平移"而不是随便改
--------------------------------------
方案硬约束里有红窗下限（``I3_*_red*_min_015C``：红窗 ≥0.15C）和绿窗下限
（``I4_down_green_min_025C``：≥0.25C），它们都是**绝对**比例约束。如果克隆时
胡乱改边界位置，很容易把某个窗口压到下限以下，整个案例直接不可行——这种
"配置错误"会伪装成"求解很慢"。所以这里：

1. 克隆只做整体平移，保持窗口宽度不变；
2. ``span``（最大平移量）由 ``verify_case_buildable`` 用**常数行检查**自适应
   收缩：把边界固定后，模型里所有"没有变量"的约束行必须在标称点成立。
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from greenwave.model import naming
from greenwave.model.builder import ModelBuilder
from greenwave.preprocess.feasibility import FeasibilityFilter
from greenwave.schema.models import ProblemInput

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(HERE, "standard_problem.json")

CYCLE = 140.0
KAPPA = 1.0
DEFAULT_TIME_LIMIT = 20.0
DEFAULT_PLAN_SPAN = 0.05
# 双周期下只有 I3 会派生出 2 个绿窗，因此"需要 2 个带槽"的需求都跟它有关。
TWO_SLOT_NODES = ("I3",)

BOUNDARY_MODES = ("fixed", "adjustable")


# ---------------------------------------------------------------------------
# 目标目录
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectiveSpec:
    id: str
    label: str
    desc: str
    balanced: bool = False


OBJECTIVES: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec(
        "GW_both_global", "上下全局绿波带宽和",
        "上行/下行各一条跨全线绿波带，weight=1，max Σβ（白板基准）"),
    ObjectiveSpec(
        "GW_both_global_balanced", "上下全局带宽均衡组×1 + 带宽和×0.5",
        "同上两条跨全线绿波带 weight=0.5，另加均衡组 z（weight=1.0）", balanced=True),
    ObjectiveSpec(
        "GW_window_both", "双向窗口绿波（逐对）",
        "每个相邻对、上下行各一条窗口绿波带，weight=1"),
    ObjectiveSpec(
        "GW_up_window_RW_down", "上绿（逐对）+ 下红（逐对）",
        "上行逐对窗口绿波 + 下行逐对红波，weight=1"),
    ObjectiveSpec(
        "RW_both_window", "双向红波（逐对）",
        "每个相邻对、上下行各一条红波带，weight=1"),
    ObjectiveSpec(
        "GW_up_global_RW_down", "上全局绿 + 下红（逐对）",
        "上行一条跨全线绿波 + 下行逐对红波（= standard_problem 原始目标）"),
)
OBJECTIVE_IDS: tuple[str, ...] = tuple(o.id for o in OBJECTIVES)
_OBJECTIVE_BY_ID = {o.id: o for o in OBJECTIVES}


def objective_spec(objective: str) -> ObjectiveSpec:
    try:
        return _OBJECTIVE_BY_ID[objective]
    except KeyError:                                  # pragma: no cover - 防御
        raise KeyError(f"unknown objective {objective!r}, "
                       f"expected one of {OBJECTIVE_IDS}") from None


# ---------------------------------------------------------------------------
# 测试分组
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupSpec:
    id: str
    label: str
    double_cycle: bool
    plan_counts: tuple[int, ...]


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec("G1", "① 无双周期 · 单方案", False, (1,)),
    GroupSpec("G2", "② 无双周期 · 多方案", False, (2, 3, 4, 5)),
    GroupSpec("G3", "③ 双周期 · 单方案", True, (1,)),
    GroupSpec("G4", "④ 双周期 · 多方案", True, (2, 3, 4, 5)),
)
_GROUP_BY_ID = {g.id: g for g in GROUPS}


def group_spec(group_id: str) -> GroupSpec:
    return _GROUP_BY_ID[group_id]


@dataclass(frozen=True)
class CaseSpec:
    """矩阵里的一个案例（一个配置 + 一种边界模式）。"""

    group: str
    objective: str
    double_cycle: bool
    n_plans: int
    boundaries: str

    @property
    def id(self) -> str:
        cyc = "D" if self.double_cycle else "S"
        bnd = "adj" if self.boundaries == "adjustable" else "fix"
        return f"{self.group}|{self.objective}|{cyc}|p{self.n_plans}|{bnd}"

    @property
    def label(self) -> str:
        spec = objective_spec(self.objective)
        return f"{spec.label} · {self.n_plans}方案 · " + \
               ("可调边界" if self.boundaries == "adjustable" else "固定边界")


def all_cases(*, plan_counts: Mapping[bool, Sequence[int]] | None = None
              ) -> list[CaseSpec]:
    """展开完整矩阵：4 组 × 6 目标 × 2 种边界模式。"""
    out: list[CaseSpec] = []
    for g in GROUPS:
        counts = (plan_counts or {}).get(g.double_cycle, g.plan_counts)
        for k in counts:
            for obj in OBJECTIVE_IDS:
                for bnd in BOUNDARY_MODES:
                    out.append(CaseSpec(g.id, obj, g.double_cycle, int(k), bnd))
    return out


# ---------------------------------------------------------------------------
# 原始 JSON 的改造函数（统一在这里，便于复用/审阅）
# ---------------------------------------------------------------------------

def load_base_raw() -> dict[str, Any]:
    with open(BASE_PATH, "r", encoding="utf-8") as f:
        return copy.deepcopy(json.load(f))


def corridor_nodes(raw: Mapping[str, Any]) -> list[str]:
    return list(raw["corridor"]["intersections"])


def adjacent_pairs(nodes: Sequence[str]) -> list[tuple[str, str]]:
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


# -- 周期 / 边界 -------------------------------------------------------------

def apply_single_cycle(raw: dict[str, Any]) -> None:
    """把 I3 从双周期降级成单周期，并删掉只对双周期有意义的 I3 约束。"""
    keep_hard = {"I3_up_end_hard_070", "I3_down_end_hard_070", "I3_sync_up_down_end"}
    keep_soft = {"I3_up_end_soft_055", "I3_down_end_soft_055"}
    for plan in raw["plans"]:
        if plan["id"] != "I3_P1":
            continue
        plan["double_cycle"] = False
        plan["hard_constraints"] = [c for c in plan.get("hard_constraints", [])
                                    if c["id"] in keep_hard]
        plan["soft_constraints"] = [c for c in plan.get("soft_constraints", [])
                                    if c["id"] in keep_soft]


def strip_adjust(raw: dict[str, Any]) -> None:
    """删掉所有可调区间：相位边界固定，只剩 φ 偏置自由。"""
    for plan in raw["plans"]:
        for d in ("up", "down"):
            for boundary in plan["directions"][d]["boundaries"]:
                boundary.pop("adjust", None)


def configure_solver(raw: dict[str, Any], *, time_limit_s: float = DEFAULT_TIME_LIMIT,
                     workers: int = 1, heuristic: bool = False,
                     mip_rel_gap: float = 0.0) -> None:
    solver = raw.setdefault("solver", {})
    solver["backend"] = solver.get("backend", "scipy")
    solver["time_limit_s"] = float(time_limit_s)
    solver["mip_rel_gap"] = float(mip_rel_gap)
    solver["num_workers"] = int(workers)
    solver.setdefault("heuristic", {})
    solver["heuristic"]["enabled"] = bool(heuristic)


def apply_grid(raw: dict[str, Any], *, cycles_s: Sequence[float],
               speed_ratios: Sequence[float]) -> None:
    """替换格点列表：同一份配置扫 ``C × κ`` 的网格。

    ``cycles_s`` 是公共周期候选，``speed_ratios`` 是速度倍率候选（κ 越大跑得越快，
    行程时间 ``τ = 距离 / (κ·设计速度)``）。
    """
    raw["grid"] = {"cycles_s": [float(c) for c in cycles_s],
                   "speed_ratios": [float(k) for k in speed_ratios]}


def clear_global_scopes(raw: dict[str, Any]) -> None:
    """清空跨需求/模板约束：它们引用了本矩阵里不存在的需求 id。"""
    raw["global_constraints"] = []
    raw["constraint_templates"] = []


# -- 目标 --------------------------------------------------------------------

def _demand(did: str, kind: str, direction: str, nodes: Sequence[str], *,
            max_bands: int = 1, weight: float = 1.0) -> dict[str, Any]:
    return {
        "id": did, "type": kind, "direction": direction, "nodes": list(nodes),
        "max_bands": int(max_bands), "weight": float(weight),
        "margin": {"delta_min_s": 0.0, "coef": 1.0},
    }


def demand_needs_two_slots(demand: Mapping[str, Any], nodes: Sequence[str]) -> bool:
    """双周期下哪些需求需要 2 个带槽：跨全线，或经过有两个绿窗的路口（I3）。"""
    seq = tuple(demand["nodes"])
    if seq == tuple(nodes) or seq == tuple(reversed(nodes)):
        return True
    return any(n in TWO_SLOT_NODES for n in demand["nodes"])


def objective_demands(objective: str, nodes: Sequence[str], *,
                      double_cycle: bool) -> list[dict[str, Any]]:
    """按目标 id 生成需求列表（max_bands 已按双周期规则算好）。"""
    pairs = adjacent_pairs(nodes)
    up_full = list(nodes)
    down_full = list(reversed(nodes))

    if objective == "GW_both_global":
        out = [_demand("GW_up_full", "green", "up", up_full, weight=1.0),
               _demand("GW_down_full", "green", "down", down_full, weight=1.0)]
    elif objective == "GW_both_global_balanced":
        out = [_demand("GW_up_full", "green", "up", up_full, weight=0.5),
               _demand("GW_down_full", "green", "down", down_full, weight=0.5)]
    elif objective == "GW_window_both":
        out = []
        for a, b in pairs:
            out.append(_demand(f"GW_up_{a}_{b}", "green", "up", [a, b]))
            out.append(_demand(f"GW_down_{b}_{a}", "green", "down", [b, a]))
    elif objective == "GW_up_window_RW_down":
        out = []
        for a, b in pairs:
            out.append(_demand(f"GW_up_{a}_{b}", "green", "up", [a, b]))
            out.append(_demand(f"RW_down_{b}_{a}", "red", "down", [b, a]))
    elif objective == "RW_both_window":
        out = []
        for a, b in pairs:
            out.append(_demand(f"RW_up_{a}_{b}", "red", "up", [a, b]))
            out.append(_demand(f"RW_down_{b}_{a}", "red", "down", [b, a]))
    elif objective == "GW_up_global_RW_down":
        out = [_demand("GW_up_full", "green", "up", up_full)]
        for a, b in pairs:
            out.append(_demand(f"RW_down_{b}_{a}", "red", "down", [b, a]))
    else:                                             # pragma: no cover - 防御
        raise KeyError(f"unknown objective {objective!r}")

    for d in out:
        d["max_bands"] = 2 if (double_cycle and demand_needs_two_slots(d, nodes)) else 1
    return out


def set_objective(raw: dict[str, Any], objective: str, *,
                  double_cycle: bool) -> None:
    """把 raw 的目标替换成矩阵里的目标（需求 + 均衡组 + 共享开关）。"""
    spec = objective_spec(objective)
    nodes = corridor_nodes(raw)
    obj = raw["objective"]
    obj["band_demands"] = objective_demands(objective, nodes, double_cycle=double_cycle)
    obj["balanced_groups"] = ([{
        "id": "BG_up_down_global",
        "demands": ["GW_up_full", "GW_down_full"],
        "weight": 1.0,
        "on_infeasible": "zero",
    }] if spec.balanced else [])
    # 双周期时需要共享绿窗才能用上第 2 个带槽；单周期时关闭，保持单方案口径。
    obj["allow_window_sharing"] = bool(double_cycle)
    obj["margin_default"] = {"delta_min_s": 0.0, "coef": 1.0, "hard_min_s": 0.0}
    obj["pareto"] = {"num_points": 0, "relax_max": 0.0, "topk_grids": 1}


# -- 多方案克隆 --------------------------------------------------------------

def _reown(constraints: Iterable[Mapping[str, Any]], plan_id: str
           ) -> list[dict[str, Any]]:
    out = []
    for c in constraints:
        c = copy.deepcopy(dict(c))
        for t in c.get("terms", []):
            atom = t.get("atom")
            if atom and atom.get("plan"):
                atom["plan"] = plan_id
            if atom and atom.get("kind") == "window_duration" and atom.get("plan"):
                atom["plan"] = plan_id
        out.append(c)
    return out


def _shift_plan(plan: Mapping[str, Any], shift: float, plan_id: str) -> dict[str, Any]:
    """整体平移一个方案的所有相位边界（宽度不变），并换 id / 去掉方案软约束。"""
    out = copy.deepcopy(dict(plan))
    out["id"] = plan_id
    for d, dp in out["directions"].items():
        for b in dp["boundaries"]:
            b["pos_r"] = float(b["pos_r"]) + float(shift)
    # 方案自带软约束 owner=intersection_loss，不进 composite 目标，克隆后无意义。
    out["soft_constraints"] = []
    out["hard_constraints"] = _reown(out.get("hard_constraints", []), plan_id)
    return out


def variant_shifts(k: int, span: float) -> list[float]:
    """k 个方案的整体平移量，均匀铺在 [-span, +span] 上。"""
    if k <= 1:
        return [0.0]
    return [-span + 2.0 * span * c / (k - 1) for c in range(k)]


def clone_plans(raw: dict[str, Any], k: int, span: float = DEFAULT_PLAN_SPAN
                ) -> dict[str, int]:
    """把每个路口的方案克隆成 k 个（整体平移），返回每个路口的方案数。"""
    if k <= 1:
        return {p["intersection"]: 1 for p in raw["plans"]}
    shifts = variant_shifts(k, span)
    new_plans: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for p in raw["plans"]:
        for c, s in enumerate(shifts):
            new_plans.append(_shift_plan(p, s, f"{p['id']}_v{c}"))
        counts[p["intersection"]] = counts.get(p["intersection"], 0) + k
    raw["plans"] = new_plans
    return counts


# ---------------------------------------------------------------------------
# 案例构建
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseBuild:
    problem: ProblemInput
    plan_span: float
    plan_counts: Mapping[str, int]
    notes: tuple[str, ...] = ()


def constant_row_violations(model) -> list[str]:
    """常数行检查：没有变量系数的约束行必须在标称点成立。

    固定边界时，方案硬约束（红窗/绿窗下限、上下行同步、边界上限）会退化成
    纯常数行。只要有一行不成立，说明这套方案变体本身不可行——不必浪费一次
    20s 求解去发现它。
    """
    bad: list[str] = []
    for row in model.cons.rows:
        if row.expr.coefs:
            continue
        lhs = float(row.expr.const)
        if row.sense == "<=":
            ok = lhs <= row.rhs + 1e-6
        elif row.sense == ">=":
            ok = lhs >= row.rhs - 1e-6
        else:
            ok = abs(lhs - row.rhs) <= 1e-6
        if not ok:
            bad.append(f"{row.meta.group}/{row.meta.label}: "
                       f"{lhs:.6f} {row.sense} {row.rhs:.6f}")
    return bad


def _model_for(raw: Mapping[str, Any]) -> ProblemInput:
    return ProblemInput.from_obj(copy.deepcopy(dict(raw)))


def _probe_ok(raw: dict[str, Any], k: int) -> list[str]:
    """用"边界固定"的版本检查方案变体是否可行（返回问题清单）。

    两条判据：
    1. ``constant_row_violations``：标称点必须满足所有退化成常数的方案硬约束；
    2. 每个路口必须**保住 k 个方案**——可行性过滤会把越界的变体直接丢掉，
       如果不管这一点，"k 方案"实验可能悄悄退化成"3 方案"。
    """
    try:
        probe = copy.deepcopy(raw)
        strip_adjust(probe)
        problem = _model_for(probe)
        errs = [i for i in problem.validate_all() if i.level == "error"]
        if errs:
            return [f"validate: {i.message}" for i in errs[:3]]
        ctx = FeasibilityFilter(problem).filter(CYCLE, KAPPA)
        bad = constant_row_violations(ModelBuilder(problem, ctx).build())
        for iid in problem.corridor.intersections:
            n = len(ctx.feasible_plans.get(iid, ()))
            if n < k:
                bad.append(f"{iid}: 只剩 {n}/{k} 个可行方案")
        return bad
    except Exception as exc:                          # noqa: BLE001 - 变体无效即可
        return [f"{type(exc).__name__}: {exc}"]


def build_case(spec: CaseSpec, *, time_limit_s: float = DEFAULT_TIME_LIMIT,
               span: float = DEFAULT_PLAN_SPAN, verify: bool = True,
               grid: tuple[Sequence[float], Sequence[float]] | None = None
               ) -> CaseBuild:
    """把 CaseSpec 变成可直接求解的 ProblemInput。

    ``grid=(cycles_s, speed_ratios)`` 可覆盖格点列表（做 C × κ 网格搜索时用）。
    """
    notes: list[str] = []
    raw = load_base_raw()
    if not spec.double_cycle:
        apply_single_cycle(raw)
    configure_solver(raw, time_limit_s=time_limit_s)
    clear_global_scopes(raw)
    set_objective(raw, spec.objective, double_cycle=spec.double_cycle)
    if grid is not None:
        apply_grid(raw, cycles_s=grid[0], speed_ratios=grid[1])

    plan_counts = {p["intersection"]: 1 for p in raw["plans"]}
    used_span = 0.0
    if spec.n_plans > 1:
        if verify:
            cur = float(span)
            for _ in range(8):
                trial = copy.deepcopy(raw)
                clone_plans(trial, spec.n_plans, cur)
                bad = _probe_ok(trial, spec.n_plans)
                if not bad:
                    break
                cur /= 2.0
            else:
                notes.append("方案变体检查始终失败，退回 span=0（各方案退化为同一套）")
                cur = 0.0
            used_span = cur
        else:
            used_span = float(span)
        plan_counts = clone_plans(raw, spec.n_plans, used_span)

    if spec.boundaries == "fixed":
        strip_adjust(raw)

    problem = _model_for(raw)
    return CaseBuild(problem=problem, plan_span=used_span,
                     plan_counts=plan_counts, notes=tuple(notes))


# ---------------------------------------------------------------------------
# 结构诊断
# ---------------------------------------------------------------------------

def case_structure(problem: ProblemInput) -> dict[str, Any]:
    """案例的规模与"能不能分解"的结构信息。"""
    from greenwave.stage1_decompose import (plan_dp_candidate,  # 局部导入避免环
                                            semantic_demand_groups)

    ctx = FeasibilityFilter(problem).filter(CYCLE, KAPPA)
    model = ModelBuilder(problem, ctx, stage1_symmetry=True,
                         min_band_width_s=1.0).build()
    n_bin = sum(1 for i, t in enumerate(model.vars.types)
                if int(t) == 1 and abs(model.vars.lbs[i] - model.vars.ubs[i]) > 1e-9)
    n_int = sum(1 for i, t in enumerate(model.vars.types)
                if int(t) == 2 and abs(model.vars.lbs[i] - model.vars.ubs[i]) > 1e-9)
    n_adjust = sum(1 for n in model.vars.names
                   if naming.is_family(n, naming.BND_DELTA))
    groups = semantic_demand_groups(problem, ctx)
    cand = plan_dp_candidate(problem, ctx, max_solves=400)
    plans_per_node = {iid: len(ctx.feasible_plans.get(iid, ()))
                      for iid in problem.corridor.intersections}
    return {
        "groups": len(groups),
        "group_sizes": [len(g.node_ids) for g in groups],
        "group_spans": [f"{g.node_ids[0]}-{g.node_ids[-1]}" for g in groups],
        "group_demands": [list(g.demand_ids) for g in groups],
        "dp_candidate": bool(cand is not None),
        "dp_solves": int(cand.n_solves) if cand is not None else None,
        "plans_per_node": plans_per_node,
        "n_adjust_vars": n_adjust,
        "adjustable": n_adjust > 0,
        "model": {
            "vars": len(model.vars),
            "binary": n_bin,
            "integer": n_int,
            "cons": len(model.cons.rows),
            "nnz": sum(len(r.expr.coefs) for r in model.cons.rows),
        },
    }


__all__ = [
    "CYCLE", "KAPPA", "DEFAULT_TIME_LIMIT", "DEFAULT_PLAN_SPAN",
    "BOUNDARY_MODES", "OBJECTIVES", "OBJECTIVE_IDS", "objective_spec",
    "ObjectiveSpec", "GroupSpec", "GROUPS", "group_spec", "CaseSpec", "all_cases",
    "load_base_raw", "corridor_nodes", "adjacent_pairs",
    "apply_single_cycle", "strip_adjust", "configure_solver", "clear_global_scopes",
    "apply_grid",
    "objective_demands", "set_objective", "demand_needs_two_slots",
    "clone_plans", "variant_shifts", "CaseBuild", "build_case",
    "constant_row_violations", "case_structure",
]
