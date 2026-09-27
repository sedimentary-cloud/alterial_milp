"""有效窗口重构与红窗补集生成。

给高中生看的小课堂
------------------
每个路口可能有很多候选方案，但最终只能选一个。
如果我们为每个“方案 × 窗口”都建一个变量，模型会非常大。

本文件用一种更聪明的办法：
- 每个路口、每个方向、每个窗口槽位只建一套有效端点变量 ŝ/ê；
- 方案选择 x 决定这些端点最后取哪个方案的值；
- 端点微调量 δ 与方案选择 x 的乘积 z = x·δ 用 McCormick 线性化。

红波带需要的是“红灯区间”。本文件从绿灯窗口的补集生成红灯区间，
并显式包含跨 0/C 的那一段。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import GWModelError
from ..preprocess.context import GridPointContext
from ..schema.models import Direction, PlanSpec, ProblemInput, Side
from .constraints import ConstraintMeta, ConstraintStore
from .expressions import LinearExpr
from .variables import GVarType, VarHandle, VariableRegistry


# 一个路口、一个方向的“有效窗口”集合。
# s/e 是端点秒值，h 是槽位可用性（选中方案是否真的有第 k 个窗口）。
@dataclass(frozen=True)
class EffectiveWindowSet:
    intersection: str
    direction: Direction
    slots: tuple[int, ...]              # 1-based slot ids
    s: Mapping[int, VarHandle]
    e: Mapping[int, VarHandle]
    h: Mapping[int, VarHandle]


# 一个路口、一个方向的“红窗补集”集合。
# start/end 是红灯区间的起止表达式，avail 是可用性表达式。
@dataclass(frozen=True)
class RedWindowSet:
    intersection: str
    direction: Direction
    slots: tuple[int, ...]
    start: Mapping[int, LinearExpr]
    end: Mapping[int, LinearExpr]
    avail: Mapping[int, LinearExpr]
    kind: Mapping[int, str]             # "gap" or "wrap"
    red_after_window: Mapping[int, int] # sequential rid -> source green window slot


class EffectiveWindowBuilder:
    """负责建立有效窗口变量 ŝ/ê/h 以及 McCormick 约束。"""

    def __init__(self, problem: ProblemInput, ctx: GridPointContext,
                 registry: VariableRegistry, store: ConstraintStore,
                 plan_vars: Mapping[str, VarHandle],
                 adj_vars: dict[tuple[str, Direction, int, Side], VarHandle],
                 z_vars: dict[tuple[str, Direction, int, Side], VarHandle]) -> None:
        self.problem = problem
        self.ctx = ctx
        self.registry = registry
        self.store = store
        self.plan_vars = plan_vars
        self.adj_vars = adj_vars
        self.z_vars = z_vars
        self.C = ctx.cycle_s

    def build(self, intersection: str, d: Direction) -> EffectiveWindowSet:
        # kmax 是本路口在这个方向上的最大窗口数，决定要建几个窗口槽位。
        plans = self.ctx.feasible_plans.get(intersection, ())
        kmax = self.ctx.max_window_slots(intersection, d)
        if kmax <= 0:
            return EffectiveWindowSet(intersection, d, (), {}, {}, {})
        # 如果某路口只有一个可行方案，就不需要方案选择 0/1 变量 x：
        # 直接把 x=1 代入表达式，并省掉 z=x·δ 的 McCormick 约束。
        fixed_single = len(plans) == 1 and plans[0].id not in self.plan_vars
        fixed_plan = plans[0] if fixed_single else None

        s_vars: dict[int, VarHandle] = {}
        e_vars: dict[int, VarHandle] = {}
        h_vars: dict[int, VarHandle] = {}
        for k in range(1, kmax + 1):
            s = self.registry.add(f"win[{intersection},{d},{k}].s", -0.5 * self.C, 2.5 * self.C,
                                  GVarType.CONTINUOUS, {"kind": "window_endpoint"})
            e = self.registry.add(f"win[{intersection},{d},{k}].e", -0.5 * self.C, 2.5 * self.C,
                                  GVarType.CONTINUOUS, {"kind": "window_endpoint"})
            h_lo, h_hi = (1.0, 1.0) if fixed_single else (0.0, 1.0)
            h = self.registry.add(f"win[{intersection},{d},{k}].h", h_lo, h_hi,
                                  GVarType.CONTINUOUS, {"kind": "window_availability"})
            s_vars[k] = s
            e_vars[k] = e
            h_vars[k] = h

            if fixed_single:
                # 单方案：x=1，窗口端点直接是 nominal*C + C·δ。
                p = fixed_plan
                assert p is not None
                if p.window_count(d) < k:
                    continue
                s_expr = LinearExpr.constant(p.nominal_endpoint_r(d, k - 1, "start") * self.C)
                e_expr = LinearExpr.constant(p.nominal_endpoint_r(d, k - 1, "end") * self.C)
                for side, expr in (("start", s_expr), ("end", e_expr)):
                    if p.endpoint_is_adjustable(d, k - 1, side):
                        delta = self.adj_vars[(p.id, d, k - 1, side)]
                        expr.add(delta, self.C)
                h_expr = LinearExpr.constant(1.0)
            else:
                # 多方案：需要 x 来选方案，并用 z 线性化 x·δ。
                s_expr = LinearExpr()
                e_expr = LinearExpr()
                h_expr = LinearExpr()
                for p in plans:
                    if p.window_count(d) < k:
                        continue
                    x = self.plan_vars[p.id]
                    h_expr.add(x, 1.0)
                    for side, expr in (("start", s_expr), ("end", e_expr)):
                        delta_idx = k - 1
                        nominal = p.nominal_endpoint_r(d, delta_idx, side)
                        expr.add(x, nominal * self.C)
                        if p.endpoint_is_adjustable(d, delta_idx, side):
                            delta = self.adj_vars[(p.id, d, delta_idx, side)]
                            z = self.registry.add(
                                f"z[{p.id},{d},{delta_idx},{side}]",
                                min(0.0, p.adjust_range(d, delta_idx, side).lo_r),
                                max(0.0, p.adjust_range(d, delta_idx, side).hi_r),
                                GVarType.CONTINUOUS,
                                {"kind": "bilinear_x_delta"},
                            )
                            self.z_vars[(p.id, d, delta_idx, side)] = z
                            expr.add(z, self.C)
                            self._add_mccormick(x, delta, z, p.adjust_range(d, delta_idx, side).lo_r,
                                                p.adjust_range(d, delta_idx, side).hi_r,
                                                p.id, d, delta_idx, side)
            self.store.add(LinearExpr.of(s, 1.0).add(s_expr, -1.0), "==", 0.0,
                           ConstraintMeta("WIN", f"effective start {intersection}/{d}/{k}",
                                          {"intersection": intersection, "direction": d, "slot": str(k)}))
            self.store.add(LinearExpr.of(e, 1.0).add(e_expr, -1.0), "==", 0.0,
                           ConstraintMeta("WIN", f"effective end {intersection}/{d}/{k}",
                                          {"intersection": intersection, "direction": d, "slot": str(k)}))
            self.store.add(LinearExpr.of(h, 1.0).add(h_expr, -1.0), "==", 0.0,
                           ConstraintMeta("WIN", f"effective availability {intersection}/{d}/{k}",
                                          {"intersection": intersection, "direction": d, "slot": str(k)}))
        return EffectiveWindowSet(intersection, d, tuple(range(1, kmax + 1)),
                                  s_vars, e_vars, h_vars)

    def _add_mccormick(self, x: VarHandle, delta: VarHandle, z: VarHandle,
                       lo: float, hi: float, plan_id: str, d: Direction, k: int, side: Side) -> None:
        # McCormick 线性化：用 4 条不等式逼出 z = x·δ。
        # x 是 0/1，δ 在 [lo, hi] 内。
        # 下面 4 条约束在 x=0 时逼 z=0，在 x=1 时逼 z=δ。
        # z <= hi*x
        expr = LinearExpr.of(z, 1.0).add(x, -hi)
        self.store.add(expr, "<=", 0.0, ConstraintMeta("MCC", f"z<=hi*x {plan_id}/{d}/{k}/{side}",
                                                       {"plan": plan_id}))
        # z >= lo*x
        expr = LinearExpr.of(z, 1.0).add(x, -lo)
        self.store.add(expr, ">=", 0.0, ConstraintMeta("MCC", f"z>=lo*x {plan_id}/{d}/{k}/{side}",
                                                       {"plan": plan_id}))
        # z <= delta - lo*(1-x)  ->  z - delta - lo*x <= -lo
        expr = LinearExpr.of(z, 1.0).add(delta, -1.0).add(x, -lo)
        self.store.add(expr, "<=", -lo, ConstraintMeta("MCC", f"z<=delta-lo(1-x) {plan_id}/{d}/{k}/{side}",
                                                        {"plan": plan_id}))
        # z >= delta - hi*(1-x)  ->  z - delta - hi*x >= -hi
        expr = LinearExpr.of(z, 1.0).add(delta, -1.0).add(x, -hi)
        self.store.add(expr, ">=", -hi, ConstraintMeta("MCC", f"z>=delta-hi(1-x) {plan_id}/{d}/{k}/{side}",
                                                       {"plan": plan_id}))


class RedWindowBuilder:
    """由有效绿窗生成红窗补集，包括跨 0/C 的 wrap 段。"""

    def __init__(self, cycle_s: float, store: ConstraintStore) -> None:
        self.C = cycle_s
        self.store = store

    def build(self, gw: EffectiveWindowSet) -> RedWindowSet:
        slots = gw.slots
        kmax = max(slots) if slots else 0
        if kmax <= 0:
            raise GWModelError("cannot build red windows without green windows")
        start: dict[int, LinearExpr] = {}
        end: dict[int, LinearExpr] = {}
        avail: dict[int, LinearExpr] = {}
        kind: dict[int, str] = {}
        red_after: dict[int, int] = {}
        rid = 0
        # 红灯区间 = 两个相邻绿灯窗口之间的空隙。
        for r in range(1, kmax):
            rid += 1
            start[rid] = LinearExpr.of(gw.e[r], 1.0)
            end[rid] = LinearExpr.of(gw.s[r + 1], 1.0)
            avail[rid] = LinearExpr.of(gw.h[r + 1], 1.0)
            kind[rid] = "gap"
            red_after[rid] = r
        # 跨 0/C 的回绕红灯区间：最后一个绿窗结束 -> 第一个绿窗开始 + C。
        for r in range(1, kmax + 1):
            rid += 1
            start[rid] = LinearExpr.of(gw.e[r], 1.0)
            end[rid] = LinearExpr.of(gw.s[1], 1.0).add_const(self.C)
            a = LinearExpr.of(gw.h[r], 1.0)
            if r + 1 <= kmax:
                a.add(gw.h[r + 1], -1.0)
            avail[rid] = a
            kind[rid] = "wrap"
            red_after[rid] = r
        return RedWindowSet(gw.intersection, gw.direction, tuple(range(1, rid + 1)),
                            start, end, avail, kind, red_after)


__all__ = ["EffectiveWindowSet", "RedWindowSet", "EffectiveWindowBuilder", "RedWindowBuilder"]
