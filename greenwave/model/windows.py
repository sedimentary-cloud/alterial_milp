"""有效窗口/红窗补集构造（v2：相位边界 + 派生窗口表达式）。

窗口端点在 v2 中不再是独立 solver 变量，而是由 plan 的相位边界直接拼出的
``LinearExpr``。这样 C4–C13 的公式形式不变，但所有窗口端点都保持为边界 δ 的
线性表达式。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import GWModelError
from ..preprocess.context import GridPointContext
from ..schema.models import Direction, PlanSpec, ProblemInput
from .constraints import ConstraintMeta, ConstraintStore
from . import naming
from .expressions import LinearExpr
from .variables import GVarType, VarHandle, VariableRegistry


@dataclass(frozen=True)
class EffectiveWindowSet:
    intersection: str
    direction: Direction
    slots: tuple[int, ...]                 # 1-based slot ids
    s: Mapping[int, LinearExpr]            # derived window start expressions
    e: Mapping[int, LinearExpr]            # derived window end expressions
    h: Mapping[int, VarHandle]             # slot availability (0/1)


@dataclass(frozen=True)
class RedWindowSet:
    intersection: str
    direction: Direction
    slots: tuple[int, ...]
    start: Mapping[int, LinearExpr]
    end: Mapping[int, LinearExpr]
    avail: Mapping[int, LinearExpr]
    kind: Mapping[int, str]                # "gap" or "wrap"
    red_after_window: Mapping[int, int]


class EffectiveWindowBuilder:
    """Build effective window expressions from per-plan boundary variables."""

    def __init__(self, problem: ProblemInput, ctx: GridPointContext,
                 registry: VariableRegistry, store: ConstraintStore,
                 plan_vars: Mapping[str, VarHandle],
                 delta_vars: Mapping[tuple[str, Direction, int], VarHandle],
                 z_vars: Mapping[tuple[str, Direction, int], VarHandle]) -> None:
        self.problem = problem
        self.ctx = ctx
        self.registry = registry
        self.store = store
        self.plan_vars = plan_vars
        self.delta_vars = delta_vars
        self.z_vars = z_vars
        self.C = ctx.cycle_s

    # -- helpers -------------------------------------------------------------
    def _plan_selected_expr(self, plan: PlanSpec, d: Direction, edge: int) -> LinearExpr:
        """Expression for one expanded edge of ``plan``, weighted by x_p.

        For a multi-plan intersection, ``boundary_vars`` stores z = x * delta,
        so the edge is ``nominal*C*x + C*z``.  For a single fixed plan it stores
        delta directly, so the edge is ``nominal*C + C*delta``.

        **方案无关的常数端点也要乘 x_p**：有效窗口端点是 Σ_p endpoint_p，
        若常数端点（周期首尾、双周期的 0.5 seam）对每个方案各加一次，
        就会被重复计成"方案个数 × 常数"。
        """
        kind, src, offset = plan.expanded_edge_sources(d)[edge]
        x = self.plan_vars.get(plan.id)
        if kind == "const":
            val = float(offset) * self.C
            return LinearExpr.constant(val) if x is None else LinearExpr.of(x, val)
        nominal = plan.nominal_boundary_r(d, src) + offset
        if x is None:
            expr = LinearExpr.constant(nominal * self.C)
        else:
            expr = LinearExpr.of(x, nominal * self.C)
        v = (self.z_vars if x is not None else self.delta_vars).get((plan.id, d, src))
        if v is not None:
            expr.add(v, self.C)
        return expr

    def _plan_window_endpoint_expr(self, plan: PlanSpec, d: Direction, k: int,
                                   side: str) -> LinearExpr:
        # ``k`` is the 1-based effective slot; PlanSpec methods are 0-based.
        w = plan.window_edge_span(d, k - 1)
        edge = w.start_edge if side == "start" else w.end_edge
        expr = self._plan_selected_expr(plan, d, edge)
        if side == "end" and w.wrap:
            # 跨 0/C 的 +C 也只属于被选中的那个方案。
            x = self.plan_vars.get(plan.id)
            if x is None:
                expr.add_const(self.C)
            else:
                expr.add(x, self.C)
        return expr

    # -- public ---------------------------------------------------------------
    def build(self, intersection: str, d: Direction) -> EffectiveWindowSet:
        plans = self.ctx.feasible_plans.get(intersection, ())
        kmax = self.ctx.max_window_slots(intersection, d)
        if kmax <= 0:
            return EffectiveWindowSet(intersection, d, (), {}, {}, {})
        fixed_single = len(plans) == 1 and plans[0].id not in self.plan_vars

        s_exprs: dict[int, LinearExpr] = {}
        e_exprs: dict[int, LinearExpr] = {}
        h_vars: dict[int, VarHandle] = {}
        for k in range(1, kmax + 1):
            h_lo, h_hi = (1.0, 1.0) if fixed_single else (0.0, 1.0)
            h = self.registry.add(naming.win_avail(intersection, d, k), h_lo, h_hi,
                                  GVarType.CONTINUOUS, {"kind": "window_availability"})
            h_vars[k] = h
            s_expr = LinearExpr()
            e_expr = LinearExpr()
            h_expr = LinearExpr()
            for p in plans:
                if p.window_count(d) < k:
                    continue
                x = self.plan_vars.get(p.id)
                if x is not None:
                    h_expr.add(x, 1.0)
                else:
                    h_expr.add_const(1.0)
                s_expr.add(self._plan_window_endpoint_expr(p, d, k, "start"), 1.0)
                e_expr.add(self._plan_window_endpoint_expr(p, d, k, "end"), 1.0)
            self.store.add(LinearExpr.of(h, 1.0).add(h_expr, -1.0), "==", 0.0,
                           ConstraintMeta("WIN", f"effective availability {intersection}/{d}/{k}",
                                          {"intersection": intersection, "direction": d, "slot": str(k)}))
            s_exprs[k] = s_expr
            e_exprs[k] = e_expr
        return EffectiveWindowSet(intersection, d, tuple(range(1, kmax + 1)),
                                  s_exprs, e_exprs, h_vars)


class RedWindowBuilder:
    """由有效绿窗表达式生成红窗补集，包括跨 0/C 的 wrap 段。"""

    def __init__(self, cycle_s: float, store: ConstraintStore) -> None:
        self.C = cycle_s
        self.store = store

    @staticmethod
    def _expr(v) -> LinearExpr:
        return LinearExpr.of(v, 1.0)

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
        for r in range(1, kmax):
            rid += 1
            start[rid] = self._expr(gw.e[r])
            end[rid] = self._expr(gw.s[r + 1])
            avail[rid] = LinearExpr.of(gw.h[r + 1], 1.0)
            kind[rid] = "gap"
            red_after[rid] = r
        for r in range(1, kmax + 1):
            rid += 1
            start[rid] = self._expr(gw.e[r])
            end[rid] = self._expr(gw.s[1]).add_const(self.C)
            a = LinearExpr.of(gw.h[r], 1.0)
            if r + 1 <= kmax:
                a.add(gw.h[r + 1], -1.0)
            avail[rid] = a
            kind[rid] = "wrap"
            red_after[rid] = r
        return RedWindowSet(gw.intersection, gw.direction, tuple(range(1, rid + 1)),
                            start, end, avail, kind, red_after)


__all__ = ["EffectiveWindowSet", "RedWindowSet", "EffectiveWindowBuilder", "RedWindowBuilder"]
