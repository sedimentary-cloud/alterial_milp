"""scipy 后端：调用 scipy.optimize.milp 求解。

给高中生看的小课堂
------------------
scipy 是一个常用的科学计算库，
它里面的 milp 可以解“混合整数线性规划”：
- 线性目标函数；
- 线性约束；
- 部分变量必须是整数或 0/1。

本文件负责把我们的变量表、约束矩阵翻译成 scipy 需要的格式，
然后调用 HiGHS 求解器。
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from ..errors import GWSolveError
from ..model.builder import ArterialModel
from ..model.constraints import ConstraintMeta, ConstraintStore
from ..model.expressions import LinearExpr
from .backend_base import (ModelSession, SolveOptions, SolveResult, SolveStatus,
                           SolverBackend)


# scipy 会话：保存模型、当前目标函数和追加的约束。
@dataclass
class ScipySession(ModelSession):
    model: ArterialModel
    backend_name: str = "scipy"
    objective: LinearExpr | None = None
    objective_sense: Literal["max", "min"] = "max"
    extra_rows: list[tuple[LinearExpr, str, float, ConstraintMeta]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.objective is None:
            self.objective = self.model.objective
            self.objective_sense = self.model.objective_sense


# scipy 后端实现。
class ScipyBackend(SolverBackend):
    name = "scipy"

    def create(self, model: ArterialModel) -> ScipySession:
        return ScipySession(model=model, objective=model.objective,
                            objective_sense=model.objective_sense)

    def set_objective(self, session: ScipySession, expr: LinearExpr,
                      sense: Literal["max", "min"]) -> None:
        session.objective = expr
        session.objective_sense = sense

    def add_constraint(self, session: ScipySession, expr: LinearExpr,
                       sense: str, rhs: float, meta: ConstraintMeta) -> int:
        session.extra_rows.append((expr.copy(), sense, float(rhs), meta))
        return len(session.extra_rows) - 1

    def solve(self, session: ScipySession, options: SolveOptions) -> SolveResult:
        # scipy 只能最小化，所以如果我们要最大化 composite，
        # 就把目标系数全部乘以 -1，算完再取反。
        t0 = time.perf_counter()
        model = session.model
        n = len(model.vars)
        obj = session.objective or LinearExpr()
        c = np.zeros(n, dtype=float)
        for idx, coef in obj.items():
            c[idx] = coef
        if session.objective_sense == "max":
            c = -c
        base_A, base_lb, base_ub = model.cons.to_sparse(n)
        for expr, sense, rhs, _ in session.extra_rows:
            row = np.zeros((1, n), dtype=float)
            for idx, coef in expr.items():
                row[0, idx] = coef
            Arow = csr_matrix(row)
            base_A = vstack([base_A, Arow], format="csr")
            const = expr.const
            if sense == "<=":
                base_lb = np.append(base_lb, -np.inf)
                base_ub = np.append(base_ub, rhs - const)
            elif sense == ">=":
                base_lb = np.append(base_lb, rhs - const)
                base_ub = np.append(base_ub, np.inf)
            else:
                base_lb = np.append(base_lb, rhs - const)
                base_ub = np.append(base_ub, rhs - const)
        bounds = Bounds(np.asarray(model.vars.lbs, dtype=float),
                        np.asarray(model.vars.ubs, dtype=float))
        integrality = np.asarray(model.vars.integrality, dtype=int)
        scipy_options = {
            "time_limit": float(options.time_limit_s),
            "presolve": bool(options.presolve),
            "disp": False,
            # 注意：不要在这里传 "threads"。
            # SciPy 的 milp 选项会把未知键原样转发给 HiGHS；新版
            # SciPy/HiGHS 对 "threads" 的处理会直接让模型状态变成
            # "Not Set"（尤其是中大型模型）。需要单线程时应使用
            # highspy 后端，或通过进程级环境变量控制。
        }
        try:
            scipy_options["mip_rel_gap"] = float(options.mip_rel_gap)
        except Exception:
            pass
        try:
            # 老版本 scipy 会把 mip_rel_gap 转发给 HiGHS 并打印
            # “Unrecognized options detected” 警告；这个警告不影响结果。
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Unrecognized options detected.*")
                res = milp(c=c, integrality=integrality, bounds=bounds,
                           constraints=LinearConstraint(base_A, base_lb, base_ub),
                           options=scipy_options)
        except Exception as exc:  # pragma: no cover - solver level failure
            return SolveResult(SolveStatus.ERROR, None, None, None,
                               time.perf_counter() - t0, str(exc))
        dt = time.perf_counter() - t0
        values = None if res.x is None else np.asarray(res.x, dtype=float)
        status = self._map_status(res)
        objective = None
        bound = None
        if res.fun is not None and np.isfinite(res.fun):
            if session.objective_sense == "max":
                objective = float(-res.fun) + obj.const
            else:
                objective = float(res.fun) + obj.const
            if getattr(res, "mip_dual_bound", None) is not None:
                b = float(res.mip_dual_bound)
                bound = (-b + obj.const) if session.objective_sense == "max" else (b + obj.const)
        return SolveResult(status=status, objective=objective, values=values, bound=bound,
                           time_s=dt, message=getattr(res, "message", ""))

    @staticmethod
    def _map_status(res) -> SolveStatus:
        st = getattr(res, "status", 4)
        if st == 0:
            return SolveStatus.OPTIMAL
        if st == 1:
            return SolveStatus.TIME_LIMIT if getattr(res, "x", None) is not None else SolveStatus.TIME_LIMIT
        if st == 2:
            return SolveStatus.INFEASIBLE
        if st == 3:
            return SolveStatus.ERROR
        return SolveStatus.ERROR


__all__ = ["ScipyBackend", "ScipySession"]
