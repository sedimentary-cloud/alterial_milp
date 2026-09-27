"""highspy 后端：直接调用 HiGHS 的 Python 接口。

给高中生看的小课堂
------------------
scipy 内部其实也用 HiGHS，但 highspy 是 HiGHS 的原生接口，
能设置更多选项，也方便以后支持 warm start。

本文件把同一个模型翻译成 highspy 的变量、约束和目标。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, vstack

from ..errors import GWSolveError
from ..model.builder import ArterialModel
from ..model.constraints import ConstraintMeta
from ..model.expressions import LinearExpr
from .backend_base import (ModelSession, SolveOptions, SolveResult, SolveStatus,
                           SolverBackend)

try:  # highspy is optional
    import highspy  # type: ignore
except Exception:  # pragma: no cover
    highspy = None  # type: ignore


# highspy 会话。
@dataclass
class HighspySession(ModelSession):
    model: ArterialModel
    backend_name: str = "highspy"
    objective: LinearExpr | None = None
    objective_sense: Literal["max", "min"] = "max"
    extra_rows: list[tuple[LinearExpr, str, float, ConstraintMeta]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.objective is None:
            self.objective = self.model.objective
            self.objective_sense = self.model.objective_sense


# highspy 后端实现。
class HighspyBackend(SolverBackend):
    name = "highspy"

    def create(self, model: ArterialModel) -> HighspySession:
        if highspy is None:
            raise GWSolveError("highspy is not installed")
        return HighspySession(model=model, objective=model.objective,
                              objective_sense=model.objective_sense)

    def set_objective(self, session: HighspySession, expr: LinearExpr,
                      sense: Literal["max", "min"]) -> None:
        session.objective = expr
        session.objective_sense = sense

    def add_constraint(self, session: HighspySession, expr: LinearExpr,
                       sense: str, rhs: float, meta: ConstraintMeta) -> int:
        session.extra_rows.append((expr.copy(), sense, float(rhs), meta))
        return len(session.extra_rows) - 1

    def solve(self, session: HighspySession, options: SolveOptions) -> SolveResult:
        if highspy is None:
            return SolveResult(SolveStatus.ERROR, None, None, None, 0.0, "highspy unavailable")
        t0 = time.perf_counter()
        model = session.model
        n = len(model.vars)
        obj = session.objective or LinearExpr()
        c = np.zeros(n, dtype=float)
        for idx, coef in obj.items():
            c[idx] = coef
        base_A, base_lb, base_ub = model.cons.to_sparse(n)
        for expr, sense, rhs, _ in session.extra_rows:
            row = np.zeros((1, n), dtype=float)
            for idx, coef in expr.items():
                row[0, idx] = coef
            base_A = vstack([base_A, csr_matrix(row)], format="csr")
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
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        # worker 内单线程，避免并行格点时相互抢 CPU。
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(options.time_limit_s))
        h.setOptionValue("mip_rel_gap", float(options.mip_rel_gap))
        h.setOptionValue("presolve", "on" if options.presolve else "off")
        h.addVars(n, [float(x) for x in model.vars.lbs], [float(x) for x in model.vars.ubs])
        h.changeColsCost(n, np.arange(n, dtype=np.int32), c)
        types = np.array([highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous
                          for t in model.vars.integrality], dtype=np.uint8)
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), types)
        if session.objective_sense == "max":
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        else:
            h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        num_rows = base_A.shape[0]
        if num_rows:
            base_A = base_A.tocsr()
            starts = np.asarray(base_A.indptr, dtype=np.int32)
            indices = np.asarray(base_A.indices, dtype=np.int32)
            values = np.asarray(base_A.data, dtype=float)
            h.addRows(num_rows, np.asarray(base_lb, dtype=float), np.asarray(base_ub, dtype=float),
                      int(values.size), starts, indices, values)
        try:
            run_status = h.run()
        except Exception as exc:  # pragma: no cover
            return SolveResult(SolveStatus.ERROR, None, None, None, time.perf_counter() - t0, str(exc))
        dt = time.perf_counter() - t0
        model_status = h.getModelStatus()
        status = self._map_status(model_status)
        values = None
        objective = None
        try:
            sol = h.getSolution()
            vals = list(sol.col_value)
            if vals:
                values = np.asarray(vals, dtype=float)
                objective = float(h.getObjectiveValue()) + obj.const
        except Exception:
            values = None
        return SolveResult(status=status, objective=objective, values=values, bound=None,
                           time_s=dt, message=str(model_status))

    @staticmethod
    def _map_status(status) -> SolveStatus:
        name = str(status)
        if "kOptimal" in name:
            return SolveStatus.OPTIMAL
        if "kInfeasible" in name or "kUnboundedOrInfeasible" in name:
            return SolveStatus.INFEASIBLE
        if "kTimeLimit" in name or "kIterationLimit" in name or "kSolutionLimit" in name:
            return SolveStatus.TIME_LIMIT
        if "kUnbounded" in name:
            return SolveStatus.ERROR
        if "kModelEmpty" in name:
            return SolveStatus.OPTIMAL
        return SolveStatus.ERROR


__all__ = ["HighspyBackend", "HighspySession"]
