"""不可行诊断：模型无解时，找出是哪一组约束互相打架。

给高中生看的小课堂
------------------
有时约束太多，彼此矛盾，导致“无解”。
例如：
- 要求带宽 >= 50s；
- 但所有绿灯窗口都只有 30s。

直接看求解器说“infeasible”是不知道原因的。
本文件给每条约束加一个松弛变量（相当于“违反量”），
然后最小化总违反量。
哪组约束的违反量不为 0，哪组就是嫌疑犯。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from ..model.builder import ArterialModel
from ..solve.backend_base import SolverBackend


# 诊断报告：嫌疑约束组和详细说明。
@dataclass(frozen=True)
class DiagnosisReport:
    culprit_groups: tuple[str, ...]
    detail: str


# 不可行诊断器。
class InfeasibilityDiagnoser:
    def __init__(self, model: ArterialModel, backend: SolverBackend | None = None) -> None:
        self.model = model
        self.backend = backend

    def diagnose(self) -> DiagnosisReport:
        model = self.model
        n = len(model.vars)
        rows = model.cons.rows
        m = len(rows)
        # original variables + one slack per row
        lbs = list(model.vars.lbs) + [0.0] * m
        ubs = list(model.vars.ubs) + [1e6] * m
        integrality = list(model.vars.integrality) + [0] * m
        c = np.zeros(n + m, dtype=float)
        for i in range(m):
            c[n + i] = 1.0

        # Row-wise relaxation: one non-negative slack variable per original row.
        # For equality rows the same slack bounds the violation in either direction.
        data: list[float] = []
        indices: list[int] = []
        indptr: list[int] = [0]
        lb_list: list[float] = []
        ub_list: list[float] = []
        for r, row in enumerate(rows):
            base = list(row.expr.items())
            rhs = row.rhs - row.expr.const
            if row.sense == "<=":
                for idx, coef in base:
                    data.append(coef); indices.append(idx)
                data.append(-1.0); indices.append(n + r)
                lb_list.append(-np.inf); ub_list.append(rhs)
                indptr.append(len(data))
            elif row.sense == ">=":
                for idx, coef in base:
                    data.append(coef); indices.append(idx)
                data.append(1.0); indices.append(n + r)
                lb_list.append(rhs); ub_list.append(np.inf)
                indptr.append(len(data))
            else:
                # expr - s <= rhs
                for idx, coef in base:
                    data.append(coef); indices.append(idx)
                data.append(-1.0); indices.append(n + r)
                lb_list.append(-np.inf); ub_list.append(rhs)
                indptr.append(len(data))
                # expr + s >= rhs
                for idx, coef in base:
                    data.append(coef); indices.append(idx)
                data.append(1.0); indices.append(n + r)
                lb_list.append(rhs); ub_list.append(np.inf)
                indptr.append(len(data))
        A = csr_matrix((np.asarray(data, dtype=float), np.asarray(indices, dtype=int),
                        np.asarray(indptr, dtype=int)), shape=(len(lb_list), n + m))
        try:
            res = milp(c=c, integrality=np.asarray(integrality, dtype=int),
                       bounds=Bounds(np.asarray(lbs, dtype=float), np.asarray(ubs, dtype=float)),
                       constraints=LinearConstraint(A, np.asarray(lb_list, dtype=float),
                                                    np.asarray(ub_list, dtype=float)),
                       options={"time_limit": 60.0, "disp": False})
        except Exception as exc:  # pragma: no cover
            return DiagnosisReport((), f"diagnostic solve failed: {exc}")
        if res.x is None:
            return DiagnosisReport(tuple(sorted({r.meta.group for r in rows})),
                                   "relaxed diagnostic model could not be solved")
        slacks = res.x[n:n + m]
        bad = [i for i, s in enumerate(slacks) if s > 1e-6]
        groups = tuple(sorted({rows[i].meta.group for i in bad}))
        details = []
        for i in bad[:30]:
            details.append(f"{rows[i].meta.group}/{rows[i].meta.label}: slack={slacks[i]:.6g}")
        return DiagnosisReport(groups, "; ".join(details) or "no violated row found")


__all__ = ["DiagnosisReport", "InfeasibilityDiagnoser"]
