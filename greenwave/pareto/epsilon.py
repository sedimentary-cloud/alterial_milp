"""阶段 2：用 ε-constraint 扫描 Pareto 前沿。

给高中生看的小课堂
------------------
我们希望同时做到两件事：
- 总带宽尽量大；
- 路口损失尽量小。

这两个目标往往互相矛盾，不能简单相加。
ε-constraint 的做法是：
- 先把总带宽固定在一个接近最优的值 B*；
- 然后允许它稍微下降 ε；
- 在这个前提下，去最小化路口损失；
- ε 从小到大取几个点，就得到一条“权衡曲线”，即 Pareto 前沿。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..model.builder import ModelBuilder
from ..model.constraints import ConstraintMeta
from ..report.solution import SolutionDecoder, SolutionRecord
from ..solve.backend_base import SolveOptions, SolveStatus


# 一个 Pareto 点：ε、实际 composite、路口损失、完整方案。
@dataclass(frozen=True)
class ParetoPoint:
    epsilon_s: float
    composite_actual_s: float
    intersection_loss_s: float
    solution: SolutionRecord


# 一条 Pareto 前沿，同时也是普通 list，方便直接遍历。
class ParetoFront(list):
    """A list of ParetoPoint that also remembers the base grid result."""

    def __init__(self, points: Iterable[ParetoPoint], base_grid=None):
        super().__init__(points)
        self.base_grid = base_grid


# ε-constraint 扫描器。
class EpsilonConstraintScanner:
    def __init__(self, problem, ctx, backend, cfg) -> None:
        self.problem = problem
        self.ctx = ctx
        self.backend = backend
        self.cfg = cfg

    def scan(self, base) -> ParetoFront:
        if base.composite_best is None:
            return ParetoFront([], base_grid=base)
        model = ModelBuilder(self.problem, self.ctx).build()
        decoder = SolutionDecoder(model)
        Bstar = float(base.composite_best)
        relax_mag = abs(Bstar) if abs(Bstar) > 1e-12 else 1.0
        points: list[ParetoPoint] = []
        # 阶段 1 返回的 B* 是求解器浮点目标值，可能比同一模型的
        # 实际可行 composite 高 1e-13 量级。l=0 的 eps=0 点若严格用
        # B* 作下界，会被数值误差误判为不可行，从而丢掉 Pareto 原点。
        # 这里加一个远小于报告精度的数值容差，避免该问题。
        num_slack = 1e-7 * max(1.0, relax_mag)
        for l in range(self.cfg.num_points + 1):
            eps = (l / max(1, self.cfg.num_points)) * self.cfg.relax_max * relax_mag
            session = self.backend.create(model)
            self.backend.set_objective(session, model.expr_loss(), "min")
            self.backend.add_constraint(
                session, model.expr_composite(), ">=", Bstar - eps - num_slack,
                ConstraintMeta("EPS", f"composite >= B*-epsilon l={l}", {"epsilon": str(eps)}),
            )
            res = self.backend.solve(session, SolveOptions(
                time_limit_s=self.problem.solver.time_limit_s,
                mip_rel_gap=self.problem.solver.mip_rel_gap,
            ))
            if res.values is None or res.status in (SolveStatus.INFEASIBLE, SolveStatus.ERROR):
                continue
            sol = decoder.decode(res.values)
            points.append(ParetoPoint(float(eps), float(sol.composite_s),
                                      float(sol.loss_s), sol))
        return ParetoFront(points, base_grid=base)


__all__ = ["ParetoPoint", "ParetoFront", "EpsilonConstraintScanner"]
