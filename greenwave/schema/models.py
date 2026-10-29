"""输入数据模型：把用户写的 JSON 变成程序能安全使用的对象。

给高中生看的小课堂
------------------
用户会用 JSON 描述：
- 一条干线有哪些路口、路段多长、速度多少；
- 每个路口有哪些候选信号方案；
- 想优化哪些方向的绿波/红波；
- 有哪些硬约束、软约束；
- 周期和速度倍率有哪些候选值。

本文件把这些 JSON 解析成不可变的 dataclass 对象。
“不可变”意思是：对象建好以后不能随便改，避免求解过程中
某段代码不小心改坏了输入数据。

单位命名约定：
- 以 _s 结尾：秒；
- 以 _r 结尾：周期比例，例如 0.45 表示 45% 的周期；
- 以 _m 结尾：米；
- 以 _mps 结尾：米/秒。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

from ..errors import ValidationIssue

Direction = Literal["up", "down"]
Side = Literal["start", "end"]
AtomKind = Literal["offset", "boundary", "window_duration", "red_duration", "bandwidth", "const"]
Unit = Literal["s", "ratio"]
Owner = Literal["band_loss", "intersection_reward", "intersection_loss"]
ConstraintScopeKind = Literal["all_plan_windows", "all_plan_red_windows", "specific_plan_windows", "all_plans", "all_windows_of_plan"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


def _check_direction(value: Any) -> Direction:
    if value not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {value!r}")
    return value  # type: ignore[return-value]


def _check_side(value: Any) -> Side:
    if value not in ("start", "end"):
        raise ValueError(f"side must be 'start' or 'end', got {value!r}")
    return value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Plan / boundary schema (v2)
# ---------------------------------------------------------------------------

# 一个相位边界：周期比例位置 + 可选的可调范围（比例偏移）。
@dataclass(frozen=True)
class BoundarySpec:
    pos_r: float
    adjust: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.adjust is not None:
            lo, hi = self.adjust
            if lo > 0.0 or hi < 0.0:
                raise ValueError("boundary adjust range must satisfy lo <= 0 <= hi")

    @property
    def lo_r(self) -> float:
        return 0.0 if self.adjust is None else float(self.adjust[0])

    @property
    def hi_r(self) -> float:
        return 0.0 if self.adjust is None else float(self.adjust[1])

    @property
    def adjustable(self) -> bool:
        return self.adjust is not None

    @classmethod
    def from_obj(cls, obj: Any) -> "BoundarySpec":
        if isinstance(obj, BoundarySpec):
            return obj
        if isinstance(obj, (int, float)):
            return cls(float(obj))
        if not isinstance(obj, Mapping):
            raise ValueError(f"boundary must be a number or object, got {obj!r}")
        pos = obj.get("pos_r", obj.get("pos"))
        if pos is None:
            raise ValueError("boundary object requires pos_r")
        adj = obj.get("adjust")
        if adj is None:
            return cls(_num(pos, "pos_r"))
        if not isinstance(adj, (list, tuple)) or len(adj) != 2:
            raise ValueError("boundary adjust must be a two-element [lo, hi] range")
        return cls(_num(pos, "pos_r"), (_num(adj[0], "adjust.lo"), _num(adj[1], "adjust.hi")))


# 一个方向声明：有序相位边界 + 每段的颜色与 margin 配置（GreenSegmentSpec）。
@dataclass(frozen=True)
class DirectionPlan:
    boundaries: tuple[BoundarySpec, ...] = ()
    green_mask: tuple[GreenSegmentSpec, ...] = field(
        default_factory=lambda: (GreenSegmentSpec(False),))

    def __post_init__(self) -> None:
        pos = [b.pos_r for b in self.boundaries]
        if any(not (0.0 < p < 1.0) for p in pos):
            raise ValueError("boundary positions must be in (0, 1)")
        for a, b in zip(pos, pos[1:]):
            if a >= b:
                raise ValueError("boundary positions must be strictly increasing")
        if len(self.green_mask) != len(self.boundaries) + 1:
            raise ValueError("green_mask length must equal boundary count + 1")
        for seg in self.green_mask:
            if not isinstance(seg, GreenSegmentSpec):
                raise ValueError(
                    "green_mask entries must be GreenSegmentSpec objects "
                    '({"green": ..., "margin": ...}), not booleans')

    @property
    def boundary_count(self) -> int:
        return len(self.boundaries)

    @property
    def mask_bool(self) -> tuple[bool, ...]:
        """只取颜色位（窗口派生、画图等只需要 True/False）。"""
        return tuple(seg.green for seg in self.green_mask)

    @classmethod
    def from_obj(cls, obj: Any) -> "DirectionPlan":
        if isinstance(obj, DirectionPlan):
            return obj
        if obj is None:
            return cls()
        if not isinstance(obj, Mapping):
            raise ValueError(f"direction plan must be an object, got {obj!r}")
        raw_bounds = obj.get("boundaries", ())
        boundaries = tuple(BoundarySpec.from_obj(b) for b in raw_bounds)
        raw_mask = obj.get("green_mask", obj.get("mask"))
        if raw_mask is None:
            raise ValueError("direction plan requires green_mask")
        green_mask = tuple(GreenSegmentSpec.from_obj(x) for x in raw_mask)
        return cls(boundaries, green_mask)


# 派生后的一个区间（绿窗 / 红窗）。边下标是“扩展 edge”下标。
@dataclass(frozen=True)
class DerivedInterval:
    rid: int
    kind: str                 # "green" | "gap" | "wrap"
    start_edge: int
    end_edge: int
    wrap: int
    start_r: float
    end_r: float
    after_green: int = 0

    @property
    def width_r(self) -> float:
        return self.end_r - self.start_r


# 兼容旧名字：消费者只需要 .start_r / .end_r / .width_r。
WindowSpec = DerivedInterval


@dataclass(frozen=True)
class AtomRef:
    kind: AtomKind
    intersection: str | None = None
    plan: str | None = None
    direction: Direction | None = None
    index: int | None = None          # boundary index (0-based)
    window: int | None = None         # derived window index / red rid
    demand: str | None = None
    const: float | None = None
    # 只有 const 原子用：'s'（默认，秒）或 'ratio'（乘周期 C）。
    unit: Unit | None = None

    def __post_init__(self) -> None:
        if self.kind == "offset":
            if self.intersection is None:
                raise ValueError("offset atom requires intersection")
        elif self.kind == "boundary":
            missing = [n for n, v in (
                ("intersection", self.intersection), ("plan", self.plan),
                ("direction", self.direction), ("index", self.index)) if v is None]
            if missing:
                raise ValueError(f"boundary atom missing fields: {missing}")
            if self.direction not in ("up", "down"):
                raise ValueError("boundary atom direction must be up/down")
            if self.index < 0:
                raise ValueError("boundary atom index must be >= 0")
        elif self.kind in ("window_duration", "red_duration"):
            if self.direction is not None:
                _check_direction(self.direction)
            if self.kind == "window_duration" and self.window is not None and self.window < 0:
                raise ValueError("window_duration atom window index must be >= 0")
            if self.kind == "red_duration" and self.window is not None and self.window < 1:
                raise ValueError("red_duration atom rid must be >= 1")
        elif self.kind == "bandwidth":
            if self.demand is None:
                raise ValueError("bandwidth atom requires demand")
        elif self.kind == "const":
            if self.const is None:
                raise ValueError("const atom requires const")
        else:  # pragma: no cover - Literal is developer-facing
            raise ValueError(f"unknown atom kind {self.kind!r}")
        if self.unit is not None:
            if self.kind != "const":
                raise ValueError("atom unit is only allowed on const atoms")
            if self.unit not in ("s", "ratio"):
                raise ValueError(f"const atom unit must be 's' or 'ratio', got {self.unit!r}")

    @classmethod
    def from_obj(cls, obj: Any) -> "AtomRef":
        if isinstance(obj, AtomRef):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"atom must be an object, got {obj!r}")
        return cls(
            kind=obj["kind"],
            intersection=obj.get("intersection"),
            plan=obj.get("plan"),
            direction=_check_direction(obj["direction"]) if obj.get("direction") is not None else None,
            index=int(obj["index"]) if obj.get("index") is not None else None,
            window=int(obj["window"]) if obj.get("window") is not None else None,
            demand=obj.get("demand"),
            const=_num(obj["const"], "const") if obj.get("const") is not None else None,
            unit=obj.get("unit"),
        )


# 线性项：一个原子乘以一个系数。
# 例如 2.0 × I2 路口偏置。
@dataclass(frozen=True)
class LinearTerm:
    atom: AtomRef
    coef: float

    @classmethod
    def from_obj(cls, obj: Any) -> "LinearTerm":
        if isinstance(obj, LinearTerm):
            return obj
        return cls(AtomRef.from_obj(obj["atom"]), _num(obj["coef"], "coef"))


# 软约束的惩罚/奖励配置。
# owner 决定这笔账记到哪里：
#   band_loss          -> 带目标的损失（从 composite 减掉）
#   intersection_reward-> 路口回报（超额满足量）
#   intersection_loss  -> 路口损失（违反量）
@dataclass(frozen=True)
class SoftPenalty:
    coef: float
    owner: Owner

    def __post_init__(self) -> None:
        if self.owner not in ("band_loss", "intersection_reward", "intersection_loss"):
            raise ValueError(f"unknown soft constraint owner {self.owner!r}")
        if self.coef <= 0.0:
            raise ValueError("soft constraint coef must be > 0")

    @classmethod
    def from_obj(cls, obj: Any) -> "SoftPenalty":
        if isinstance(obj, SoftPenalty):
            return obj
        return cls(_num(obj["coef"], "coef"), obj["owner"])


def _parse_rhs_fields(obj: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """把统一写法 ``rhs + rhs_unit`` 与旧写法 ``rhs_s/rhs_ratio`` 解析成内部字段。

    ``rhs_s`` 与 ``rhs_ratio`` 可以同时提供：建模时 RHS = rhs_ratio×C + rhs_s。

    - rhs_unit == "s"      -> rhs_s
    - rhs_unit == "ratio"  -> rhs_ratio
    - 未写 rhs_unit 时，``rhs`` 默认按秒处理，保持旧 JSON 兼容。
    """
    rhs = obj.get("rhs")
    unit = obj.get("rhs_unit")
    if rhs is None and unit is not None:
        raise ValueError(
            "rhs_unit requires rhs; use rhs_s/rhs_ratio or provide rhs with rhs_unit")
    if rhs is not None:
        if obj.get("rhs_s") is not None or obj.get("rhs_ratio") is not None:
            raise ValueError("rhs and rhs_s/rhs_ratio cannot be provided together")
        if unit is None:
            unit = "s"
        if unit == "s":
            return _num(rhs, "rhs"), None
        if unit == "ratio":
            return None, _num(rhs, "rhs")
        raise ValueError(f"rhs_unit must be 's' or 'ratio', got {unit!r}")
    rhs_s = obj.get("rhs_s")
    rhs_ratio = obj.get("rhs_ratio")
    return (_num(rhs_s, "rhs_s") if rhs_s is not None else None,
            _num(rhs_ratio, "rhs_ratio") if rhs_ratio is not None else None)


# 一条用户自定义约束：线性表达式 + sense + 右边常数 + 硬/软属性。
@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    terms: tuple[LinearTerm, ...]
    sense: Literal["<=", ">=", "=="]
    hard: bool
    rhs_s: float | None = None
    rhs_ratio: float | None = None
    soft: SoftPenalty | None = None

    def __post_init__(self) -> None:
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"unknown constraint sense {self.sense!r}")
        if self.rhs_s is None and self.rhs_ratio is None:
            raise ValueError(
                "rhs_s and rhs_ratio cannot both be missing; they are additive "
                "when both are provided (rhs = rhs_ratio*C + rhs_s)")
        if self.hard:
            if self.soft is not None:
                raise ValueError("hard constraint must not carry soft penalty")
        else:
            if self.soft is None:
                raise ValueError("soft constraint requires soft penalty")

    def rhs_for_cycle(self, cycle_s: float) -> float:
        """RHS = rhs_ratio × C + rhs_s（两项都可选，同时给就是相加）。"""
        total = 0.0
        if self.rhs_ratio is not None:
            total += self.rhs_ratio * cycle_s
        if self.rhs_s is not None:
            total += self.rhs_s
        return total

    @classmethod
    def from_obj(cls, obj: Any) -> "ConstraintSpec":
        if isinstance(obj, ConstraintSpec):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"constraint must be an object, got {obj!r}")
        soft_obj = obj.get("soft")
        rhs_s, rhs_ratio = _parse_rhs_fields(obj)
        return cls(
            id=str(obj["id"]),
            terms=tuple(LinearTerm.from_obj(t) for t in obj.get("terms", ())),
            sense=obj.get("sense", "<="),
            hard=bool(obj.get("hard", True)),
            rhs_s=rhs_s,
            rhs_ratio=rhs_ratio,
            soft=SoftPenalty.from_obj(soft_obj) if soft_obj is not None else None,
        )


# ---------------------------------------------------------------------------
# Scoped constraint templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintScope:
    """约束模板要展开到哪些 plan / direction / window 上。"""

    kind: ConstraintScopeKind = "all_plan_windows"
    directions: tuple[Direction, ...] = ("up", "down")
    plan_ids: tuple[str, ...] | None = None
    window_indices: tuple[int, ...] | None = None
    skip_double_cycle_copies: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("all_plan_windows", "all_plan_red_windows",
                             "specific_plan_windows", "all_plans", "all_windows_of_plan"):
            raise ValueError(f"unknown constraint scope kind {self.kind!r}")
        if not self.directions:
            raise ValueError("constraint scope directions must not be empty")
        for d in self.directions:
            _check_direction(d)
        if self.window_indices is not None and any(k < 0 for k in self.window_indices):
            raise ValueError("constraint scope window_indices must be >= 0")
        if self.kind in ("all_windows_of_plan", "specific_plan_windows") and not self.plan_ids:
            raise ValueError(f"constraint scope kind {self.kind!r} requires plan_ids")
        if self.kind == "specific_plan_windows" and not self.window_indices:
            raise ValueError("constraint scope kind 'specific_plan_windows' requires window_indices")

    @classmethod
    def from_obj(cls, obj: Any) -> "ConstraintScope":
        if isinstance(obj, ConstraintScope):
            return obj
        if obj is None:
            return cls()
        if not isinstance(obj, Mapping):
            raise ValueError(f"constraint scope must be an object, got {obj!r}")
        raw_dirs = obj.get("directions")
        directions = (tuple(_check_direction(d) for d in raw_dirs)
                      if raw_dirs is not None else ("up", "down"))
        plan_ids = obj.get("plan_ids")
        window_indices = obj.get("window_indices")
        return cls(
            kind=obj.get("kind", "all_plan_windows"),
            directions=tuple(directions),
            plan_ids=None if plan_ids is None else tuple(str(x) for x in plan_ids),
            window_indices=(None if window_indices is None
                            else tuple(int(x) for x in window_indices)),
            skip_double_cycle_copies=bool(obj.get("skip_double_cycle_copies", True)),
        )


@dataclass(frozen=True)
class ConstraintTemplate:
    """一条需要按 scope 展开的约束模板。

    模板里的 ``window_duration`` 等作用域原子可以不写 plan/direction/window，
    由建模器在展开时填入当前 scope 的取值。
    """

    id: str
    scope: ConstraintScope
    terms: tuple[LinearTerm, ...]
    sense: Literal["<=", ">=", "=="]
    hard: bool
    rhs_s: float | None = None
    rhs_ratio: float | None = None
    soft: SoftPenalty | None = None

    def __post_init__(self) -> None:
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"unknown constraint sense {self.sense!r}")
        if self.rhs_s is None and self.rhs_ratio is None:
            raise ValueError(
                "rhs_s and rhs_ratio cannot both be missing; they are additive "
                "when both are provided (rhs = rhs_ratio*C + rhs_s)")
        if self.hard:
            if self.soft is not None:
                raise ValueError("hard constraint template must not carry soft penalty")
        else:
            if self.soft is None:
                raise ValueError("soft constraint template requires soft penalty")

    def rhs_for_cycle(self, cycle_s: float) -> float:
        """RHS = rhs_ratio × C + rhs_s（两项都可选，同时给就是相加）。"""
        total = 0.0
        if self.rhs_ratio is not None:
            total += self.rhs_ratio * cycle_s
        if self.rhs_s is not None:
            total += self.rhs_s
        return total

    @classmethod
    def from_obj(cls, obj: Any) -> "ConstraintTemplate":
        if isinstance(obj, ConstraintTemplate):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"constraint template must be an object, got {obj!r}")
        soft_obj = obj.get("soft")
        rhs_s, rhs_ratio = _parse_rhs_fields(obj)
        return cls(
            id=str(obj["id"]),
            scope=ConstraintScope.from_obj(obj.get("scope")),
            terms=tuple(LinearTerm.from_obj(t) for t in obj.get("terms", ())),
            sense=obj.get("sense", "<="),
            hard=bool(obj.get("hard", True)),
            rhs_s=rhs_s,
            rhs_ratio=rhs_ratio,
            soft=SoftPenalty.from_obj(soft_obj) if soft_obj is not None else None,
        )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

# 一个信号控制候选方案（v2：相位边界建模）。
# 干线只感受每个方向由边界和 green_mask 派生出的绿灯窗口；
# 边界本身是一等变量，可以被任意约束引用。
@dataclass(frozen=True)
class PlanSpec:
    id: str
    intersection: str
    directions: Mapping[Direction, DirectionPlan]
    double_cycle: bool = False
    min_phase_s: float = 1e-3
    hard_constraints: tuple[ConstraintSpec, ...] = ()
    soft_constraints: tuple[ConstraintSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.min_phase_s < 0.0:
            raise ValueError("min_phase_s must be >= 0")
        for d in ("up", "down"):
            dp = self.directions.get(d)
            if dp is None:
                continue
            if self.double_cycle:
                for b in dp.boundaries:
                    if not (0.0 < b.pos_r < 0.5):
                        raise ValueError(
                            f"plan {self.id!r} double_cycle {d} boundaries must be in (0, 0.5)")
            else:
                for b in dp.boundaries:
                    if not (0.0 < b.pos_r < 1.0):
                        raise ValueError(
                            f"plan {self.id!r} {d} boundaries must be in (0, 1)")

    # -- declared access ------------------------------------------------------
    def direction_plan(self, d: Direction) -> DirectionPlan:
        return self.directions.get(d, DirectionPlan((), (GreenSegmentSpec(False),)))

    def boundary_count(self, d: Direction) -> int:
        return self.direction_plan(d).boundary_count

    def nominal_boundary_r(self, d: Direction, j: int) -> float:
        return self.direction_plan(d).boundaries[j].pos_r

    def boundary_is_adjustable(self, d: Direction, j: int) -> bool:
        return self.direction_plan(d).boundaries[j].adjustable

    def boundary_adjust_range(self, d: Direction, j: int):
        b = self.direction_plan(d).boundaries[j]
        return None if b.adjust is None else (b.lo_r, b.hi_r)

    # -- expanded edges (double-cycle aware) ----------------------------------
    def _expanded_boundary_specs(self, d: Direction):
        """Interior expanded boundaries as (nominal_r, source_kind, source_idx, offset)."""
        dp = self.direction_plan(d)
        if not self.double_cycle:
            return tuple((b.pos_r, "boundary", j, 0.0) for j, b in enumerate(dp.boundaries))
        specs = [(b.pos_r, "boundary", j, 0.0) for j, b in enumerate(dp.boundaries)]
        specs.append((0.5, "const", -1, 0.5))
        specs.extend((b.pos_r + 0.5, "boundary", j, 0.5) for j, b in enumerate(dp.boundaries))
        specs.sort(key=lambda x: x[0])
        return tuple(specs)

    def expanded_edge_ratios(self, d: Direction) -> tuple[float, ...]:
        specs = self._expanded_boundary_specs(d)
        return (0.0,) + tuple(s[0] for s in specs) + (1.0,)

    def expanded_edge_sources(self, d: Direction) -> tuple[tuple[str, int, float], ...]:
        specs = self._expanded_boundary_specs(d)
        return (("const", 0, 0.0),) + tuple((s[1], s[2], s[3]) for s in specs) + (("const", 1, 1.0),)

    def expanded_segments(self, d: Direction) -> tuple[GreenSegmentSpec, ...]:
        segs = self.direction_plan(d).green_mask
        if not self.double_cycle:
            return segs
        return segs + segs

    def expanded_mask(self, d: Direction) -> tuple[bool, ...]:
        mask = self.direction_plan(d).mask_bool
        if not self.double_cycle:
            return mask
        return mask + mask

    # -- derived green windows ------------------------------------------------
    @staticmethod
    def _windows_from_ratios(ratios, mask, plan_id: str, d: Direction):
        if mask and mask[0] and mask[-1]:
            raise ValueError(
                f"plan {plan_id!r} {d} green_mask first and last intervals are both green; "
                f"rotate the plan so the complete green interval lies inside one cycle")
        out: list[DerivedInterval] = []
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append(DerivedInterval(
                rid=len(out) + 1, kind="green",
                start_edge=i, end_edge=j + 1, wrap=0,
                start_r=ratios[i], end_r=ratios[j + 1],
            ))
            i = j + 1
        return tuple(out)

    def derived_windows(self, d: Direction) -> tuple[DerivedInterval, ...]:
        return self._windows_from_ratios(self.expanded_edge_ratios(d),
                                         self.expanded_mask(d), self.id, d)

    def derived_windows_from_boundary_r(self, d: Direction,
                                        boundary_r: Mapping[int, float]):
        """Derived windows using adjusted boundary values (or nominal if absent)."""
        ratios = []
        for edge in range(len(self.expanded_edge_sources(d))):
            kind, src, offset = self.expanded_edge_sources(d)[edge]
            if kind == "const":
                ratios.append(float(offset))
            else:
                ratios.append(float(boundary_r.get(src, self.nominal_boundary_r(d, src)))
                              + offset)
        return self._windows_from_ratios(tuple(ratios), self.expanded_mask(d), self.id, d)

    # 兼容旧调用名：主读者仍然使用 effective_windows。
    def effective_windows(self, d: Direction) -> tuple[DerivedInterval, ...]:
        return self.derived_windows(d)

    def window_count(self, d: Direction) -> int:
        return len(self.derived_windows(d))

    def window_edge_span(self, d: Direction, k: int) -> DerivedInterval:
        return self.derived_windows(d)[k]

    def window_margins(self, d: Direction, k: int):
        """第 k 个绿窗的 (start margin, end margin)。

        start 取形成该窗口的第一段绿的 ``margin_start``；
        end 取形成该窗口的最后一段绿的 ``margin_end``。
        同色内部边界的配置不会被返回（窗口边界必是颜色变化边界）。
        """
        w = self.window_edge_span(d, k)
        segs = self.expanded_segments(d)
        start = segs[w.start_edge].margin_start if 0 <= w.start_edge < len(segs) else None
        j = w.end_edge - 1
        end = segs[j].margin_end if 0 <= j < len(segs) else None
        return start, end

    def is_copy_window(self, d: Direction, k: int) -> bool:
        """For double-cycle plans, second-half derived windows are copies."""
        if not self.double_cycle:
            return False
        return self.window_edge_span(d, k).start_r >= 0.5 - 1e-12

    def nominal_window_endpoint_r(self, d: Direction, k: int, side: Side) -> float:
        w = self.window_edge_span(d, k)
        return w.start_r if side == "start" else w.end_r

    # -- derived red intervals -------------------------------------------------
    def derived_red_intervals(self, d: Direction) -> tuple[DerivedInterval, ...]:
        green = self.derived_windows(d)
        if not green:
            return ()
        ratios = self.expanded_edge_ratios(d)
        out: list[DerivedInterval] = []
        K = len(green)
        for i in range(K):
            if i < K - 1:
                start_edge = green[i].end_edge
                end_edge = green[i + 1].start_edge
                wrap = 0
                kind = "gap"
                after = i + 1
            else:
                start_edge = green[-1].end_edge
                end_edge = green[0].start_edge
                wrap = 1
                kind = "wrap"
                after = K
            out.append(DerivedInterval(
                rid=i + 1, kind=kind,
                start_edge=start_edge, end_edge=end_edge, wrap=wrap,
                start_r=ratios[start_edge], end_r=ratios[end_edge] + wrap,
                after_green=after,
            ))
        return tuple(out)

    def red_window_count(self, d: Direction) -> int:
        return len(self.derived_red_intervals(d))

    def window_endpoint_bound_r(self, d: Direction, k: int, side: Side):
        """Nominal endpoint bounds (ratio) including adjustable ranges and wrap."""
        w = self.window_edge_span(d, k)
        edge = w.start_edge if side == "start" else w.end_edge
        sources = self.expanded_edge_sources(d)
        kind, src, offset = sources[edge]
        wrap = 0 if side == "start" else w.wrap
        if kind == "const":
            val = float(offset)
            return val + wrap, val + wrap
        b = self.direction_plan(d).boundaries[src]
        lo = b.pos_r + offset + (b.lo_r if b.adjust is not None else 0.0) + wrap
        hi = b.pos_r + offset + (b.hi_r if b.adjust is not None else 0.0) + wrap
        return lo, hi

    @classmethod
    def from_obj(cls, obj: Any) -> "PlanSpec":
        if isinstance(obj, PlanSpec):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"plan must be an object, got {obj!r}")
        if "windows" in obj and "directions" not in obj:
            raise ValueError(
                "v2 schema no longer accepts 'windows'; use directions[d].boundaries + green_mask")
        directions_in = obj.get("directions", {}) or {}
        directions: dict[Direction, DirectionPlan] = {}
        for d in ("up", "down"):
            if d in directions_in:
                directions[d] = DirectionPlan.from_obj(directions_in[d])
        return cls(
            id=str(obj["id"]),
            intersection=str(obj["intersection"]),
            directions=directions,
            double_cycle=bool(obj.get("double_cycle", False)),
            min_phase_s=_num(obj.get("min_phase_s", 1e-3), "min_phase_s"),
            hard_constraints=tuple(ConstraintSpec.from_obj(c) for c in obj.get("hard_constraints", ())),
            soft_constraints=tuple(ConstraintSpec.from_obj(c) for c in obj.get("soft_constraints", ())),
        )


# ---------------------------------------------------------------------------
# Corridor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentSpec:
    """一段干线路段。

    ``distance_m`` is the legacy/common length used for both directions.
    Directional lengths may be supplied independently through
    ``distance_up_m`` and ``distance_down_m``; when present they override the
    common value for that direction only.
    """

    from_: str
    to: str
    distance_m: float | None
    speed_up_mps: float
    speed_down_mps: float
    distance_up_m: float | None = None
    distance_down_m: float | None = None

    def distance_for(self, d: Direction) -> float:
        """Return the effective segment length for direction ``d``."""
        if d == "up" and self.distance_up_m is not None:
            return float(self.distance_up_m)
        if d == "down" and self.distance_down_m is not None:
            return float(self.distance_down_m)
        if self.distance_m is not None:
            return float(self.distance_m)
        raise ValueError(
            f"segment {self.from_}->{self.to} has no distance for direction {d!r}"
        )

    @classmethod
    def from_obj(cls, obj: Any) -> "SegmentSpec":
        if isinstance(obj, SegmentSpec):
            return obj
        distance_m = obj.get("distance_m")
        distance_up_m = obj.get("distance_up_m")
        distance_down_m = obj.get("distance_down_m")
        return cls(
            from_=str(obj.get("from", obj.get("from_"))),
            to=str(obj["to"]),
            distance_m=_num(distance_m, "distance_m") if distance_m is not None else None,
            speed_up_mps=_num(obj["speed_up_mps"], "speed_up_mps"),
            speed_down_mps=_num(obj["speed_down_mps"], "speed_down_mps"),
            distance_up_m=_num(distance_up_m, "distance_up_m") if distance_up_m is not None else None,
            distance_down_m=_num(distance_down_m, "distance_down_m") if distance_down_m is not None else None,
        )


# 干线：有序路口 + 相邻路段。
@dataclass(frozen=True)
class CorridorSpec:
    corridor_id: str
    intersections: tuple[str, ...]
    segments: tuple[SegmentSpec, ...]

    def index(self, intersection_id: str) -> int:
        return self.intersections.index(intersection_id)

    def speed_mps(self, seg_index: int, d: Direction) -> float:
        seg = self.segments[seg_index]
        return seg.speed_up_mps if d == "up" else seg.speed_down_mps

    def distance_m(self, seg_index: int, d: Direction) -> float:
        return self.segments[seg_index].distance_for(d)

    def segment_for_direction(self, intersection_id: str, d: Direction) -> tuple[int, SegmentSpec]:
        """Return segment index and segment leaving ``intersection_id`` along ``d``."""
        idx = self.index(intersection_id)
        if d == "up":
            if idx >= len(self.segments):
                raise IndexError(f"no up segment from {intersection_id}")
            return idx, self.segments[idx]
        if idx <= 0:
            raise IndexError(f"no down segment from {intersection_id}")
        return idx - 1, self.segments[idx - 1]

    @classmethod
    def from_obj(cls, obj: Any) -> "CorridorSpec":
        if isinstance(obj, CorridorSpec):
            return obj
        return cls(
            corridor_id=str(obj["corridor_id"]),
            intersections=tuple(str(x) for x in obj["intersections"]),
            segments=tuple(SegmentSpec.from_obj(s) for s in obj["segments"]),
        )


# ---------------------------------------------------------------------------
# Objective / grid / solver
# ---------------------------------------------------------------------------

# 一个相位边界上的 margin 配置：带宽端离这个绿灯窗口边界至少留多少秒。
# 只挂在 green_mask 的绿段上（见 GreenSegmentSpec），不再有 demand/全局默认值。
@dataclass(frozen=True)
class MarginSpec:
    # 软 margin 目标：低于该值产生罚项（0 表示没有软损失）。
    delta_min_s: float = 0.0
    coef: float = 1.0
    # 硬 margin 下限：低于该值直接不可行；0 表示不启用硬约束。
    hard_min_s: float = 0.0

    def to_obj(self) -> dict[str, float]:
        return {"delta_min_s": self.delta_min_s, "coef": self.coef,
                "hard_min_s": self.hard_min_s}

    @classmethod
    def from_obj(cls, obj: Any) -> "MarginSpec":
        if isinstance(obj, MarginSpec):
            return obj
        if obj is None:
            return cls()
        return cls(_num(obj.get("delta_min_s", 0.0), "delta_min_s"),
                   _num(obj.get("coef", 1.0), "coef"),
                   _num(obj.get("hard_min_s", 0.0), "hard_min_s"))


# 一个相位区间（相邻边界之间）的颜色与 margin 配置。
#   green        : 这段是否绿灯；
#   margin_start : 本段左边界上的 margin（仅当本段是绿、且左边界两侧颜色不同才生效）；
#   margin_end   : 本段右边界上的 margin（同上）。
# 同色边界、或红段上的 margin 配置会被忽略（schema 校验给 warning）。
@dataclass(frozen=True)
class GreenSegmentSpec:
    green: bool = False
    margin_start: MarginSpec | None = None
    margin_end: MarginSpec | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> "GreenSegmentSpec":
        if isinstance(obj, GreenSegmentSpec):
            return obj
        if isinstance(obj, bool):
            # 简写：只给颜色、不带 margin。正式格式仍是对象。
            return cls(green=obj)
        if not isinstance(obj, Mapping):
            raise ValueError(f"green segment must be an object, got {obj!r}")
        if "green" not in obj:
            raise ValueError(f"green segment requires a 'green' boolean: {obj!r}")
        margin = obj.get("margin")
        start = None
        end = None
        if margin is not None:
            if not isinstance(margin, Mapping):
                raise ValueError(f"green segment margin must be an object, got {margin!r}")
            if margin.get("start") is not None:
                start = MarginSpec.from_obj(margin["start"])
            if margin.get("end") is not None:
                end = MarginSpec.from_obj(margin["end"])
        return cls(green=bool(obj["green"]), margin_start=start, margin_end=end)

    def to_obj(self) -> dict[str, Any]:
        out: dict[str, Any] = {"green": self.green}
        if self.margin_start is not None or self.margin_end is not None:
            out["margin"] = {
                "start": None if self.margin_start is None else self.margin_start.to_obj(),
                "end": None if self.margin_end is None else self.margin_end.to_obj(),
            }
        return out


# 带边界亲和性配置（绿波、红波共用）：
# 带的一端离参考窗口的对应端点多远开始罚。
#   free_s     : 前 free_s 秒免费；
#   coef       : 超出部分每秒罚多少（单位同 composite，即秒）；
#   hard_max_s : 可选的硬上限，None 表示不加硬约束。
@dataclass(frozen=True)
class EdgeSpec:
    free_s: float = 0.0
    hard_max_s: float | None = None
    coef: float = 1.0

    def __post_init__(self) -> None:
        if self.free_s < 0:
            raise ValueError("edge free_s must be >= 0")
        if self.coef < 0:
            raise ValueError("edge coef must be >= 0")
        if self.hard_max_s is not None and self.hard_max_s < self.free_s:
            raise ValueError("edge hard_max_s must be >= free_s")

    @classmethod
    def from_obj(cls, obj: Any) -> "EdgeSpec | None":
        if isinstance(obj, EdgeSpec):
            return obj
        if obj is None:
            return None
        if not isinstance(obj, Mapping):
            raise ValueError(f"edge spec must be an object, got {obj!r}")
        hard = obj.get("hard_max_s")
        return cls(
            free_s=_num(obj.get("free_s", 0.0), "free_s"),
            hard_max_s=(_num(hard, "hard_max_s") if hard is not None else None),
            coef=_num(obj.get("coef", 1.0), "coef"),
        )


# 一条带的边界亲和配置：起点侧、终点侧各自独立（可只开一侧，或给不同系数）。
# 绿波（green_boundary）：带端离本路口绿窗对应端点多远开始罚；
# 红波（red_boundary）  ：带端离上游绿窗对应端点多远开始罚。
@dataclass(frozen=True)
class BoundaryAffinitySpec:
    start: EdgeSpec | None = None
    end: EdgeSpec | None = None

    def __post_init__(self) -> None:
        # 两端都可以为空；全空表示不启用。
        pass

    @property
    def enabled(self) -> bool:
        return self.start is not None or self.end is not None

    @classmethod
    def from_obj(cls, obj: Any) -> "BoundaryAffinitySpec":
        if isinstance(obj, BoundaryAffinitySpec):
            return obj
        if obj is None:
            return cls()
        if not isinstance(obj, Mapping):
            raise ValueError(f"boundary affinity spec must be an object, got {obj!r}")
        return cls(
            start=EdgeSpec.from_obj(obj.get("start")),
            end=EdgeSpec.from_obj(obj.get("end")),
        )


# 旧名字保留：红波边界亲和用的就是这两张表。
RedEdgeSpec = EdgeSpec
RedBoundarySpec = BoundaryAffinitySpec




# 一条优化需求：绿波带或红波带。
@dataclass(frozen=True)
class BandDemandSpec:
    id: str
    type: Literal["green", "red"]
    direction: Direction
    nodes: tuple[str, ...]
    max_bands: int | None = None
    weight: float = 1.0
    red_boundary: RedBoundarySpec | None = None
    green_boundary: BoundaryAffinitySpec | None = None

    def __post_init__(self) -> None:
        if self.type not in ("green", "red"):
            raise ValueError("demand type must be green/red")
        _check_direction(self.direction)
        if self.red_boundary is not None and self.type != "red":
            raise ValueError(f"red_boundary is only valid for red demand {self.id!r}")
        if self.green_boundary is not None and self.type != "green":
            raise ValueError(f"green_boundary is only valid for green demand {self.id!r}")

    @classmethod
    def from_obj(cls, obj: Any) -> "BandDemandSpec":
        if isinstance(obj, BandDemandSpec):
            return obj
        if "margin" in obj:
            raise ValueError(
                "demand-level margin was removed; configure margin on each green segment: "
                '{"green": true, "margin": {"start": {...}, "end": {...}}}')
        return cls(
            id=str(obj["id"]),
            type=obj.get("type", "green"),
            direction=_check_direction(obj["direction"]),
            nodes=tuple(str(x) for x in obj["nodes"]),
            max_bands=int(obj["max_bands"]) if obj.get("max_bands") is not None else None,
            weight=_num(obj.get("weight", 1.0), "weight"),
            red_boundary=(RedBoundarySpec.from_obj(obj.get("red_boundary"))
                          if obj.get("red_boundary") is not None else None),
            green_boundary=(BoundaryAffinitySpec.from_obj(obj.get("green_boundary"))
                            if obj.get("green_boundary") is not None else None),
        )



# 均衡组：让组内“每个需求的总带宽”的最小值尽量大。
# 例如组内有上行绿波和下行绿波，希望两个方向都不要太差，
# 而不是让一个方向特别宽、另一个方向为零。
@dataclass(frozen=True)
class BalancedGroupSpec:
    """均衡组：让组内“目标归一化后的总带宽”最小值尽量大。

    成员粒度：默认按需求级聚合带宽 Β_b = Σ_q β_{b,q}。
    如果提供 targets_s = {demand_id: target_s}，则约束为：

        z_g <= Β_b / target_b

    默认 target_b = 1s，此时 z_g 的单位仍是秒。
    若某个 target_b != 1，则 z_g 表示“相对于目标值的比例”。

    on_infeasible：
    - "zero"：任一成员不存在时，z=0（默认，语义最稳）；
    - "skip"：只对存在的成员取最小；建议配合 min_existing 软约束，
      避免求解器故意丢掉最难做的成员。
    """

    id: str
    demands: tuple[str, ...]
    weight: float = 1.0
    on_infeasible: Literal["zero", "skip"] = "zero"
    # demand_id -> 目标带宽（秒）；缺省为 1.0，即绝对均衡。
    targets_s: Mapping[str, float] = field(default_factory=dict)
    # 期望至少有多少个成员存在；None 表示不设置该软目标。
    min_existing: int | None = None
    # min_existing 短缺罚系数，单位仍是目标函数里的“秒”。
    min_existing_penalty_s: float = 0.0

    def target_for(self, demand_id: str) -> float:
        return float(self.targets_s.get(demand_id, 1.0))

    @classmethod
    def from_obj(cls, obj: Any) -> "BalancedGroupSpec":
        if isinstance(obj, BalancedGroupSpec):
            return obj
        # 兼容旧字段 missing_policy / missing_penalty_s：
        # 旧 penalize 映射成 skip + min_existing_penalty_s。
        legacy_policy = obj.get("missing_policy")
        on_infeasible = obj.get("on_infeasible", legacy_policy if legacy_policy in ("zero", "skip") else "zero")
        targets_raw = obj.get("targets_s", {}) or {}
        targets = {str(k): _num(v, f"targets_s[{k}]") for k, v in targets_raw.items()}
        min_pen = obj.get("min_existing_penalty_s")
        if min_pen is None and legacy_policy == "penalize":
            min_pen = obj.get("missing_penalty_s", 0.0)
        return cls(
            id=str(obj["id"]),
            demands=tuple(str(x) for x in obj.get("demands", ())),
            weight=_num(obj.get("weight", 1.0), "weight"),
            on_infeasible=on_infeasible,
            targets_s=targets,
            min_existing=int(obj["min_existing"]) if obj.get("min_existing") is not None else None,
            min_existing_penalty_s=_num(min_pen or 0.0, "min_existing_penalty_s"),
        )


# Pareto 前沿扫描配置：扫描多少个 ε 点、最大放宽多少、取前几个格点。
@dataclass(frozen=True)
class ParetoConfig:
    num_points: int = 4
    relax_max: float = 0.3
    topk_grids: int = 1

    @classmethod
    def from_obj(cls, obj: Any) -> "ParetoConfig":
        if isinstance(obj, ParetoConfig):
            return obj
        if obj is None:
            return cls()
        return cls(int(obj.get("num_points", 4)), _num(obj.get("relax_max", 0.3), "relax_max"),
                   int(obj.get("topk_grids", 1)))


# 一个聚合贡献项：把聚合值 t 显式按符号计入某个目标/损失。
# 不要求解器推断符号；target/sign/weight/direction 全部写在配置里，
# 合法性由 AggregateSpec 校验：
#   min 聚合（上包络，精确方向=最大化）：
#       target=band_objective,      sign=+1, direction=maximize
#       target=intersection_reward, sign=+1, direction=maximize
#   max 聚合（下包络，精确方向=最小化）：
#       target=band_objective,      sign=-1, direction=minimize
#       target=intersection_loss,   sign=+1, direction=minimize
@dataclass(frozen=True)
class AggregateContributionSpec:
    target: Literal["band_objective", "intersection_reward", "intersection_loss"]
    sign: int
    weight: float
    direction: Literal["maximize", "minimize"]

    def __post_init__(self) -> None:
        if self.target not in ("band_objective", "intersection_reward", "intersection_loss"):
            raise ValueError(f"unknown aggregate contribution target {self.target!r}")
        if self.sign not in (-1, 1):
            raise ValueError(f"aggregate contribution sign must be +1 or -1, got {self.sign!r}")
        if self.weight <= 0.0:
            raise ValueError("aggregate contribution weight must be > 0")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"aggregate contribution direction must be maximize/minimize, got {self.direction!r}")

    @classmethod
    def from_obj(cls, obj: Any) -> "AggregateContributionSpec":
        if isinstance(obj, AggregateContributionSpec):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"aggregate contribution must be an object, got {obj!r}")
        missing = [k for k in ("target", "sign", "direction") if k not in obj]
        if missing:
            raise ValueError(
                "aggregate contribution requires explicit "
                f"target/sign/direction (missing {missing}); legacy auto-sign is not supported")
        return cls(target=str(obj["target"]),
                   sign=int(obj["sign"]),
                   weight=_num(obj.get("weight", 1.0), "weight"),
                   direction=str(obj["direction"]))

    def to_obj(self) -> dict[str, Any]:
        return {"target": self.target, "sign": self.sign,
                "weight": self.weight, "direction": self.direction}


# 一个具名聚合：对一组线性表达式取 max/min，得到一个标量值 t，
# 再通过 contributions 计入目标/损失。表达式只允许 boundary/const 原子；
# 所有 boundary 原子必须属于同一个 plan（`plan` 字段）。
@dataclass(frozen=True)
class AggregateSpec:
    id: str
    op: Literal["max", "min"]
    expressions: tuple[tuple[LinearTerm, ...], ...]
    plan: str | None = None
    contributions: tuple[AggregateContributionSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.op not in ("max", "min"):
            raise ValueError(f"aggregate op must be 'max' or 'min', got {self.op!r}")
        if not self.expressions:
            raise ValueError("aggregate requires at least one expression")
        for i, terms in enumerate(self.expressions):
            if not terms:
                raise ValueError(f"aggregate {self.id!r} expression {i} is empty")
            for t in terms:
                if t.atom.kind not in ("boundary", "const"):
                    raise ValueError(
                        f"aggregate {self.id!r} only supports boundary/const atoms, "
                        f"got {t.atom.kind!r}")
        for c in self.contributions:
            if c.target == "band_objective":
                legal = ((self.op == "min" and c.sign == 1 and c.direction == "maximize")
                         or (self.op == "max" and c.sign == -1 and c.direction == "minimize"))
            elif c.target == "intersection_reward":
                legal = (self.op == "min" and c.sign == 1 and c.direction == "maximize")
            else:  # intersection_loss
                legal = (self.op == "max" and c.sign == 1 and c.direction == "minimize")
            if not legal:
                raise ValueError(
                    f"aggregate {self.id!r} contribution {c.to_obj()} is not exact for op={self.op!r}; "
                    "legal combinations: min+band_objective(+1,maximize), "
                    "min+intersection_reward(+1,maximize), "
                    "max+band_objective(-1,minimize), max+intersection_loss(+1,minimize)")

    @classmethod
    def from_obj(cls, obj: Any) -> "AggregateSpec":
        if isinstance(obj, AggregateSpec):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"aggregate must be an object, got {obj!r}")
        raw_exprs = obj.get("expressions", ())
        expressions = tuple(tuple(LinearTerm.from_obj(t) for t in terms)
                            for terms in raw_exprs)
        contributions = tuple(AggregateContributionSpec.from_obj(c)
                              for c in obj.get("contributions", ()))
        return cls(id=str(obj["id"]),
                   op=obj.get("op", "min"),
                   expressions=expressions,
                   plan=(str(obj["plan"]) if obj.get("plan") is not None else None),
                   contributions=contributions)

    def to_obj(self) -> dict[str, Any]:
        return {
            "id": self.id, "op": self.op, "plan": self.plan,
            "expressions": [[{"atom": _atom_obj(t.atom), "coef": t.coef}
                             for t in terms] for terms in self.expressions],
            "contributions": [c.to_obj() for c in self.contributions],
        }


# 目标配置：所有带需求、默认 margin、Pareto 配置、窗口共享开关。
@dataclass(frozen=True)
class ObjectiveConfig:
    band_demands: tuple[BandDemandSpec, ...]
    red_boundary_default: RedBoundarySpec = field(default_factory=RedBoundarySpec)
    # 绿波边界亲和：默认关闭；用户显式配置 green_boundary_default 或需求级
    # green_boundary 才生效（不做隐式默认，避免用户不知道目标里多了一项）。
    green_boundary_default: BoundaryAffinitySpec = field(
        default_factory=BoundaryAffinitySpec)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    allow_window_sharing: bool = False
    balanced_groups: tuple[BalancedGroupSpec, ...] = ()

    @classmethod
    def from_obj(cls, obj: Any) -> "ObjectiveConfig":
        if isinstance(obj, ObjectiveConfig):
            return obj
        if "margin_default" in obj:
            raise ValueError(
                "objective.margin_default was removed; configure margin per green segment "
                "inside each plan's directions[direction].green_mask")
        return cls(
            band_demands=tuple(BandDemandSpec.from_obj(d) for d in obj.get("band_demands", ())),
            red_boundary_default=RedBoundarySpec.from_obj(obj.get("red_boundary_default")),
            green_boundary_default=BoundaryAffinitySpec.from_obj(
                obj.get("green_boundary_default")),
            pareto=ParetoConfig.from_obj(obj.get("pareto")),
            allow_window_sharing=bool(obj.get("allow_window_sharing", False)),
            balanced_groups=tuple(BalancedGroupSpec.from_obj(g)
                                  for g in obj.get("balanced_groups", ())),
        )


# 格点配置：周期候选和速度倍率候选。
@dataclass(frozen=True)
class GridConfig:
    cycles_s: tuple[float, ...]
    speed_ratios: tuple[float, ...]

    @classmethod
    def from_obj(cls, obj: Any) -> "GridConfig":
        if isinstance(obj, GridConfig):
            return obj
        return cls(
            cycles_s=tuple(_num(x, "cycle_s") for x in obj["cycles_s"]),
            speed_ratios=tuple(_num(x, "speed_ratio") for x in obj["speed_ratios"]),
        )


# 贪心初始解配置。
@dataclass(frozen=True)
class HeuristicConfig:
    enabled: bool = True
    refine_rounds: int = 3

    @classmethod
    def from_obj(cls, obj: Any) -> "HeuristicConfig":
        if isinstance(obj, HeuristicConfig):
            return obj
        if obj is None:
            return cls()
        return cls(bool(obj.get("enabled", True)), int(obj.get("refine_rounds", 3)))


# 多阶段快速优化配置。
#
# 默认关闭，保证原 MILP 接口行为不变；开启后 GridRunner 会在每个格点走
# “固定边界组合优化 + 固定组合 LP + 交替迭代”的启发式路径。
@dataclass(frozen=True)
class MultistageConfig:
    enabled: bool = False
    max_rounds: int = 3
    restarts: int = 1
    # 空值表示沿用 solver.time_limit_s；显式设置可分别控制两个阶段。
    stage1_time_limit_s: float | None = None
    stage2_time_limit_s: float | None = None
    improvement_tol_s: float = 1e-4
    fallback_to_full_milp: bool = True
    # Stage-1 是否按变量-约束耦合图自动拆成独立子问题。
    decompose_stage1: bool = True
    # 至少要有两个规模达到该阈值的独立分量，才真正启用分解。
    decompose_min_component_size: int = 2
    # Stage-1 的"方案枚举 + 链式 DP"（多方案路口）：固定方案后问题可分解，
    # 于是按语义组枚举方案组合、再沿走廊做链式 DP。
    decompose_dp: bool = True
    # 枚举+DP 的子问题求解次数上限（Σ_g Π_{i∈g}|P_i|），超过就放弃 DP 走整模。
    decompose_dp_max_solves: int = 400
    # 预估 DP 耗时超过该秒数时，先用同样的短预算试一次整模，证到最优就用整模。
    decompose_dp_probe_s: float = 1.0
    # Stage-1 是否启用 band slot 对称破缺与正宽度下界。
    stage1_symmetry_breaking: bool = True
    stage1_min_band_width_s: float = 1.0

    @classmethod
    def from_obj(cls, obj: Any) -> "MultistageConfig":
        if isinstance(obj, MultistageConfig):
            return obj
        if obj is None:
            return cls()
        return cls(
            enabled=bool(obj.get("enabled", False)),
            max_rounds=int(obj.get("max_rounds", 3)),
            restarts=int(obj.get("restarts", 1)),
            stage1_time_limit_s=(
                _num(obj["stage1_time_limit_s"], "stage1_time_limit_s")
                if obj.get("stage1_time_limit_s") is not None else None),
            stage2_time_limit_s=(
                _num(obj["stage2_time_limit_s"], "stage2_time_limit_s")
                if obj.get("stage2_time_limit_s") is not None else None),
            improvement_tol_s=_num(obj.get("improvement_tol_s", 1e-4), "improvement_tol_s"),
            fallback_to_full_milp=bool(obj.get("fallback_to_full_milp", True)),
            decompose_stage1=bool(obj.get("decompose_stage1", True)),
            decompose_min_component_size=int(obj.get("decompose_min_component_size", 2)),
            decompose_dp=bool(obj.get("decompose_dp", True)),
            decompose_dp_max_solves=int(obj.get("decompose_dp_max_solves", 400)),
            decompose_dp_probe_s=_num(obj.get("decompose_dp_probe_s", 1.0),
                                      "decompose_dp_probe_s"),
            stage1_symmetry_breaking=bool(obj.get("stage1_symmetry_breaking", True)),
            stage1_min_band_width_s=_num(obj.get("stage1_min_band_width_s", 1.0),
                                         "stage1_min_band_width_s"),
        )


# 求解器配置：后端、时限、gap、并行数、贪心开关、多阶段开关。
@dataclass(frozen=True)
class SolverConfig:
    backend: Literal["scipy", "highspy"] = "scipy"
    time_limit_s: float = 300.0
    mip_rel_gap: float = 0.0
    num_workers: int = 8
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    multistage: MultistageConfig = field(default_factory=MultistageConfig)

    @classmethod
    def from_obj(cls, obj: Any) -> "SolverConfig":
        if isinstance(obj, SolverConfig):
            return obj
        if obj is None:
            return cls()
        return cls(
            backend=obj.get("backend", "scipy"),
            time_limit_s=_num(obj.get("time_limit_s", 300.0), "time_limit_s"),
            mip_rel_gap=_num(obj.get("mip_rel_gap", 0.0), "mip_rel_gap"),
            num_workers=int(obj.get("num_workers", 8)),
            heuristic=HeuristicConfig.from_obj(obj.get("heuristic")),
            multistage=MultistageConfig.from_obj(obj.get("multistage")),
        )


# ---------------------------------------------------------------------------
# Top-level problem
# ---------------------------------------------------------------------------

# 顶层问题对象：把干线、方案、目标、格点、求解器全部装在一起。
@dataclass(frozen=True)
class ProblemInput:
    corridor: CorridorSpec
    plans: tuple[PlanSpec, ...]
    objective: ObjectiveConfig
    grid: GridConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    global_constraints: tuple[ConstraintSpec, ...] = ()
    constraint_templates: tuple[ConstraintTemplate, ...] = ()
    # 具名聚合（boundary 的线性组合取 max/min），只通过 contributions 进目标/损失。
    aggregates: tuple[AggregateSpec, ...] = ()

    # -- convenience lookups -----------------------------------------------------
    def plan(self, plan_id: str) -> PlanSpec:
        for p in self.plans:
            if p.id == plan_id:
                return p
        raise KeyError(plan_id)

    def plans_of(self, intersection: str) -> tuple[PlanSpec, ...]:
        return tuple(p for p in self.plans if p.intersection == intersection)

    def aggregates_affect_composite(self) -> bool:
        """聚合是否影响阶段 1 的 composite（即存在 band_objective 贡献）。"""
        return any(c.target == "band_objective"
                   for a in self.aggregates for c in a.contributions)

    def demand(self, demand_id: str) -> BandDemandSpec:
        for d in self.objective.band_demands:
            if d.id == demand_id:
                return d
        raise KeyError(demand_id)

    def demand_or_none(self, demand_id: str) -> BandDemandSpec | None:
        for d in self.objective.band_demands:
            if d.id == demand_id:
                return d
        return None

    def red_boundary_for(self, demand: BandDemandSpec) -> RedBoundarySpec:
        if demand.red_boundary is not None:
            return demand.red_boundary
        return self.objective.red_boundary_default

    def green_boundary_for(self, demand: BandDemandSpec) -> BoundaryAffinitySpec:
        if demand.green_boundary is not None:
            return demand.green_boundary
        return self.objective.green_boundary_default

    def validate_all(self) -> list[ValidationIssue]:
        from .validation import validate_problem
        return validate_problem(self)

    @classmethod
    def from_obj(cls, obj: Any) -> "ProblemInput":
        if isinstance(obj, ProblemInput):
            return obj
        return cls(
            corridor=CorridorSpec.from_obj(obj["corridor"]),
            plans=tuple(PlanSpec.from_obj(p) for p in obj["plans"]),
            objective=ObjectiveConfig.from_obj(obj["objective"]),
            grid=GridConfig.from_obj(obj["grid"]),
            solver=SolverConfig.from_obj(obj.get("solver")),
            global_constraints=tuple(ConstraintSpec.from_obj(c) for c in obj.get("global_constraints", ())),
            constraint_templates=tuple(ConstraintTemplate.from_obj(c)
                                       for c in obj.get("constraint_templates", ())),
            aggregates=tuple(AggregateSpec.from_obj(a)
                             for a in obj.get("aggregates", ())),
        )

    # pydantic-like alias for readability at call sites
    model_dump = None  # replaced below by method; placeholder for type checkers


def _atom_obj(a: AtomRef) -> dict[str, Any]:
    return {"kind": a.kind, "intersection": a.intersection, "plan": a.plan,
            "direction": a.direction, "index": a.index, "window": a.window,
            "demand": a.demand, "const": a.const, "unit": a.unit}


def _edge_obj(spec: EdgeSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {"free_s": spec.free_s, "hard_max_s": spec.hard_max_s, "coef": spec.coef}


def _boundary_obj(spec: BoundaryAffinitySpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {"start": _edge_obj(spec.start), "end": _edge_obj(spec.end)}


def _problem_model_dump(self: ProblemInput) -> dict[str, Any]:
    # Minimal dump used by multiprocessing payloads.  It deliberately emits
    # the same keys accepted by ``from_obj``.
    def cmap(c: ConstraintSpec | ConstraintTemplate) -> dict[str, Any]:
        out = {
            "id": c.id,
            "terms": [{"atom": _atom_obj(t.atom), "coef": t.coef} for t in c.terms],
            "sense": c.sense,
            "hard": c.hard,
            "rhs_s": c.rhs_s,
            "rhs_ratio": c.rhs_ratio,
            "soft": None if c.soft is None else {"coef": c.soft.coef, "owner": c.soft.owner},
        }
        if isinstance(c, ConstraintTemplate):
            out["scope"] = {
                "kind": c.scope.kind,
                "directions": list(c.scope.directions),
                "plan_ids": None if c.scope.plan_ids is None else list(c.scope.plan_ids),
                "window_indices": None if c.scope.window_indices is None else list(c.scope.window_indices),
                "skip_double_cycle_copies": c.scope.skip_double_cycle_copies,
            }
        return out

    return {
        "corridor": {
            "corridor_id": self.corridor.corridor_id,
            "intersections": list(self.corridor.intersections),
            "segments": [{"from": s.from_, "to": s.to,
                          "distance_m": s.distance_m,
                          "distance_up_m": s.distance_up_m,
                          "distance_down_m": s.distance_down_m,
                          "speed_up_mps": s.speed_up_mps, "speed_down_mps": s.speed_down_mps}
                         for s in self.corridor.segments],
        },
        "plans": [{
            "id": p.id, "intersection": p.intersection, "double_cycle": p.double_cycle,
            "min_phase_s": p.min_phase_s,
            "directions": {
                d: {
                    "boundaries": [
                        {"pos_r": b.pos_r,
                         "adjust": None if b.adjust is None else [b.lo_r, b.hi_r]}
                        for b in p.direction_plan(d).boundaries
                    ],
                    "green_mask": [seg.to_obj() for seg in p.direction_plan(d).green_mask],
                }
                for d in ("up", "down")
            },
            "hard_constraints": [cmap(c) for c in p.hard_constraints],
            "soft_constraints": [cmap(c) for c in p.soft_constraints],
        } for p in self.plans],
        "objective": {
            "band_demands": [{
                "id": b.id, "type": b.type, "direction": b.direction, "nodes": list(b.nodes),
                "max_bands": b.max_bands, "weight": b.weight,
                "red_boundary": _boundary_obj(b.red_boundary),
                "green_boundary": _boundary_obj(b.green_boundary),
            } for b in self.objective.band_demands],
            "red_boundary_default": _boundary_obj(self.objective.red_boundary_default),
            "green_boundary_default": _boundary_obj(self.objective.green_boundary_default),
            "pareto": {"num_points": self.objective.pareto.num_points,
                       "relax_max": self.objective.pareto.relax_max,
                       "topk_grids": self.objective.pareto.topk_grids},
            "allow_window_sharing": self.objective.allow_window_sharing,
            "balanced_groups": [{
                "id": g.id,
                "demands": list(g.demands),
                "weight": g.weight,
                "on_infeasible": g.on_infeasible,
                "targets_s": dict(g.targets_s),
                "min_existing": g.min_existing,
                "min_existing_penalty_s": g.min_existing_penalty_s,
            } for g in self.objective.balanced_groups],
        },
        "grid": {"cycles_s": list(self.grid.cycles_s), "speed_ratios": list(self.grid.speed_ratios)},
        "solver": {
            "backend": self.solver.backend,
            "time_limit_s": self.solver.time_limit_s,
            "mip_rel_gap": self.solver.mip_rel_gap,
            "num_workers": self.solver.num_workers,
            "heuristic": {"enabled": self.solver.heuristic.enabled,
                          "refine_rounds": self.solver.heuristic.refine_rounds},
        },
        "global_constraints": [cmap(c) for c in self.global_constraints],
        "constraint_templates": [cmap(c) for c in self.constraint_templates],
        "aggregates": [a.to_obj() for a in self.aggregates],
    }


ProblemInput.model_dump = _problem_model_dump  # type: ignore[assignment]


__all__ = [
    "Direction", "Side", "Unit", "BoundarySpec", "DirectionPlan", "DerivedInterval",
    "WindowSpec", "AtomRef", "LinearTerm", "SoftPenalty", "ConstraintSpec",
    "ConstraintScope", "ConstraintTemplate", "PlanSpec",
    "SegmentSpec", "CorridorSpec", "MarginSpec", "GreenSegmentSpec",
    "EdgeSpec", "BoundaryAffinitySpec",
    "RedEdgeSpec", "RedBoundarySpec",
    "BandDemandSpec",
    "BalancedGroupSpec", "AggregateContributionSpec", "AggregateSpec",
    "ParetoConfig", "ObjectiveConfig", "GridConfig", "HeuristicConfig",
    "SolverConfig", "MultistageConfig", "ProblemInput",
]
