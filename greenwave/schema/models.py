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
AtomKind = Literal["offset", "endpoint", "window_duration", "red_duration", "bandwidth", "const"]
Unit = Literal["s", "ratio"]
Owner = Literal["composite", "intersection_loss"]
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
# Plan / window schema
# ---------------------------------------------------------------------------

# 一个绿灯窗口：只能用“周期比例”描述，不能用绝对秒。
# 例如 start_r=0.1, end_r=0.3 表示从周期的 10% 到 30% 是绿灯。
@dataclass(frozen=True)
class WindowSpec:
    start_r: float
    end_r: float

    @property
    def width_r(self) -> float:
        return self.end_r - self.start_r

    @classmethod
    def from_obj(cls, obj: Any) -> "WindowSpec":
        if isinstance(obj, WindowSpec):
            return obj
        if isinstance(obj, Mapping):
            return cls(_num(obj["start_r"], "start_r"), _num(obj["end_r"], "end_r"))
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            return cls(_num(obj[0], "start_r"), _num(obj[1], "end_r"))
        raise ValueError(f"cannot parse window spec from {obj!r}")


# 窗口端点允许微调的范围，单位仍是周期比例。
# 必须满足 lo_r <= 0 <= hi_r，也就是名义值必须在允许范围内。
@dataclass(frozen=True)
class EndpointRange:
    lo_r: float
    hi_r: float

    @classmethod
    def from_obj(cls, obj: Any) -> "EndpointRange":
        if isinstance(obj, EndpointRange):
            return obj
        if isinstance(obj, Mapping):
            return cls(_num(obj["lo_r"], "lo_r"), _num(obj["hi_r"], "hi_r"))
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            return cls(_num(obj[0], "lo_r"), _num(obj[1], "hi_r"))
        raise ValueError(f"cannot parse endpoint range from {obj!r}")


# 一个窗口的 start / end 两端分别允许怎么微调。
# 没写的那一端就是固定值，不能动。
@dataclass(frozen=True)
class WindowAdjustSpec:
    start: EndpointRange | None = None
    end: EndpointRange | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> "WindowAdjustSpec":
        if isinstance(obj, WindowAdjustSpec):
            return obj
        if obj is None:
            return cls()
        if not isinstance(obj, Mapping):
            raise ValueError(f"window adjust spec must be an object, got {obj!r}")
        return cls(
            start=EndpointRange.from_obj(obj["start"]) if obj.get("start") is not None else None,
            end=EndpointRange.from_obj(obj["end"]) if obj.get("end") is not None else None,
        )


# 约束表达式里的一个“原子”引用。
# 例如：某个路口的偏置 φ、某个窗口端点的秒值、某条带的总带宽、一个常数。
@dataclass(frozen=True)
class AtomRef:
    kind: AtomKind
    intersection: str | None = None
    plan: str | None = None
    direction: Direction | None = None
    window: int | None = None
    side: Side | None = None
    demand: str | None = None
    const: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "offset":
            if self.intersection is None:
                raise ValueError("offset atom requires intersection")
        elif self.kind == "endpoint":
            missing = [n for n, v in (
                ("intersection", self.intersection), ("plan", self.plan),
                ("direction", self.direction), ("window", self.window),
                ("side", self.side)) if v is None]
            if missing:
                raise ValueError(f"endpoint atom missing fields: {missing}")
            if self.direction not in ("up", "down"):
                raise ValueError("endpoint atom direction must be up/down")
            if self.side not in ("start", "end"):
                raise ValueError("endpoint atom side must be start/end")
        elif self.kind in ("window_duration", "red_duration"):
            # 可以写成具体 plan/direction/window，也可以留空由 ConstraintTemplate
            # 的 scope 在建模时填充；side 对时长原子没有意义。
            # window_duration 的 window 是绿灯窗口有效索引；
            # red_duration 的 window 是红窗补集区间索引。
            if self.side is not None:
                raise ValueError(f"{self.kind} atom must not specify side")
        elif self.kind == "bandwidth":
            if self.demand is None:
                raise ValueError("bandwidth atom requires demand")
        elif self.kind == "const":
            if self.const is None:
                raise ValueError("const atom requires const")
        else:  # pragma: no cover - Literal is developer-facing
            raise ValueError(f"unknown atom kind {self.kind!r}")

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
            window=int(obj["window"]) if obj.get("window") is not None else None,
            side=_check_side(obj["side"]) if obj.get("side") is not None else None,
            demand=obj.get("demand"),
            const=_num(obj["const"], "const") if obj.get("const") is not None else None,
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


# 软约束的惩罚配置：违反 1 秒要罚多少钱（这里单位仍是目标函数里的“秒”）。
# owner 决定这笔账记到 composite 还是 intersection_loss。
@dataclass(frozen=True)
class SoftPenalty:
    coef: float
    owner: Owner

    @classmethod
    def from_obj(cls, obj: Any) -> "SoftPenalty":
        if isinstance(obj, SoftPenalty):
            return obj
        return cls(_num(obj["coef"], "coef"), obj["owner"])


def _parse_rhs_fields(obj: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """把统一写法 ``rhs + rhs_unit`` 与旧写法 ``rhs_s/rhs_ratio`` 解析成内部字段。

    - rhs_unit == "s"      -> rhs_s
    - rhs_unit == "ratio"  -> rhs_ratio
    - 未写 rhs_unit 时，``rhs`` 默认按秒处理，保持旧 JSON 兼容。
    """
    rhs = obj.get("rhs")
    unit = obj.get("rhs_unit")
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
        if (self.rhs_s is None) == (self.rhs_ratio is None):
            raise ValueError("exactly one of rhs_s / rhs_ratio must be provided")
        if self.hard:
            if self.soft is not None:
                raise ValueError("hard constraint must not carry soft penalty")
        else:
            if self.soft is None:
                raise ValueError("soft constraint requires soft penalty")

    def rhs_for_cycle(self, cycle_s: float) -> float:
        if self.rhs_s is not None:
            return self.rhs_s
        assert self.rhs_ratio is not None
        return self.rhs_ratio * cycle_s

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
        if (self.rhs_s is None) == (self.rhs_ratio is None):
            raise ValueError("exactly one of rhs_s / rhs_ratio must be provided")
        if self.hard:
            if self.soft is not None:
                raise ValueError("hard constraint template must not carry soft penalty")
        else:
            if self.soft is None:
                raise ValueError("soft constraint template requires soft penalty")

    def rhs_for_cycle(self, cycle_s: float) -> float:
        if self.rhs_s is not None:
            return self.rhs_s
        assert self.rhs_ratio is not None
        return self.rhs_ratio * cycle_s

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

# 一个信号控制候选方案。
# 干线优化只关心“每个方向有哪些绿灯窗口”，不关心内部相位怎么排。
@dataclass(frozen=True)
class PlanSpec:
    id: str
    intersection: str
    windows: Mapping[Direction, tuple[WindowSpec, ...]]
    double_cycle: bool = False
    adjustable: Mapping[Direction, Mapping[int, WindowAdjustSpec]] = field(default_factory=dict)
    hard_constraints: tuple[ConstraintSpec, ...] = ()
    soft_constraints: tuple[ConstraintSpec, ...] = ()

    # -- effective window access -------------------------------------------------
    def _original_windows(self, d: Direction) -> tuple[WindowSpec, ...]:
        return tuple(self.windows.get(d, ()))

    def _effective_entries(self, d: Direction) -> tuple[tuple[WindowSpec, int, bool], ...]:
        """Return ``(window, original_index, is_copy)`` sorted by start ratio."""
        originals = self._original_windows(d)
        entries: list[tuple[WindowSpec, int, bool]] = [(w, i, False) for i, w in enumerate(originals)]
        if self.double_cycle:
            for i, w in enumerate(originals):
                entries.append((WindowSpec(w.start_r + 0.5, w.end_r + 0.5), i, True))
        entries.sort(key=lambda e: (e[0].start_r, e[0].end_r, e[1], e[2]))
        return tuple(entries)

    def effective_windows(self, d: Direction) -> tuple[WindowSpec, ...]:
        return tuple(e[0] for e in self._effective_entries(d))

    def effective_original_index(self, d: Direction, k: int) -> int:
        return self._effective_entries(d)[k][1]

    def effective_is_copy(self, d: Direction, k: int) -> bool:
        return self._effective_entries(d)[k][2]

    def window_count(self, d: Direction) -> int:
        return len(self._effective_entries(d))

    def nominal_endpoint_r(self, d: Direction, k: int, side: Side) -> float:
        w = self._effective_entries(d)[k][0]
        return w.start_r if side == "start" else w.end_r

    def adjust_range(self, d: Direction, k: int, side: Side) -> EndpointRange | None:
        orig = self.effective_original_index(d, k)
        spec = self.adjustable.get(d, {}).get(orig)
        if spec is None:
            return None
        return spec.start if side == "start" else spec.end

    def endpoint_is_adjustable(self, d: Direction, k: int, side: Side) -> bool:
        return self.adjust_range(d, k, side) is not None

    def endpoint_original_key(self, d: Direction, k: int, side: Side) -> tuple[int, Side]:
        """Key identifying the *declared* adjustable endpoint for double-cycle copies."""
        return (self.effective_original_index(d, k), side)

    @classmethod
    def from_obj(cls, obj: Any) -> "PlanSpec":
        if isinstance(obj, PlanSpec):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"plan must be an object, got {obj!r}")
        windows_in = obj.get("windows", {})
        windows: dict[Direction, tuple[WindowSpec, ...]] = {}
        for d in ("up", "down"):
            windows[d] = tuple(WindowSpec.from_obj(w) for w in windows_in.get(d, ()))
        adjustable_in = obj.get("adjustable", {}) or {}
        adjustable: dict[Direction, dict[int, WindowAdjustSpec]] = {}
        for d in ("up", "down"):
            per = adjustable_in.get(d, {}) or {}
            adjustable[d] = {int(k): WindowAdjustSpec.from_obj(v) for k, v in per.items()}
        return cls(
            id=str(obj["id"]),
            intersection=str(obj["intersection"]),
            windows=windows,
            double_cycle=bool(obj.get("double_cycle", False)),
            adjustable=adjustable,
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

# 绿波带的 margin 配置：希望带宽离绿灯窗口边界至少留多少秒。
@dataclass(frozen=True)
class MarginSpec:
    # 软 margin 目标：低于该值产生罚项。
    delta_min_s: float = 2.0
    coef: float = 1.0
    # 硬 margin 下限：低于该值直接不可行；0 表示不启用硬约束。
    hard_min_s: float = 0.0

    @classmethod
    def from_obj(cls, obj: Any) -> "MarginSpec":
        if isinstance(obj, MarginSpec):
            return obj
        if obj is None:
            return cls()
        return cls(_num(obj.get("delta_min_s", 2.0), "delta_min_s"),
                   _num(obj.get("coef", 1.0), "coef"),
                   _num(obj.get("hard_min_s", 0.0), "hard_min_s"))


# 红波带边界亲和性配置：红波带边界离上游绿灯区间边界多远开始罚。
@dataclass(frozen=True)
class RedEdgeSpec:
    free_s: float = 0.0
    hard_max_s: float | None = None
    coef: float = 1.0

    def __post_init__(self) -> None:
        if self.free_s < 0:
            raise ValueError("red edge free_s must be >= 0")
        if self.coef < 0:
            raise ValueError("red edge coef must be >= 0")
        if self.hard_max_s is not None and self.hard_max_s < self.free_s:
            raise ValueError("red edge hard_max_s must be >= free_s")

    @classmethod
    def from_obj(cls, obj: Any) -> "RedEdgeSpec | None":
        if isinstance(obj, RedEdgeSpec):
            return obj
        if obj is None:
            return None
        if not isinstance(obj, Mapping):
            raise ValueError(f"red edge spec must be an object, got {obj!r}")
        hard = obj.get("hard_max_s")
        return cls(
            free_s=_num(obj.get("free_s", 0.0), "free_s"),
            hard_max_s=(_num(hard, "hard_max_s") if hard is not None else None),
            coef=_num(obj.get("coef", 1.0), "coef"),
        )


@dataclass(frozen=True)
class RedBoundarySpec:
    start: RedEdgeSpec | None = None
    end: RedEdgeSpec | None = None

    def __post_init__(self) -> None:
        # 两端都可以为空；全空表示不启用。
        pass

    @property
    def enabled(self) -> bool:
        return self.start is not None or self.end is not None

    @classmethod
    def from_obj(cls, obj: Any) -> "RedBoundarySpec":
        if isinstance(obj, RedBoundarySpec):
            return obj
        if obj is None:
            return cls()
        if not isinstance(obj, Mapping):
            raise ValueError(f"red boundary spec must be an object, got {obj!r}")
        return cls(
            start=RedEdgeSpec.from_obj(obj.get("start")),
            end=RedEdgeSpec.from_obj(obj.get("end")),
        )


# 一条优化需求：绿波带或红波带。
@dataclass(frozen=True)
class BandDemandSpec:
    id: str
    type: Literal["green", "red"]
    direction: Direction
    nodes: tuple[str, ...]
    max_bands: int | None = None
    weight: float = 1.0
    margin: MarginSpec | None = None
    red_boundary: RedBoundarySpec | None = None

    def __post_init__(self) -> None:
        if self.type not in ("green", "red"):
            raise ValueError("demand type must be green/red")
        _check_direction(self.direction)
        if self.red_boundary is not None and self.type != "red":
            raise ValueError(f"red_boundary is only valid for red demand {self.id!r}")

    @classmethod
    def from_obj(cls, obj: Any) -> "BandDemandSpec":
        if isinstance(obj, BandDemandSpec):
            return obj
        return cls(
            id=str(obj["id"]),
            type=obj.get("type", "green"),
            direction=_check_direction(obj["direction"]),
            nodes=tuple(str(x) for x in obj["nodes"]),
            max_bands=int(obj["max_bands"]) if obj.get("max_bands") is not None else None,
            weight=_num(obj.get("weight", 1.0), "weight"),
            margin=MarginSpec.from_obj(obj.get("margin")) if obj.get("margin") is not None else None,
            red_boundary=(RedBoundarySpec.from_obj(obj.get("red_boundary"))
                          if obj.get("red_boundary") is not None else None),
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


# 目标配置：所有带需求、默认 margin、Pareto 配置、窗口共享开关。
@dataclass(frozen=True)
class ObjectiveConfig:
    band_demands: tuple[BandDemandSpec, ...]
    margin_default: MarginSpec = field(default_factory=MarginSpec)
    red_boundary_default: RedBoundarySpec = field(default_factory=RedBoundarySpec)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    allow_window_sharing: bool = False
    balanced_groups: tuple[BalancedGroupSpec, ...] = ()

    @classmethod
    def from_obj(cls, obj: Any) -> "ObjectiveConfig":
        if isinstance(obj, ObjectiveConfig):
            return obj
        return cls(
            band_demands=tuple(BandDemandSpec.from_obj(d) for d in obj.get("band_demands", ())),
            margin_default=MarginSpec.from_obj(obj.get("margin_default")),
            red_boundary_default=RedBoundarySpec.from_obj(obj.get("red_boundary_default")),
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


# 求解器配置：后端、时限、gap、并行数、贪心开关。
@dataclass(frozen=True)
class SolverConfig:
    backend: Literal["scipy", "highspy"] = "scipy"
    time_limit_s: float = 300.0
    mip_rel_gap: float = 0.0
    num_workers: int = 8
    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)

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

    # -- convenience lookups -----------------------------------------------------
    def plan(self, plan_id: str) -> PlanSpec:
        for p in self.plans:
            if p.id == plan_id:
                return p
        raise KeyError(plan_id)

    def plans_of(self, intersection: str) -> tuple[PlanSpec, ...]:
        return tuple(p for p in self.plans if p.intersection == intersection)

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

    def margin_for(self, demand: BandDemandSpec) -> MarginSpec:
        return demand.margin if demand.margin is not None else self.objective.margin_default

    def red_boundary_for(self, demand: BandDemandSpec) -> RedBoundarySpec:
        if demand.red_boundary is not None:
            return demand.red_boundary
        return self.objective.red_boundary_default

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
        )

    # pydantic-like alias for readability at call sites
    model_dump = None  # replaced below by method; placeholder for type checkers


def _problem_model_dump(self: ProblemInput) -> dict[str, Any]:
    # Minimal dump used by multiprocessing payloads.  It deliberately emits
    # the same keys accepted by ``from_obj``.
    def win(w: WindowSpec) -> list[float]:
        return [w.start_r, w.end_r]

    def cmap(c: ConstraintSpec | ConstraintTemplate) -> dict[str, Any]:
        out = {
            "id": c.id,
            "terms": [{"atom": a, "coef": t.coef} for t in c.terms for a in [{
                "kind": t.atom.kind,
                "intersection": t.atom.intersection,
                "plan": t.atom.plan,
                "direction": t.atom.direction,
                "window": t.atom.window,
                "side": t.atom.side,
                "demand": t.atom.demand,
                "const": t.atom.const,
            }]],
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
            "windows": {d: [win(w) for w in p.windows.get(d, ())] for d in ("up", "down")},
            "adjustable": {d: {str(k): {"start": None if v.start is None else [v.start.lo_r, v.start.hi_r],
                                         "end": None if v.end is None else [v.end.lo_r, v.end.hi_r]}
                                for k, v in (p.adjustable.get(d) or {}).items()}
                           for d in ("up", "down")},
            "hard_constraints": [cmap(c) for c in p.hard_constraints],
            "soft_constraints": [cmap(c) for c in p.soft_constraints],
        } for p in self.plans],
        "objective": {
            "band_demands": [{
                "id": b.id, "type": b.type, "direction": b.direction, "nodes": list(b.nodes),
                "max_bands": b.max_bands, "weight": b.weight,
                "margin": None if b.margin is None else {
                    "delta_min_s": b.margin.delta_min_s,
                    "coef": b.margin.coef,
                    "hard_min_s": b.margin.hard_min_s,
                },
                "red_boundary": None if b.red_boundary is None else {
                    "start": None if b.red_boundary.start is None else {
                        "free_s": b.red_boundary.start.free_s,
                        "hard_max_s": b.red_boundary.start.hard_max_s,
                        "coef": b.red_boundary.start.coef,
                    },
                    "end": None if b.red_boundary.end is None else {
                        "free_s": b.red_boundary.end.free_s,
                        "hard_max_s": b.red_boundary.end.hard_max_s,
                        "coef": b.red_boundary.end.coef,
                    },
                },
            } for b in self.objective.band_demands],
            "margin_default": {
                "delta_min_s": self.objective.margin_default.delta_min_s,
                "coef": self.objective.margin_default.coef,
                "hard_min_s": self.objective.margin_default.hard_min_s,
            },
            "red_boundary_default": {
                "start": None if self.objective.red_boundary_default.start is None else {
                    "free_s": self.objective.red_boundary_default.start.free_s,
                    "hard_max_s": self.objective.red_boundary_default.start.hard_max_s,
                    "coef": self.objective.red_boundary_default.start.coef,
                },
                "end": None if self.objective.red_boundary_default.end is None else {
                    "free_s": self.objective.red_boundary_default.end.free_s,
                    "hard_max_s": self.objective.red_boundary_default.end.hard_max_s,
                    "coef": self.objective.red_boundary_default.end.coef,
                },
            },
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
    }


ProblemInput.model_dump = _problem_model_dump  # type: ignore[assignment]


__all__ = [
    "Direction", "Side", "Unit", "WindowSpec", "EndpointRange", "WindowAdjustSpec",
    "AtomRef", "LinearTerm", "SoftPenalty", "ConstraintSpec",
    "ConstraintScope", "ConstraintTemplate", "PlanSpec",
    "SegmentSpec", "CorridorSpec", "MarginSpec", "RedEdgeSpec", "RedBoundarySpec",
    "BandDemandSpec",
    "BalancedGroupSpec", "ParetoConfig", "ObjectiveConfig", "GridConfig", "HeuristicConfig",
    "SolverConfig", "ProblemInput",
]
