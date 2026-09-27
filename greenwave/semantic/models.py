"""高层语义规则的数据模型。

这一层只负责把用户 JSON 解析成不可变对象，不做寻址、不做编译。
真正的红窗台账、端点解析、约束生成在 resolver/compiler 中完成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..schema.models import Direction, SoftPenalty

RuleKind = Literal["green_at_cycle_start", "green_at_cycle_end", "sync", "duration_limit"]
Enforcement = Literal["hard", "soft"]
SyncRelation = Literal["start_start", "end_end", "start_end", "end_start"]


def _num(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


def _direction(value: Any) -> Direction:
    if value not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {value!r}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class WindowRef:
    kind: Literal["green", "red"]
    index: int | None = None          # green: 0-based declaration index; red: 1-based rid
    position: str | None = None       # green only: "first" | "last"
    after_green: int | None = None    # red only: 1-based green slot

    @classmethod
    def from_obj(cls, obj: Any) -> "WindowRef":
        if isinstance(obj, WindowRef):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"window reference must be an object, got {obj!r}")
        kind = obj.get("kind")
        if kind not in ("green", "red"):
            raise ValueError("window.kind must be 'green' or 'red'")
        if kind == "green":
            has_index = obj.get("index") is not None
            has_pos = obj.get("position") is not None
            if has_index == has_pos:
                raise ValueError("green window requires exactly one of index/position")
            if has_index:
                idx = int(obj["index"])
                if idx < 0:
                    raise ValueError("green window index must be >= 0")
                return cls("green", index=idx)
            pos = str(obj["position"])
            if pos not in ("first", "last"):
                raise ValueError("green window position must be 'first' or 'last'")
            return cls("green", position=pos)
        # red
        has_index = obj.get("index") is not None
        has_after = obj.get("after_green") is not None
        if has_index == has_after:
            raise ValueError("red window requires exactly one of index/after_green")
        if has_index:
            idx = int(obj["index"])
            if idx < 1:
                raise ValueError("red window index must be >= 1")
            return cls("red", index=idx)
        after = int(obj["after_green"])
        if after < 1:
            raise ValueError("red window after_green must be >= 1")
        return cls("red", after_green=after)


@dataclass(frozen=True)
class Target:
    intersection: str
    plan: str
    direction: Direction
    window: WindowRef

    @classmethod
    def from_obj(cls, obj: Any) -> "Target":
        if isinstance(obj, Target):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"target must be an object, got {obj!r}")
        return cls(
            intersection=str(obj["intersection"]),
            plan=str(obj["plan"]),
            direction=_direction(obj["direction"]),
            window=WindowRef.from_obj(obj["window"]),
        )


@dataclass(frozen=True)
class SemanticRule:
    id: str
    kind: RuleKind
    target: Target | None = None
    targets: tuple[Target, ...] = ()
    relation: SyncRelation | None = None
    sense: str | None = None
    value: float | None = None
    unit: str | None = None
    enforcement: Enforcement = "hard"
    penalty: SoftPenalty | None = None

    @classmethod
    def from_obj(cls, obj: Any) -> "SemanticRule":
        if isinstance(obj, SemanticRule):
            return obj
        if not isinstance(obj, Mapping):
            raise ValueError(f"semantic rule must be an object, got {obj!r}")
        raw_targets = tuple(Target.from_obj(t) for t in obj.get("targets", ()) or ())
        target = Target.from_obj(obj["target"]) if obj.get("target") is not None else None
        penalty_obj = obj.get("penalty")
        return cls(
            id=str(obj["id"]),
            kind=obj["kind"],
            target=target,
            targets=raw_targets,
            relation=obj.get("relation"),
            sense=obj.get("sense"),
            value=(_num(obj["value"], "value") if obj.get("value") is not None else None),
            unit=obj.get("unit"),
            enforcement=obj.get("enforcement", "hard"),
            penalty=SoftPenalty.from_obj(penalty_obj) if penalty_obj is not None else None,
        )


@dataclass(frozen=True)
class SemanticInput:
    rules: tuple[SemanticRule, ...] = field(default_factory=tuple)

    @classmethod
    def from_obj(cls, obj: Any) -> "SemanticInput":
        if isinstance(obj, SemanticInput):
            return obj
        if isinstance(obj, Mapping):
            raw = obj.get("semantic_rules", obj.get("rules", ()))
        elif isinstance(obj, (list, tuple)):
            raw = obj
        else:
            raise ValueError("semantic input must be a list or an object with semantic_rules")
        return cls(tuple(SemanticRule.from_obj(r) for r in raw))


__all__ = ["WindowRef", "Target", "SemanticRule", "SemanticInput",
           "RuleKind", "Enforcement", "SyncRelation"]
