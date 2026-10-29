"""变量命名的**唯一事实源**。

为什么要这个模块
----------------
以前变量名是散在各处用 f-string 拼的，于是出现了"同一个字母在不同族里表示不同东西"：
`e` 既是绿带存在性又是窗口终点、`n` 既是回绕周期又是节点记号、`k` 在绿带里是窗口槽
而在红带里是周期偏移、`m` 既是边界个数又是红带上游回绕…… 读日志和读代码都要猜。

命名约定
--------
* **前缀 = 变量族**，能自解释：`gw_` 绿波 / `rw_` 红波 / `bnd_` 相位边界 /
  `win_` 路口窗口 / `bal_` 均衡组 / `soft_` 软约束；
* 方括号里是**键**，顺序固定，含义见各构造函数；
* **不用单字母键**：绿窗槽写 `slot`、上游绿窗槽 `upslot`、红窗槽 `rid`、
  下游周期偏移 `cshift`、链上路口写路口 id（`node`）；
* 解析一律走本模块的 `family_of` / `keys_of` / `plan_of`，不要在别处手写
  `startswith("xx[")` 或切片——改名时才不会漏。

与文档的对应关系见 `docs/MILP建模教学-绿波篇.md` §0.3、§3.1。
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 族名（前缀）
# ---------------------------------------------------------------------------

PLAN_ON: Final = "plan_on"            # 方案选择 0/1
OFFSET: Final = "offset"              # 周期偏置 φ
BND_DELTA: Final = "bnd_delta"        # 相位边界微调 δ
BND_XDELTA: Final = "bnd_xdelta"      # x·δ 的 McCormick 线性化
WIN_AVAIL: Final = "win_avail"        # 绿窗槽是否被选中方案提供（h）

GW_START: Final = "gw_start"          # 绿带起点 u（链首到达时刻）
GW_WIDTH: Final = "gw_width"          # 绿带宽度 β
GW_ON: Final = "gw_on"                # 绿带存在性 e
GW_WIN: Final = "gw_win"              # 绿带在链上某节点用哪个绿窗槽 a
GW_CYCLE: Final = "gw_cycle"          # 绿带回绕周期下标 n
GW_MARGIN_LO: Final = "gw_margin_lo"  # 绿带左侧 margin 短缺
GW_MARGIN_HI: Final = "gw_margin_hi"  # 绿带右侧 margin 短缺
GW_EDGE_LO: Final = "gw_edge_lo"      # 绿波边界亲和（起点侧 slack）
GW_EDGE_HI: Final = "gw_edge_hi"      # 绿波边界亲和（终点侧 slack）

RW_START: Final = "rw_start"          # 红带起点 u（**下游**路口到达时刻）
RW_WIDTH: Final = "rw_width"          # 红带宽度 β
RW_ON: Final = "rw_on"                # 红带存在性 e
RW_UP_WIN: Final = "rw_up_win"        # 红带选上游第几个绿窗槽 ag
RW_DOWN_WIN: Final = "rw_down_win"    # 红带选下游第几个红窗 + 周期偏移 ar
RW_UP_CYCLE: Final = "rw_up_cycle"    # 红带上游回绕周期下标 m
RW_EDGE_START: Final = "rw_edge_start"  # 红波边界亲和（起点侧 slack）
RW_EDGE_END: Final = "rw_edge_end"      # 红波边界亲和（终点侧 slack）

AGG: Final = "agg"                    # 具名聚合（max/min 一组线性表达式）
BAL_MIN: Final = "bal_min"            # 均衡组：组内归一化最小带宽 z
BAL_PRESENT: Final = "bal_present"    # 均衡组：成员需求是否至少有一条带 y
BAL_DEFICIT: Final = "bal_deficit"    # 均衡组：min_existing 短缺

SOFT_SLACK: Final = "soft_slack"      # 软约束违反量 σ（含方案软约束与用户软约束）
SOFT_REWARD: Final = "soft_reward"    # 软约束的超额满足量（路口回报）

#: 全部族名（供校验/遍历用）
ALL_FAMILIES: Final[tuple[str, ...]] = (
    PLAN_ON, OFFSET, BND_DELTA, BND_XDELTA, WIN_AVAIL,
    GW_START, GW_WIDTH, GW_ON, GW_WIN, GW_CYCLE, GW_MARGIN_LO, GW_MARGIN_HI,
    GW_EDGE_LO, GW_EDGE_HI,
    RW_START, RW_WIDTH, RW_ON, RW_UP_WIN, RW_DOWN_WIN, RW_UP_CYCLE,
    RW_EDGE_START, RW_EDGE_END,
    AGG, BAL_MIN, BAL_PRESENT, BAL_DEFICIT, SOFT_SLACK, SOFT_REWARD,
)

#: margin 短缺的两侧
MARGIN_SIDES: Final[tuple[str, str]] = ("lo", "hi")


def _fmt(family: str, *keys: object) -> str:
    return f"{family}[{','.join(str(k) for k in keys)}]"


# ---------------------------------------------------------------------------
# 构造：每个变量族的唯一生成函数
# ---------------------------------------------------------------------------

def plan_on(intersection: str, plan_id: str) -> str:
    """路口选了哪个方案（0/1）。键：路口 id、方案 id。"""
    return _fmt(PLAN_ON, intersection, plan_id)


def offset(intersection: str) -> str:
    """路口周期偏置 φ（秒）。"""
    return _fmt(OFFSET, intersection)


def bnd_delta(plan_id: str, direction: str, boundary: int) -> str:
    """相位边界微调量 δ（比例域）。键：方案 id、方向、**边界下标**（不是链位置）。"""
    return _fmt(BND_DELTA, plan_id, direction, boundary)


def bnd_xdelta(plan_id: str, direction: str, boundary: int) -> str:
    """`x·δ` 的线性化辅助变量。键同 `bnd_delta`。"""
    return _fmt(BND_XDELTA, plan_id, direction, boundary)


def win_avail(intersection: str, direction: str, slot: int) -> str:
    """绿窗槽是否被选中方案提供（`h`，天然 0/1）。键：路口、方向、绿窗槽。"""
    return _fmt(WIN_AVAIL, intersection, direction, slot)


def gw_start(demand: str, slot: int) -> str:
    """绿带起点 `u`（**链首**到达时刻，链首 = 该需求 nodes[0]）。"""
    return _fmt(GW_START, demand, slot)


def gw_width(demand: str, slot: int) -> str:
    """绿带宽度 β（秒）。"""
    return _fmt(GW_WIDTH, demand, slot)


def gw_on(demand: str, slot: int) -> str:
    """绿带存在性 e（0/1）。"""
    return _fmt(GW_ON, demand, slot)


def gw_win(demand: str, slot: int, node: str, win_slot: int) -> str:
    """绿带在某节点使用哪个绿窗槽 `a`。键：需求、带槽、**链上路口 id**、绿窗槽。"""
    return _fmt(GW_WIN, demand, slot, node, win_slot)


def gw_cycle(demand: str, slot: int, node: str) -> str:
    """绿带在该节点的回绕周期下标 `n`（整数）。"""
    return _fmt(GW_CYCLE, demand, slot, node)


def gw_margin_lo(demand: str, slot: int, node: str, win_slot: int,
                 plan: str | None = None) -> str:
    """绿带左侧 margin 短缺（带宽起点离绿窗起点太近的部分）。"""
    if plan is None:
        return _fmt(GW_MARGIN_LO, demand, slot, node, win_slot)
    return _fmt(GW_MARGIN_LO, demand, slot, node, win_slot, plan)


def gw_margin_hi(demand: str, slot: int, node: str, win_slot: int,
                 plan: str | None = None) -> str:
    """绿带右侧 margin 短缺（带宽终点离绿窗终点太近的部分）。"""
    if plan is None:
        return _fmt(GW_MARGIN_HI, demand, slot, node, win_slot)
    return _fmt(GW_MARGIN_HI, demand, slot, node, win_slot, plan)


def gw_edge_lo(demand: str, slot: int, node: str, win_slot: int) -> str:
    """绿波边界亲和（起点侧）：带起点比绿窗起点晚多少，超出 free_s 的部分。"""
    return _fmt(GW_EDGE_LO, demand, slot, node, win_slot)


def gw_edge_hi(demand: str, slot: int, node: str, win_slot: int) -> str:
    """绿波边界亲和（终点侧）：带终点比绿窗终点早多少，超出 free_s 的部分。"""
    return _fmt(GW_EDGE_HI, demand, slot, node, win_slot)


def rw_start(demand: str, slot: int) -> str:
    """红带起点 `u`（**下游**路口到达时刻；上游位置 = u − τ）。"""
    return _fmt(RW_START, demand, slot)


def rw_width(demand: str, slot: int) -> str:
    """红带宽度 β（秒）。"""
    return _fmt(RW_WIDTH, demand, slot)


def rw_on(demand: str, slot: int) -> str:
    """红带存在性 e（0/1）。"""
    return _fmt(RW_ON, demand, slot)


def rw_up_win(demand: str, slot: int, up_slot: int) -> str:
    """红带选上游第几个绿窗槽 `ag`。"""
    return _fmt(RW_UP_WIN, demand, slot, up_slot)


def rw_down_win(demand: str, slot: int, rid: int, cycle_shift: int) -> str:
    """红带选下游第几个红窗 + 周期偏移 `ar`。键：需求、带槽、红窗槽 `rid`、周期偏移。"""
    return _fmt(RW_DOWN_WIN, demand, slot, rid, cycle_shift)


def rw_up_cycle(demand: str, slot: int) -> str:
    """红带上游回绕周期下标 `m`（整数，与绿带的 `gw_cycle` 对称）。"""
    return _fmt(RW_UP_CYCLE, demand, slot)


def rw_edge_start(demand: str, slot: int, up_slot: int) -> str:
    """红波边界亲和：红带起点比上游绿窗起点晚多少（超出 free_s 的部分）。"""
    return _fmt(RW_EDGE_START, demand, slot, up_slot)


def rw_edge_end(demand: str, slot: int, up_slot: int) -> str:
    """红波边界亲和：红带终点比上游绿窗终点早多少（超出 free_s 的部分）。"""
    return _fmt(RW_EDGE_END, demand, slot, up_slot)


def agg(aggregate_id: str) -> str:
    """具名聚合的标量变量名。"""
    return _fmt(AGG, aggregate_id)


def bal_min(group: str) -> str:
    """均衡组：组内归一化带宽的最小值 `z`。"""
    return _fmt(BAL_MIN, group)


def bal_present(group: str, demand: str) -> str:
    """均衡组：成员需求是否至少存在一条带 `y`（0/1）。"""
    return _fmt(BAL_PRESENT, group, demand)


def bal_deficit(group: str) -> str:
    """均衡组：`min_existing` 的成员数短缺。"""
    return _fmt(BAL_DEFICIT, group)


def soft_slack(name_base: str, tag: str) -> str:
    """软约束违反量 σ。`name_base` 形如 `I2_P1.I2_up_end_soft_055` 或约束 id。"""
    return _fmt(SOFT_SLACK, f"{name_base}.{tag}")


def soft_reward(name_base: str) -> str:
    """软约束的超额满足量 ρ（owner=intersection_reward）。"""
    return _fmt(SOFT_REWARD, f"{name_base}.reward")


def edge_name(demand: str, slot: int, node: str, win_slot: int, side: str) -> str:
    """按 `side ∈ {"lo","hi"}` 取绿波边界亲和变量名（兼容 "s"/"e"）。"""
    if side in ("lo", "s"):
        return gw_edge_lo(demand, slot, node, win_slot)
    if side in ("hi", "e"):
        return gw_edge_hi(demand, slot, node, win_slot)
    raise ValueError(f"unknown edge side {side!r}")


def margin_name(demand: str, slot: int, node: str, win_slot: int, side: str,
                plan: str | None = None) -> str:
    """按 `side ∈ {"lo","hi"}` 取 margin 短缺变量名（兼容旧代码里的 "s"/"e"）。

    margin 逐方案配置后，同一 (需求, 带槽, 节点, 窗口, 侧) 可能对应多个方案的
    slack；`plan` 非空时把它放进变量名以免重名。
    """
    if side in ("lo", "s"):
        return gw_margin_lo(demand, slot, node, win_slot, plan)
    if side in ("hi", "e"):
        return gw_margin_hi(demand, slot, node, win_slot, plan)
    raise ValueError(f"unknown margin side {side!r}")


# ---------------------------------------------------------------------------
# 解析：不要在别处手写 startswith / 字符串切片
# ---------------------------------------------------------------------------

def family_of(name: str) -> str:
    """取变量名的族名；不含方括号时返回整串。"""
    idx = name.find("[")
    return name if idx < 0 else name[:idx]


def keys_of(name: str) -> tuple[str, ...]:
    """取变量名的键（方括号内按逗号切分）；没有键时返回空元组。"""
    idx = name.find("[")
    if idx < 0 or not name.endswith("]"):
        return ()
    body = name[idx + 1:-1]
    return tuple(part for part in body.split(",")) if body else ()


def is_family(name: str, family: str) -> bool:
    return family_of(name) == family


def plan_of(name: str) -> str | None:
    """取"这条变量属于哪个方案"，用于多方案时给未选中方案的变量置零。

    * `bnd_delta` / `bnd_xdelta`：键的第一项就是方案 id；
    * `soft_slack`：`name_base` 形如 `<plan_id>.<constraint_id>.<pos|neg>`，
      取第一个点之前的部分；没有点（用户全局软约束）则返回 None。
    """
    family = family_of(name)
    if family in (BND_DELTA, BND_XDELTA):
        idx = name.find("[")
        body = name[idx + 1:-1]
        return body.split(",", 1)[0] if "," in body else body
    if family in (SOFT_SLACK, SOFT_REWARD):
        body = name[len(family) + 1:-1]
        return body.split(".", 1)[0] if "." in body else None
    return None


def offset_intersection(name: str) -> str | None:
    """`offset[I2]` → `I2`；不是 offset 族则 None。"""
    if not is_family(name, OFFSET):
        return None
    keys = keys_of(name)
    return keys[0] if keys else None


__all__ = [
    "PLAN_ON", "OFFSET", "BND_DELTA", "BND_XDELTA", "WIN_AVAIL",
    "GW_START", "GW_WIDTH", "GW_ON", "GW_WIN", "GW_CYCLE",
    "GW_MARGIN_LO", "GW_MARGIN_HI", "GW_EDGE_LO", "GW_EDGE_HI",
    "RW_START", "RW_WIDTH", "RW_ON", "RW_UP_WIN", "RW_DOWN_WIN",
    "RW_UP_CYCLE", "RW_EDGE_START", "RW_EDGE_END",
    "AGG", "BAL_MIN", "BAL_PRESENT", "BAL_DEFICIT", "SOFT_SLACK", "SOFT_REWARD",
    "ALL_FAMILIES", "MARGIN_SIDES",
    "plan_on", "offset", "bnd_delta", "bnd_xdelta", "win_avail",
    "gw_start", "gw_width", "gw_on", "gw_win", "gw_cycle",
    "gw_margin_lo", "gw_margin_hi", "margin_name",
    "gw_edge_lo", "gw_edge_hi", "edge_name",
    "rw_start", "rw_width", "rw_on", "rw_up_win", "rw_down_win",
    "rw_up_cycle", "rw_edge_start", "rw_edge_end",
    "agg", "bal_min", "bal_present", "bal_deficit", "soft_slack", "soft_reward",
    "family_of", "keys_of", "is_family", "plan_of", "offset_intersection",
]
