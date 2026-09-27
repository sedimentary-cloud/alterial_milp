# 干线绿波 / 红波协调优化 MILP 算法包 — 设计文档

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 文档状态 | 待评审（规范已锁定，可据此开发） |
| 目标读者 | 算法 / 后端开发工程师 |
| 求解器基线 | Python + scipy.optimize.milp（HiGHS），预留 highspy 后端 |

---

## 1. 概述

### 1.1 问题背景

选定一条干线（含上行、下行两个方向），干线穿过 N 个信号控制路口，路口之间由路段连接。每个路段的上/下行各有距离与设计车速。每个路口有若干候选信号控制方案。需要决策：

1. 每个路口选择哪一个信号控制方案；
2. 公共信号周期 C（从候选格点中遍历）；
3. 干线设计速度倍率 κ（从候选格点中遍历，统一缩放所有路段速度）；
4. 每个路口的周期起始偏置 φ；
5. 在被选方案声明的可调范围内，微调绿灯窗口端点。

使得干线层面的绿波 / 红波协调效果最优，同时兼顾路口自身因协调而产生的损失。

### 1.2 核心抽象

**干线看不到路口内部的具体相位方案，只能感受到路口在干线上下行方向上的绿灯窗口。** 因此每个信控方案被抽象为：

- 每个方向若干个绿灯窗口（一个周期内 ≤ 3 个，双周期方案复制后 ≤ 6 个），以周期比例表示起止；
- 窗口端点的可调范围与端点之间的线性约束（硬 / 软）；
- 方案被选中时才生效的全部约束。

干线优化在这个抽象层上进行：带（band）= 时间-空间图上连续穿过一系列路口绿灯窗口的时间区间。

### 1.3 关键概念定义

| 概念 | 定义 |
|---|---|
| 绿波带（链式） | 一条时间区间，沿某方向连续穿过给定有序路口列表（≥ 2 个）中每个路口的某一个绿灯窗口；其宽度为该区间长度，由链上最紧的窗口决定，宽度计一次 |
| 相邻对绿带 | 节点列表长度 = 2 的链式绿带（统一 schema 的特例） |
| 红波带 | 上游路口**一个**绿灯窗口（平移行程时间 τ 后）与下游邻接路口**一个**红灯区间的重叠时间区间。**只能在相邻两个路口之间定义，结构上不可能穿过 3 个及以上路口连续传播**（越过下游路口后定义它的红灯区间已不存在） |
| 带需求（BandDemand） | 用户声明的一条优化目标单元：{方向, 有序节点列表, 类型(绿/红), 每周期最大带数, 权重, margin 参数}。红带需求强制节点数 = 2 |
| 窗口绿波 | 起讫路口为干线中间某两路口的链式绿带（部分干线绿波），与全程绿波同为链式带需求的特例 |
| composite band objective | 干线目标 = 所有带需求的带宽加权和 − margin 短缺罚 − owner 为 composite 的软约束罚（单位：秒） |
| 路口损失（intersection loss） | 路口窗口端点线性组合的软约束违反量 × 强度系数之和（单位：秒），仅选中方案激活 |
| margin（端点净距） | 绿带端点到其所用绿灯窗口对应端点的距离；不足期望净距 δ_min 的部分计为短缺并罚款（红带无 margin） |

### 1.4 多目标结构

两个目标，**不可加权求和**：

1. composite band objective（最大化）；
2. intersection loss（最小化）。

采用两阶段 ε-constraint：

- 阶段 1：在每个 (C, κ) 格点上最大化 composite，得 B*(C, κ)；
- 阶段 2：在最优格点（可配 top-k）上，依次求解 `min L s.t. composite ≥ B* − ε_l`，ε_l 从 0 递增，得到 pareto 前沿上的若干解。

### 1.5 设计原则

1. **格点化是结构而非技巧**：C 与 κ 固定后，全部行程时间为常数，模型内无任何双线性项（除方案选择 × 微调量，用标准方法线性化）；
2. **存在性指示**：所有带、所有带-窗口指派均有存在性 / 指派二进制，约束经 big-M 激活，模型在数据合法时恒可行（带可以全部不存在）；
3. **统一软约束机制**：路口损失、margin 罚、用户软约束共用同一套「线性表达式 + 违反量 × 系数 + owner 归属」机制；
4. **约束表达式系统**：用户自定义约束 = 预定义变量原子的线性组合与常数比较，声明硬 / 软；
5. **求解器可替换**：建模层不依赖 scipy 特有 API，求解适配层可切换 highspy。

### 1.6 非目标（本期不做）

- 路口内部相位相序优化（由方案候选集外生给定）；
- 排队、溢流、公交优先等交通流动态；
- 多干线 / 网络级协调（架构上预留）；
- 双周期路口的原生建模（本期用「公共周期内复制 + 一致性约束」表达，见 3.3.5）。

---

## 2. 术语与符号

### 2.1 集合与索引

| 符号 | 含义 |
|---|---|
| `I = {1..N}` | 路口集合，沿干线物理顺序编号 |
| `d ∈ {+, −}` | 方向，+ 上行（1→N），− 下行（N→1） |
| `P_i` | 路口 i 的候选方案集合 |
| `W_{i,p,d}` | 方案 p 在方向 d 的绿灯窗口列表（有序、不重叠） |
| `K_{i,p,d} = |W_{i,p,d}|` | 窗口数，≤ 3（双周期复制后 ≤ 6） |
| `B` | 带需求集合 |
| `S_b = [n_1..n_m]` | 需求 b 的有序节点列表，**按行驶方向排序**（绿带 m ≥ 2，红带 m = 2） |
| `Q_b` | 需求 b 每周期的最大带槽数（绿带默认 = 链上各路口最大窗口数的最小值；红带默认 3，可配） |
| `q ∈ {1..Q_b}` | 带槽位编号 |

### 2.2 格点常数（每个 (C, κ) 格点内为常数）

| 符号 | 含义 |
|---|---|
| `C` | 公共周期（秒） |
| `κ` | 速度倍率 |
| `τ^d_i` | 方向 d 从路口 i 到沿 d 下一相邻路口的行程时间 = `dist_i^d / (κ · v_i^d)` |
| `T^d_i` | 沿方向 d 从参考锚点到路口 i 的累计行程时间（锚点 = 0） |
| `s̄_{i,p,d,k}, ē_{i,p,d,k}` | 窗口 k 的相对起止（周期比例），0 ≤ s̄ < ē，复制窗口允许 ē ≤ 2 |

### 2.3 方向与节点顺序约定

- 所有带需求的节点列表按**行驶方向**排序：下行链写 `[N, N−1, …, 1]`；
- 红带需求 `[i, j]` 中 i 为上游、j 为下游（沿行驶方向）；
- 绿带链上节点 n_j 的累计行程时间 `T_j` 从链首（T=0）沿行驶方向累计。

---

## 3. 输入数据模型（Schema 规范）

所有输入用一个 JSON / pydantic 模型表达。以下字段名即 schema 字段名。

### 3.1 Corridor（干线）

```json
{
  "corridor_id": "string",
  "intersections": ["I1", "I2", "..."],
  "segments": [
    {
      "from": "I1", "to": "I2",
      "distance_m": 520.0,
      "speed_up_mps": 11.1,
      "speed_down_mps": 10.5
    }
  ]
}
```

- `segments[i]` 连接 `intersections[i]` 与 `intersections[i+1]`，数量 = N−1；
- 速度为设计速度初值，格点遍历时统一乘 κ。

### 3.2 Intersection（路口）

```json
{
  "id": "I1",
  "plans": ["P1", "P2"]
}
```

### 3.3 Plan（信控方案）

#### 3.3.1 基本结构

```json
{
  "id": "I1_P1",
  "intersection": "I1",
  "double_cycle": false,
  "windows": {
    "up":   [[0.0, 0.2], [0.5, 0.9]],
    "down": [[0.3, 0.6]]
  },
  "adjustable": {
    "up":   {"0": {"start": [-0.02, 0.02], "end": [-0.05, 0.05]}},
    "down": {}
  },
  "hard_constraints": [ /* 见 3.6 */ ],
  "soft_constraints": [ /* 见 3.6 */ ]
}
```

- 窗口以**周期比例** `[start, end)` 给出，0 ≤ start < end；双周期复制产生的窗口允许 end ≤ 2（展开表示，**禁止 mod 回 [0,1)**，见 3.3.5）；
- 同一方向窗口有序且不重叠（校验规则 V3）；
- `adjustable`：声明窗口端点的微调范围（单位：比例，相对名义值的偏移 [lo, hi]，lo ≤ 0 ≤ hi）。未声明的端点固定。

#### 3.3.2 窗口端点的变量引用

约束表达式中引用端点的**秒值**：

```
endpoint(intersection, plan, direction, window_index, side)
  side ∈ {"start", "end"}
  秒值 = (s̄ + δ) · C      （C 为当前格点周期，δ 为微调量，固定端点 δ ≡ 0）
```

#### 3.3.3 方案内约束

- `hard_constraints`：端点秒值的线性组合与常数比较，方案选中时必须满足；
- `soft_constraints`：同形，但可违反；违反量（秒）× `coef` 计入**路口损失**（owner 固定为 intersection_loss）；
- 典型用例：最优绿信比偏离（`end − start ∈ [0.38C, 0.42C]` → 秒值区间软约束）、最小绿（硬）、两窗口间距固定（硬等式）。

#### 3.3.4 方案在格点下的可行性

方案在周期 C 下可行 ⟺ 其窗口秒值 + 可调范围 + 硬约束构成的多面体非空。可行性由预处理层逐 `(i, p, C)` 精确判定（见 §5）。**固定窗口方案在周期 C 下违反硬约束时：仅剔除 + 诊断记录，不自动修复**（已锁定的语义）。

#### 3.3.5 双周期方案（`double_cycle: true`）

双周期路口建模为公共周期 C 内的复制方案：

1. 原窗口 `(s, e)` 复制为 `(s, e)` 与 `(s+0.5, e+0.5)`；复制后 end > 1 的窗口保持展开表示（如 `(1.0, 1.4)`）；
2. 若端点可调，两份对应端点的微调量用**一致性等式**锁定：`δ_copy2 = δ_copy1`（即 copy2 端点 = copy1 端点 + 0.5，比例域）。**该等式必须显式生成**——缺失时求解器会让两个半周期不对称，等价于偷偷变成全周期方案；
3. 复制后窗口数上限 6，带槽数上界自适应（见 4.8）；
4. 复制窗口同样参与 3.3.4 的可行性检查（含一致性等式）。

### 3.4 BandDemand（带需求）

```json
{
  "id": "WB_up_full",
  "type": "green",              
  "direction": "up",
  "nodes": ["I1", "I2", "I3", "I4"],
  "max_bands": null,            
  "weight": 1.0,
  "margin": { "delta_min_s": 2.0, "coef": 1.0 }
}
```

- `type`: `"green"`（nodes ≥ 2）/ `"red"`（nodes 必须 = 2）；
- `nodes` 按行驶方向排序；
- `max_bands`: null → 自动（绿带 = 链上各路口 `max_p K_{i,p,d}` 的最小值；红带 = 3）；
- `weight`：目标中该需求聚合带宽的权重，默认 1.0；
- `margin.delta_min_s`：期望净距（秒），红带忽略；全局默认值在 ObjectiveConfig 中，需求级可覆盖；
- `margin.coef`：短缺罚系数，默认 1.0（短缺 1 秒 = 带宽 1 秒）。

目标配置示例的对应关系：

| 用户目标 | 配置 |
|---|---|
| 单向全程绿波 | 1 条 green 需求，nodes = 全程 |
| 窗口绿波 | 1 条 green 需求，nodes = 中间区段 |
| 双向绿波 | 2 条 green 需求（上下行各一） |
| 单绿单红 | 1 条 green + 另一方向每相邻对 1 条 red |
| 双向红波 | 两方向每相邻对各 1 条 red |
| 仅相邻对绿波和最大 | 每相邻对 1 条 green（nodes 长度 2） |

### 3.5 ObjectiveConfig（目标配置）

```json
{
  "band_demands": ["WB_up_full", "..."],
  "margin_default": { "delta_min_s": 2.0, "coef": 1.0 },
  "pareto": { "num_points": 4, "relax_max": 0.3, "topk_grids": 1 }
}
```

### 3.6 ConstraintExpr（约束表达式系统）

用户自定义约束（方案内硬/软约束、跨路口约束、带宽约束）统一为：

```json
{
  "id": "c1",
  "terms": [
    {"atom": {"kind": "offset", "intersection": "I2"}, "coef": 1.0},
    {"atom": {"kind": "offset", "intersection": "I1"}, "coef": -1.0},
    {"atom": {"kind": "bandwidth", "demand": "WB_up_full"}, "coef": 0.0}
  ],
  "sense": "<=",
  "rhs": 30.0,
  "hard": true,
  "soft": {"coef": 5.0, "owner": "composite"}
}
```

原子（atom）种类：

| kind | 引用 | 单位 |
|---|---|---|
| `offset` | 路口偏置 φ_i | 秒 |
| `endpoint` | 方案窗口端点秒值（3.3.2） | 秒 |
| `bandwidth` | 某带需求的**聚合带宽**（同需求多带宽度直接加和 Β_b = Σ_q β_{b,q}） | 秒 |
| `const` | 常数 | — |

规则：

- `hard: true` → 硬约束；`hard: false` → 必须给 `soft.coef` 与 `soft.owner ∈ {"composite", "intersection_loss"}`；
- 方案内约束可引用 `endpoint` 原子，仅在方案选中时生效（big-M 激活）；
- 全局约束（跨路口、带宽类）不得引用 `endpoint` 原子（避免方案归属歧义）；
- 违反量定义：`expr ≤ rhs` 的违反 = max(0, expr − rhs)；`expr ≥ rhs` 对称；`=` 拆成两个方向各计违反。

### 3.7 GridConfig（格点配置）

```json
{
  "cycles_s": [70, 80, 90, 100, 110, 120],
  "speed_ratios": [0.9, 1.0, 1.1]
}
```

### 3.8 SolverConfig（求解配置）

```json
{
  "backend": "scipy",
  "time_limit_s": 300,
  "mip_rel_gap": 0.0,
  "num_workers": 8,
  "heuristic": {"enabled": true, "refine_rounds": 3}
}
```

### 3.9 输入校验规则（前置校验，全部在建模前执行）

| 编号 | 规则 |
|---|---|
| V1 | 窗口比例 0 ≤ start < end ≤ 2（双周期允许 >1） |
| V2 | 同方向窗口有序且不重叠 |
| V3 | 可调范围 lo ≤ 0 ≤ hi，且调整后窗口仍满足 V1/V2 的可能性由预处理 LP 判定 |
| V4 | 红带需求 nodes 长度 = 2；绿带 ≥ 2；节点在干线上且按行驶方向排序 |
| V5 | 带需求节点列表沿行驶方向连续（中间不跳路口） |
| V6 | 软约束必须声明 coef 与 owner；硬约束不得声明 soft |
| V7 | 全局约束不得引用 endpoint 原子 |
| V8 | 周期、速度、距离为正；倍率 > 0 |
| V9 | 每个路口至少 1 个方案（格点级可行性由 §5 判定） |
| V10 | margin.delta_min_s ≥ 0 且小于链上最小窗口宽度（给出警告而非报错） |

---

## 4. 数学模型（单个格点 (C, κ) 内）

以下所有常数均已按当前格点常量化（τ、T、窗口秒值名义量）。

### 4.1 决策变量

| 变量 | 类型 | 含义 |
|---|---|---|
| `x_{i,p}` | 二进制 | 路口 i 选择方案 p |
| `φ_i` | 连续 [0, C) | 路口 i 周期起始偏置；锚点路口固定 φ = 0（消除平移对称，**必做**） |
| `δ^s_{i,p,d,k}, δ^e_{i,p,d,k}` | 连续 [lo, hi] | 可调端点微调量（比例域）；仅对声明可调的端点建立 |
| `z^s_{i,p,d,k}, z^e_{i,p,d,k}` | 连续 | 线性化辅助变量 z = x·δ（仅可调端点） |
| `ŝ_{i,d,k}, ê_{i,d,k}` | 连续 | **有效窗口**端点秒值（槽位 k = 1..K_i^max，见 4.2） |
| `h_{i,d,k}` | 连续 [0,1] | 槽位 k 是否被选中方案提供（自动 0/1） |
| `u_{b,q}` | 连续 [0, C) | 带槽 q 在参考时间轴上的起点 |
| `β_{b,q}` | 连续 [0, C] | 带槽 q 的宽度；带终点 v = u + β（允许 v > C，最大 2C，跨 0 点的带由此表达） |
| `e_{b,q}` | 二进制 | 带槽 q 存在性 |
| `a_{b,q,j,k}` | 二进制 | 绿带槽 q 在链上节点 n_j 使用窗口槽位 k |
| `n_{b,q,j}` | 整数 [0, ⌈T_j^span/C⌉] | 绿带在节点 n_j 的回绕周期数；链首固定为 0 |
| `ag_{b,q,g}` | 二进制 | 红带槽 q 的上游绿窗槽指派（只负责上游一侧） |
| `ar_{b,q,r,k}` | 二进制 | 红带槽 q 的下游红窗实例指派（红窗槽 r + 下游周期偏移 k） |
| `σ^{m,s}_{b,q,j}, σ^{m,e}_{b,q,j}` | 连续 ≥ 0 | 绿带在节点 n_j 的左/右端 margin 短缺 |
| `σ_c` | 连续 ≥ 0 | 软约束 c 的违反量 |

### 4.2 有效窗口重构（关键工程决策，必须按此实现）

带-窗口指派**不按** `(带, 路口, 方案, 窗口)` 建立，而按 `(带, 路口, 方向, 槽位)` 建立。对每个 `(i, d)`，槽位 `k = 1..K_i^max`，`K_i^max = max_p K_{i,p,d}`（≤ 3，双周期 ≤ 6）。

定义常数 `A_{i,p,d,k} = 1` 当且仅当方案 p 在方向 d 至少有 k 个窗口。

```
槽位可用性：  h_{i,d,k} = Σ_p A_{i,p,d,k} · x_{i,p}          （Σx=1 ⟹ h ∈ {0,1} 自动成立）
有效端点：    ŝ_{i,d,k} = Σ_p [ s̄_{i,p,d,k}·C·x_{i,p} + C·z^s_{i,p,d,k} ]
             ê_{i,d,k} = Σ_p [ ē_{i,p,d,k}·C·x_{i,p} + C·z^e_{i,p,d,k} ]
```

其中固定端点无 z 项。双线性项 `z = x·δ` 用标准 McCormick（二进制 × 有界连续）线性化：

```
z ≤ δ^hi·x        z ≥ δ^lo·x
z ≤ δ − δ^lo·(1−x)   z ≥ δ − δ^hi·(1−x)
```

效果：指派二进制数量从 `带数 × N × 方案数 × 3` 降到 `带数 × N × 3`；窗口端点的方案依赖全部吸收进连续变量 ŝ/ê。

双周期复制：复制窗口直接作为额外资位（k = 4..6），其端点名义值 = 原窗口 + 0.5（比例），微调量一致性等式 `δ_copy = δ_orig` 按 3.3.5 生成后，复制槽位的 z 变量直接复用原槽位的 δ（无需新建微调变量）。

### 4.3 约束全集

记号：绿带需求 b 的链上第 j 个节点 `n_j`，沿方向 `d_b` 的累计行程时间 `T_j`（链首 T_1 = 0）；M 的取值规范见 4.6。

**C1 方案选择**

```
Σ_p x_{i,p} = 1                         ∀i
```

**C2 方案内硬约束**（方案 p 的每条硬约束 `expr(端点秒值) R rhs`）

```
expr ≤ rhs + M·(1 − x_{i,p})            （sense 为 ≤；≥、= 对称处理）
```

注意端点秒值 = `(s̄ + δ)·C`，其中 δ 经 C1 与 z 的联动仅在 x=1 时有意义；M(1−x) 保证未选中时松弛。

**C3 方案内软约束**（owner = intersection_loss）

```
expr ≤ rhs + σ_c + M·(1 − x_{i,p})
σ_c ≥ 0
```

**C4 绿带指派**

```
Σ_k a_{b,q,j,k} = e_{b,q}               ∀b, q, 链上节点 n_j
a_{b,q,j,k} ≤ h_{n_j, d_b, k}           ∀b, q, j, k
```

存在则每个节点恰用一个窗口槽位；槽位必须被选中方案提供。

**C5 绿带包含约束**（带区间平移后落在所用窗口内）

```
u_{b,q} + T_j ≥ φ_{n_j} + ŝ_{n_j,d_b,k} + n_{b,q,j}·C − M·(1 − a_{b,q,j,k})
u_{b,q} + T_j + β_{b,q} ≤ φ_{n_j} + ê_{n_j,d_b,k} + n_{b,q,j}·C + M·(1 − a_{b,q,j,k})
                                          ∀b, q, j, k
```

带宽无需 min 表达式：上述不等式天然让 β 被链上最紧窗口卡住，最大化目标自动顶到最紧处（MAXBAND 标准技巧）。

**C6 绿带存在性**

```
β_{b,q} ≤ C · e_{b,q}                   ∀b, q
```

e = 0 时 β = 0，且 C4 使所有 a = 0，C5 全部被 big-M 松弛——不缩小解空间。

**C7 多带破对称与窗口独占**

```
u_{b,q} ≤ u_{b,q+1}                     ∀b, q = 1..Q_b−1        （按起点排序）
Σ_q a_{b,q,j,k} ≤ 1                     ∀b, j, k                （窗口独占）
```

窗口独占的保守性论证与开关见 4.7。

**C8 绿带 margin 短缺**（仅绿带；红带无 margin）

```
σ^{m,s}_{b,q,j} ≥ δ^min_b − (u_{b,q} + T_j − φ_{n_j} − ŝ_{n_j,d_b,k} − n_{b,q,j}·C) − M·(1 − a_{b,q,j,k})
σ^{m,e}_{b,q,j} ≥ δ^min_b − (φ_{n_j} + ê_{n_j,d_b,k} + n_{b,q,j}·C − u_{b,q} − T_j − β_{b,q}) − M·(1 − a_{b,q,j,k})
                                          ∀b, q, j, k
```

短缺在目标中被最小化（composite 的减项），未指派的槽位约束自动松弛，σ 取 0。

**C9 红带构造（因式分解版）**

红带需求 b = (上游 i → 下游 j，方向 d)，时间轴取下游路口 j 的周期轴，`u_{b,q} ∈ [0, C)`。

下游红窗槽由有效绿窗补集生成：设 j 方向 d 的有效绿窗排序后为 `[ŝ_1, ê_1], …, [ŝ_K, ê_K]`（K = K_{j,d} 实际可用槽数），则红窗槽：

```
r = 1..K−1:  [ê_r, ŝ_{r+1}]              （绿窗间隙）
r = K:       [ê_K, ŝ_1 + C]              （跨 0/C 段，必须显式生成，否则跨周期红波建不出来）
```

红窗槽 r 的可用性 `hR_{j,d,r}` 与端点均为有效绿窗端点的线性组合（r ≤ K−1 时需要 h_{j,d,r+1} = 1；r = K 需要 h_{j,d,1} = 1）。

**指派变量因式分解**。不再建“上游绿窗 × 下游红窗 × 周期偏移”的联合二进制 aR，而是拆成：

```
ag_{b,q,g}       带槽 q 使用上游绿窗槽 g
ar_{b,q,r,k}     带槽 q 使用下游红窗实例 (r,k)，k 为下游周期偏移
Σ_g ag_{b,q,g} = e_{b,q}
Σ_{r,k} ar_{b,q,r,k} = e_{b,q}
ag ≤ h_{i,d,g}；ar ≤ hR_{j,d,r}
```

上游两条包含约束只挂 `ag`：

```
u_{b,q}          ≥ φ_i + ŝ_{i,d,g} + τ_{i→j} − M·(1 − ag_{b,q,g})
u_{b,q} + β      ≤ φ_i + ê_{i,d,g} + τ_{i→j} + M·(1 − ag_{b,q,g})
```

下游两条包含约束只挂 `ar`（k 放在下游红窗实例上）：

```
u_{b,q}          ≥ φ_j + rstart_r + k·C − M·(1 − ar_{b,q,r,k})
u_{b,q} + β      ≤ φ_j + rend_r   + k·C + M·(1 − ar_{b,q,r,k})
```

耦合完全由共享变量 `u_{b,q}、β_{b,q}` 完成。等价性：旧联合变量 `aR=1` 激活的恰好是“g 的两条 + (r,k) 的两条”四条不等式；因式分解后 `ag_g=1, ar_{r,k}=1` 激活同一组不等式，可行域不变。只有当约束必须引用“g 与 r 的配对本身”时才需要联合变量；C9 不包含这类规则。

**k 预剪枝**。对每个 `(g,r,k)`，存在 `Δφ = φ_j − φ_i ∈ (−C, C)` 使两个区间相交的必要条件是：

```
( ŝ_g + τ − re_r − kC,  ê_g + τ − rs_r − kC ) ∩ (−C, C) ≠ ∅
```

预处理用端点上下界计算上式；丢弃所有与任何对方槽位都不兼容的 `ag`、`ar` 变量，只保留幸存实例。因式分解后先建“兼容对”集合，再投影出 `ag_slots` 与 `ar_keys`，避免独立剪枝错删配对。

**时间分离**。不再使用“每个 (g,r,k) 实例至多宿主一条带”的独占约束，而是对任意两条都存在的带强制时间不重叠：

```
u_{b,q} + β_{b,q} ≤ u_{b,q2} + M·(2 − e_{b,q} − e_{b,q2})    ∀ q < q2
```

只加相邻 q 不够（中间槽不存在时 q 与 q+2 仍可能重叠）。同一相邻对同方向的时间重叠红带可以合并成一条更宽的带，最大化目标不会偏好拆分，因此该约束安全；槽位对称，按 q 编号排序不丢最优解。

**C13 带宽有效不等式**：

```
β_{b,q} ≤ Σ_g W_g · ag_{b,q,g} + C·(1 − e_{b,q})
β_{b,q} ≤ Σ_{r,k} Wr_{r,k} · ar_{b,q,r,k} + C·(1 − e_{b,q})
β_{b,q} ≤ C · e_{b,q}
```

其中 `W_g`、`Wr_{r,k}` 分别是所选上游绿窗/下游红窗实例的最大可能宽度。C13 补回 big-M 在 LP 松弛中丢失的“带宽 ≤ 所选窗口宽度”信息。

**C10 用户全局约束**（3.6 的表达式，原子为 φ、聚合带宽 Β_b = Σ_q β_{b,q}、常数）

```
硬： expr R rhs
软： expr ≤ rhs + σ_c   （sense ≥ 对称；= 拆双向）
```

**C11 双周期一致性**（见 3.3.5）：复制端点微调量等式。

### 4.4 目标函数

```
composite = Σ_b weight_b · Σ_q β_{b,q}
          − Σ_{b(green),q,j} κ^m_b · (σ^{m,s}_{b,q,j} + σ^{m,e}_{b,q,j})
          − Σ_{c ∈ soft, owner=composite} coef_c · σ_c

L         = Σ_{c ∈ soft, owner=intersection_loss} coef_c · σ_c
```

- 单位统一为秒；margin 短缺与软罚的系数无量纲；
- 簿记规则：**margin 类罚项全部归 composite**（已锁定）；方案内端点软约束全部归 L；用户全局软约束按声明的 owner 归集。

### 4.5 两阶段求解

**阶段 1（逐格点）**：`max composite` → 记录 `B*(C, κ)`、求解状态、 incumbent。

**阶段 2（ε-constraint，逐最优点）**：对 top-k 格点（默认 k=1，按 B* 排序）：

```
for l = 0..K_pareto:
    ε_l = (l / K_pareto) · relax_max · B*
    min  L
    s.t. composite ≥ B* − ε_l
         （其余约束不变）
```

- **l = 0 必须求解**：阶段 1 只保证 composite 最大，达到 B* 的解损失可能天差地别，ε=0 给出 pareto 前沿的低损失端点；
- 报告每个点的**实际 composite 值**（求解器只保证下界，实际可能更高）；
- 每个 ε 点输出完整方案（见 §8）。

### 4.6 big-M 取值规范

- 按每个约束两端项的**最大可能差**取紧，量级为 `T_max + 2C`（通常数百到一千余秒）；
- **禁止**使用 1e6 量级的通用大 M（HiGHS 数值稳定性与 LP 松弛质量）；
- 实现上按约束类型集中定义 M 常量（包含约束、方案激活、软约束各自一类），写入模型构建器的常量表。

### 4.7 窗口独占的保守性说明（设计决策记录）

同一方向多条带在同一路口共用同一窗口槽位时：

- 若两条带在链上**所有**路口共用同一窗口，它们可合并为一条带，总宽度不变——禁止无损失；
- 仅当两条带在**部分**路口共用窗口、在其余路口使用不同窗口时，独占约束才切掉真实最优（轻度保守）。

默认启用窗口独占（C7），schema 预留 `allow_window_sharing` 开关：关闭独占、改为「同槽位则时间不重叠」的条件析取约束（每对带槽每窗口槽一个 big-M 析取，约束数平方级增长）。本期默认不开启。

### 4.8 模型规模估算与自适应裁剪

设 N = 15、方案 ≤ 5、窗口 ≤ 3、带需求 4 条、每条带槽 ≤ 3：

| 项 | 量级 |
|---|---|
| x | 75 |
| 绿带指派 a | 3 槽 × 15 节点 × 3 窗口 ≈ 135 / 需求 |
| 红带指派 ag + ar | 3 槽 × (3 绿 + 3 红 × k) ≈ 27~45 / 需求（因式分解后） |
| 回绕整数 n | 3 槽 × 15 ≈ 45 / 需求 |
| 连续变量 | 数百 |

裁剪规则（建模器必须实现）：

1. **带槽数自适应**：`Q_b = min(用户指定, 链上各路口 max_p K_{i,p,d} 的最小值)`；全单窗口方案时 Q_b = 1，不建多带变量（用户明确要求的优化）；
2. **回绕上界局部化**：`n_{b,q,j} ≤ ⌈T_j/C⌉`，按需求实际跨度取，不用全局最大值；
3. **不可行方案剔除**：预处理判定不可行的 (i, p) 在当前格点不建 x 变量（或固定为 0）；
4. **固定端点不建 δ/z**；
5. **k（红带周期偏移）上界** = `⌈τ/C⌉ + 1`，逐需求取。
6. **红带兼容性预剪枝**：对 `(g,r,k)` 用 `Δφ∈(−C,C)` 与绝对容纳性两层判定，只建幸存 `ag/ar`；
7. **C13 有效不等式**：每条带每侧一条，收紧 LP 松弛（见 C9）。

---

## 5. 预处理层：方案可行性过滤

### 5.1 为什么必须精确而非启发式

格点遍历时，窗口秒值 = 比例 × C，而硬约束通常以秒表达（如最小绿 15s）。比例 0.2 的绿在 C=100s 时为 20s，到 C=60s 只剩 12s——**方案可行性随周期变化**。过滤层逐 `(i, p, C)` 精确判定，不可行方案在该格点直接剔除。判定本身是微型 LP（见下），精确且毫秒级，没有理由用启发式引入"误判不可行"的新失败模式。

### 5.2 判定方法

- **固定窗口方案**：窗口秒值逐条代入硬约束验证，纯算术；
- **可调端点方案**：以微调量 δ 为变量、硬约束 + 可调边界 + 窗口有序不重叠 + 双周期一致性等式为约束，解一个可行性 LP（变量 ≤ 12，约束数十条，scipy.optimize.linprog / HiGHS）。

### 5.3 格点级结论

- 某路口全部方案不可行 → **该格点不可行**，跳过并记录诊断（哪个路口、违反哪条约束、违反量），不进入求解器；
- 部分方案不可行 → 剔除并记录（诊断级别 INFO）；
- 固定窗口方案不可行时**仅剔除 + 诊断，不自动修复**（已锁定语义）。

### 5.4 不变式（架构原则）

经过滤后，MILP **恒可行**：每路口至少一个可行方案，所有带可不存在（e=0, β=0），软约束可付费。因此 MILP 返回 infeasible 只可能来自：

1. 用户跨路口硬约束互相冲突；
2. 带宽类硬约束不可满足（如"红波宽度 ≥ 20s"）；
3. 建模 bug。

不可行诊断工具（§7.4）按此分类定位。

---

## 6. 启发式初始解模块（贪心 + 块坐标 refinement）

### 6.1 定位与价值

scipy.optimize.milp 不支持 warm start，本模块不为加速单次求解，其价值是：

1. 给出 composite 的**下界**与可行 incumbent；
2. 给格点**排序打分**（分数高的格点先解、给足时间预算）；
3. MILP 超时时的 **fallback** 输出；
4. **正确性对照**：MILP 解必须不差于贪心解，否则建模有 bug——这是最有力的回归断言。

### 6.2 算法流程

```
输入：格点 (C, κ)、过滤后的可行方案集、带需求列表
输出：可行解 (x, φ, δ, 带结构) 与 composite 估计值

Step 1  选方案：每个路口取路口损失最小（软约束常量部分）的可行方案；
        若方案含可调端点，用 §5.2 的 LP 求使软约束违反最小的端点值。
Step 2  定偏置（逐路口解耦）：
        固定锚点带起点 u（取锚点窗口起点 + δ_min）。
        关键结构：给定 u 后，路口 i 的包含约束只含 φ_i 自己 ⟹ φ_i 可逐路口独立优化。
        对每个路口 i：
          - 对每条经过 i 的带需求，理想到达时刻 = (u + T_i) mod C；
          - 候选 φ_i 只取有限个事件点：各窗口槽的起点/终点对齐到达时刻 ± δ_min
            （每窗口 2 条边 × 每需求 1 个到达时刻，候选数 = O(窗口数 × 需求数)）；
          - 枚举全部候选，取「各需求 margin 达标情况 − 短缺罚」总分最大者。
Step 3  构造带：φ 固定后，对每条绿带需求，链上各窗口平移求交，
        构造性地算出每条带槽的真实宽度 β̂（对固定偏置这是精确计算，非估计）；
        红带需求逐对计算 (g, r, k) 重叠实例，取最宽的 ≤ Q_b 个。
Step 4  块坐标 refinement（默认 3 轮）：
        固定 φ → 每路口重选方案（局部贡献 = 带 margin 收益 − 路口损失最大者）；
        固定方案 → 重算 φ（同 Step 2）；
        收敛或轮数用尽后输出。
```

### 6.3 复杂度

每格点：O(N × 窗口数 × 需求数 × 候选数) + 数个小 LP，毫秒~十毫秒级，可随格点并行。

---

## 7. 求解流程

### 7.1 总体流水线

```
输入校验（V1–V10）
   │
   ▼
逐格点（并行，进程池）：
   ① 预处理过滤（§5）── 格点不可行 → 记录诊断，跳过
   ② 贪心初始解（§6）── 记录 composite 下界与格点分数
   ③ 构建 MILP（§4，含自适应裁剪）
   ④ 阶段 1 求解：max composite → B*(C, κ)
   │
   ▼
汇总全部格点的 B* 排序表
   │
   ▼
阶段 2：对 top-k 格点（默认 k=1）做 ε-constraint 扫描（§4.5）
   │
   ▼
输出报告（§8）
```

### 7.2 并行策略

- 格点间天然独立：`concurrent.futures.ProcessPoolExecutor`，worker 数 = `num_workers`；
- 每个进程内 HiGHS 单线程（不依赖求解器内部并行，行为可复现）；
- 提交顺序按贪心分数降序（早完成的往往是好格点，便于设置整体时间预算与提前收敛）；
- 每个格点记录：求解状态（optimal / feasible / infeasible / time_limit / skipped）、B*、gap、耗时。

### 7.3 格点时间预算

- 基础 `time_limit_s` 对每个格点生效；
- 可选策略：贪心分数低于当前最优 B* 的格点收紧 time_limit（分数是下界，若下界已低于已有最优，该格点翻盘点仅在于 MILP 上限，给较少时间）。

### 7.4 不可行诊断工具

scipy 无法获取 IIS。提供「分块软化」调试器：按约束组（C1–C11、用户约束逐组）依次替换为带大罚系数的软约束重解，定位使模型恢复可行的最小约束组集合，输出诊断报告。触发方式：MILP 返回 infeasible 时自动运行（可在 SolverConfig 关闭）。

---

## 8. 输出规范

### 8.1 结果 JSON

```json
{
  "grid_results": [
    {
      "cycle_s": 90, "speed_ratio": 1.0,
      "status": "optimal",
      "composite_best": 42.5,
      "heuristic_bound": 38.0,
      "solve_time_s": 12.3,
      "dropped_plans": [{"intersection": "I3", "plan": "I3_P2", "reason": "min_green 15s violated (12.0s)"}]
    }
  ],
  "pareto": {
    "grid": {"cycle_s": 90, "speed_ratio": 1.0},
    "points": [
      {
        "epsilon": 0.0,
        "composite_actual": 42.5,
        "intersection_loss": 18.0,
        "solution": {
          "plan_selection": {"I1": "I1_P2", "I2": "I2_P1"},
          "offsets_s": {"I1": 0.0, "I2": 33.5},
          "adjustments": {"I2_P1.up.0.end": "+0.03 (ratio)"},
          "bands": [
            {"demand": "WB_up_full", "slot": 1, "exists": true,
             "width_s": 21.0, "start_at_anchor_s": 5.0,
             "windows": {"I1": 1, "I2": 2}}
          ],
          "constraint_violations": [{"id": "c7", "violation_s": 3.0, "cost": 15.0, "owner": "composite"}]
        }
      }
    ]
  }
}
```

### 8.2 时距图（time-space diagram）

每个 pareto 点输出一张图（matplotlib）：横轴时间（0–C 或展开到 2C）、纵轴路口位置；每路口按方向绘制绿灯窗口条带；绿带画连续阴影区、红带画相邻对之间的重叠区；标注带宽与 margin。用于人工校核与汇报。

---

## 9. 软件架构与模块划分

```
greenwave/
├── schema/          # pydantic 数据模型：Corridor / Plan / BandDemand / ObjectiveConfig …
│   └── validation.py# 校验规则 V1–V10
├── preprocess/
│   └── feasibility.py   # §5 方案可行性过滤（固定方案算术检查 + 可调方案 LP）
├── heuristic/
│   └── greedy.py        # §6 贪心初始解与格点打分
├── model/
│   ├── variables.py     # 变量注册表（统一命名、索引、边界、integrality）
│   ├── windows.py       # 有效窗口重构（4.2）、红窗补集生成、双周期复制
│   ├── constraints.py   # C1–C11 约束工厂
│   ├── objective.py     # composite / L 目标装配、owner 簿记
│   └── builder.py       # 单格点模型构建入口（含 4.8 自适应裁剪）
├── solve/
│   ├── backend_base.py  # 求解器抽象接口（solve / status / objective / values）
│   ├── backend_scipy.py # scipy.optimize.milp 实现（基线）
│   └── backend_highspy.py # highspy 实现（预留，支持 warm start / 更全选项）
├── grid/
│   └── runner.py        # 格点遍历、进程池并行、时间预算、状态汇总
├── pareto/
│   └── epsilon.py       # 阶段 2 ε-constraint 扫描
├── diagnostics/
│   └── infeasible.py    # 分块软化不可行定位（7.4）
└── report/
    ├── solution.py      # 结果 JSON 装配（8.1）
    └── diagram.py       # 时距图（8.2）
tests/
```

关键设计约束：

1. **建模层与求解器解耦**：`model/` 只产出变量表 + 稀疏线性约束矩阵 + 目标向量 + integrality + bounds，不 import scipy；`solve/` 负责翻译与调用；
2. **约束可追溯**：每条约束携带 `(组号 C1–C11, 语义标签, 相关对象 id)` 元数据，供诊断工具与调试输出使用；
3. **表达式系统单一入口**：方案内约束、用户全局约束、margin、目标罚项全部经由同一个 LinearExpression 构件生成，禁止散落的临时拼装；
4. **配置驱动**：带需求、目标簿记、pareto 参数、求解参数全部来自输入 JSON，无硬编码业务参数。

## 10. 详细类与方法设计

本章给出类 / 方法 / 属性级设计。签名为 Python 风格示意，实现时可微调，但**职责划分、命名、单位约定、不可变性约定不得变更**。

### 10.1 通用约定

1. **单位后缀强制**：秒 `_s`、周期比例 `_r`、米 `_m`、米每秒 `_mps`。字段、变量、方法名一律遵守，禁止无量纲暗示的名字（如 `start`、`width`）出现在公共接口；
2. **schema 对象不可变**：全部 pydantic `frozen=True`；`GridPointContext` 为 frozen dataclass；求解期对象（`ArterialModel`、`SolveResult`）可变性按本章定义；
3. **变量命名规范**（用于调试、日志、诊断）：
   `x[I1,P1]`、`phi[I2]`、`win[I2,up,1].s` / `.e` / `.h`、`adj[I2_P1,up,0,end]`、`band[GW_up,1].u` / `.beta` / `.e`、`asg[GW_up,1,I2,2]`、`n[GW_up,1,I2]`、`marg[GW_up,1,I2].s` / `.e`、`red[RW_12,1].u` / `.beta` / `.e`、`asgR[RW_12,1,g1,r2,k0]`、`viol[c7]`；
4. **错误体系**：`GWValidationError`（输入校验，携带 V 编号）、`GWInfeasibleGridPoint`（格点不可行，携带诊断记录，由 GridRunner 捕获而非上抛）、`GWModelError`（建模内部错误）、`GWSolveError`（求解器层错误）；
5. **日志**：`logging.getLogger("greenwave.<module>")`；格点级日志必须带 `cycle_s / kappa` context 字段；
6. **pydantic 校验器只查静态合法性**（V1–V10）；**格点相关可行性一律在 preprocess 层判定**，不混入 schema。

### 10.2 schema 包

```python
Direction = Literal["up", "down"]
Side = Literal["start", "end"]

class WindowSpec(BaseModel, frozen=True):
    start_r: float            # 周期比例
    end_r: float              # 0 <= start_r < end_r <= 2（双周期复制允许 >1）
    @property
    def width_r(self) -> float: ...

class EndpointRange(BaseModel, frozen=True):
    lo_r: float               # 校验 lo_r <= 0 <= hi_r
    hi_r: float

class WindowAdjustSpec(BaseModel, frozen=True):
    start: EndpointRange | None = None
    end: EndpointRange | None = None

class AtomRef(BaseModel, frozen=True):
    kind: Literal["offset", "endpoint", "bandwidth", "const"]
    intersection: str | None = None     # offset / endpoint 必填
    plan: str | None = None             # endpoint 必填
    direction: Direction | None = None  # endpoint 必填
    window: int | None = None           # endpoint 必填
    side: Side | None = None            # endpoint 必填
    demand: str | None = None           # bandwidth 必填
    const: float | None = None          # const 必填
    # 校验器：kind 与必填字段组合必须合法，多余字段必须为 None

class LinearTerm(BaseModel, frozen=True):
    atom: AtomRef
    coef: float

class SoftPenalty(BaseModel, frozen=True):
    coef: float               # 无量纲强度系数
    owner: Literal["composite", "intersection_loss"]

class ConstraintSpec(BaseModel, frozen=True):
    id: str
    terms: tuple[LinearTerm, ...]
    sense: Literal["<=", ">=", "=="]
    rhs_s: float | None = None      # 秒值
    rhs_ratio: float | None = None  # 比例值，预处理换算 rhs_s = rhs_ratio × C；二者恰填其一
    hard: bool
    soft: SoftPenalty | None = None
    # 校验器：hard=True ⟹ soft 为 None；hard=False ⟹ soft 必填

class PlanSpec(BaseModel, frozen=True):
    id: str
    intersection: str
    double_cycle: bool = False
    windows: dict[Direction, tuple[WindowSpec, ...]]
    adjustable: dict[Direction, dict[int, WindowAdjustSpec]] = {}
    hard_constraints: tuple[ConstraintSpec, ...] = ()
    soft_constraints: tuple[ConstraintSpec, ...] = ()

    def effective_windows(self, d: Direction) -> tuple[WindowSpec, ...]:
        """双周期时返回复制展开后的窗口（3.3.5），否则原样返回。"""
    def window_count(self, d: Direction) -> int: ...
    def nominal_endpoint_r(self, d: Direction, k: int, side: Side) -> float: ...
    def adjust_range(self, d: Direction, k: int, side: Side) -> EndpointRange | None: ...

class SegmentSpec(BaseModel, frozen=True):
    from_: str
    to: str
    distance_m: float
    speed_up_mps: float
    speed_down_mps: float

class CorridorSpec(BaseModel, frozen=True):
    corridor_id: str
    intersections: tuple[str, ...]
    segments: tuple[SegmentSpec, ...]      # 数量 = N−1（校验）
    def index(self, intersection_id: str) -> int: ...
    def speed_mps(self, seg_index: int, d: Direction) -> float: ...

class MarginSpec(BaseModel, frozen=True):
    delta_min_s: float = 2.0
    coef: float = 1.0

class BandDemandSpec(BaseModel, frozen=True):
    id: str
    type: Literal["green", "red"]
    direction: Direction
    nodes: tuple[str, ...]          # 按行驶方向排序；red 恰 2 个，green ≥ 2（校验 V4/V5）
    max_bands: int | None = None    # None → 自动（4.8 规则 1）
    weight: float = 1.0
    margin: MarginSpec | None = None  # None → 用 ObjectiveConfig.margin_default；red 忽略

class ParetoConfig(BaseModel, frozen=True):
    num_points: int = 4
    relax_max: float = 0.3
    topk_grids: int = 1

class ObjectiveConfig(BaseModel, frozen=True):
    band_demands: tuple[BandDemandSpec, ...]
    margin_default: MarginSpec = MarginSpec()
    pareto: ParetoConfig = ParetoConfig()

class GridConfig(BaseModel, frozen=True):
    cycles_s: tuple[float, ...]
    speed_ratios: tuple[float, ...]

class HeuristicConfig(BaseModel, frozen=True):
    enabled: bool = True
    refine_rounds: int = 3

class SolverConfig(BaseModel, frozen=True):
    backend: Literal["scipy", "highspy"] = "scipy"
    time_limit_s: float = 300.0
    mip_rel_gap: float = 0.0
    num_workers: int = 8
    heuristic: HeuristicConfig = HeuristicConfig()

class ProblemInput(BaseModel, frozen=True):
    corridor: CorridorSpec
    plans: tuple[PlanSpec, ...]
    objective: ObjectiveConfig
    grid: GridConfig
    solver: SolverConfig = SolverConfig()
    global_constraints: tuple[ConstraintSpec, ...] = ()   # 3.6 用户全局约束（C10）

    def plan(self, plan_id: str) -> PlanSpec: ...
    def plans_of(self, intersection: str) -> tuple[PlanSpec, ...]: ...
    def demand(self, demand_id: str) -> BandDemandSpec: ...
    def validate_all(self) -> list[ValidationIssue]:
        """执行 V1–V10；返回 issue 列表，error 级 issue 非空时调用方抛 GWValidationError。"""
```

### 10.3 GridPointContext（核心中间对象）

格点常量化产物，**preprocess 的输出、heuristic 与 model 的唯一事实来源**：

```python
@dataclass(frozen=True)
class DemandTravelTable:
    demand_id: str
    nodes: tuple[str, ...]            # 按行驶方向
    cum_T_s: tuple[float, ...]        # 链首 = 0
    span_s: float
    n_max: tuple[int, ...]            # 每节点回绕上界 ⌈T_j/C⌉（4.8 规则 2）

@dataclass(frozen=True)
class DroppedPlanRecord:
    intersection: str
    plan: str
    cycle_s: float
    violations: tuple[str, ...]       # 人类可读原因，如 "min_green 15s violated (12.0s)"

@dataclass(frozen=True)
class GridPointContext:
    cycle_s: float
    kappa: float
    tau_s: dict[tuple[str, Direction], float]       # (路段起点, 方向) → 行程时间
    feasible_plans: dict[str, tuple[PlanSpec, ...]] # 路口 → 本格点可行方案（非空，否则格点跳过）
    dropped: tuple[DroppedPlanRecord, ...]
    travel: dict[str, DemandTravelTable]            # demand_id → 行程时间表

    def max_window_slots(self, intersection: str, d: Direction) -> int:
        """max over feasible plans of window_count（含双周期复制）。"""
    def demand_max_bands(self, demand: BandDemandSpec) -> int:
        """4.8 规则 1：min(用户指定, 链上各路口 max_window_slots 的最小值)；red 默认 3。"""
```

### 10.4 preprocess 包

```python
@dataclass(frozen=True)
class PlanFeasibility:
    feasible: bool
    violations: tuple[str, ...] = ()

class FeasibilityFilter:
    def __init__(self, problem: ProblemInput) -> None:
        self._lp_cache: dict[tuple[str, float], PlanFeasibility] = {}  # (plan_id, cycle_s)

    def check_plan(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        """含可调端点 → _check_lp；否则 _check_fixed。"""
    def _check_fixed(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        """窗口秒值逐条代入硬约束，纯算术。"""
    def _check_lp(self, plan: PlanSpec, cycle_s: float) -> PlanFeasibility:
        """以 δ 为变量的可行性 LP（§5.2），含窗口有序、双周期一致性等式；结果入缓存。"""
    def filter(self, cycle_s: float, kappa: float) -> GridPointContext:
        """逐路口检查；某路口全部方案不可行 → raise GWInfeasibleGridPoint（含诊断）。"""
```

### 10.5 model 包

```python
class GVarType(Enum):
    CONTINUOUS = 0
    BINARY = 1
    INTEGER = 2

@dataclass(frozen=True)
class VarHandle:
    index: int
    name: str

class VariableRegistry:
    def add(self, name: str, lb: float, ub: float, vtype: GVarType,
            meta: dict | None = None) -> VarHandle: ...
    def by_name(self, name: str) -> VarHandle: ...
    # 只读数组视图（供求解器适配层）：names / lbs / ubs / integrality / metas

class LinearExpr:
    """dict[int, float] + const；链式构建。"""
    @classmethod
    def of(cls, h: VarHandle, coef: float = 1.0) -> "LinearExpr": ...
    def add(self, h: VarHandle, coef: float) -> "LinearExpr": ...
    def add_const(self, c: float) -> "LinearExpr": ...

@dataclass(frozen=True)
class ConstraintMeta:
    group: str    # "C1".."C11" | "PLAN_HARD" | "PLAN_SOFT" | "USER" | "EPS"
    label: str
    refs: dict[str, str]   # 相关对象 id（路口/方案/需求/槽位），供诊断与调试

class ConstraintStore:
    def add(self, expr: LinearExpr, sense: str, rhs: float, meta: ConstraintMeta) -> int:
        """返回行号。"""
    def to_sparse(self) -> tuple["csr_matrix", "np.ndarray", "np.ndarray"]: ...

@dataclass(frozen=True)
class BigMTable:
    containment: float        # 包含约束：T_max + 2C
    plan_activation: float    # 方案激活：按约束两端最大差
    soft: float               # 软约束松弛
    @classmethod
    def compute(cls, ctx: GridPointContext) -> "BigMTable": ...   # 4.6 规范

@dataclass(frozen=True)
class EffectiveWindowSet:
    slots: tuple[int, ...]
    s: dict[int, VarHandle]   # ŝ
    e: dict[int, VarHandle]   # ê
    h: dict[int, VarHandle]   # 槽位可用性

@dataclass(frozen=True)
class RedWindowSet:
    slots: tuple[int, ...]
    start: dict[int, LinearExpr]   # 端点为有效绿窗端点的线性组合（C9）
    end: dict[int, LinearExpr]
    avail: dict[int, LinearExpr]   # hR

class EffectiveWindowBuilder:
    """4.2：建 ŝ/ê/h、δ/z 变量与 McCormick 约束；双周期复制槽位复用原 δ（C11 联动）。"""
    def build(self, intersection: str, d: Direction) -> EffectiveWindowSet: ...

class RedWindowBuilder:
    """C9：由 EffectiveWindowSet 生成红窗补集（含跨 0/C 段）。"""
    def build(self, intersection: str, d: Direction, gw: EffectiveWindowSet) -> RedWindowSet: ...

@dataclass
class ArterialModel:
    """求解器无关的模型表示 + 解码元数据。访问器供目标装配、ε 约束、解解码三方共用。"""
    vars: VariableRegistry
    cons: ConstraintStore
    objective: LinearExpr | None
    objective_sense: Literal["max", "min"]
    meta: "ModelMeta"

    def v_offset(self, i: str) -> VarHandle: ...
    def v_win(self, i: str, d: Direction, slot: int, side: Side) -> VarHandle: ...
    def v_adjust(self, plan_id: str, d: Direction, k: int, side: Side) -> VarHandle | None: ...
    def v_band_u(self, demand: str, q: int) -> VarHandle: ...
    def v_band_beta(self, demand: str, q: int) -> VarHandle: ...
    def v_band_e(self, demand: str, q: int) -> VarHandle: ...
    def v_soft(self, constraint_id: str) -> VarHandle: ...
    def expr_bandwidth(self, demand: str) -> LinearExpr: ...   # Β_b = Σ_q β
    def expr_composite(self) -> LinearExpr: ...                # 4.4
    def expr_loss(self) -> LinearExpr: ...

class ModelBuilder:
    def __init__(self, problem: ProblemInput, ctx: GridPointContext) -> None: ...
    def build(self) -> ArterialModel:
        """按序执行（每步约束打组标签，§9 可追溯要求）：
        _build_plan_selection(C1) → _build_effective_windows(4.2)
        → _build_plan_constraints(C2/C3) → _build_green_bands(C4–C8)
        → _build_red_bands(C9) → _build_user_constraints(C10)
        → _build_double_cycle_links(C11) → _assemble_objective(4.4)
        """
```

### 10.6 solve 包

```python
class SolveStatus(Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIME_LIMIT = "time_limit"
    ERROR = "error"

@dataclass(frozen=True)
class SolveOptions:
    time_limit_s: float
    mip_rel_gap: float
    presolve: bool = True

@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    objective: float | None
    values: "np.ndarray | None"      # 与 VariableRegistry 索引对齐
    bound: float | None
    time_s: float
    message: str = ""

class ModelSession(Protocol):
    """后端持有的求解器侧模型句柄。"""
    backend_name: str

class SolverBackend(ABC):
    name: str
    @abstractmethod
    def create(self, model: ArterialModel) -> ModelSession: ...
    @abstractmethod
    def solve(self, session: ModelSession, options: SolveOptions) -> SolveResult: ...
    @abstractmethod
    def set_objective(self, session: ModelSession, expr: LinearExpr,
                      sense: Literal["max", "min"]) -> None: ...
    @abstractmethod
    def add_constraint(self, session: ModelSession, expr: LinearExpr,
                       sense: str, rhs: float, meta: ConstraintMeta) -> int: ...

class ScipyBackend(SolverBackend):
    """scipy.optimize.milp 实现。注意：integrality 只分 0=连续 / 1=整数，
    BINARY = integrality 1 + bounds [0,1]；INTEGER = integrality 1 + 整数界。
    增量接口（set_objective / add_constraint）通过重建 milp 输入实现，
    但复用已装配的稀疏矩阵块，避免全量重装配。"""

class HighspyBackend(SolverBackend):
    """预留。支持 warm start：solve() 接受可选 basis/incumbent 参数。"""
```

### 10.7 heuristic 包

```python
@dataclass(frozen=True)
class ConstructedBand:
    demand: str
    slot: int
    exists: bool
    width_s: float
    start_s: float
    windows: dict[str, int]          # 路口 → 使用窗口槽位

@dataclass(frozen=True)
class GreedySolution:
    plan_selection: dict[str, str]
    offsets_s: dict[str, float]
    adjustments_r: dict[tuple[str, Direction, int, Side], float]
    bands: tuple[ConstructedBand, ...]
    composite_est: float             # 下界（6.1）

class GreedyConstructor:
    def __init__(self, problem: ProblemInput, ctx: GridPointContext) -> None: ...
    def run(self) -> GreedySolution:
        """§6.2 四步：_select_plans → _compute_offsets → _construct_bands → _refine。"""
    def _select_plans(self) -> dict[str, PlanSpec]: ...
    def _candidate_offsets(self, i: str, u_anchor_s: float,
                           selection: dict[str, PlanSpec]) -> list[float]:
        """事件点枚举：各窗口槽起点/终点对齐 (u_anchor + T_i) mod C ± δ_min。"""
    def _score_offset(self, i: str, phi_s: float,
                      selection: dict[str, PlanSpec]) -> float: ...
    def _construct_bands(self, selection, offsets) -> tuple[ConstructedBand, ...]:
        """φ 固定后链上窗口平移求交，精确计算带槽宽度；红带取最宽 ≤ Q_b 个重叠实例。"""
    def _refine(self, sol: GreedySolution, rounds: int) -> GreedySolution: ...
```

### 10.8 grid 包

```python
@dataclass(frozen=True)
class GridPointResult:
    cycle_s: float
    kappa: float
    status: SolveStatus | Literal["skipped_infeasible"]
    composite_best: float | None
    heuristic: GreedySolution | None
    solution: "SolutionRecord | None"
    dropped: tuple[DroppedPlanRecord, ...]
    time_s: float
    message: str = ""

class GridRunner:
    def __init__(self, problem: ProblemInput,
                 backend_factory: Callable[[], SolverBackend],
                 cfg: SolverConfig) -> None: ...
    def run(self) -> "GridRunReport":
        """两阶段：
        A. 全部格点并行跑 filter + greedy（毫秒级），得分数与不可行格点；
        B. 可行格点按贪心分数降序提交进程池跑阶段 1 MILP（7.2/7.3）。
        """
    @staticmethod
    def _worker(payload: dict) -> GridPointResult:
        """进程池入口。payload 必须可 pickle：problem.model_dump() + cycle_s + kappa +
        solver 配置；worker 内重建 ProblemInput → FeasibilityFilter → GreedyConstructor
        → ModelBuilder → backend.solve。"""

@dataclass(frozen=True)
class GridRunReport:
    results: tuple[GridPointResult, ...]
    def best(self, k: int = 1) -> list[GridPointResult]:
        """按 composite_best 降序取 top-k（跳过非 optimal/feasible）。"""
```

### 10.9 pareto 包

```python
@dataclass(frozen=True)
class ParetoPoint:
    epsilon_s: float
    composite_actual_s: float        # 用解向量重算的实际值，不取下界（4.5）
    intersection_loss_s: float
    solution: "SolutionRecord"

class EpsilonConstraintScanner:
    def __init__(self, problem: ProblemInput, ctx: GridPointContext,
                 backend: SolverBackend, cfg: ParetoConfig) -> None: ...
    def scan(self, base: GridPointResult) -> list[ParetoPoint]:
        """复用同一 ModelSession：
        1. set_objective(model.expr_loss(), "min")；
        2. l = 0..num_points：add_constraint(expr_composite() >= B* − ε_l, meta=EPS)，
           solve，用 values 重算 composite 实际值；
        3. 每点经 SolutionDecoder 解码。"""
```

### 10.10 diagnostics 包

```python
@dataclass(frozen=True)
class DiagnosisReport:
    culprit_groups: tuple[str, ...]  # 使模型恢复可行的最小约束组集合
    detail: str

class InfeasibilityDiagnoser:
    def __init__(self, model: ArterialModel, backend: SolverBackend) -> None: ...
    def diagnose(self) -> DiagnosisReport:
        """按 ConstraintMeta.group 分组（C1–C11 / PLAN_HARD / USER），
        逐组替换为大罚软约束重解（7.4）；GridRunner 在 status=INFEASIBLE 时自动调用。"""
```

### 10.11 report 包

```python
@dataclass(frozen=True)
class ViolationRecord:
    constraint_id: str
    violation_s: float
    cost: float
    owner: str

@dataclass(frozen=True)
class SolutionRecord:
    plan_selection: dict[str, str]
    offsets_s: dict[str, float]
    adjustments_r: dict[tuple[str, Direction, int, Side], float]
    bands: tuple[ConstructedBand, ...]
    violations: tuple[ViolationRecord, ...]
    composite_s: float
    loss_s: float

class SolutionDecoder:
    """仅用 ArterialModel 访问器 + meta 解码，不触碰求解器对象。"""
    def __init__(self, model: ArterialModel) -> None: ...
    def decode(self, values: "np.ndarray") -> SolutionRecord: ...

class ReportBuilder:
    @staticmethod
    def build(grid_report: GridRunReport,
              pareto: list[list[ParetoPoint]]) -> dict:
        """装配 8.1 结果 JSON。"""

class TimeSpaceDiagram:
    @staticmethod
    def render(solution: SolutionRecord, problem: ProblemInput,
               ctx: GridPointContext, out_path: str) -> None:
        """8.2 时距图，matplotlib。"""
```

### 10.12 主流程编排

```python
def run(problem_json: dict) -> dict:
    problem = ProblemInput(**problem_json)
    issues = problem.validate_all()
    if any(i.level == "error" for i in issues):
        raise GWValidationError(issues)
    backend_factory = lambda: make_backend(problem.solver.backend)
    grid_report = GridRunner(problem, backend_factory, problem.solver).run()
    bases = grid_report.best(k=problem.objective.pareto.topk_grids)
    pareto = [EpsilonConstraintScanner(problem, ctx_of(r), backend_factory(),
                                       problem.objective.pareto).scan(r) for r in bases]
    return ReportBuilder.build(grid_report, pareto)
```

---

## 11. 求解器适配层

### 11.1 基线：scipy.optimize.milp

- 线性约束用 `scipy.sparse`（csr/coo）装配；`integrality` 数组标注二进制(1)/整数(2)；
- 已知限制：无指示约束（本设计全部 big-M，无影响）、**无 warm start**、HiGHS 选项暴露有限（time_limit、mip_rel_gap、presolve、disp 可用）；
- 每格点每次求解独立构建模型（阶段 2 的 ε 扫描在同一格点上复用模型、仅改目标与一条追加约束——适配层需提供 `add_constraint` / `set_objective` 增量接口，避免重复装配）。

### 11.2 预留：highspy

同一 HiGHS 求解器，但 API 完整：warm start（ε 扫描与格点间收益大）、回调、更多选项。接口与 `backend_base` 对齐后切换，建模层零改动。

## 12. 测试与验证策略

### 11.1 解析解回归（必须）

合成案例：等距路口（间距 s）、全相同单窗口方案（窗口 [a, b]）、固定 κ。经典 MAXBAND 情形有解析最优：

- 当 `s/v mod C = 0`：完美绿波，β* = 窗口宽度；
- 一般情形 β* 可手算。

断言 MILP 阶段 1 的 B* 与解析值一致（容差 1e-4）。在此基础上逐步叠加：双方向、多窗口、多带、红带、微调、软约束，每步一个回归案例。

### 11.2 不变式断言（性质测试）

随机生成合法输入，断言：

1. 经过滤后 MILP 恒可行（5.4）；
2. MILP 阶段 1 的 B* ≥ 贪心下界（6.1）；
3. 带区间在所有链上节点落在所用窗口内（解的构造性校验，独立于模型重算）；
4. 周期回绕正确性：把解中所有绝对时刻 mod C 后重算带宽，与 β 一致；
5. 红带 = 上游绿窗平移 ∩ 下游红窗，重叠宽度与 β 一致；
6. 双周期方案两个半周期端点一致。

### 11.3 求解正确性抽查

小规模案例（N ≤ 4）用暴力枚举（方案组合 × 偏置网格）验证 MILP 最优性。

## 13. 已知限制与扩展路线

| 项 | 状态 |
|---|---|
| 窗口独占轻度保守 | 已知，预留 `allow_window_sharing` 开关（4.7） |
| 双周期 | 经复制 + 一致性等式表达（3.3.5）；原生半周期建模为后续扩展 |
| warm start | scipy 不支持，预留 highspy 后端（11.2） |
| 多干线 / 网络 | 架构预留（带需求 schema 与干线解耦），本期不实现 |
| 排队 / 溢流 | 非目标（1.6） |
| 路段速度逐段独立调整 | 本期仅统一倍率 κ；逐段变量化会使 τ 成为变量（× 带时间 = 双线性），需另立项评估 |

---

## 附录 A：完整符号表

| 符号 | 类型 | 含义 |
|---|---|---|
| C, κ | 格点参数 | 公共周期（秒）、速度倍率 |
| τ^d_i, T^d_i | 常数 | 行程时间、累计行程时间（秒） |
| s̄, ē | 常数 | 窗口相对起止（比例） |
| x_{i,p} | 二进制 | 方案选择 |
| φ_i | 连续 | 周期起始偏置（秒） |
| δ, z | 连续 | 端点微调量（比例）、x·δ 线性化辅助 |
| ŝ, ê, h | 连续 | 有效窗口端点（秒）、槽位可用性 |
| u, β, e | 连续/二进制 | 带起点、宽度、存在性 |
| a, ag, ar | 二进制 | 绿带窗口槽指派、红带上游绿窗指派、红带下游红窗实例指派 |
| n | 整数 | 回绕周期数 |
| σ^m, σ_c | 连续 ≥ 0 | margin 短缺、软约束违反 |
| Β_b | 表达式 | 需求 b 的聚合带宽 Σ_q β_{b,q} |
| B* | 标量 | 阶段 1 最优 composite |

## 附录 B：最小输入示例

```json
{
  "corridor": {
    "corridor_id": "demo",
    "intersections": ["I1", "I2", "I3"],
    "segments": [
      {"from": "I1", "to": "I2", "distance_m": 500, "speed_up_mps": 11.1, "speed_down_mps": 11.1},
      {"from": "I2", "to": "I3", "distance_m": 500, "speed_up_mps": 11.1, "speed_down_mps": 11.1}
    ]
  },
  "plans": [
    {
      "id": "I1_P1", "intersection": "I1", "double_cycle": false,
      "windows": {"up": [[0.0, 0.45]], "down": [[0.5, 0.95]]},
      "adjustable": {"up": {"0": {"end": [-0.05, 0.05]}}, "down": {}},
      "hard_constraints": [
        {"id": "mg_up", "terms": [
          {"atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1", "direction": "up", "window": 0, "side": "end"}, "coef": 1.0},
          {"atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1", "direction": "up", "window": 0, "side": "start"}, "coef": -1.0}
        ], "sense": ">=", "rhs": 15.0, "hard": true}
      ],
      "soft_constraints": [
        {"id": "split_up", "terms": [
          {"atom": {"kind": "endpoint", "intersection": "I1", "plan": "I1_P1", "direction": "up", "window": 0, "side": "end"}, "coef": 1.0}
        ], "sense": "<=", "rhs_ratio": 0.42, "hard": false, "soft": {"coef": 2.0, "owner": "intersection_loss"}}
      ]
    }
  ],
  "objective": {
    "band_demands": [
      {"id": "GW_up", "type": "green", "direction": "up", "nodes": ["I1", "I2", "I3"],
       "max_bands": null, "weight": 1.0, "margin": {"delta_min_s": 2.0, "coef": 1.0}}
    ],
    "margin_default": {"delta_min_s": 2.0, "coef": 1.0},
    "pareto": {"num_points": 4, "relax_max": 0.3, "topk_grids": 1}
  },
  "grid": {"cycles_s": [80, 90, 100], "speed_ratios": [1.0]},
  "solver": {"backend": "scipy", "time_limit_s": 120, "mip_rel_gap": 0.0, "num_workers": 3,
             "heuristic": {"enabled": true, "refine_rounds": 3}}
}
```

注：`rhs_ratio` 为便捷字段：表示"rhs = 比例 × 当前周期"，预处理时换算为秒。

## 附录 C：合成测试案例清单

| 编号 | 案例 | 预期 |
|---|---|---|
| T1 | 3 路口等距 500m、v=40km/h、C=90s、单窗口 [0,0.45] | 完美绿波，β* = 0.45C = 40.5s |
| T2 | 同 T1 但 C=80s | β* 手算值（行程时间 45s mod 80 的余量决定） |
| T3 | 双向绿波，上下行冲突结构 | 验证方案选择与偏置权衡 |
| T4 | 双窗口方案 + 2 带槽 | 验证指派与窗口独占 |
| T5 | 红波相邻对 × 3 | 验证红窗补集（含跨 0/C 段）与重叠计算 |
| T6 | 可调端点 + 软约束 | 验证微调、路口损失与 ε 扫描 |
| T7 | 某方案在 C=60s 违反 min green | 验证过滤剔除 + 诊断记录 |
| T8 | 双周期方案 | 验证复制窗口与一致性等式 |

---

*文档结束。评审通过后按 §9 模块划分进入开发。*
