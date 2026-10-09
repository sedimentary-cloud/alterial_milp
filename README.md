# 干线绿波 / 红波协调优化 MILP 算法包

> 完整配置字段、物理意义与数学推导见 [docs/配置手册与数学原理.md](docs/配置手册与数学原理.md)。
> 想从零搞懂建模的每一行（变量、约束、放松、窗口选择）看 [docs/MILP建模教学-绿波篇.md](docs/MILP建模教学-绿波篇.md)，其中的手算案例可用 `python examples/tutorial_small_cases.py` 复现。

本目录是根据《干线绿波红波协调优化 MILP 设计文档》实现的 solver-independent 建模层 +
scipy / highspy 求解适配层。运行环境为 Windows 侧 conda 环境 `artery_milp` 对应的
Linux prefix（WSL 下使用 `python3.11`）。

## 安装依赖

环境已包含：

- Python 3.11
- numpy
- scipy（`scipy.optimize.milp`，HiGHS 基线）
- matplotlib
- highspy（可选后端）

## 输入要点（v2：相位边界建模）

- 路段长度支持统一值 `distance_m`，也支持分方向覆盖：`distance_up_m` / `distance_down_m`；
  某方向未单独给出时回退到 `distance_m`。
- 上下行设计速度继续使用 `speed_up_mps` / `speed_down_mps`。
- 每个 `plan.directions[d]` 声明一串有序相位边界 `boundaries: [{pos_r, adjust}]`
  和等长的 `green_mask`；绿灯窗口/红窗是**派生量**。
- `green_mask` 长度 = 边界数 + 1；一个方向必须同时有绿区间和红区间；
  首尾区间不能同时为绿（用户需要自己轮转输入，让完整绿区间落在一个周期内）。
- 双周期方案在半周期 `(0, 0.5)` 内声明边界，0.5 是强制边界；复制窗口复用同一组 δ。
- `min_phase_s` 默认 `1e-3`，是吸收 MILP 容差的数值保护；业务最小相位请显式配置。

## 快速开始

```bash
python -m greenwave examples/minimal_problem.json --out examples/minimal_result.json \
    --diagram-dir examples/diagrams
```

Python API：

```python
import json
from greenwave import run

problem = json.load(open("examples/minimal_problem.json", encoding="utf-8"))
result = run(problem)
print(result["grid_results"])
print(result["pareto"]["points"][0]["solution"]["bands"])
```

### 多阶段快速优化（可选）

原完整 MILP 接口保持不变。若需要更快的工程启发式，可显式调用：

```python
from greenwave import run_multistage

result = run_multistage(problem, max_rounds=3, restarts=2)
```

或在问题 JSON 中开启：

```json
{
  "solver": {
    "multistage": {
      "enabled": true,
      "max_rounds": 3,
      "restarts": 2,
      "stage1_time_limit_s": 30.0,
      "stage2_time_limit_s": 10.0,
      "fallback_to_full_milp": true,
      "decompose_stage1": true,
      "decompose_min_component_size": 2,
      "stage1_symmetry_breaking": true,
      "stage1_min_band_width_s": 1.0
    }
  }
}
```

多阶段算法先固定相位边界重选组合变量，再固定组合用 LP 调整 `φ/δ/u/β`，
交替迭代；它是启发式，不保证全局最优。找不到可行解时默认回退完整 MILP。

#### Stage-1 语义子问题分解

`decompose_stage1`（默认开）在 Stage-1 内部按**交通语义**把问题切成互不耦合的
子问题，再各自求解并拼回整条干线。条件是：

1. 优化目标都是"局部"的带——每条需求只跨少数几个连续路口（相邻对窗口绿波 /
   窗口红波是典型形态），没有横跨多个连通分量的带；
2. 跨带的约束不连接不同子问题：`global_constraints`、`balanced_groups`、
   `constraint_templates`、方案自带约束里如果引用了不同组的需求或 φ，这些组会
   被并回一组；
3. 子问题之间共享的路口是多方案路口时同样并组（那时窗口可用性 `h` 和方案选择
   `x` 被多条带共用）。

原理：Stage-1 里 δ 已固定，模型里唯一被多条带共用的自由变量是路口偏置 `φ`，
而 `φ` 在约束里只以 `u - φ` 出现——它只是"公共平移"（gauge）。因此按共享路口
把需求分组、各组独立求解、再把相邻组的相对偏置串起来，期望得到**精确**分解
（全局最优 = Σ 组最优）。

但"gauge 平移一定可以吸收"并不总成立，为此做了两件事：

1. **把红带模型补完整**：红带的上游包含式原来只有 `u - τ - φ_up`，缺一个周期下标，
   `u` 一旦被平移推出 `[0, C]` 就表示不出原来的摆放（表现是"把整组做时间平移后最优值
   会掉"）。现在给红带加了与绿带 `n` 对称的整数下标 `m`：
   `s_g ≤ u - τ - φ_up - m·C ≤ e_g`（范围按 `φ∈[0,C]`、`τ`、窗口位置推出）。
   修完后红带模型和绿带一样对时间平移**严格不变**，而且整模本身也变强——
   `up_green_down_red`（1 条跨全线绿波 + 5 条下行红波）的目标值从 280.0 提升到
   **296.76（+6%）**：以前有一部分物理可行的红带摆放模型根本表示不出来。
2. **两道关卡**：分解出来的解除了要过"代回整模逐行校验"，还要过**最优性证书**——
   各组再自由（不钉 φ）求一次得到上界 `Σ f_g^free`，拼装解必须达到它，否则放弃、
   回退整模 MIP。所以分解**要么给出经证明的全局最优，要么回退**。

代价：红带多了几个整数下标变量，红波类目标（如 `up_green_down_red`）的单次整模
由约 2.7s 变成约 10.5s（把 `m` 固定到最优值后只要 0.67s，说明耗时全在搜索这几个
整数）；绿波类目标不受影响。后续可把 `m` 改成"和下游 `(r,k)` 一样枚举进 `ag`
赋值"的纯 0/1 编法来回收这部分时间。

实测（`examples/standard_problem.json` 的"双周期 mb=2 × 双向窗口绿波"，
6 路口 / 10 条相邻对绿波）：Stage-1 从 8.4s+5.0s 降到 0.18s+0.23s，
多阶段总耗时 13.5s → 0.45s，目标值不变。

红波场景：如果把接口 φ 钉住会丢自由度（`_certify` 判定失败），分解自动回退整模
（实测"上绿 + 下红 + 多方案"的案例回退后与整模、与 32 个方案指派的暴力枚举三方
一致 = 362.0）。所以目前**可分解的收益主要落在绿波（窗口绿波）目标上**。

`decompose_min_component_size` 仍然表示"至少要有两个规模达标的组才值得分解"，
这里的"规模"指子问题自由变量个数。

#### 方案枚举 + 链式 DP（多方案路口）

多方案路口会让上面这条规则把组并回去（`x` 与窗口可用性 `h` 被多条带同时引用），
于是分解失效。但**把方案固定下来之后**，每个路口的窗口就是常数，耦合消失，分解
依然精确。所以 Stage-1 还带一条 "方案枚举 + 链式 DP" 路径（`decompose_dp`，默认开）：

- 忽略"多方案并组"重新分组（`plan_dp_candidate`），要求组数 ≥2 且构成链
  （相邻组只共享 1 个路口）；
- 对每个组枚举它覆盖路口的方案组合，得到 `f_g(lo方案, hi方案)`；
- 沿走廊做链式 DP：`全局最优 = max_π Σ_g f_g`，状态是分隔路口的方案；
- 接口路口的 φ 用"从左到右链式固定"对齐（与普通分解一致），最后按最优指派重解
  各组并拼装、回填整模逐行校验。子问题按"钉住的 φ 范围"推导可行窗口键，可能比
  整模少几个键（例如红波的 `rw_down_win[...]` 赋值变量），拼装时把这些键置 0，再交给逐行
  校验兜底——校验不过就回退整模，所以不会放过错误解；这一步收掉了
  "双周期 5 方案 × 双向红波"这类案例上 DP 因缺键而放弃的情形
  （该案例从 20s 未证最优 → 2.98s 已证最优）。

代价从 `Π_i |P_i|`（暴力方案枚举）降到 `Σ_g Π_{i∈g}|P_i|`（子问题求解次数）。

收益判定器（避免"DP 比整模还慢"）：

1. 先跑 DP 的第一个子问题，实测单次耗时 → 估 `t_dp = 求解次数 × t_sub`；
2. `t_dp ≤ decompose_dp_probe_s`（默认 1.0s）→ 直接做 DP；
3. 否则先给整模 `decompose_dp_probe_s` 的短预算——它若在预算内证到 optimal 就用
   整模，否则用 DP 兜底（DP 给的是精确最优，也可以当超时兜底）；
4. `decompose_dp_max_solves`（默认 400）限制子问题求解次数，超过就放弃 DP。

实测（`examples/standard_problem.json` 改成每路口多方案、边界固定）：

| 案例 | 暴力方案空间 | 整模 | 枚举+DP | 加速 |
|---|---|---|---|---|
| 2 方案/路口（20 次子求解） | 64 | 3.86s | **1.09s** | 3.5× |
| 3 方案/路口（39 次子求解） | 729 | 46.16s | **1.86s** | 24.9× |

两种情况下目标值与整模**完全一致**（差异在数值容差内）。

## 模块结构

```text
greenwave/
├── schema/          # 不可变输入模型 + 静态校验 V1–V10
├── preprocess/      # 格点常量化 + 方案可行性 LP 过滤
├── heuristic/       # 贪心初始解 / 块坐标 refinement
├── model/           # 变量注册表、派生窗口、C1–C13、目标装配（变量名统一由 naming.py 生成）
├── solve/           # scipy 后端（基线）与 highspy 后端
├── grid/            # 多格点并行遍历
├── pareto/          # 两阶段 epsilon-constraint
├── diagnostics/     # 分块松弛不可行诊断
└── report/          # 结果 JSON + 时距图
tests/
└── test_document_cases.py   # T1–T8 与性质回归
```

## 标准综合案例

问题定义 JSON：`examples/standard_problem.json`。

`run_standard.py` 定义了一个 6 路口、140s 周期的综合案例：

- 上行全局绿波 + 下行 5 个相邻对红波；
- 方案硬约束、方案软约束（路口损失）；
- 绿波带 margin 短缺罚；
- 全局硬约束（上行带宽 >= 15s）；
- 全局软约束（上行带宽尽量 >= 20s）；
- 中间路口可调绿灯结束点，首末路口只调周期偏置；
- I2/I3/I4 的上下行绿灯必须同时结束。

运行：

```bash
python run_standard.py
```

输出位于 `examples/standard_output/`。

## 时距图

本仓库内置一组示例时距图，生成脚本：

```bash
python examples/generate_diagrams.py
```

## 均衡组目标

除了“带宽加权求和最大”，现在还支持 `balanced_groups`：
让组内需求总带宽的最小值尽量大。

示例：

```json
"balanced_groups": [
  {
    "id": "balance_up_down",
    "demands": ["GW_up", "GW_down"],
    "weight": 5.0,
    "on_infeasible": "zero",
    "targets_s": {"GW_up": 30.0, "GW_down": 20.0},
    "min_existing": 2,
    "min_existing_penalty_s": 0.5
  }
]
```

`on_infeasible` 可选：

- `zero`：任一需求缺失则 z=0，默认严格模式；
- `skip`：跳过缺失需求，只对存在的需求取最小，建议配合 `min_existing` 软约束。

`targets_s` 用于归一化均衡，例如上行 30s、下行 20s 的 3:2 目标。

完整均衡组示例：`examples/balanced_group_example.json`。

## 性能测试（全矩阵）

`examples/benchmark_matrix.py` 对标准案例跑一张完整矩阵，每个案例都对比**整模 vs 分阶段**：

| 维度 | 取值 |
|---|---|
| 分组 | ① 无双周期·单方案 ② 无双周期·多方案 ③ 双周期·单方案 ④ 双周期·多方案 |
| 方案数/路口 | 单方案组 1；多方案组 2 / 3 / 4 / 5 |
| 相位边界 | 固定（1.1–1.4 与 2.x 对应项）· 可调（1.5–1.8 对应项） |
| 目标 | 上下全局绿波带宽和 · 全局均衡组×1+带宽和×0.5 · 双向窗口绿波（逐对）· 上绿逐对+下红逐对 · 双向红波（逐对）· 上全局绿+下红逐对 |

合计 4 组 × 6 目标 × 2 种边界模式 × 各自方案数 = 120 个案例；每案例每种方法跑 2 次，
单次求解上限 20s（命中上限记为"未证最优"，同时记录当时的可行解与对偶界）。

对比口径：

- **固定相位边界**：δ≡0，没有 stage-2 可言。"整模" =
  `solve_fixed_boundaries(decompose=False)`，"分阶段" =
  `solve_fixed_boundaries(decompose=True)`（先试语义分解 / 方案枚举 DP，不行才回退整模）。
- **可调相位边界**：整模 = 一次性完整 MILP（δ 与组合变量一起解）；分阶段 =
  `solve_multistage` 全链路（stage-1 组合 → stage-2 LP 连续优化）。

```bash
python examples/benchmark_matrix.py                    # 跑全矩阵
python examples/benchmark_matrix.py --limit 4          # 先试 4 个案例
python examples/benchmark_matrix.py --groups G1,G3 --reps 1
python examples/benchmark_matrix.py --no-diagrams      # 不画时空图（默认画）
python examples/benchmark_matrix.py --resume           # 断点续跑（口径变了会自动重跑）
python examples/benchmark_matrix.py --html-only        # 只重生成报告
```

输出：

- `examples/bench_output/bench_matrix.json`：逐次求解的原始记录（墙钟/求解器时间、状态、
  目标值、对偶界与相对 gap、实际走的路径、模型规模、语义组与 DP 候选等结构信息）；
- `examples/bench_output/bench_matrix.html`：自包含报告，浏览器直接打开——矩阵表
  （整模/分阶段时间 + 加速比 + 目标值 + 路径标记）、log-log 散点、可筛选明细表、口径说明；
  明细表里的"时空图"列可直接点开对应图片；
- **时空图改成"行内渲染"，不再输出 PNG 文件**：每个案例每种方法的解
  （φ、各相位边界微调 δ、选中方案、各条波带）都内嵌在 JSON/HTML 里，
  在报告明细表的「时空图」列点 <b>整模 / 分阶段</b> 即可就地渲染
  （用该方法多次重复里目标值最好的那次解）。
  这样省掉了原先 120 个目录 / 240 张 PNG（约 **47 MB**）。
  想仍然落盘成 PNG 就加 `--png-diagrams`（配 `--diagram-dir` 改目录名），
  `--no-diagrams` 则完全不内嵌时距图数据。

> 时空图的浏览器渲染器与格点搜索报告**共用一份**（`examples/ts_canvas.py`），
> 画法与库里的 `TimeSpaceDiagram` 一致：红/绿灯条按周期平铺、绿波带按 `u+cum_T(κ)` 连线、
> **红波带**用与库函数 `_red_band_display_times` 同一套"上下游展示副本"搜索
> （已逐条对账：最大偏差 `0.00e+00`）。多方案案例只画被选中的那个方案的相位。
>
> `--png-diagrams` 走 matplotlib 落盘，那条路径的标题默认用 ASCII
> （本机没装"中英文全覆盖"的字体，matplotlib 不做逐字回退，纯 CJK 回退字体会让
> 拉丁字母变方框）；装了 Noto Sans CJK / 微软雅黑 之类完整字体时会自动切成中文标题。

### 精细格点搜索（C × κ 热力图）

`examples/benchmark_grid_search.py` 针对**一个固定配置**扫公共周期 `C` 与速度倍率 `κ`
的二维网格，只跑**整模**，并把结果渲染成**可交互** HTML 热力图：

- 固定案例：标准案例 · **无双周期 · 单方案 · 可调相位边界**，
  目标 **上下全局带宽均衡组×1.0 + 上下绿波带宽和×0.5**（`composite = z + 0.5(β↑+β↓)`）；
- 默认网格 `C ∈ [60,180]s × κ ∈ [0.3,1.5]`，**58×44 = 2552 个格点**，实测 **87s** 跑完
  （单格点 ~35 ms，含建模）。步长由 `--step-scale`（默认 **0.75**）决定：
  C 步长 2.105s、κ 步长 0.0279，都是旧默认（33 点）步长的 3/4；
  也可以 `--cells N` 固定成 N×N，或 `--cycle-step/--kappa-step` 直接给步长；
- **指标按周期归一化**（与库里的 matplotlib 版 `GridHeatmapDiagram` 同口径）：
  **打开就是归一化热力图**——下拉框默认选中 `composite_objective_ratio`，
  而 canvas 的初始指标直接读该选中项（同一事实源，不会出现"下拉框显示归一化、
  热力图却画绝对秒"的不一致），色标值域也随之取归一化指标的 min/max。
  `*_ratio` = 秒值 ÷ 该格点的 C（周期比例，无量纲），损失取负号 `negative_*`
  使"越大越好"统一；绝对秒指标放在下拉框的"绝对秒"分组，`wall_s` 是时间开销不归一化；
- 输出 `examples/bench_output/grid_search.json` + `grid_search.html`：
  指标下拉（归一组 + 绝对秒组）、
  三种配色、悬停看数值、点击钉住看完整记录、★ 跳到最优、Top-15 表；
- **点击任意格点就地渲染该 (C, κ) 解的时距图**（纯 canvas，不生成 PNG 文件）：
  每点的 φ、各相位边界微调 δ、选中方案、各条波带的 u/β 都内嵌在 HTML 里，
  渲染器与 benchmark 报告共用（`examples/ts_canvas.py`，含红波带画法），
  可勾选灯条 / 波带 / 周期分界，也可一键下载当前时距图 PNG；
- **★ / "跳到最优" / Top-15 都按归一化目标 `composite / C` 判定**（不是绝对秒）。

```bash
python examples/benchmark_grid_search.py                 # 33×33
python examples/benchmark_grid_search.py --cells 41      # 更细
python examples/benchmark_grid_search.py --workers 4     # 并行（会干扰耗时指标）
python examples/benchmark_grid_search.py --html-only     # 只重画 HTML
```

实测地形（2552 点全部可行且 optimal）：

- **两个口径给出不同的最优点**（这正是归一化的意义）：
  归一化口径最优在 `C≈83.2s, κ≈0.356`（`composite/C = 0.600`，即带宽占到周期的 60%）；
  绝对秒口径最优在 `C≈177.9s, κ=0.300`（`composite = 92.48s`，但 `composite/C` 只有 0.520）
  —— 绝对秒会把最优点推向大周期、并顶在 κ 下边界上；
- 最优点总是 `β↑ = β↓ = z`（均衡组目标导致两方向带宽相等）；
- 存在一条明显的**对角暗带 `κ·C ≈ 80.8`**（等价于"全线行程 ≈ 1.13 个周期"）：
  两侧都更亮，`span/C ≈ 1.7` 与 `≈ 3.1` 是好区域；
- 若关心**绝对秒**口径，需要把 κ 往下扩（现在的最优点在 κ=0.3 的边界上）；
  归一化口径的最优点在区间内部，不需要扩。

配置改造函数统一放在 `examples/bench_configs.py`（目标定义、单/双周期改造、多方案克隆、
自适应收缩的方案平移量），报告渲染在 `examples/bench_report.py`。

#### 实测（120 案例全矩阵，C=140s、κ=1、上限 20s、每例 2 次的中位数）

| 分组 | 边界 | 整模中位 | 分阶段中位 | 加速中位 | ≥1.25× | <0.8× | 整模触上限 |
|---|---|---|---|---|---|---|---|
| ① 无双周期·单方案 | 固定 | 0.128s | 0.047s | **2.04×** | 4/6 | 0 | 0 |
| ① 无双周期·单方案 | 可调 | 0.440s | 0.236s | 1.66× | 3/6 | 3 | 0 |
| ② 无双周期·多方案 | 固定 | 0.432s | 0.259s | 1.03× | 9/24 | 0 | 0 |
| ② 无双周期·多方案 | 可调 | 0.680s | 0.804s | 1.24× | 12/24 | 7 | 0 |
| ③ 双周期·单方案 | 固定 | 6.686s | 0.329s | **16.76×** | 3/6 | 0 | 1 |
| ③ 双周期·单方案 | 可调 | 5.504s | 0.953s | 2.89× | 3/6 | 2 | 2 |
| ④ 双周期·多方案 | 固定 | 7.600s | 1.408s | 1.84× | 12/24 | 0 | 8 |
| ④ 双周期·多方案 | 可调 | 17.156s | 3.389s | 2.08× | 14/24 | 2 | 11 |

按目标（跨方案数、跨边界合并）：

| 目标 | 加速中位 | 整模中位 | 分阶段中位 | 整模触上限 |
|---|---|---|---|---|
| 上下全局绿波带宽和 | 0.99× | 0.125s | 0.130s | 0 |
| 全局均衡组×1 + 带宽和×0.5 | 0.98× | 0.127s | 0.140s | 0 |
| 双向窗口绿波（逐对） | 2.58× | 3.440s | 0.958s | 3 |
| 上绿（逐对）+ 下红（逐对） | **5.36×** | 11.357s | 1.138s | 10 |
| 双向红波（逐对） | **5.19×** | 9.945s | 0.889s | 9 |
| 上全局绿 + 下红（逐对） | 1.00× | 1.592s | 1.837s | 0 |

三条结论：

1. **可分解的"逐对/局部"目标收益最大**：双周期单方案下语义分解把整模
   6.7s→0.33s（中位 16.8×，单例最高 62×）；双周期多方案下方案枚举 DP 把
   20s 证不动的案例变成 0.7–3.6s 的**已证最优**。
2. **跨全线的全局目标（上下全局绿、上全局绿+下红）不可分解**：固定边界下分阶段
   与整模走同一条路（1.0× 属噪声）；可调边界下分阶段是多阶段启发式，比一次性
   整模慢（0.22–0.99×）且目标值可能更低（实测最多 −1.4%）。全局目标该用整模。
3. **分解的正确定量**：120 个案例里，整模有 44 次撞 20s 上限（22 个案例），
   分阶段 **0 次**；在整模已证最优的 51 个固定边界案例上，分阶段目标值最大偏差
   **3e-6**（求解器容差级）。反过来，整模 20s 内没能证明最优的 9 个固定边界案例里，
   分阶段给出的都是**已证最优**（其中 1 例比整模 20s 的可行解还好 0.667）。

## 时距图读法

- 横轴：时间（秒）；纵轴：沿干线累计距离（米，向上递增）。
- 每个路口画两条紧贴在一起的不透明水平灯条：路口线以上为上行，以下为下行；
  上行绿灯窗口为绿色，下行绿灯窗口为湖蓝色，红灯时段统一为红色；均按周期重复绘制。
- 绿波带为斜向带：上行带从左下向右上倾斜并绘制在上行灯条中心线上；
  下行带从左上向右下倾斜并绘制在下行灯条中心线上；带子按周期重复绘制以铺满可见时间范围。
- 红波带为相邻路口之间的红色斜向重叠区，斜率由该路段行程时间决定。
- 上行/下行绿波带使用不同颜色和 hatch 纹理，便于区分。

输出位于 `examples/diagrams/`，包含：

- 单向完美绿波
- 双向绿波
- 多窗口多带槽
- 相邻对红波
- 双周期方案
- 上下行不等长路段
- 上行绿波 + 下行红波
- Pareto 前沿（可调节端点 + 路口损失软约束）

Pareto 前沿文件为 `examples/diagrams/08_pareto_frontier.png`，单独生成：

```bash
python examples/generate_pareto.py
```


## 运行测试

```bash
cd tests && python -m unittest test_multistage test_document_cases \
    test_phase1_schema test_phase2_feasibility test_phase3_model \
    test_phase4_semantic test_phase5_output test_semantic test_sharing
```

## 设计要点

- 建模层不依赖 scipy 求解 API；`model/` 只输出变量表、稀疏约束、目标向量。
- 相位边界是决策变量；绿灯窗口由 `boundaries + green_mask` 派生，窗口端点是
  边界 δ 的线性表达式。
- 双线性项 `z = x·δ` 用 McCormick 线性化。
- 红窗是整周期绿窗的补集，plan-native 编号 `rid = 1..K`；`rid=K` 是循环意义上
  最后一段红，可能跨 0/C。
- 红波带只能存在于相邻路口间。
- 周期 `C` 与速度倍率 `κ` 格点化；阶段 1 最大化 composite，阶段 2 用
  ε-constraint 扫描 Pareto 前沿。
