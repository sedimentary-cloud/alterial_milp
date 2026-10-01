# 干线绿波 / 红波协调优化 MILP 算法包

> 完整配置字段、物理意义与数学推导见 [docs/配置手册与数学原理.md](docs/配置手册与数学原理.md)。

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

## 模块结构

```text
greenwave/
├── schema/          # 不可变输入模型 + 静态校验 V1–V10
├── preprocess/      # 格点常量化 + 方案可行性 LP 过滤
├── heuristic/       # 贪心初始解 / 块坐标 refinement
├── model/           # 变量注册表、派生窗口、C1–C13、目标装配
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

## 性能测试

`examples/benchmark_grid.py` 会分别用 `num_workers=1/2/3` 跑 4×3 格点，
输出单格点耗时、总墙钟时间和并行加速比。

`examples/profile_pipeline.py` 会拆解单格点的各个步骤耗时：
解析、校验、预处理、贪心、建模、求解、解码。

```bash
python examples/benchmark_grid.py
```

## 格点搜索热力图

`examples/plot_grid_heatmaps.py` 演示周期 C × 速度倍率 κ 的格点搜索，
并生成 2×2 布局的四张热力图：

- Bandwidth objective / C
- Negative band-margin loss / C
- Negative intersection loss / C
- Composite objective / C

四个指标都除以周期 C，变成无量纲的“周期比例”，可跨周期比较。
色阶统一为淡绿（好）→ 白色（中等）→ 淡红（不好），
并按每个指标自身的实际最小/最大值自适应。

不可行或被预处理跳过的格点显示为灰色；密集格点下格子内数字字号会自适应缩小，
小到不可读时自动省略数字，只保留颜色信息。

运行：

```bash
python examples/plot_grid_heatmaps.py
```

输出：`examples/diagrams/09_grid_heatmaps.png`。

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
python -m unittest discover -s tests -v
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
