"""Compare adjustable phase boundaries vs fixed boundaries + offset-only.

Runs the same 3x3 standard-case experiment twice:
  A. baseline: boundaries may adjust (delta variables exist)
  B. fixed   : all boundary adjusts removed, only phi offsets are free
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import benchmark_configs as bc  # noqa: E402


def main() -> None:
    rows = []
    for factor in bc.FACTORS:
        for objective in bc.OBJECTIVES:
            base = bc._run_one(objective, factor, fixed_boundaries=False)
            fixed = bc._run_one(objective, factor, fixed_boundaries=True)
            rows.append((factor, objective, base, fixed))
            print(f"done {factor:11s} {objective:16s} "
                  f"adjust={base.get('solve_s', 0):.3f}s "
                  f"fixed={fixed.get('solve_s', 0):.3f}s")

    hdr = (f"{'双周期配置':<12}{'目标':<16}"
           f"{'可调求解(s)':>12}{'固定求解(s)':>12}{'倍数':>8}"
           f"{'可调二进制':>10}{'固定二进制':>10}"
           f"{'可调约束':>9}{'固定约束':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for factor, objective, base, fixed in rows:
        a = base.get("solve_s", float("nan"))
        b = fixed.get("solve_s", float("nan"))
        ratio = a / b if b and b > 0 else float("nan")
        print(f"{bc.FACTOR_LABEL[factor]:<12}{bc.OBJECTIVE_LABEL[objective]:<16}"
              f"{a:>12.3f}{b:>12.3f}{ratio:>8.2f}"
              f"{base.get('binary', 0):>10}{fixed.get('binary', 0):>10}"
              f"{base.get('cons', 0):>9}{fixed.get('cons', 0):>9}")

    print("\n最耗时（固定边界版本）Top 3：")
    for factor, objective, base, fixed in sorted(
            rows, key=lambda r: r[3].get("solve_s", -1), reverse=True)[:3]:
        print(f"  {bc.FACTOR_LABEL[factor]} + {bc.OBJECTIVE_LABEL[objective]} "
              f": {fixed.get('solve_s', 0):.3f}s "
              f"(binary={fixed.get('binary', 0)})")


if __name__ == "__main__":
    main()
