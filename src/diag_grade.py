"""分级公式核验：打印"风险 × 置信度 × 面积"三维下的得分与等级矩阵。

用途：确认分级公式具备足够梯度（不会出现"只有 A 和 D"这类退化），
并作为设计说明书中分级标定表的生成依据。

用法：
    python -m src.diag_grade
"""
from __future__ import annotations

import argparse
from collections import Counter

from .infer import severity_score
from .utils import load_config, severity_grade


def main() -> None:
    ap = argparse.ArgumentParser(description="严重度分级公式核验")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sv = cfg["severity"]

    print("=" * 76)
    print("分级配置")
    print(f"  权重      : {sv['weights']}")
    print(f"  风险基线分: {sv['risk_score']}")
    print(f"  面积 ref={sv.get('area_ref')} transform={sv.get('area_transform')} cap={sv.get('area_cap')}")
    print("  等级区间  : " + "  ".join(f"{g['level']}≤{g['max']}" for g in sv["grades"]))
    print("=" * 76)

    print(f"{'风险':<8}{'置信度':<9}{'面积占比':<11}{'得分':<9}{'等级':<6}{'处置'}")
    print("-" * 76)
    counter: Counter = Counter()
    for risk in ("high", "mid", "low"):
        for conf in (0.60, 0.80, 0.95):
            for area in (0.003, 0.01, 0.03, 0.08, 0.20, 0.40):
                s = severity_score(cfg, risk, conf, area)
                g = severity_grade(cfg, s)
                counter[g["level"]] += 1
                print(f"{risk:<8}{conf:<9}{area:<11}{s:<9.1f}{g['level']:<6}{g['action']}")
    print("-" * 76)
    print("等级覆盖统计（理论网格）：" + "  ".join(
        f"{lv}: {counter.get(lv, 0)}" for lv in ("A", "B", "C", "D", "E")))
    print("=" * 76)

    # 洁净样本（无检出）单独说明
    print("说明：洁净样本不产生检出，代码路径直接判定 A 级（合格/放行），不参与上式。")


if __name__ == "__main__":
    main()
