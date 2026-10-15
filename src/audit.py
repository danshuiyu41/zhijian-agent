"""项目成熟度自评：量化"距离真实可用还差多少"。

动机：本项目的指标是在与训练同源的合成数据上测得的，属于闭环自证。
本脚本把这类风险显式列出并给出量化依据，避免自评时自我感觉良好。

用法：
    python -m src.audit
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(p: str) -> str:
    f = ROOT / p
    return f.read_text(encoding="utf-8", errors="ignore") if f.exists() else ""


def main() -> None:
    print("=" * 78)
    print("项目成熟度自评（含风险项量化）")
    print("=" * 78)

    # ---- 代码与文档规模 ----
    src = list((ROOT / "src").glob("*.py"))
    src_lines = sum(len(read(f"src/{f.name}").splitlines()) for f in src)
    docs = list((ROOT / "docs").glob("*.md"))
    doc_lines = sum(len(read(f"docs/{f.name}").splitlines()) for f in docs)
    web = read("web/index.html")
    css = read("web/design-system.css")
    print("\n【规模】")
    print(f"  后端模块 {len(src)} 个，{src_lines} 行代码")
    print(f"  文档     {len(docs)} 份，{doc_lines} 行")
    print(f"  前端     HTML {len(web.splitlines())} 行 + CSS {len(css.splitlines())} 行")

    # ---- 指标 ----
    m = json.loads(read("runs/metrics.json") or "{}")
    if m:
        t = m.get("test", {})
        print("\n【实测指标】")
        print(f"  测试集准确率 {t.get('accuracy')}　宏平均 F1 {t.get('macro_f1')}　ECE {t.get('ece')}")
        print(f"  参数量 {m.get('num_params_m')} M　设备 {m.get('device')}")

    rob = json.loads(read("runs/robustness.json") or "{}")
    if rob:
        print("\n【鲁棒性（受控退化）】")
        for k, v in rob.get("levels", {}).items():
            print(f"  {k:<24} 漏检率 {v.get('miss_rate')}　缺陷分类准确率 {v.get('defect_class_accuracy')}")

    # ---- 知识库 ----
    kb = json.loads(read("knowledge/quality_kb.json") or "{}")
    docs_kb = kb.get("docs", [])
    diag = [d for d in docs_kb if d["meta"].get("type") == "排查顺序"]
    print("\n【知识库】")
    print(f"  条目 {len(docs_kb)} 条（其中诊断型 {len(diag)} 条）")

    # ---- 未知缺陷识别效果 ----
    ood = json.loads(read("runs/ood_eval.json") or "{}")
    if ood:
        print("\n【未知缺陷识别（开集识别）】")
        print(f"  特征库 {ood.get('known_samples')} 条样本参与评估，"
              f"阈值 {ood.get('recommended_threshold')}")
        print(f"  已知类通过率 {ood.get('recommended_known_pass_rate')}"
              f"　未知类拦截率 {ood.get('recommended_unknown_reject_rate')}"
              f"　距离分离度 {ood.get('separation')} 倍")

    # ---- 风险清单 ----
    print("\n" + "=" * 78)
    print("风险清单（按严重程度排序）")
    print("=" * 78)
    risks = [
        ("严重", "测试集与训练集同源",
         "随机种子同为 42、由同一生成器产生，100% 准确率只证明"
         "「学会了模拟器」，不能证明能检真实缺陷。评委一问数据来源即暴露。"),
        ("高", "未知缺陷识别能力有限（已实现，但只是安全阀）",
         "已加开集识别：特征 k-NN 距离超阈值即告警。实测在"
         "「已知通过率≥95%」前提下，只能拦住约一半的合成未知样本。"
         "更根本的困难是滑窗机制：一张图有几十个窗口，"
         "只要有一个窗口离已知簇近，整图就会被判为已知。"),
        ("高", "知识库阈值无现场标定",
         "排查条目里的「20 摄氏度」「45% 阈值」等数字是通用经验值，"
         "未在具体产线标定过，严格说只能算示例。"),
        ("中", "缺陷定位为窗口级精度",
         "滑窗分类给出的是「窗口内有该缺陷」，无法给出缺陷的实际边界与尺寸，"
         "对需要测长的场景（如划伤深度、长度判定）不够用。"),
        ("中", "无真实产线数据验证",
         "所有指标均来自合成数据，未在任何真实工件上做过盲测。"),
        ("低", "光照与成像条件未纳入考量",
         "未针对具体产线做光源与相机标定，实际部署需现场调试。"),
    ]
    for i, (lvl, name, desc) in enumerate(risks, 1):
        print(f"\n  {i}. [{lvl}] {name}")
        for line in re.findall(r".{1,34}", desc):
            print(f"       {line}")

    print("\n" + "=" * 78)
    print("可改进项（按性价比排序，均可落地）")
    print("=" * 78)
    todos = [
        ("窗口聚合策略（改进未知缺陷识别）", "高", "中",
         "当前取整图窗口距离的最小值，只要一个窗口像已知类就放行。"
         "改为按窗口距离的稳健分位数（如中位数或 80 分位）聚合，"
         "可显著提升整图级的未知拦截率。"),
        ("测试集按生成参数切分", "高", "中",
         "让训练/测试使用不同的生成参数（尺度、对比度、随机种子），"
         "使指标反映泛化能力而非记忆能力。需重训约 40 分钟。"),
        ("真实公开数据集盲测", "高", "低",
         "接入通道已实现，直接把公开真实数据集作为独立测试集跑一遍，"
         "诚实给出合成→真实的差距。这是最有力的可信度证据。"),
        ("阈值自动标定", "中", "低",
         "用企业自己的正常样本统计得分分布，自动推算置信度阈值与面积参考值，"
         "替代当前写死在配置里的通用经验值。"),
        ("在界面上采集数据", "中", "中",
         "检测页增加「标注并归档为训练样本」，让现场边用边积累数据，"
         "解决真实数据采集的落地难题。"),
    ]
    for i, (name, gain, cost, desc) in enumerate(todos, 1):
        print(f"\n  {i}. {name}　（收益 {gain} / 成本 {cost}）")
        for line in re.findall(r".{1,34}", desc):
            print(f"       {line}")

    print("\n" + "=" * 78)
    print("结论：工程完成度高、演示链路完整，但**实证强度不足**。")
    print("      最该补的是「用独立数据证明它真的有用」，而不是继续加功能。")
    print("=" * 78)


if __name__ == "__main__":
    main()
