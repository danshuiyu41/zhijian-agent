"""获奖差距分析：把"拿二等奖"拆成可执行的检查项。

动机：比赛评分发生在评委面对材料的几分钟内，决定因素与代码质量并不同构。
本脚本按"评委实际看到的东西"逐项盘点当前材料，指出离获奖还差什么，
避免把时间花在评委会忽略的地方。

用法：
    python -m src.award_gap
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exists(p: str) -> bool:
    return (ROOT / p).exists()


def size_kb(p: str) -> float:
    f = ROOT / p
    return round(f.stat().st_size / 1024, 1) if f.exists() else 0.0


def main() -> None:
    print("=" * 80)
    print("获奖差距分析（对照评委实际看到的三份材料 + 现场答辩）")
    print("=" * 80)

    # ---------------- 1. 提交材料 ----------------
    print("\n【一、提交材料：作品简介 + 设计文档 + 演示视频】")
    items = [
        ("作品简介文档", "docs/作品简介.md", True, "含 3 幅核心功能截图、≤400 字简介、队长联系方式"),
        ("软件设计文档", "docs/软件设计文档.md", True, "按模板、PDF 格式、含开发全周期"),
        ("演示视频", "docs/演示视频脚本.md", False, "仅脚本就绪，**视频本身必须自己录**"),
    ]
    for name, path, done, note in items:
        mark = "[有]" if exists(path) else "[缺]"
        print(f"  {mark} {name:<12} {size_kb(path):>7.1f} KB　{note}")

    # ---------------- 2. 指标证据链 ----------------
    print("\n【二、指标证据链：评委判断「真做过还是编的」看这里】")
    m = json.loads((ROOT / "runs/metrics.json").read_text(encoding="utf-8")) if exists("runs/metrics.json") else {}
    rob = json.loads((ROOT / "runs/robustness.json").read_text(encoding="utf-8")) if exists("runs/robustness.json") else {}
    ood = json.loads((ROOT / "runs/ood_eval.json").read_text(encoding="utf-8")) if exists("runs/ood_eval.json") else {}
    ev = [
        ("合成数据测试集准确率", f"{m.get('test', {}).get('accuracy')}", "有，但**同源**，说服力弱"),
        ("受控退化实验（鲁棒性）", "L3 漏检 15%" if rob else "缺", "有，这项加分"),
        ("未知缺陷识别（开集）", f"拦截率 {ood.get('recommended_unknown_reject_rate')}" if ood else "缺", "有，是差异化亮点"),
        ("**真实数据上的指标**", "**缺**", "**最致命**：一问数据来源即暴露"),
        ("混淆矩阵/训练曲线图", "有" if exists("runs/figures/confusion_matrix.png") else "缺", "有"),
        ("可解释性验证（热力图）", "有" if exists("runs/figures/heatmap_study.png") else "缺", "有，且实测落在缺陷上"),
    ]
    for name, val, note in ev:
        print(f"  · {name:<22} {str(val):<16} {note}")

    # ---------------- 3. 现场演示 ----------------
    print("\n【三、现场演示：决赛答辩的决定性环节】")
    print("  评委最常做的三件事：")
    print("    ① 让你现场跑一遍——**服务挂了就直接出局**")
    print("    ② 挑一个缺陷图问「为什么判这个等级」——答不上来就掉分")
    print("    ③ 问「数据哪来的」「跟商用方案比差在哪」——这是分水岭")

    # ---------------- 4. 差距与动作 ----------------
    print("\n" + "=" * 80)
    print("拿二等奖及以上：必须补的四件事（按性价比排序）")
    print("=" * 80)
    actions = [
        ("A", "拿一组真实数据，重训并给指标", "决定性",
         "三条路任选：① 工程训练中心自己划伤/磨痕造缺陷；"
         "② 找一家小厂拍 200~500 张；③ 用公开真实数据集（接入通道已实现）。"
         "哪怕准确率掉到 70%，只要是真数据上测的，说服力都强过合成数据的 100%。"),
        ("B", "录一支 4 分半的演示视频", "决定性",
         "分镜脚本已写好（docs/演示视频脚本.md）。关键：**必须有实物拍摄环节**，"
         "手机对着一个真实带划痕的金属件拍，比对着屏幕点鼠标有说服力得多。"),
        ("C", "准备一页纸的答辩问答卡", "高",
         "三个必答问题写成卡片：数据来源、为什么不用 YOLO、跟商用方案比差在哪。"
         "现有文档里答案都有，但答辩时需要**一眼能看到**，不能现场翻文档。"),
        ("D", "把「未知缺陷识别」做成主动演示点", "高",
         "现场喂一张模型没见过的图，让它主动说「这个我识别不了，转人工」。"
         "绝大多数参赛作品都是闭集分类，这一下就把差距拉开了。"),
    ]
    for tag, name, gain, desc in actions:
        print(f"\n  [{tag}] {name}　（重要性 {gain}）")
        import re as _re
        for line in _re.findall(r".{1,36}", desc):
            print(f"       {line}")

    print("\n" + "=" * 80)
    print("结论：当前材料约在**二等奖边缘**。")
    print("      A、B 两项做完，冲一等奖有竞争力；只做 C、D，稳在二三等奖之间。")
    print("=" * 80)


if __name__ == "__main__":
    main()
