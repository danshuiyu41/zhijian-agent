"""提交材料缺口检查：对照官方要求逐项核对，列出还缺什么。

依据：2026 年赛事通知（首都师范大学转发版）第四条「提交作品」明确列出交什么，
以及第三条「参赛设计要求」里的硬性约束。

用法：
    python -m src.check_deliverables
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESK = Path.home() / "Desktop" / "智检Agent-上传材料"


def has(p: Path) -> bool:
    return p.exists()


def size_kb(p: Path) -> float:
    return round(p.stat().st_size / 1024, 1) if p.exists() else 0.0


def main() -> None:
    print("=" * 80)
    print("提交材料缺口检查（对照 2026 赛事通知第四条）")
    print("=" * 80)

    # 官方列的提交物：作品简介、软件设计文档、作品源代码、软件安装包、视频短片
    official = [
        ("作品简介", "平台网页富文本 + 3 幅核心功能截图", [
            ("docs/上传文案.md", "简介文本（398 字）"),
            ("docs/upload/cover_350.png", "封面 350×350"),
            ("docs/upload/effect_1.jpg", "效果图 1/5"),
            ("docs/upload/effect_2.jpg", "效果图 2/5"),
            ("docs/upload/effect_3.jpg", "效果图 3/5"),
            ("docs/upload/effect_4.jpg", "效果图 4/5"),
            ("docs/upload/effect_5.jpg", "效果图 5/5"),
        ]),
        ("软件设计文档", "PDF，按大赛模板", [
            ("docs/软件设计文档.md", "源文件"),
            ("docs/pdf/软件设计文档.pdf", "PDF"),
            ("docs/作品核心资料.md", "源文件"),
            ("docs/pdf/作品核心资料.pdf", "PDF"),
        ]),
        ("作品源代码", "完整可运行工程", [
            ("src", "后端 61 个模块"),
            ("web", "前端"),
            ("config", "配置"),
            ("knowledge", "知识库"),
            ("README.md", "说明"),
            (".gitignore", "忽略规则"),
            (".vscode", "VSCode 配置"),
        ]),
        ("软件安装包", "**通知里明确要求**", [
            ("src", "随源码一起分发"),
            ("models/zhijian_best.pt", "模型权重"),
            ("models/feature_bank.npz", "开集特征库"),
            ("requirements.txt", "依赖清单"),
            ("docs/安装与部署说明.md", "安装说明"),
            ("start.py", "一键启动"),
        ]),
        ("作品简介补充", "附件材料", [
            ("docs/团队分工说明.md", "团队合作佐证"),
            ("docs/AI使用与原创性声明.md", "满足第三条(6)"),
            ("requirements.txt", "依赖清单"),
            ("docs/安装与部署说明.md", "部署说明"),
        ]),
        ("视频短片", "MP4，≤5 分钟，≤150MB", [
            ("docs/演示视频脚本.md", "分镜脚本（成片待录）"),
        ]),
        ("作品申报表", "PDF，按模板", [
            ("docs/作品申报表.md", "源文件"),
            ("docs/pdf/作品申报表.pdf", "PDF"),
        ]),
    ]

    missing: list[str] = []
    for group, need, files in official:
        print(f"\n【{group}】{need}")
        if not files:
            print("   [缺 ] 完全没有")
            missing.append(group)
            continue
        for rel, note in files:
            p = ROOT / rel
            mark = "OK " if has(p) else "缺 "
            print(f"   [{mark}] {rel:<36} {note}" + (f"  {size_kb(p)} KB" if has(p) else ""))

    # ---------------- 官方硬性约束 ----------------
    print("\n" + "=" * 80)
    print("官方硬性约束（第三条 参赛设计要求）——不满足可能被判无效")
    print("=" * 80)
    constraints = [
        ("(5) 必须具备公网联通性", "待办",
         "现在只跑在 127.0.0.1，评委访问不到。见 docs/云服务器部署手册.md"),
        ("(6) 借助 AI 生成的部分须明示", "部分满足",
         "设计文档 10.2 节有提及，但未逐项列出「哪些是自己设计、用了什么工具」，需补专项声明"),
        ("(6) 历年/其他大赛作品不可重复参赛", "满足", "本作品为新作"),
        ("(8) 涉及疆域的地图须标注审图号", "不涉及", "本项目无地图"),
        ("初赛评审 + 决赛现场展示", "待办", "现场演示需准备（本机 + 公网双方案）"),
    ]
    for name, status, note in constraints:
        flag = "OK " if status == "满足" else ("不涉及" if status == "不涉及" else "缺 ")
        print(f"  [{flag}] {name:<28} {status}")
        print(f"          {note}")

    # ---------------- 建议补的 ----------------
    print("\n" + "=" * 80)
    print("建议补充（非强制，但影响得分）")
    print("=" * 80)
    suggestions = [
        ("软件安装包 / 离线部署包", "官方明确要求",
         "打一个 zip：源码 + 模型权重 + 依赖清单 + 一键启动说明，评委可本地跑"),
        ("AI 工具使用与原创性声明", "官方明确要求",
         "逐项说明哪些模块是团队设计、哪些借用了 AI 辅助、用了什么工具"),
        ("团队分工说明", "评审含团队合作维度",
         "文档里完全没体现分工，评委无从判断团队合作"),
        ("访问二维码", "提升评委体验",
         "把带 token 的公网地址生成二维码，评委扫码即进"),
        ("演示视频成片", "决定第一印象", "只有脚本，需录 4 分半 MP4"),
        ("真实数据指标", "最大加分项",
         "当前指标全部来自合成数据留出集，补一组真实数据指标能显著提升可信度"),
    ]
    for name, why, note in suggestions:
        print(f"\n  · {name}　（{why}）")
        print(f"      {note}")

    # ---------------- 桌面同步状态 ----------------
    print("\n" + "=" * 80)
    print("桌面材料目录状态")
    print("=" * 80)
    if not DESK.exists():
        print(f"  [缺 ] 桌面目录不存在：{DESK}")
    else:
        for sub in ("图片素材", "文档", "PDF待提交", "源代码"):
            d = DESK / sub
            n = len(list(d.rglob("*"))) if d.exists() else 0
            print(f"  [{'OK ' if d.exists() else '缺 '}] {sub:<10} {n} 项")
        print(f"\n  提示：桌面缺「软件安装包」，若要做需在此新建目录")

    print("\n" + "=" * 80)
    if missing:
        print(f"官方明确要求但完全缺失：{'、'.join(missing)}")
    print("最该补的三件：① 软件安装包　② AI 使用声明　③ 团队分工说明")
    print("=" * 80)


if __name__ == "__main__":
    main()
