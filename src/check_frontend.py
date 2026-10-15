"""前端静态自检：核对 HTML 中的 id 与 JS 引用是否一致。

为什么要这一步
--------------
前端没有构建链路，元素 id 写错不会报错，只会在运行时静默失效（点了没反应）。
本脚本做三件事：
1. 提取 HTML 中所有 `id="..."`，与 JS 中 `$('#...')` / `getElementById` 的引用比对，
   报告"引用了但不存在"和"定义了但未使用"；
2. 校验页签 data-tab 与 section id="tab-*" 一一对应；
3. 校验 HTML 内不再包含硬编码的内联样式表（应统一走 design-system.css）。

用法：
    python -m src.check_frontend
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "web" / "index.html"
CSS = ROOT / "web" / "design-system.css"

PASS, FAIL, WARN = "  [通过]", "  [失败]", "  [提示]"


def main() -> None:
    if not HTML.exists():
        raise SystemExit(f"未找到 {HTML}")
    text = HTML.read_text(encoding="utf-8")
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{PASS if cond else FAIL} {name}" + (f"　{detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("=" * 72)
    print("前端静态自检")
    print("=" * 72)

    # 1) 设计系统
    check("引用外部设计系统", "design-system.css" in text)
    check("不含内联 <style> 块", "<style" not in text,
          f"{len(re.findall(r'<style', text))} 处")
    check("样式文件存在", CSS.exists(), f"{CSS.stat().st_size} 字节" if CSS.exists() else "缺失")

    # 2) id 与 JS 引用一致性
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    refs = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", text))
    refs |= set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", text))
    missing = sorted(refs - ids)
    unused = sorted(ids - refs)
    check("JS 引用的元素都存在", not missing, "缺失: " + "、".join(missing) if missing else f"{len(refs)} 个引用全部命中")
    if unused:
        print(f"{WARN} 定义了但 JS 未直接引用（可能仅用于 CSS 选择）: {'、'.join(unused)}")

    # 3) 页签一致性
    tabs = re.findall(r'data-tab="([^"]+)"', text)
    sections = re.findall(r'<section id="tab-([^"]+)"', text)
    check("页签与区块一一对应", sorted(tabs) == sorted(sections),
          f"页签 {tabs} ｜ 区块 {sections}")

    # 4) 页签数量（4 个：检测 / 看板 / 工单 / 诊断）
    check("页签数量为 4", len(tabs) == 4, f"实际 {len(tabs)} 个：{'、'.join(tabs)}")

    # 5) 关键 id 必须存在（渲染路径上的硬依赖）
    required = ["drop", "file", "batch", "resultIdle", "resultBusy", "resultError",
                "resultBody", "gradeBig", "verdictText", "verdictSub", "mini", "advice",
                "detWrap", "shotOrigin", "shotBox", "shotHeat", "reportLink", "againBtn",
                "samples", "recent", "stats", "gradeDist", "defectDist", "chart", "insights",
                "orderList", "woStat", "q", "askBtn", "answer", "kbq", "kbBtn", "kbList",
                "modelInfo", "status"]
    absent = [r for r in required if r not in ids]
    check("关键渲染节点齐全", not absent, "缺失: " + "、".join(absent) if absent else f"{len(required)} 个节点")

    # 6) 深色模式
    check("适配深色模式", "prefers-color-scheme" in text or
          ("prefers-color-scheme" in CSS.read_text(encoding="utf-8") if CSS.exists() else False))

    # 7) 无外部网络依赖（离线可用）
    externals = re.findall(r'(?:src|href)="(https?://[^"]+)"', text)
    check("无外部网络依赖（离线可用）", not externals,
          "外部资源: " + "、".join(externals) if externals else "全部为本地资源")

    print("=" * 72)
    if failures:
        print(f"自检未通过，{len(failures)} 项失败：" + "、".join(failures))
        sys.exit(1)
    print("前端静态自检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    main()
