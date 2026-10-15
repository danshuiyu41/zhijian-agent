"""静态检测会引起横向溢出的元素（无法开浏览器时的替代手段）。

排查思路：横向滚动几乎总由这几类元素造成——
1. 有固定/最小宽度的元素（input 默认 min-width、canvas 固有尺寸、宽表格）；
2. 网格/弹性子项未设 min-width:0，被内容撑开（CSS 网格的经典坑）；
3. 绝对定位元素超出父容器；
4. 内部横向滚动容器未加 overflow-x（本项目的样本条、表格）。

用法：
    python -m src.check_overflow
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
    css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
    html = HTML.read_text(encoding="utf-8")
    problems: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{PASS if cond else FAIL} {name}" + (f"　{detail}" if detail else ""))
        if not cond:
            problems.append(name)

    print("=" * 74)
    print("横向溢出静态检查")
    print("=" * 74)

    # 1) 页面级兜底
    has_html_overflow = bool(re.search(r"html\s*\{[^}]*overflow-x\s*:\s*hidden", css))
    has_body_overflow = bool(re.search(r"body\s*\{[^}]*overflow-x\s*:\s*hidden", css))
    check("html/body 有横向兜底（overflow-x: hidden）", has_html_overflow and has_body_overflow,
          f"html={has_html_overflow} body={has_body_overflow}")

    # 2) 网格子项是否允许收缩
    #
    # 关键：CSS 网格/弹性子项默认 min-width:auto，内容（长文本、canvas、表格）
    # 会把轨道撑破父容器，导致整页横向滚动。常见修法是给子项 min-width:0。
    cols2 = re.search(r"\.cols-2\s*\{([^}]*)\}", css)
    has_min0 = bool(re.search(r"minmax\(0,\s*1fr\)", cols2.group(1))) if cols2 else False
    check("两栏网格使用 minmax(0,1fr)（允许收缩）", has_min0,
          cols2.group(1).strip()[:70] if cols2 else "未找到 .cols-2")

    # 3) canvas 自适应
    has_canvas = '.chart' in css or 'canvas' in css
    canvas_rule = re.search(r"canvas\s*\{([^}]*)\}", css)
    check("canvas 宽度自适应父容器",
          bool(canvas_rule and ("max-width" in canvas_rule.group(1) or "width" in canvas_rule.group(1))),
          canvas_rule.group(1).strip()[:70] if canvas_rule else "未找到 canvas 规则")

    # 4) 宽表格需要容器内滚动（移动端否则会撑破页面）
    has_table_wrap = "table-wrap" in css
    check("宽表格有横向滚动容器（.table-wrap）", has_table_wrap,
          "" if has_table_wrap else "移动端宽表格会撑破页面")

    # 5) 样本条内部滚动
    samples_rule = re.search(r"\.samples\s*\{([^}]*)\}", css)
    check(".samples 容器内部横向滚动",
          bool(samples_rule and "overflow-x" in samples_rule.group(1)),
          samples_rule.group(1).strip()[:70] if samples_rule else "")

    # 6) 是否存在明显超宽的固定值
    #
    # 注意用 `(?<![\w-])` 排除 `max-width` / `min-width` 中的 `width`：
    # 早期版本只写 `(width|min-width)`，会把 `max-width:1240px` 当成固定宽度误报。
    fixed = re.findall(r"(?<![\w-])(width|min-width)\s*:\s*(\d{4,})px", css)
    fixed = [(k, v) for k, v in fixed if k != "max-width"]
    check("无过大的固定像素宽度（max-width 不计）", not fixed,
          str(fixed[:5]) if fixed else "")

    # 7) 主容器宽度是否收敛
    main_rule = re.search(r"main\s*\{([^}]*)\}", css)
    if main_rule:
        mw = re.search(r"max-width\s*:\s*(\d+)px", main_rule.group(1))
        val = int(mw.group(1)) if mw else 0
        check("主容器 max-width 合理（≤1240px）", 0 < val <= 1240, f"{val}px")
    else:
        check("找到 main 规则", False)

    # 8) 移动端断点
    breakpoints = sorted({int(x) for x in re.findall(r"max-width:\s*(\d+)px", css)})
    check("存在移动端断点", any(b <= 760 for b in breakpoints), f"断点 {breakpoints}")

    # 9) HTML 中是否有内联固定宽度
    inline_w = re.findall(r'style="[^"]*width\s*:\s*(\d{3,})px', html)
    check("无内联固定宽度", not inline_w, str(inline_w[:5]) if inline_w else "")

    print("=" * 74)
    if problems:
        print(f"发现 {len(problems)} 项可能导致横向溢出的问题：")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("未发现横向溢出隐患")
    print("=" * 74)


if __name__ == "__main__":
    main()
