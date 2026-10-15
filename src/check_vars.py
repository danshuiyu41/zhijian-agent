"""检查 HTML/JS 中引用的 CSS 变量是否真实存在。

动机：重命名主题变量（本项目把 --accent 改成 --c1/--c2/--c3）后，
JS 里内联的 `style="background: var(--accent)"` 不会报错，只是**静默失效**，
表现为"某一条进度条没颜色"这类很难定位的视觉 bug。本脚本把这类引用一次性查出来。

用法：
    python -m src.check_vars
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "web" / "design-system.css"
HTML = ROOT / "web" / "index.html"
PASS, FAIL = "  [通过]", "  [失败]"


def main() -> None:
    css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
    html = HTML.read_text(encoding="utf-8")
    defined = set(re.findall(r"(--[A-Za-z0-9-]+)\s*:", css))

    print("=" * 74)
    print("CSS 变量引用一致性检查（CSS + HTML/JS）")
    print("=" * 74)
    print(f"  design-system.css 中定义了 {len(defined)} 个变量")
    print("-" * 74)

    failures: list[str] = []

    def check(name: str, missing: list[str], extra: str = "") -> None:
        ok = not missing
        print(f"{PASS if ok else FAIL} {name}" + (f"　未定义: {'、'.join(sorted(set(missing)))}" if missing else extra))
        if not ok:
            failures.append(name)

    # 1) CSS 内部引用
    css_used = set(re.findall(r"var\(\s*(--[A-Za-z0-9-]+)", css))
    check("样式表中引用的变量均已定义", sorted(css_used - defined), f"　共 {len(css_used)} 处引用")

    # 2) HTML/JS 内联引用（这是本次出问题的地方）
    html_used = set(re.findall(r"var\(\s*(--[A-Za-z0-9-]+)", html))
    check("HTML/JS 内联引用的变量均已定义", sorted(html_used - defined),
          f"　共 {len(html_used)} 处引用")

    # 3) 定义但从未使用（提示，不算失败）
    all_used = css_used | html_used
    unused = sorted(defined - all_used)
    if unused:
        print(f"  [提示] 已定义但未被引用：{'、'.join(unused[:12])}"
              + ("…" if len(unused) > 12 else ""))

    print("=" * 74)
    if failures:
        print(f"未通过 {len(failures)} 项")
        sys.exit(1)
    print("变量引用一致性检查通过")
    print("=" * 74)


if __name__ == "__main__":
    main()
