"""打印当前主题的关键色值（便于人工核对与写文档）。

用法：
    python -m src.show_theme
"""
from __future__ import annotations

import re
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "web" / "design-system.css"


def block(text: str, dark: bool) -> str:
    marker = "prefers-color-scheme: light"
    idx = text.find(marker)
    if dark:
        head = text[:idx] if idx > 0 else text
        s = head.find(":root")
        return head[s:head.find("}", s) + 1]
    if idx < 0:
        return ""
    s = text.find(":root", idx)
    return text[s:text.find("}", s) + 1]


def main() -> None:
    t = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
    vars_ = ["bg", "panel", "panel-2", "panel-hi", "text", "text-2", "text-3",
             "c1", "c2", "c3", "lv-A", "lv-B", "lv-C", "lv-D", "lv-E",
             "glass-brd", "edge-blue"]
    for label, dark in (("深色主题（默认）", True), ("浅色主题（备选）", False)):
        b = block(t, dark)
        print("=" * 62)
        print(label)
        print("=" * 62)
        for v in vars_:
            m = re.search(r"--" + re.escape(v) + r":\s*([^;]+);", b)
            print(f"  --{v:<12s} {m.group(1).strip() if m else '(未定义)'}")
        print()


if __name__ == "__main__":
    main()
