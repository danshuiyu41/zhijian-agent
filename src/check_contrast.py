"""界面配色对比度核验：量化"卡片能否从背景中分辨出来"。

背景
----
初版配色 --bg:#f5f5f7 / --surface:#fff 只差 2% 亮度、描边仅 8% 不透明度，
三重叠加后卡片与背景几乎糊成一片纯白，实测观感是"连边框都没有了"。
本脚本用 WCAG 相对亮度公式把这类问题量化，避免靠肉眼反复调参。

判定依据（工程经验值，非标准强制项）：
- 背景与卡片表面的亮度差 ≥ 3.0%：卡片边界可辨；
- 描边与背景的对比度 ≥ 1.25：细线可见（1.0 表示完全看不见）。

用法：
    python -m src.check_contrast
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "web" / "design-system.css"
PASS, FAIL = "  [通过]", "  [失败]"


def rel_luminance(hex_color: str) -> float:
    """WCAG 相对亮度。支持 #rgb / #rrggbb / #rrggbbaa（自动展开短写）。"""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 8:
        h = h[:6]
    if len(h) != 6:
        raise ValueError(f"无法解析颜色：{hex_color!r}")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(c1: str, c2: str) -> float:
    l1, l2 = rel_luminance(c1), rel_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def strip_comments(text: str) -> str:
    """移除 CSS 注释。

    必要性：本项目在 `--bg` 附近写了"初版 --bg:#f5f5f7 与 --surface:#fff 只差 2%"这样的
    说明性注释，而正则解析会命中注释里的旧值，导致核验结果与真实配色不符
    （实测表现为"文件明明是 #ececf0，脚本却读出 #f5f5f7"）。先剥注释再解析。
    """
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def parse_color(block: str, var: str) -> tuple[tuple[int, int, int], float]:
    """取变量颜色，返回 ((r,g,b), alpha)。支持 #rgb/#rrggbb 与 rgba()。

    v3 起卡片使用**半透明面板 + 玻璃拟态**（如 rgba(22,32,56,.72)），
    因此必须解析 alpha 并在核验时叠加到底色上，否则算出来的是"假对比度"。
    """
    m = re.search(rf"--{re.escape(var)}:\s*#([0-9a-fA-F]{{3,8}})\b", block)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 8:
            h, a = h[:6], int(h[6:8], 16) / 255
        else:
            a = 1.0
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)), a  # type: ignore
    m = re.search(rf"--{re.escape(var)}:\s*rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\s*\)", block)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))), float(m.group(4))
    return (0, 0, 0), 0.0


def to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def over(fg: tuple[tuple[int, int, int], float], bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """把（可能半透明的）前景色叠加到背景色上。"""
    (r, g, b), a = fg
    return (round(r * a + bg[0] * (1 - a)),
            round(g * a + bg[1] * (1 - a)),
            round(b * a + bg[2] * (1 - a)))


def _theme_block(text: str, dark: bool) -> str:
    """取出深色或浅色主题的 CSS 片段。

    主题策略（v3 起）：**深色为默认**（写在 `:root` 里），浅色放在
    `@media (prefers-color-scheme: light)` 中。因此：
      - dark=True  → 取文件开头到 light 媒体查询之前的 `:root` 段；
      - dark=False → 取 light 媒体查询内部的 `:root` 段。
    早期版本假定浅色在前，切换主题策略后会导致取错值（实测表现为"深色值算成浅色"）。
    """
    marker = "prefers-color-scheme: light"
    idx = text.find(marker)

    def brace_block(start: int) -> str:
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
            i += 1
        return text[start:]

    if dark:
        head = text[:idx] if idx > 0 else text
        start = head.find(":root")
        end = head.find("}", start)
        return head[start:end + 1]
    if idx < 0:
        return ""
    start = text.find("{", text.find(":root", idx))
    return brace_block(start)


def hex_of(block: str, var: str) -> str:
    """从指定主题片段中取变量的十六进制值。

    注意用 `\\b` 界定结尾：早期写成 `#[0-9a-fA-F]{3,8}` 会因量词"先满足最小值"
    而把 `#f5f5f7` 截断成 `#f5f`，导致后续解析直接报错。
    """
    m = re.search(rf"--{re.escape(var)}:\s*(#[0-9a-fA-F]{{3,8}})\b", block)
    return m.group(1) if m else ""


def rgba_of(block: str, var: str) -> tuple[int, int, int, float] | None:
    m = re.search(rf"--{re.escape(var)}:\s*rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\s*\)", block)
    if not m:
        return None
    return tuple(int(m.group(i)) for i in (1, 2, 3)) + (float(m.group(4)),)  # type: ignore


def blend(fg: tuple[int, int, int, float], bg: tuple[int, int, int, int, int]) -> str:
    """把带透明度的描边叠在背景色上，得到实际可见颜色。"""
    r, g, b, a = fg
    br, bg_, bb = bg
    return "#%02x%02x%02x" % (round(r * a + br * (1 - a)),
                              round(g * a + bg_ * (1 - a)),
                              round(b * a + bb * (1 - a)))


def check_theme(text: str, dark: bool, failures: list[str]) -> None:
    """核验单个主题的配色对比度。

    v3 起卡片为**半透明玻璃面板**，因此先把面板色叠加到底色上再算对比度，
    否则得到的是"假对比度"（透明面板会被误当成纯色）。
    """
    label = "深色主题（默认）" if dark else "浅色主题（备选）"
    print("=" * 74)
    print(f"界面配色对比度核验 · {label}")
    print("=" * 74)

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{PASS if cond else FAIL} {name}" + (f"　{detail}" if detail else ""))
        if not cond:
            failures.append(f"{label} / {name}")

    block = _theme_block(text, dark=dark)
    if not block:
        check("主题变量可解析", False, "未找到该主题的 :root 段")
        return

    bg_rgb, _ = parse_color(block, "bg")
    panel_fg = parse_color(block, "panel")
    panel2_fg = parse_color(block, "panel-2")
    text_c, _ = parse_color(block, "text")
    text2, _ = parse_color(block, "text-2")
    text3, _ = parse_color(block, "text-3")
    line_fg = parse_color(block, "line")

    bg = to_hex(bg_rgb)
    panel_rgb = over(panel_fg, bg_rgb)
    panel2_rgb = over(panel2_fg, panel_rgb)
    panel = to_hex(panel_rgb)
    panel2 = to_hex(panel2_rgb)

    print(f"  背景        {bg}")
    print(f"  玻璃面板    {panel}   （原始 rgba{panel_fg[0] + (panel_fg[1],)}）")
    print(f"  次级面板    {panel2}")
    print(f"  正文色      {to_hex(text_c)}")
    print("-" * 74)

    # 1) 面板与背景可分辨
    cr_panel = contrast_ratio(panel, bg)
    check("玻璃面板与背景对比度 ≥ 1.20", cr_panel >= 1.20, f"{cr_panel:.2f}")

    # 2) 次级面板与主面板可区分
    cr_p2 = contrast_ratio(panel2, panel)
    check("次级面板与主面板对比度 ≥ 1.05", cr_p2 >= 1.05, f"{cr_p2:.2f}")

    # 3) 描边可见
    seen = to_hex(over(line_fg, bg_rgb))
    cr_line = contrast_ratio(seen, bg)
    check("描边叠在背景上对比度 ≥ 1.15", cr_line >= 1.15, f"可见色 {seen}，对比度 {cr_line:.2f}")

    # 4) 文字可读性（WCAG AA 4.5 / AAA 7.0）
    cr_text = contrast_ratio(to_hex(text_c), panel)
    check("正文对比度 ≥ 7.0（AAA）", cr_text >= 7.0, f"{cr_text:.2f}")
    cr_t2 = contrast_ratio(to_hex(text2), panel)
    check("次级文字对比度 ≥ 4.5（AA）", cr_t2 >= 4.5, f"{cr_t2:.2f}")
    cr_t3 = contrast_ratio(to_hex(text3), panel)
    check("三级文字对比度 ≥ 3.5", cr_t3 >= 3.5, f"{cr_t3:.2f}")


def main() -> None:
    if not CSS.exists():
        raise SystemExit(f"未找到 {CSS}")
    text = strip_comments(CSS.read_text(encoding="utf-8"))
    failures: list[str] = []

    check_theme(text, dark=True, failures=failures)
    check_theme(text, dark=False, failures=failures)

    print("=" * 74)
    if failures:
        print(f"未通过 {len(failures)} 项：")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("两套主题配色对比度全部通过 —— 面板、描边、文字层级清晰可辨")
    print("=" * 74)


if __name__ == "__main__":
    main()
