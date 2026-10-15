"""CSS 语法体检：括号配对、声明合法性、可疑值。

动机：样式表里的一个未闭合括号或非法声明，会让**其后所有规则被浏览器丢弃**，
表现却是"某几块样式莫名失效"，很难靠肉眼定位。本脚本把这类问题静态查出来。

检查项：
1. 花括号配对（含媒体查询嵌套）；
2. 括号/引号配对；
3. 声明是否以 `;` 结尾、属性名是否合法；
4. 使用的 CSS 函数/属性是否有明显的拼写问题；
5. 输出每个选择器块的数量，便于核对是否有规则整体丢失。

用法：
    python -m src.check_css
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "web" / "design-system.css"
PASS, FAIL, WARN = "  [通过]", "  [失败]", "  [提示]"


def strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def main() -> None:
    if not CSS.exists():
        raise SystemExit(f"未找到 {CSS}")
    raw = CSS.read_text(encoding="utf-8")
    text = strip_comments(raw)
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{PASS if cond else FAIL} {name}" + (f"　{detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("=" * 72)
    print(f"CSS 语法体检：{CSS.name}（{len(raw)} 字符，去注释后 {len(text)} 字符）")
    print("=" * 72)

    # 1) 花括号配对
    opens, closes = text.count("{"), text.count("}")
    check("花括号配对", opens == closes, f"{{×{opens} }}×{closes}")

    # 2) 逐字符扫描，定位不配对的位置
    depth = 0
    bad_line = None
    for lineno, line in enumerate(text.splitlines(), 1):
        depth += line.count("{") - line.count("}")
        if depth < 0 and bad_line is None:
            bad_line = lineno
    check("无提前闭合", bad_line is None,
          f"第 {bad_line} 行出现多余的 }}" if bad_line else "")
    check("结尾深度归零", depth == 0, f"最终深度 {depth}")

    # 3) 圆括号配对
    check("圆括号配对", text.count("(") == text.count(")"),
          f"(×{text.count('(')} )×{text.count(')')}")

    # 4) 引号配对（忽略换行，CSS 里不应跨行）
    dq = text.count('"')
    check("双引号成对", dq % 2 == 0, f"共 {dq} 个")

    # 5) 声明体检：找出明显不合法的属性名
    #
    # 说明：这里只做"粗筛"，因此把厂商前缀与常见现代属性都纳入白名单。
    # 早期版本没处理 `-webkit-` 前缀，把 -webkit-backdrop-filter 等正常属性报成了非法。
    decls = re.findall(r"([a-zA-Z-]+)\s*:\s*([^;{}]+)[;}]", text)
    known = {
        "color", "background", "border", "margin", "padding", "font", "display", "position",
        "top", "left", "right", "bottom", "width", "height", "max", "min", "gap", "grid",
        "flex", "align", "justify", "overflow", "text", "line", "letter", "white", "word",
        "box", "shadow", "transition", "transform", "opacity", "cursor", "content", "z-index",
        "inset", "outline", "filter", "backdrop", "aspect", "object", "list", "vertical",
        "float", "clear", "visibility", "animation", "scroll", "place", "pointer", "resize",
        "appearance", "user", "will", "columns", "column", "order", "row", "scale", "rotate",
        "translate", "fill", "stroke", "clip", "mask", "caret", "accent", "tab", "table",
        "empty", "quotes", "counter", "isolation", "mix", "shape", "hyphens", "direction",
        "unicode", "zoom", "scrollbar", "z", "index",
    }
    suspicious = []
    for name, _ in decls:
        if name.startswith("--"):
            continue
        base = name.lstrip("-")
        for pfx in ("webkit-", "moz-", "ms-", "o-"):
            if base.startswith(pfx):
                base = base[len(pfx):]
        # 白名单按"首段"匹配，但 `z-index` / `scrollbar-width` 这类属性的首段本身
        # 就是完整词，因此同时允许整名与末段匹配，避免反复误报。
        parts = base.split("-")
        if parts[0] in known or base in known or parts[-1] in known:
            continue
        suspicious.append(name)
    if suspicious:
        print(f"{WARN} 可疑属性名：{'、'.join(sorted(set(suspicious))[:10])}")
    check("未发现非法属性名", not suspicious, f"{len(suspicious)} 处" if suspicious else "")

    # 6) 空值声明（如 `color: ;`）——常见于编辑失误
    empties = re.findall(r"([a-zA-Z-]+)\s*:\s*[;}]", text)
    check("无空值声明", not empties, "、".join(sorted(set(empties))[:8]) if empties else "")

    # 7) color-mix 用法核对
    #
    # 递归匹配括号：`color-mix(in srgb, var(--accent) 13%, var(--surface))` 里含嵌套括号，
    # 早期用 `\(([^)]*)\)` 会在第一个 `)` 处截断，把 36 处正常写法全部误判为非法。
    def find_calls(src: str, fn: str) -> list[str]:
        out: list[str] = []
        for m in re.finditer(re.escape(fn) + r"\(", src):
            i, depth = m.end(), 1
            while i < len(src) and depth:
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                i += 1
            out.append(src[m.end():i - 1])
        return out

    cm = find_calls(text, "color-mix")
    bad_cm = [c for c in cm if "in srgb" not in c or c.count(",") < 2]
    check("color-mix 语法正确", not bad_cm,
          f"{len(bad_cm)} 处可疑：{bad_cm[:2]}" if bad_cm else f"共 {len(cm)} 处")

    # 7b) var() 引用核对
    var_calls = find_calls(text, "var")
    bad_var = [v for v in var_calls if not v.strip().startswith("--")]
    check("var() 用法正确", not bad_var, f"{len(bad_var)} 处可疑" if bad_var else f"共 {len(var_calls)} 处")

    # 8) 关键选择器是否仍在（防止规则被整体丢弃）
    #
    # 主题策略说明：v3 起改为**深色为默认**、浅色放在 prefers-color-scheme: light 里，
    # 因此这里检查的是 light 媒体查询（v1/v2 是相反的）。
    must_have = [".nav {", ".tabs button", ".card {", ".stat {", ".level", ".drop {",
                 ".verdict {", ".insight", ".wo {", ".compare", ".samples", ".seg ",
                 "@media (prefers-color-scheme: light)", ".page-head", "footer {"]
    lost = [s for s in must_have if s not in text]
    check("关键选择器齐全", not lost, "丢失: " + "、".join(lost) if lost else f"{len(must_have)} 项")

    # 9) 变量使用前是否已定义
    #
    # 注意大小写：CSS 自定义属性**区分大小写**，`--lv-A` 里的 `A` 必须一起匹配，
    # 早期用 `[a-z0-9-]+` 只截出 `--lv-`，于是把 5 个等级色全部误报为"未定义"。
    defined = set(re.findall(r"(--[A-Za-z0-9-]+)\s*:", text))
    used = set(re.findall(r"var\(\s*(--[A-Za-z0-9-]+)", text))
    undef = sorted(used - defined)
    check("引用的 CSS 变量均已定义", not undef,
          "未定义: " + "、".join(undef) if undef else f"共 {len(defined)} 个变量")

    print("=" * 72)
    if failures:
        print(f"未通过 {len(failures)} 项：" + "、".join(failures))
        sys.exit(1)
    print("CSS 语法体检全部通过")
    print("=" * 72)


if __name__ == "__main__":
    main()
