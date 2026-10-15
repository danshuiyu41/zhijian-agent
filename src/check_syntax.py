"""全量源码编译体检（比正则扫描可靠得多）。

背景与教训
----------
中文全角引号（U+201C/U+201D）与半角双引号在等宽字体下几乎无法分辨，
写进字符串字面量会提前结束字符串，报 `SyntaxError: invalid syntax`。
本项目在写中文文档时因此踩坑三次。

最初尝试用正则扫描"奇数个半角引号 + 含全角引号"来定位，
但**正则无法理解三引号文档字符串**，会把合法写法误报成隐患（实测误报过自己）。

更可靠的做法是直接用 Python 编译器做地面真值：
  - `py_compile` 能通过的，就是语法合法；
  - 不能通过的，直接给出文件与行号。
因此本脚本只做两件事：全量编译 + 语法错误归类提示。

用法：
    python -m src.check_syntax
"""
from __future__ import annotations

import argparse
import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL = "  [通过]", "  [失败]"

# 常见语法错误 → 成因提示（针对本项目的高频坑）
HINTS = [
    ("invalid syntax", "常见成因：字符串里混入了中文全角引号 “ ”（与 \" 极难分辨），"
                       "或 f-string 里出现了反斜杠、中文引号。"),
    ("unterminated string", "字符串未闭合：多为全角引号截断了字面量。"),
    ("f-string expression part cannot include a backslash",
     "Python 3.10 的 f-string 中不允许反斜杠，请先把值算到变量再插值。"),
    ("cannot assign to", "把关键字或字面量写在了赋值左侧。"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="全量源码编译体检")
    ap.add_argument("--dir", default="src")
    ap.add_argument("--also", nargs="*", default=["start.py"],
                    help="除 src/ 外还要检查的文件")
    args = ap.parse_args()

    files = sorted((ROOT / args.dir).glob("*.py"))
    for extra in args.also:
        p = ROOT / extra
        if p.exists():
            files.append(p)

    print("=" * 78)
    print(f"源码编译体检：共 {len(files)} 个文件")
    print("=" * 78)

    errors: list[tuple[Path, str]] = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            try:
                py_compile.compile(str(f), cfile=str(Path(td) / (f.stem + ".pyc")),
                                   doraise=True)
            except py_compile.PyCompileError as e:
                errors.append((f, str(e)))
            except SyntaxError as e:
                errors.append((f, f"{f}:{e.lineno}: {e.msg}"))

    if errors:
        for f, msg in errors:
            print(f"\n{FAIL} {f.relative_to(ROOT)}")
            for line in str(msg).strip().splitlines()[-4:]:
                print(f"       {line}")
            for key, hint in HINTS:
                if key in str(msg):
                    print(f"       → {hint}")
                    break
        print("\n" + "=" * 78)
        print(f"{len(errors)} 个文件存在语法错误，必须先修复")
        print("=" * 78)
        sys.exit(1)

    print(f"{PASS} 全部 {len(files)} 个文件语法正确")
    print("=" * 78)


if __name__ == "__main__":
    main()
