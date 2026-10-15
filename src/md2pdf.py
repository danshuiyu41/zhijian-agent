"""把 Markdown 文档转成可提交的 PDF（零额外依赖）。

实现方式
--------
Markdown → 自带打印样式的 HTML → Edge/Chrome 无头模式 `--print-to-pdf`。
不引入 markdown/pandoc/weasyprint 等依赖：本项目的 md 只用标题、表格、列表、
代码块、引用这几类语法，自己写个小转换器更可控（也便于控制中文字体与分页）。

用法：
    python -m src.md2pdf                          # 转换提交清单里的四份文档
    python -m src.md2pdf docs/训练原理.md          # 转换指定文件
    python -m src.md2pdf --all                    # 转换 docs/ 下全部 md
    python -m src.md2pdf --no-edge                # 只产出 HTML（浏览器 Ctrl+P 打印）
"""
from __future__ import annotations

import argparse
import html as _html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

from .utils import ROOT, get_logger, resolve  # noqa: E402

LOG = get_logger("md2pdf")

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

# 默认转换清单：平台要求提交的文档
DEFAULT_DOCS = [
    "docs/软件设计文档.md",
    "docs/作品核心资料.md",
    "docs/作品简介.md",
]


CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans SC", sans-serif;
  font-size: 10.5pt; line-height: 1.7; color: #1a1a1a; margin: 0;
}
h1 { font-size: 20pt; margin: 0 0 6pt; padding-bottom: 5pt; border-bottom: 2px solid #1a56db; }
h2 { font-size: 14pt; margin: 16pt 0 6pt; padding-left: 7pt; border-left: 4px solid #1a56db;
     page-break-after: avoid; }
h3 { font-size: 12pt; margin: 12pt 0 5pt; color: #1f2937; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 10pt 0 4pt; color: #374151; page-break-after: avoid; }
p { margin: 4pt 0; }
ul, ol { margin: 4pt 0; padding-left: 18pt; }
li { margin: 2pt 0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9.5pt;
       background: #f3f4f6; padding: 1pt 3pt; border-radius: 2pt; }
pre { background: #f8fafc; border: 1px solid #e5e7eb; border-left: 3px solid #1a56db;
      border-radius: 4pt; padding: 7pt 9pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; line-height: 1.5; }
table { width: 100%; border-collapse: collapse; margin: 7pt 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f1f5f9; font-weight: 600; white-space: nowrap; }
blockquote { margin: 7pt 0; padding: 6pt 10pt; background: #fffbeb;
             border-left: 3px solid #f59e0b; color: #44403c; }
hr { border: 0; border-top: 1px solid #e5e7eb; margin: 12pt 0; }
strong { font-weight: 600; }
a { color: #1a56db; text-decoration: none; }
.doc-meta { color: #6b7280; font-size: 9pt; margin-bottom: 12pt; }
"""


def _inline(s: str) -> str:
    """行内语法：转义 → 代码 → 粗体 → 斜体 → 链接。

    实现说明：替换目标一律用 lambda 回调构造，不在替换串里写 `\\\\1` 这类反向引用，
    以免转义层级过多导致难以定位的语法问题。
    """
    s = _html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", lambda m: "<code>" + m.group(1) + "</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: "<strong>" + m.group(1) + "</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", lambda m: "<em>" + m.group(1) + "</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: '<a href="' + m.group(2) + '">' + m.group(1) + "</a>", s)
    return s


def md_to_html(md: str, title: str) -> str:
    """极简 Markdown → HTML。只支持本项目用到的语法。"""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_buf: list[str] = []
    list_tag: str | None = None
    in_table = False

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    while i < len(lines):
        ln = lines[i]

        # 代码块
        if ln.strip().startswith("```"):
            if not in_code:
                close_list(); close_table()
                in_code, code_buf = True, []
            else:
                out.append("<pre><code>" + _html.escape("\n".join(code_buf)) + "</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue

        # 表格
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not in_table:
                close_list()
                out.append("<table><thead><tr>" +
                           "".join(f"<th>{_inline(c)}</th>" for c in cells) +
                           "</tr></thead><tbody>")
                in_table = True
                i += 1
                # 跳过对齐行
                if i < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i]):
                    i += 1
                continue
            if re.match(r"^\s*\|[\s:|-]+\|\s*$", ln):
                i += 1
                continue
            out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            i += 1
            continue
        close_table()

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            close_list()
            lv = len(m.group(1))
            out.append(f"<h{lv}>{_inline(m.group(2))}</h{lv}>")
            i += 1
            continue

        # 分隔线
        if re.match(r"^\s*(-{3,}|_{3,}|\*{3,})\s*$", ln):
            close_list()
            out.append("<hr/>")
            i += 1
            continue

        # 引用
        if ln.strip().startswith(">"):
            close_list()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + "<br/>".join(_inline(x) for x in buf) + "</blockquote>")
            continue

        # 列表
        m = re.match(r"^\s*([-*+]|\d+\.)\s+(.*)$", ln)
        if m:
            tag = "ul" if m.group(1) in "-*+" else "ol"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        # 空行
        if not ln.strip():
            close_list()
            i += 1
            continue

        # 普通段落
        close_list()
        out.append(f"<p>{_inline(ln)}</p>")
        i += 1

    close_list(); close_table()
    body = "\n".join(out)
    return (f"<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"/>"
            f"<title>{_html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")


def find_browser() -> str | None:
    for p in EDGE_CANDIDATES:
        if Path(p).exists():
            return p
    for name in ("msedge", "chrome", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    return None


def html_to_pdf(browser: str, html_path: Path, pdf_path: Path) -> bool:
    profile = resolve("runs/edgeprofile")
    profile.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}",
           html_path.as_uri()]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception as e:
        LOG.warning("调用浏览器失败：%s", e)
        return False
    return pdf_path.exists() and pdf_path.stat().st_size > 2000


def convert(md_path: Path, out_dir: Path, browser: str | None) -> tuple[bool, str]:
    md = md_path.read_text(encoding="utf-8")
    title = next((l.lstrip("# ").strip() for l in md.splitlines() if l.startswith("#")), md_path.stem)
    html = md_to_html(md, title)
    hp = out_dir / (md_path.stem + ".html")
    hp.write_text(html, encoding="utf-8")
    if browser is None:
        return False, f"已生成 HTML（未找到浏览器）：{hp.name}"
    pp = out_dir / (md_path.stem + ".pdf")
    if html_to_pdf(browser, hp, pp):
        return True, f"PDF {pp.stat().st_size / 1024:.0f} KB ← {md_path.name}"
    return False, f"PDF 生成失败（HTML 已生成：{hp.name}）"


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown 转 PDF")
    ap.add_argument("files", nargs="*", help="待转换的 md 文件")
    ap.add_argument("--all", action="store_true", help="转换 docs/ 下全部 md")
    ap.add_argument("--out", default="docs/pdf", help="输出目录")
    ap.add_argument("--no-edge", action="store_true", help="只生成 HTML，不转 PDF")
    args = ap.parse_args()

    if args.all:
        targets = sorted((ROOT / "docs").glob("*.md"))
    elif args.files:
        targets = [resolve(f) for f in args.files]
    else:
        targets = [resolve(f) for f in DEFAULT_DOCS]

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    browser = None if args.no_edge else find_browser()

    print("=" * 76)
    print("Markdown → PDF")
    print("=" * 76)
    print(f"  浏览器: {browser or '未找到（只输出 HTML）'}")
    print(f"  输出至: {out_dir}")
    print("-" * 76)
    ok_n = 0
    for t in targets:
        if not t.exists():
            print(f"  [跳过] {t.name} 不存在")
            continue
        ok, msg = convert(t, out_dir, browser)
        print(f"  [{'OK' if ok else '!!'}] {msg}")
        ok_n += 1 if ok else 0
    print("-" * 76)
    print(f"  完成 {ok_n} / {len(targets)}")
    if ok_n < len(targets) and browser is None:
        print("  提示：打开生成的 .html，用浏览器 Ctrl+P → 目标选“另存为 PDF”即可")
    print("=" * 76)


if __name__ == "__main__":
    main()
