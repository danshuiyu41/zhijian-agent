"""对比"服务器实际返回的页面"与"磁盘文件"，排查前端不更新的原因。

这类问题的典型成因：
1. 服务进程启动早于文件修改，模板/静态内容被缓存；
2. 浏览器缓存了旧的 HTML/CSS；
3. 服务器指向了另一份 web 目录。

用法：
    python -m src.check_served
"""
from __future__ import annotations

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8077"


def fetch(path: str) -> str:
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return r.read().decode("utf-8", "ignore")


def main() -> None:
    print("=" * 72)
    print("服务器返回内容 vs 磁盘文件 对比")
    print("=" * 72)

    try:
        html = fetch("/")
    except Exception as e:
        raise SystemExit(f"无法访问服务：{e}")

    disk_html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    disk_css = (ROOT / "web" / "design-system.css").read_text(encoding="utf-8")

    def h(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    print(f"  首页 HTML   服务端 {len(html):>7} 字符 sha={h(html)}")
    print(f"              磁盘   {len(disk_html):>7} 字符 sha={h(disk_html)}")
    same_html = h(html) == h(disk_html)
    print("              一致 [OK]" if same_html else "              不一致 [FAIL] —— 服务进程持有旧内容或指向其他目录")

    try:
        css = fetch("/static/design-system.css")
        print(f"  样式 CSS    服务端 {len(css):>7} 字符 sha={h(css)}")
        print(f"              磁盘   {len(disk_css):>7} 字符 sha={h(disk_css)}")
        print("              一致 [OK]" if h(css) == h(disk_css) else "              不一致 [FAIL]")
    except Exception as e:
        print(f"  样式获取失败：{e}")

    print("-" * 72)
    print("页面结构特征：")
    tabs = re.findall(r'data-tab="(\w+)"', html)
    legacy = 'class="logo"' in html      # 旧版导航标记（Python 3.10 的 f-string 不允许反斜杠，先算好）
    has_status = 'id="status"' in html
    has_statustext = 'statusText' in html
    print(f"  页签            {tabs}")
    print(f"  引用设计系统    {'design-system.css' in html}")
    print(f"  含内联 <style>  {'<style' in html}")
    print(f"  旧版标记 header.logo  {legacy}")
    print(f"  状态元素 id=status    {has_status}")
    print(f"  状态文本 statusText   {has_statustext}")
    m = re.search(r'<span class="status"[^>]*>.{0,120}', html, re.S)
    print(f"  状态片段        {m.group(0) if m else '未匹配'}")
    print("=" * 72)

    sys.exit(0 if same_html else 1)


if __name__ == "__main__":
    main()
