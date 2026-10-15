"""一键运行全部自检（VSCode 里按 F5 选「⑦ 全部校验」即可）。

顺序执行所有检查脚本，最后汇总通过情况。任何一项失败都会在汇总中标红。

用法：
    python check_all.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

CHECKS: list[tuple[str, list[str], bool]] = [
    # (显示名, 命令, 是否需要服务在线)
    ("源码语法体检", ["-m", "src.check_syntax"], False),
    ("前端静态自检", ["-m", "src.check_frontend"], False),
    ("CSS 变量一致性", ["-m", "src.check_vars"], False),
    ("CSS 语法体检", ["-m", "src.check_css"], False),
    ("配色对比度", ["-m", "src.check_contrast"], False),
    ("横向溢出检查", ["-m", "src.check_overflow"], False),
    ("主干与 Grad-CAM 自检", ["-m", "src.selfcheck_model"], False),
    ("端到端链路自检", ["-m", "src.selftest_e2e", "--per-class", "1"], False),
    ("接口测试（需服务在线）", ["-m", "src.test_api"], True),
]


def main() -> None:
    print("=" * 78)
    print("智检 Agent · 全部自检")
    print("=" * 78)
    results: list[tuple[str, bool, float]] = []

    for name, args, needs_server in CHECKS:
        print(f"\n▶ {name}")
        t0 = time.time()
        p = subprocess.run([PY, "-u", *args], cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8", errors="ignore")
        dt = time.time() - t0
        ok = p.returncode == 0
        results.append((name, ok, dt))
        tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()][-2:]
        for l in tail:
            print("   " + l.strip())
        if not ok:
            err = [l for l in (p.stderr or "").strip().splitlines() if l.strip()][-3:]
            for l in err:
                print("   ! " + l.strip())
            if needs_server:
                print("   ! 提示：接口测试需要先启动服务（python start.py）")

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    ok_n = 0
    for name, ok, dt in results:
        flag = "通过" if ok else "失败"
        mark = "  " if ok else "!!"
        print(f"  {mark} {name:<24} {flag}   {dt:5.1f}s")
        ok_n += 1 if ok else 0
    print("-" * 78)
    print(f"  {ok_n} / {len(results)} 项通过")
    if ok_n < len(results):
        print("  失败项多为环境问题：请确认解释器为 D:\\python\\python.exe，")
        print("  并在运行接口测试前先执行 python start.py")
    print("=" * 78)
    sys.exit(0 if ok_n == len(results) else 1)


if __name__ == "__main__":
    main()
