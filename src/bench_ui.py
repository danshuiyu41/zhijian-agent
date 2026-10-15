"""测量界面实际链路的响应耗时（与 bench_speed 的区别）。

为什么要单独测：
  `bench_speed` 测的是**纯推理**（91ms）。但界面走的是
  `/api/inspect/agent?heatmap=true`，其中包含：
    1. 多尺度滑窗推理
    2. 候选融合 + NMS
    3. **Grad-CAM 热力图**（需反向传播，最慢的一环）
    4. 智能体检索与工单生成
    5. 落库 + 标注图编码成 base64
  现场演示用的是这条链路，所以必须测它，而不是测纯推理。

用法：
    python -m src.bench_ui            # 需要服务已启动
    python -m src.bench_ui --n 5
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8077"


def post_file(path: str, fp: str, fields: dict, timeout: int = 300) -> tuple[dict, float]:
    boundary = "----zj" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    name = Path(fp).name
    data = Path(fp).read_bytes()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + path, data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out, (time.time() - t0) * 1000


def main() -> None:
    # 注意：`global` 必须在函数体最前面，不能出现在 BASE 已被引用之后
    # （argparse 默认值里引用了 BASE，会把 global 位置判定为非法）。
    global BASE

    ap = argparse.ArgumentParser(description="界面链路耗时实测")
    ap.add_argument("--n", type=int, default=5, help="每种模式测几张")
    ap.add_argument("--base", default="http://127.0.0.1:8077")
    args = ap.parse_args()

    BASE = args.base.rstrip("/")

    files = sorted(glob.glob("data/raw/*/*.png"))
    if not files:
        raise SystemExit("未找到测试图像")
    step = max(1, len(files) // args.n)
    picks = files[::step][: args.n]

    print("=" * 76)
    print(f"界面链路耗时实测（{len(picks)} 张，目标 {BASE}）")
    print("=" * 76)
    print("注意：耗时含网络与索引图/热力图编码，反映的是**现场演示的真实体感**")
    print("-" * 76)

    def run(label: str, fields: dict) -> float:
        ts = []
        for f in picks:
            try:
                _, ms = post_file("/api/inspect/agent", f, fields)
                ts.append(ms)
            except Exception as e:
                print(f"  {label}: 请求失败 {type(e).__name__}: {e}")
                return -1
        med = statistics.median(ts)
        p90 = sorted(ts)[int(len(ts) * 0.9) - 1] if len(ts) > 1 else ts[0]
        print(f"  {label:<34} 中位 {med:>8.0f} ms　最慢 {max(ts):>8.0f} ms")
        return med

    m_heat = run("① 检测 + 热力图（界面默认）", {"heatmap": "true", "agent": "true", "persist": "true"})
    m_noheat = run("② 检测，不生成热力图", {"heatmap": "false", "agent": "true", "persist": "true"})
    m_agent = run("③ 检测，无热力图、不落库", {"heatmap": "false", "agent": "false", "persist": "false"})

    print("-" * 76)
    if m_heat > 0 and m_noheat > 0:
        print(f"  热力图带来的额外耗时：{m_heat - m_noheat:.0f} ms"
              f"（占 {100 * (m_heat - m_noheat) / max(m_heat, 1):.0f}%）")
    print()
    print("结论与建议：")
    if m_heat > 1000:
        print(f"  ⚠ 界面默认链路 {m_heat:.0f} ms（约 {1000 / m_heat:.1f} 件/秒）——")
        print("    现场演示能接受，但**不要用它做产线节拍论证**。")
        print(f"  建议：演示时先说「检测 {m_noheat:.0f} ms」，再单独展示热力图作为复核功能，")
        print("        把两者分开讲，避免评委把热力图耗时当成检测耗时。")
    else:
        print(f"  界面链路 {m_heat:.0f} ms，现场演示体感良好。")
    print("=" * 76)


if __name__ == "__main__":
    main()
