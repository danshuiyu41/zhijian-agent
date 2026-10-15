"""界面链路耗时分解：定位那 1.7 秒到底花在哪。

背景：`bench_ui` 实测界面链路约 1.7 秒，但"关掉热力图"并不变快，
说明此前的归因（认为瓶颈是 Grad-CAM 反向传播）是错的。
本脚本在**同进程内**逐段计时，把链路拆开，找出真正的瓶颈。

注意：这是进程内测，不含网络与 HTTP 解析，因此与 bench_ui 的差值
即为"网络 + 上传 + 序列化"的开销。

用法：
    python -m src.bench_breakdown
    python -m src.bench_breakdown --n 3
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402

from .utils import get_logger, load_config, resolve  # noqa: E402

LOG = get_logger("bench.split")


def main() -> None:
    ap = argparse.ArgumentParser(description="界面链路耗时分解")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    import base64
    import json

    from .agent import QualityAgent
    from .db import Database
    from .infer import DefectInspector

    insp = DefectInspector.from_config(args.config)
    agent = QualityAgent.from_config(args.config)
    db = Database.from_config(args.config)

    files = sorted(glob.glob("data/raw/*/*.png"))
    step = max(1, len(files) // args.n)
    picks = files[::step][: args.n]

    def med(fn) -> float:
        ts = []
        for f in picks:
            t0 = time.time()
            fn(f)
            ts.append((time.time() - t0) * 1000)
        return statistics.median(ts)

    print("=" * 78)
    print(f"界面链路耗时分解（{len(picks)} 张，同进程测，不含网络）")
    print("=" * 78)

    # 0) 文件读取
    t_read = med(lambda f: open(f, "rb").read())

    # 1) 滑窗推理（无热力图）
    t_detect = med(lambda f: insp.inspect(f, want_heatmap=False))

    # 2) 热力图（单独测）
    def heat_only(f: str):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        insp.heatmap(g)
    t_heat = med(heat_only)

    # 3) 标注图 + base64（含二次读盘）
    def annotate_b64(f: str):
        r, _ = insp.inspect(f, want_heatmap=False)
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        insp.annotate_base64(g, r.detections)
    t_annot = med(annotate_b64) - t_detect     # 扣除检测部分

    # 4) 智能体
    r0, _ = insp.inspect(picks[0], want_heatmap=False)
    payload0 = r0.to_dict()
    t_agent = statistics.median([
        (lambda t0: (agent.analyze(payload0), (time.time() - t0) * 1000)[1])(time.time())
        for _ in range(3)
    ])

    # 5) 落库
    def db_write(_f: str):
        db.save_inspection(dict(payload0), image_path=picks[0])
    t_db = med(db_write)

    # 6) JSON 序列化（响应体）
    def json_ser(_f: str):
        json.dumps({"result": payload0, "viz": {"heatmap_png": "x" * 130000}})
    t_json = med(json_ser)

    rows = [
        ("① 读取上传文件", t_read, "1 次磁盘读"),
        ("② 滑窗推理（检测）", t_detect, "双尺度滑窗 + 融合 + NMS"),
        ("③ Grad-CAM 热力图", t_heat, "反向传播"),
        ("④ 标注图绘制 + base64", max(t_annot, 0), "含二次读盘"),
        ("⑤ 智能体检索与建议", t_agent, "知识库检索 + 工单"),
        ("⑥ 落库（SQLite）", t_db, "含 WAL 写盘"),
        ("⑦ JSON 序列化", t_json, "含 base64 大字段"),
    ]
    print(f"  {'环节':<26}{'中位耗时':>10}{'占比':>9}   说明")
    print("-" * 78)
    total = sum(max(v, 0) for _, v, _ in rows) or 1
    for name, v, note in rows:
        v = max(v, 0)
        print(f"  {name:<26}{v:>8.0f} ms{v / total * 100:>8.0f}%   {note}")
    print("-" * 78)
    print(f"  {'合计（不含网络）':<26}{total:>8.0f} ms")

    print("=" * 78)
    print("结论：")
    top = sorted(rows, key=lambda x: -max(x[1], 0))[:3]
    for i, (name, v, _) in enumerate(top, 1):
        print(f"  {i}. {name} —— {max(v, 0):.0f} ms（{max(v, 0) / total * 100:.0f}%）")
    print(f"\n  提示：bench_ui 实测界面约 1700 ms，本表合计 {total:.0f} ms，")
    print(f"        差值为 HTTP 上传/解析/网络开销。")
    print("=" * 78)


if __name__ == "__main__":
    main()
