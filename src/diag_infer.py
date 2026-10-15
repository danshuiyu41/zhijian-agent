"""检测结果诊断：输出框尺寸、面积占比与严重度分解，用于校准分级阈值。

用法：
    python -m src.diag_infer --n 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

from .infer import DefectInspector  # noqa: E402
from .utils import class_list, load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="检测结果诊断")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n", type=int, default=2, help="每类抽检张数")
    ap.add_argument("--fast", action="store_true", help="跳过定位细化（只测分类与耗时）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    insp = DefectInspector.from_config(args.config)
    if args.fast:
        insp.score_threshold = float(cfg["model"]["score_threshold"])

    print(f"图像尺寸基准：{cfg['dataset']['synthetic']['base_size']}×{cfg['dataset']['synthetic']['base_size']}")
    print(f"窗口 {insp.window}  步长 {insp.stride}  尺度 {insp.scales}  阈值 {insp.score_threshold}")
    print("严重度权重:", cfg["severity"]["weights"], "| area_ref:", cfg["severity"].get("area_ref"),
          "| transform:", cfg["severity"].get("area_transform"))
    print("=" * 108)
    print(f"{'类别':<12}{'等级':<6}{'缺陷':<12}{'置信度':<9}{'框(w×h)':<14}{'面积占比':<10}"
          f"{'风险项':<9}{'置信项':<9}{'面积项':<9}{'总分':<7}{'耗时ms':<8}")
    print("=" * 108)

    w = cfg["severity"]["weights"]
    root = resolve(cfg["paths"]["data_raw"])
    zh = {c["code"]: c["zh"] for c in class_list(cfg)}
    zh["clean"] = "洁净"
    for code in [c["code"] for c in class_list(cfg)]:
        d = root / code
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        step = max(1, len(files) // args.n)
        for f in files[::step][:args.n]:
            res, _ = insp.inspect(f, want_heatmap=False)
            row = f"{zh[code]:<12}{res.grade:<6}"
            if res.detections:
                t = res.detections[0]
                x1, y1, x2, y2 = t.bbox
                bw, bh = x2 - x1, y2 - y1
                rv = cfg["severity"]["risk_score"].get(t.risk, 0.35)
                ratio = t.area_ratio / max(cfg["severity"].get("area_ref", 0.05), 1e-6)
                av = min(1.0, ratio ** 0.5) if cfg["severity"].get("area_transform") == "sqrt" else min(1.0, ratio)
                print(row + f"{t.class_zh:<12}{t.confidence:<9.3f}{f'{bw}×{bh}':<14}"
                      f"{t.area_ratio * 100:<10.2f}{w['risk'] * rv * 100:<9.1f}"
                      f"{w['confidence'] * t.confidence * 100:<9.1f}{w['area'] * av * 100:<9.1f}"
                      f"{t.severity_score:<7.1f}{res.elapsed_ms:<8.0f}")
            else:
                print(row + "—")
    print("=" * 108)


if __name__ == "__main__":
    main()
