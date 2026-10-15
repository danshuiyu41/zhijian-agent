"""可视化核验：为每类缺陷生成"原始图 / 标注图 / 热力图"三联图。

用途：
1. 人工核验检测框是否收敛（本项目开发中曾出现框链式膨胀覆盖全图、中文标签乱码两个问题）；
2. 直接把三联图作为设计说明书与作品简介的插图素材。

用法：
    python -m src.visualize_result
    python -m src.visualize_result --per-class 2 --out runs/figures/detection_samples.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import class_list, get_logger, load_config, resolve  # noqa: E402

LOG = get_logger("viz.result")


def _label_bar(w: int, text: str, height: int = 26) -> np.ndarray:
    bar = np.full((height, w, 3), 26, np.uint8)
    cv2.putText(bar, text.encode("ascii", "replace").decode(), (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def main() -> None:
    ap = argparse.ArgumentParser(description="生成检测三联图用于人工核验")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--out", default="runs/figures/detection_samples.png")
    ap.add_argument("--tile", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from .infer import DefectInspector

    insp = DefectInspector.from_config(args.config)
    root = resolve(cfg["paths"]["data_raw"])
    rows: list[np.ndarray] = []

    for code in [c["code"] for c in class_list(cfg)] + ["clean"]:
        d = root / code
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        if not files:
            continue
        step = max(1, len(files) // args.per_class)
        picks = files[::step][:args.per_class]
        tiles: list[np.ndarray] = []
        for f in picks:
            res, viz = insp.inspect(f, want_heatmap=True)
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            origin = cv2.cvtColor(cv2.resize(gray, (args.tile, args.tile)), cv2.COLOR_GRAY2BGR)
            ann = insp.annotate(gray, res.detections)
            heat = insp.heatmap(gray)
            strip = np.hstack([
                cv2.resize(origin, (args.tile, args.tile)),
                cv2.resize(ann, (args.tile, args.tile)),
                cv2.resize(heat, (args.tile, args.tile)),
            ])
            tiles.append(strip)
            top = res.detections[0] if res.detections else None
            LOG.info("%-16s grade=%s defects=%d %s", code, res.grade, len(res.detections),
                     f"top={top.class_zh} {top.confidence:.2f} {top.bbox}" if top else "")
        if not tiles:
            continue
        strip = np.hstack(tiles)
        bar = np.full((26, strip.shape[1], 3), 26, np.uint8)
        cv2.putText(bar, f"{code}   [origin | annotated | grad-cam]", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, strip]))

    if not rows:
        raise SystemExit("未生成任何可视化，请检查数据集是否为空")
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=18) for r in rows]
    out_img = np.vstack(rows)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), out_img)
    LOG.info("三联图已保存：%s  尺寸=%s", out, out_img.shape[:2])


if __name__ == "__main__":
    main()
