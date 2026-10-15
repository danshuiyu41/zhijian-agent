"""数据集可视化抽样：生成类别拼图，用于人工核验数据质量。

为什么需要它
------------
合成数据必须"肉眼可见地像缺陷"，否则训练出的模型没有工程意义。本脚本把每类
若干样本拼成一张带标签的大图，供开发者/质检专家快速目视评估，也可直接作为
设计说明书中的"数据集示例"插图。

用法：
    python -m src.visualize_data --per-class 6 --out runs/figures/dataset_samples.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .utils import class_list, get_logger, load_config, resolve

LOG = get_logger("viz.data")


def build_montage(root: Path, classes: list[str], per_class: int, tile: int = 160) -> np.ndarray:
    rows = []
    for cls in classes:
        d = root / cls
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if d.exists() else []
        if not files:
            continue
        step = max(1, len(files) // per_class)
        picks = files[::step][:per_class]
        tiles = []
        for f in picks:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (0, 0), (tile - 1, tile - 1), (80, 80, 80), 1)
            tiles.append(img)
        while len(tiles) < per_class:
            tiles.append(np.full((tile, tile, 3), 40, np.uint8))
        row = np.hstack(tiles)
        label_bar = np.full((28, row.shape[1], 3), 25, np.uint8)
        cv2.putText(label_bar, f"{cls}  (n={len(files)})", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([label_bar, row]))
    if not rows:
        raise SystemExit("未找到任何图像")
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=20) for r in rows]
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集抽样可视化")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--root", default=None, help="数据根目录，默认 config.paths.data_raw")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--out", default="runs/figures/dataset_samples.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = resolve(args.root) if args.root else resolve(cfg["paths"]["data_raw"])
    classes = [c["code"] for c in class_list(cfg)] + ["clean"]
    montage = build_montage(root, classes, args.per_class)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage)
    LOG.info("拼图已保存：%s  尺寸=%s", out, montage.shape[:2])


if __name__ == "__main__":
    main()
