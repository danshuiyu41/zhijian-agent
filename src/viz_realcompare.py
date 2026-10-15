"""生成"合成 vs 真实"对比示意图，用于说明数据来源策略。

动机：答辩时评委最关心"你有没有真实数据"。一张并排对比图能直接说明
两者的视觉差异有多大、为什么必须用真实数据微调，比口头解释有效得多。

用法：
    python -m src.viz_realcompare
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import get_logger, load_config, resolve  # noqa: E402

LOG = get_logger("viz.real")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成/真实数据对比图")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="runs/figures/synth_vs_real.png")
    ap.add_argument("--tile", type=int, default=230)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from .viz_heatmap import draw_cjk_text as T

    rows: list[np.ndarray] = []

    def strip(title: str, files: list[str], note: str) -> np.ndarray | None:
        if not files:
            return None
        tiles = []
        for f in files[:5]:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (args.tile, args.tile), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(bgr, (0, 0), (args.tile - 1, args.tile - 1), (70, 70, 70), 1)
            tiles.append(bgr)
        if not tiles:
            return None
        while len(tiles) < 5:
            tiles.append(np.full((args.tile, args.tile, 3), 22, np.uint8))
        body = np.hstack(tiles)
        bar = np.full((56, body.shape[1], 3), 22, np.uint8)
        bar = T(bar, title, (12, 8), 19, (255, 244, 234))
        bar = T(bar, note, (12, 32), 13, (163, 129, 111))
        return np.vstack([bar, body])

    # 合成数据
    synth = sorted(glob.glob(str(resolve(cfg["paths"]["data_raw"]) / "*" / "*.png")))[:5]
    r1 = strip("合成数据（程序生成）", synth,
               "优点：标签免费、类别均衡、可无限生成　缺点：纹理是算出来的，与真实表面有差距")
    if r1 is not None:
        rows.append(r1)

    # 真实数据（若已接入）
    real = []
    for d in ("data/real/gc10det", "data/real/kolektorsdd2", "data/custom"):
        real += sorted(glob.glob(str(resolve(d) / "**" / "*.png"), recursive=True))[:3]
        real += sorted(glob.glob(str(resolve(d) / "**" / "*.jpg"), recursive=True))[:3]
    r2 = strip("真实数据（工厂实拍 / 自采）", real[:5],
               "优点：真实光照、真实纹理、真实缺陷形态　缺点：数量少、类别不均衡、标注费时")
    if r2 is not None:
        rows.append(r2)

    if not rows:
        raise SystemExit("没有可用的图像")

    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=22) for r in rows]

    head = np.full((72, width, 3), 15, np.uint8)
    head = T(head, "数据来源策略：合成预训练 + 真实数据微调", (20, 12), 27, (255, 244, 234))
    head = T(head, "合成数据解决「冷启动没数据」，真实数据保证「测得准」——两者配合使用",
             (22, 46), 14, (163, 129, 111))
    img = np.vstack([head] + rows)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    LOG.info("对比图已保存：%s  尺寸=%s", out, img.shape[:2])


if __name__ == "__main__":
    main()
