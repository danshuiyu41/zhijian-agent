"""生成热力图专题三联图：原始图 / 缺陷标注 / Grad-CAM 热力图。

用途：
1. 直观核验"模型到底在看哪里"——热力图高亮区应与真实缺陷位置吻合，
   否则说明模型可能依赖背景纹理等虚假特征（这是工业 AI 最常见的坑）；
2. 作为设计说明书"可解释性"章节的插图素材。

用法：
    python -m src.viz_heatmap
    python -m src.viz_heatmap --per-class 2 --out runs/figures/heatmap_study.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import class_list, get_logger, load_config, resolve  # noqa: E402

LOG = get_logger("viz.heatmap")


def draw_cjk_text(canvas: np.ndarray, text: str, xy: tuple[int, int],
                  size: int = 16, color: tuple[int, int, int] = (240, 244, 255),
                  center: bool = False) -> np.ndarray:
    """在图像上绘制中文文本（OpenCV 无法渲染中文，统一交给 PIL）。

    本函数被流程图与其它示意图生成脚本复用，避免各处重复实现字体探测。
    """
    from PIL import Image, ImageDraw

    from .infer import DefectInspector

    font = DefectInspector._find_cjk_font(size)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])
    x, y = xy
    if center:
        try:
            box = draw.textbbox((0, 0), text, font=font)
            x -= (box[2] - box[0]) // 2
        except Exception:
            x -= len(text) * size // 4
    try:
        draw.text((x, y), text, font=font, fill=rgb)
    except Exception:
        cv2.putText(canvas, text.encode("ascii", "replace").decode(), xy,
                    cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 1, cv2.LINE_AA)
        return canvas
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_cjk_bar(width: int, text: str, height: int = 28) -> np.ndarray:
    """绘制带中文的标题条。

    踩坑记录：OpenCV 的 putText 只支持 Hershey 矢量字体，**无法渲染中文**，
    实测会显示成 "??????"。这里统一交给 PIL，并复用 infer.py 的字体探测逻辑。
    """
    from PIL import Image, ImageDraw

    from .infer import DefectInspector

    bar = np.full((height, width, 3), 242, np.uint8)
    font = DefectInspector._find_cjk_font(15)
    pil = Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        draw.text((10, 6), text, font=font, fill=(60, 60, 60))
    except Exception:
        # 字体不可用时退回 ASCII 提示，而不是画出乱码
        cv2.putText(bar, text.encode("ascii", "replace").decode(), (10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        return bar
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成热力图专题图")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--tile", type=int, default=300)
    ap.add_argument("--out", default="runs/figures/heatmap_study.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from .infer import DefectInspector

    insp = DefectInspector.from_config(args.config)
    root = resolve(cfg["paths"]["data_raw"])
    rows: list[np.ndarray] = []

    codes = [c["code"] for c in class_list(cfg)] + ["clean"]
    zh = {c["code"]: c["zh"] for c in class_list(cfg)}
    zh["clean"] = "洁净"

    for code in codes:
        d = root / code
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        if not files:
            continue
        step = max(1, len(files) // args.per_class)
        picks = files[::step][:args.per_class]
        strips = []
        for f in picks:
            res, _ = insp.inspect(f, want_heatmap=False)
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            # 热力图：默认用模型预测类别，保证展示的是"模型判定依据"而非任意类别
            heat = insp.heatmap(gray)
            ann = insp.annotate(gray, res.detections)
            strips.append(np.hstack([
                cv2.cvtColor(cv2.resize(gray, (args.tile, args.tile)), cv2.COLOR_GRAY2BGR),
                cv2.resize(ann, (args.tile, args.tile)),
                cv2.resize(heat, (args.tile, args.tile)),
            ]))
            LOG.info("%-16s 检出=%d 等级=%s", code, len(res.detections), res.grade)
        if not strips:
            continue
        strip = np.hstack(strips)
        bar = _draw_cjk_bar(strip.shape[1],
                            f"{code} / {zh.get(code, '')}    原图 | 缺陷标注 | Grad-CAM 热力图")
        rows.append(np.vstack([bar, strip]))

    if not rows:
        raise SystemExit("未生成任何内容，请检查数据集")
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=245) for r in rows]
    img = np.vstack(rows)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    LOG.info("热力图专题图已保存：%s  尺寸=%s", out, img.shape[:2])


if __name__ == "__main__":
    main()
