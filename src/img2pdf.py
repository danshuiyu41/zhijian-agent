"""把图片文件转成 PDF（比赛材料常见需求）。

平台常要求 PDF，但手头只有照片或截图时，需要做图片→PDF 的转换。
两种模式：
  · 每张图一页（默认）；
  · 自动裁边 + 增强（--clean）：照片常有灰边、偏色、透视，做轻度清理后更清晰。

用法：
    python -m src.img2pdf 图1.jpg 图2.jpg -o docs/pdf/申报表.pdf
    python -m src.img2pdf 图1.jpg 图2.jpg -o out.pdf --clean
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import get_logger, resolve  # noqa: E402

LOG = get_logger("img2pdf")


def load_pages(paths: list[str]) -> list[np.ndarray]:
    """读取图片，统一转 BGR 三通道。"""
    pages: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(resolve(p)), cv2.IMREAD_COLOR)
        if img is None:
            LOG.warning("无法读取：%s", p)
            continue
        pages.append(img)
    return pages


def clean_document(img: np.ndarray) -> np.ndarray:
    """文档照片轻度清理：自动裁掉背景边 + 灰度 + 对比度拉伸。

    注意：这是**轻度**处理，不做透视校正（那需要手动标定四角）。
    目标是让小票/表格照片更清晰、更省体积，而不是做出版级扫描。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 用大津法把较亮的纸张区域分离出来，据此裁掉四周深色背景
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 取最大轮廓
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # 只在裁掉的比例合理时采用（避免把整张图裁没）
        if 0.25 * img.shape[0] * img.shape[1] < w * h < 0.985 * img.shape[0] * img.shape[1]:
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            gray = gray[y0:y1, x0:x1]

    # 对比度拉伸：把 2%~98% 分位映射到 0~255
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    if hi - lo > 10:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_pdf(pages: list[np.ndarray], out_path: Path, dpi: int = 150,
             quality: int = 90) -> bool:
    """把多张图写成多页 PDF。用 JPEG 编码内嵌以控制体积。"""
    from PIL import Image

    if not pages:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs = []
    for p in pages:
        h, w = p.shape[:2]
        # 按 dpi 缩放：A4 宽 8.27 英寸
        target_w = int(8.27 * dpi)
        if w > target_w:
            scale = target_w / w
            p = cv2.resize(p, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        imgs.append(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)))
    imgs[0].save(str(out_path), "PDF", save_all=True, append_images=imgs[1:],
                 resolution=dpi, quality=quality, optimize=True)
    return out_path.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="图片转 PDF")
    ap.add_argument("images", nargs="+", help="按顺序排列的图片路径")
    ap.add_argument("-o", "--out", required=True, help="输出 PDF 路径")
    ap.add_argument("--clean", action="store_true", help="裁边 + 灰度 + 对比度增强")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--quality", type=int, default=90, help="JPEG 质量 1~95")
    args = ap.parse_args()

    pages = load_pages(args.images)
    if not pages:
        raise SystemExit("没有可用图片")
    if args.clean:
        pages = [clean_document(p) for p in pages]

    out = resolve(args.out)
    ok = save_pdf(pages, out, dpi=args.dpi, quality=args.quality)
    if ok:
        print(f"OK  {out}  {out.stat().st_size / 1024:.0f} KB  {len(pages)} 页")
    else:
        raise SystemExit("PDF 生成失败")


if __name__ == "__main__":
    main()
