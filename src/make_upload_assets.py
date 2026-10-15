"""生成比赛上传所需的图片素材（作品封面 + 效果图，自动满足体积限制）。

平台要求（见上传页面提示）：
  · 作品封面：350×350，≤ 500 KB
  · 作品效果图：每张 ≤ 1 MB，最多 5 张

本脚本产出一套符合要求的素材到 `docs/upload/`：
  1. cover_350.png       —— 封面（程序绘制，深色科技风，与系统界面一致）
  2. effect_1..5.png     —— 效果图（从 runs/figures 挑选并压缩到 1MB 以内）

用法：
    python -m src.make_upload_assets
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import get_logger, resolve  # noqa: E402

LOG = get_logger("upload")

COVER_SIZE = 350
COVER_LIMIT = 500 * 1024
EFFECT_LIMIT = 1024 * 1024

# 效果图选取：按"最能说明问题"排序，并给出配图说明
EFFECT_PLAN = [
    ("runs/figures/detection_samples.png", "原图 / 缺陷标注 / Grad-CAM 三联对照"),
    ("runs/figures/heatmap_study.png", "六类缺陷的可解释性验证"),
    ("runs/figures/confusion_matrix.png", "测试集混淆矩阵"),
    ("runs/figures/training_flow.png", "训练原理流程图"),
    ("runs/figures/robustness.png", "鲁棒性退化实验"),
    # 备选（若上面某张缺失会自动往后取）
    ("runs/figures/training_curves.png", "训练曲线"),
    ("runs/figures/calibration.png", "置信度校准"),
    ("runs/figures/dataset_samples.png", "数据集样例"),
]


def _font(size: int):
    from .infer import DefectInspector
    return DefectInspector._find_cjk_font(size)


def _text(img, s: str, xy, size: int, color, center: bool = False):
    from PIL import Image, ImageDraw
    font = _font(size)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    x, y = xy
    if center:
        try:
            b = d.textbbox((0, 0), s, font=font)
            x -= (b[2] - b[0]) // 2
        except Exception:
            x -= len(s) * size // 4
    rgb = (color[2], color[1], color[0])
    try:
        d.text((x, y), s, font=font, fill=rgb)
    except Exception:
        cv2.putText(img, s.encode("ascii", "replace").decode(), xy,
                    cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 1, cv2.LINE_AA)
        return img
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_cover() -> bool:
    """绘制 350×350 封面。

    设计取舍（前两版偏丑的教训）：
      v1 堆了网格线 + 四角取景框 + 检测框 + 底部色条 + 三行小字，元素过多、
          层次混乱，缩到 350px 后更显拥挤。
      本版改为**极简 + 单一焦点**：
        · 去掉了网格线与四角框，只保留一层极淡渐变光晕营造质感；
        · 主标题放大到 44px，成为唯一视觉焦点；
        · 副标题与一句话定位各居一行，信息层级清晰；
        · 用一个圆形 Logo 标记替代杂乱的装饰图形；
        · 底部只留一条细渐变色条，不放小字（小字在 350px 下根本看不清）。
    """
    S = COVER_SIZE
    img = np.full((S, S, 3), 13, np.uint8)

    # 两层柔和光晕：左上青、右下紫。先大半径低透明铺底，再收小半径提亮焦点。
    for cx, cy, r, c, a, blur in ((70, 52, 230, (238, 211, 34), 0.16, 26),
                                  (296, 300, 220, (246, 92, 139), 0.14, 26),
                                  (185, 96, 120, (238, 211, 34), 0.10, 30)):
        ov = img.copy()
        cv2.circle(ov, (cx, cy), r, c, -1)
        img = cv2.addWeighted(ov, a, img, 1 - a, 0)
        img = cv2.GaussianBlur(img, (0, 0), blur)

    C1 = (238, 211, 34)

    # 圆形 Logo 标记（居中偏上）
    cy0 = 90
    cv2.circle(img, (S // 2, cy0), 29, C1, 2, cv2.LINE_AA)
    img = _text(img, "智", (S // 2, cy0 - 17), 29, C1, center=True)

    # 主标题：唯一焦点
    img = _text(img, "智检 Agent", (S // 2, 136), 42, (255, 246, 238), center=True)

    # 渐变色条：居中，宽度约为画布的一半
    y = 196
    for x in range(78, S - 78):
        t = (x - 78) / max(1, (S - 156))
        r = int(34 + (139 - 34) * t)
        g = int(211 + (92 - 211) * t)
        b = int(238 + (246 - 238) * t)
        img[y, x] = (b, g, r)
        img[y + 1, x] = (int(b * 0.6), int(g * 0.6), int(r * 0.6))

    # 副标题两行
    img = _text(img, "工业表面缺陷智能检测", (S // 2, 214), 16, (206, 218, 240), center=True)
    img = _text(img, "与处置智能体系统", (S // 2, 238), 16, (206, 218, 240), center=True)

    # 一句话定位
    img = _text(img, "检出 · 分级 · 根因 · 工单", (S // 2, 282), 13, (148, 164, 194), center=True)

    out = resolve("docs/upload/cover_350.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    size = out.stat().st_size
    ok = size <= COVER_LIMIT
    LOG.info("封面 %s  %s B（上限 500 KB）%s", out.name, f"{size:,}", "OK" if ok else "超限")
    return ok


def _save_under_limit(src: Path, dst: Path, limit: int) -> tuple[bool, int]:
    """把图像保存到 dst，必要时逐步降采样/降质，直到 ≤ limit。"""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False, 0
    work = img
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42):
        w = max(1, int(work.shape[1] * scale))
        h = max(1, int(work.shape[0] * scale))
        resized = cv2.resize(work, (w, h), interpolation=cv2.INTER_AREA) if scale < 1.0 else work
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok and len(buf) <= limit:
            dst.write_bytes(buf.tobytes())
            return True, len(buf)
        for q in (80, 70, 60):
            ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok and len(buf) <= limit:
                dst.write_bytes(buf.tobytes())
                return True, len(buf)
    return False, 0


def main() -> None:
    out_dir = resolve("docs/upload")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("比赛上传素材生成")
    print("=" * 74)
    ok = make_cover()
    print(f"  {'[OK]' if ok else '[超限]'} cover_350.png  （350×350）")

    used = 0
    print("\n  效果图（每张 ≤ 1 MB，最多 5 张）：")
    for src_rel, desc in EFFECT_PLAN:
        if used >= 5:
            break
        src = resolve(src_rel)
        if not src.exists():
            continue
        used += 1
        dst = out_dir / f"effect_{used}.jpg"
        good, size = _save_under_limit(src, dst, EFFECT_LIMIT)
        if good:
            print(f"    [OK] effect_{used}.jpg  {size/1024:7.1f} KB  ← {src.name}")
            print(f"         {desc}")
        else:
            used -= 1
            print(f"    [跳过] {src.name} 压缩后仍超限")

    print("\n" + "=" * 74)
    print(f"素材已生成到 docs/upload/（封面 1 张 + 效果图 {used} 张）")
    print("=" * 74)


if __name__ == "__main__":
    main()
