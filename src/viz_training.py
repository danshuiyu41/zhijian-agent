"""生成「训练原理」示意图（交付物插图 + 文档配图）。

背景：设计说明书与答辩都需要一张能一眼看懂训练流程的图。
本脚本用极简的圆角矩形 + 箭头绘制流程，配色与前端设计系统保持一致（深色 + 青蓝渐变）。

用法：
    python -m src.viz_training
    python -m src.viz_training --out runs/figures/training_flow.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import get_logger, load_config, resolve  # noqa: E402

LOG = get_logger("viz.train")

# 配色说明：OpenCV 用 BGR 通道序，以下值由设计系统的 HEX 手工转换而来，
# 命名对应 web/design-system.css 的 token（C1/C2/C3 与等级色）。
BG = (15, 8, 5)          # #05080f 深靛黑
PANEL = (46, 30, 22)     # 卡片底（略亮于背景，形成层次）
PANEL_2 = (58, 38, 28)
C1 = (238, 211, 34)      # #22d3ee 青  → BGR(34,211,238)
C2 = (246, 130, 59)      # #3b82f6 蓝  → BGR(59,130,246)
C3 = (246, 92, 139)      # #8b5cf6 紫  → BGR(139,92,246)
LV_A = (94, 197, 34)     # #22c55e 绿  → BGR(34,197,94)
LV_C = (11, 158, 245)    # #f59e0b 橙  → BGR(11,158,245)
TXT = (255, 244, 234)    # #eaf4ff
SUB = (163, 129, 111)    # #6f81a3 次级文字


def rrect(img, x, y, w, h, r, color, thickness=-1):
    """圆角矩形。用四边形 + 四角圆填充，比"两个矩形叠加"更整洁。"""
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        # 描边：画四条直线 + 四段圆弧，避免出现"缺角"
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        for (cx, cy, a0) in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                             (x + r, y + h - r, 90), (x + w - r, y + h - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, thickness, cv2.LINE_AA)


def arrow(img, x1, y1, x2, y2, color, t=2):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA, tipLength=0.28)


def box(img, x, y, w, h, title, lines, accent, draw_fn):
    rrect(img, x, y, w, h, 14, PANEL, -1)
    rrect(img, x, y, w, h, 14, accent, 2)
    # 左侧色条
    rrect(img, x, y + 12, 5, h - 24, 3, accent, -1)
    img[:] = draw_fn(img, title, (x + 22, y + 26), 19, accent)
    yy = y + 60
    for ln in lines:
        img[:] = draw_fn(img, ln, (x + 22, yy), 15, TXT)
        yy += 27


def main() -> None:
    ap = argparse.ArgumentParser(description="生成训练原理流程图")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="runs/figures/training_flow.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from .viz_heatmap import draw_cjk_text as T

    W, H = 1500, 1000
    img = np.full((H, W, 3), BG, np.uint8)
    # 背景辉光
    for cx, cy, r, c in ((120, 60, 460, C1), (1400, 80, 420, C3), (760, 980, 520, C2)):
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, c, -1)
        img = cv2.addWeighted(overlay, 0.07, img, 0.93, 0)

    img = T(img, "智检 Agent · 缺陷检测模型训练原理", (56, 40), 34, TXT)
    img = T(img, "看图 → 猜标签 → 被纠正 → 调参数，循环几千次", (58, 90), 17, SUB)

    # ── 第一行：数据准备 ──
    y1 = 140
    bw, bh, gap = 440, 250, 46
    x1 = 56
    box(img, x1, y1, bw, bh, "① 造数据（自带答案）",
        ["程序合成缺陷图：1400 张",
         "划痕 / 夹杂 / 斑块 / 麻点",
         "氧化铁皮 / 龟裂 / 正常",
         "生成时就知道标签 → 标注免费"],
        C1, T)

    box(img, x1 + bw + gap, y1, bw, bh, "② 切小块（滑窗）",
        ["大图切成 224×224 小块",
         "每块继承整图的标签",
         "一张图 → 多个训练样本",
         "1400 张 → 几千个样本"],
        C2, T)

    x3 = x1 + 2 * (bw + gap)
    box(img, x3, y1, bw, bh, "③ 分三份（不许漏题）",
        ["训练集 980 张：用来学",
         "验证集 210 张：训练中抽查",
         "测试集 210 张：最后考一次",
         "三份互不重叠"],
        C3, T)

    arrow(img, x1 + bw + 8, y1 + bh // 2, x1 + bw + gap - 8, y1 + bh // 2, C1, 2)
    arrow(img, x1 + 2 * bw + gap + 8, y1 + bh // 2, x3 - 8, y1 + bh // 2, C2, 2)

    # ── 第二行：训练循环 ──
    y2 = y1 + bh + 74
    img = T(img, "训练循环（重复几千次）", (58, y2 - 42), 20, C1)

    lw = 300
    box(img, 56, y2, lw, 210, "前向：让模型猜",
        ["输入一个小块",
         "输出 7 个概率",
         "（正常/划痕/夹杂…）"], C1, T)
    box(img, 56 + lw + 30, y2, lw, 210, "算误差：看差多少",
        ["正确答案是「划痕」",
         "模型给 88% → 误差小",
         "模型给 20% → 误差大",
         "（交叉熵损失）"], LV_C, T)
    box(img, 56 + 2 * (lw + 30), y2, lw, 210, "反向：调参数",
        ["按误差方向微调",
         "几百万个内部参数",
         "让下次猜得更准",
         "（反向传播 + 梯度下降）"], C3, T)
    box(img, 56 + 3 * (lw + 30), y2, lw + 40, 210, "难点：防死记硬背",
        ["训练时随机把缺陷缩小",
         "随机降低对比度",
         "让模型见过「难样本」",
         "（否则测试 100% 但实战就废）"], LV_A, T)

    for i in range(3):
        xa = 56 + i * (lw + 30) + lw + 6
        arrow(img, xa, y2 + 105, xa + 18, y2 + 105, C2, 2)

    # 回环箭头
    cv2.arrowedLine(img, (56 + 3 * (lw + 30) + lw + 40, y2 + 210),
                    (56 + 3 * (lw + 30) + lw + 40, y2 + 250), C1, 2, cv2.LINE_AA, tipLength=0.2)
    cv2.line(img, (56 + 3 * (lw + 30) + lw + 40, y2 + 250), (80, y2 + 250), C1, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (80, y2 + 250), (80, y2 + 218), C1, 2, cv2.LINE_AA, tipLength=0.2)
    img = T(img, "循环几千次，直到误差不再下降", (110, y2 + 232), 15, C1)

    # ── 第三行：结果 ──
    y3 = y2 + 300
    rrect(img, 56, y3, W - 112, 110, 14, PANEL_2, -1)
    rrect(img, 56, y3, W - 112, 110, 14, LV_A, 2)
    img = T(img, "结果（用 210 张从没见过的图考试）", (86, y3 + 26), 20, LV_A)
    img = T(img, "测试集准确率 100%　|　参数量仅 1.08 M　|　CPU 上 40 分钟训完　|　真实落地：换工厂照片重训即可",
            (86, y3 + 66), 17, TXT)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    LOG.info("训练原理流程图已保存：%s  尺寸=%s", out, img.shape[:2])


if __name__ == "__main__":
    main()
