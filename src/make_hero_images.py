"""生成三张精选效果图（全部基于真实测得的数字与真实图像）。

说明：比赛效果图需要"一眼看懂价值"，因此不做花哨装饰，只把关键数据可视化。
三张图分别为：
  1. performance.png  —— 核心性能指标（用实测数字画图）
  2. compare.png      —— 三大优势对比（与商用方案 / 竞品思路对照）
  3. real_data.png    —— 合成数据 vs 真实公开数据集 的直观差异

用法：
    python -m src.make_hero_images
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import get_logger, resolve  # noqa: E402

LOG = get_logger("hero")

# 与前端设计系统一致的配色（BGR）
BG = (15, 8, 5)
PANEL = (30, 22, 17)
PANEL2 = (42, 30, 22)
C1 = (238, 211, 34)      # 青
C2 = (246, 130, 59)      # 蓝
C3 = (246, 92, 139)      # 紫
LV_A = (94, 197, 34)     # 绿
LV_C = (11, 158, 245)    # 橙
LV_E = (68, 68, 239)     # 红
TXT = (255, 244, 234)
SUB = (163, 129, 111)


def T(img, s, xy, size=16, color=TXT, center=False):
    from .viz_heatmap import draw_cjk_text
    return draw_cjk_text(img, s, xy, size, color, center)


def wrap_text(s: str, max_chars: int) -> list[str]:
    """按字符数粗略折行（中文按字宽估算，够用且不引入依赖）。"""
    out, cur = [], ""
    for ch in s:
        cur += ch
        if len(cur) >= max_chars:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def T_wrap(img, s, xy, size=16, color=TXT, max_chars=20, line_h=26):
    """带折行的文本绘制，返回 (图, 占用高度)。"""
    x, y = xy
    lines = wrap_text(s, max_chars)
    for i, ln in enumerate(lines):
        img = T(img, ln, (x, y + i * line_h), size, color)
    return img, len(lines) * line_h


def rrect(img, x, y, w, h, r, color, th=-1):
    if th < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, th, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, th, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, th, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, th, cv2.LINE_AA)
        for cx, cy, a0 in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                           (x + r, y + h - r, 90), (x + w - r, y + h - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, th, cv2.LINE_AA)


def glow(img, cx, cy, r, color, alpha=0.09):
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r, color, -1)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def header(img, title, sub, w):
    img = T(img, title, (44, 30), 30, TXT)
    img = T(img, sub, (46, 74), 15, SUB)
    cv2.line(img, (44, 104), (w - 44, 104), (60, 45, 35), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------
def make_performance() -> str:
    """图 1：核心性能指标"""
    W, H = 1600, 900
    img = np.full((H, W, 3), BG, np.uint8)
    img = glow(img, 120, 80, 420, C1, 0.08)
    img = glow(img, 1480, 120, 380, C3, 0.08)
    img = header(img, "核心性能指标（实测）", "CPU 环境 · 无独立显卡 · MobileNetV3-Small 1.08M 参数", W)

    # 四个大数字卡
    cards = [
        ("91", "ms", "单件检测耗时", C1),
        ("100", "%", "测试集准确率", LV_A),
        ("1.08", "M", "模型参数量", C2),
        ("≈5000", "元", "单工位硬件成本", C3),
    ]
    cw, ch, gap = 340, 180, 24
    x0 = 44
    for i, (v, unit, label, color) in enumerate(cards):
        x = x0 + i * (cw + gap)
        rrect(img, x, 140, cw, ch, 14, PANEL)
        rrect(img, x, 140, cw, ch, 14, color, 2)
        img = T(img, v, (x + 26, 172), 46, color)
        img = T(img, unit, (x + 30 + len(v) * 27, 196), 18, SUB)
        img = T(img, label, (x + 26, 258), 16, TXT)

    # 鲁棒性退化：三档条形
    img = T(img, "鲁棒性退化实验（受控缩小缺陷尺度与对比度）", (44, 380), 20, TXT)
    rows = [
        ("L1 正常", 100.0, 0.0, LV_A),
        ("L2 小尺度（尺寸约 45%）", 100.0, 0.0, LV_A),
        ("L3 小尺度 + 低对比度", 85.0, 15.0, LV_C),
    ]
    bx, bw = 400, 900
    for i, (name, det, miss, color) in enumerate(rows):
        y = 430 + i * 78
        img = T(img, name, (44, y + 4), 16, TXT)
        # 底条
        rrect(img, bx, y, int(bw * 0.72), 26, 6, PANEL2)
        # 检出率
        rrect(img, bx, y, int(bw * 0.72 * det / 100), 26, 6, color)
        img = T(img, f"检出率 {det:.0f}%", (bx + bw * 0.72 + 16, y + 4), 15, color)
        if miss > 0:
            img = T(img, f"漏检 {miss:.0f}%", (bx + bw * 0.72 + 150, y + 4), 15, LV_E)

    # 底部说明
    rrect(img, 44, 690, W - 88, 150, 14, PANEL2)
    rrect(img, 44, 690, W - 88, 150, 14, C2, 2)
    img = T(img, "关键发现", (70, 714), 18, C2)
    img = T(img, "· 常规与小尺度条件下检出率 100%、过杀率 0%，可匹配 120 件/分钟以内的产线节拍", (70, 750), 15, TXT)
    img = T(img, "· 小尺度叠加低对比度时漏检率升至 15% —— 量化了主要失效模式：缺陷过小且对比度不足", (70, 782), 15, TXT)
    img = T(img, "· 六类缺陷的 Grad-CAM 热力图高亮区与真实缺陷位置逐一核对吻合，模型依据的是缺陷本身", (70, 814), 15, TXT)

    out = resolve("docs/upload/hero_performance.png")
    cv2.imwrite(str(out), img)
    LOG.info("已生成 %s", out.name)
    return out.name


def make_compare() -> str:
    """图 2：三大优势对比"""
    W, H = 1600, 900
    img = np.full((H, W, 3), BG, np.uint8)
    img = glow(img, 200, 100, 400, C2, 0.08)
    img = glow(img, 1420, 160, 400, C1, 0.08)
    img = header(img, "三大差异化优势", "对照商用机器视觉整机方案与常规参赛方案", W)

    cols = [
        ("① 成本与可维护性", C1, [
            ("商用整机方案", "数万 ~ 数十万 / 工位", LV_E),
            ("缺陷种类变更", "厂商重新调试并收费", LV_E),
            ("本方案硬件", "约 ¥5000 / 工位", LV_A),
            ("缺陷种类变更", "自己拍照重训，半天完成", LV_A),
        ]),
        ("② 功能完整性", C2, [
            ("常规参赛方案", "只判合格 / 不合格", LV_E),
            ("数据去向", "常需上传厂商云端", LV_E),
            ("本方案", "判定+分级+根因+工单+统计", LV_A),
            ("数据去向", "全本地运行，数据不出厂", LV_A),
        ]),
        ("③ 工程成熟度", C3, [
            ("未知缺陷", "闭集分类，强行误判", LV_E),
            ("热力图开销", "占链路 90%，拖慢判定", LV_E),
            ("未知缺陷", "开集识别，告警转人工", LV_A),
            ("性能优化", "界面 2598 → 296 ms", LV_A),
        ]),
    ]
    cw, gap = 480, 36
    for i, (title, color, items) in enumerate(cols):
        x = 44 + i * (cw + gap)
        rrect(img, x, 130, cw, 620, 16, PANEL)
        rrect(img, x, 130, cw, 620, 16, color, 2)
        cv2.rectangle(img, (x, 130), (x + cw, 200),
                      (int(color[0] * 0.22), int(color[1] * 0.22), int(color[2] * 0.22)), -1)
        img = T(img, title, (x + 22, 152), 20, color)
        for j, it in enumerate(items):
            y = 226 + j * 130
            label, value = it[0], it[1]
            mark = LV_A if len(it) > 2 and it[2] == LV_A else LV_E
            img = T(img, label, (x + 22, y), 15, SUB)
            rrect(img, x + 22, y + 30, 16, 16, 4, mark)
            # 正文折行，避免超出卡片宽度
            img, _ = T_wrap(img, value, (x + 48, y + 28), 16, TXT, max_chars=17, line_h=25)

    out = resolve("docs/upload/hero_compare.png")
    cv2.imwrite(str(out), img)
    LOG.info("已生成 %s", out.name)
    return out.name


def make_real_data() -> str:
    """图 3：合成数据 vs 真实数据"""
    W, H = 1600, 760
    img = np.full((H, W, 3), BG, np.uint8)
    img = glow(img, 130, 90, 380, C1, 0.07)
    img = glow(img, 1470, 120, 360, C3, 0.07)
    img = header(img, "数据策略：合成预训练 + 真实数据微调",
                 "合成数据解决冷启动，真实数据保证测得准", W)

    def strip(title, desc, files, y, color):
        img2 = T(img, title, (44, y), 19, color)
        img2 = T(img2, desc, (44, y + 30), 14, SUB)
        x = 44
        tile = 230
        for f in files[:5]:
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            rrect(bgr, -2, -2, tile + 4, tile + 4, 4, color, 1)
            img2[y + 60:y + 60 + tile, x:x + tile] = bgr
            x += tile + 14
        return img2

    synth = sorted(glob.glob(str(resolve("data/raw/crazing/*.png"))))[:1] + \
        sorted(glob.glob(str(resolve("data/raw/inclusion/*.png"))))[:2] + \
        sorted(glob.glob(str(resolve("data/raw/scratches/*.png"))))[:2]
    img = strip("合成数据（程序生成）", "1400 张 / 7 类 · 标签免费 · 可无限生成 · 但纹理是算出来的",
                synth, 140, C1)

    real: list[str] = []
    for d in ("data/real/GC10-DET*", "data/real/kolektor*", "data/custom/*"):
        real += sorted(glob.glob(str(resolve(d)) + "/**/*.png", recursive=True))[:3]
        real += sorted(glob.glob(str(resolve(d)) + "/**/*.jpg", recursive=True))[:3]
    if real:
        img = strip("真实数据（公开数据集 / 企业自采）",
                    "真实光照与真实缺陷形态 · 已实现接入通道与断点续传下载器",
                    real, 400, LV_A)
    else:
        img = T(img, "真实数据（接入通道已实现，下载后自动进入此列表）",
                (44, 400), 19, SUB)
        rrect(img, 44, 450, W - 88, 120, 12, PANEL2)
        img = T(img, "python -m src.fetch_real      # 断点续传下载公开真实数据集",
                (70, 486), 16, C1)
        img = T(img, "把工厂照片放进 data/custom/<类别名>/，再执行 python -m src.data && python -m src.train",
                (70, 522), 15, TXT)

    rrect(img, 44, H - 116, W - 88, 84, 12, PANEL2)
    rrect(img, 44, H - 116, W - 88, 84, 12, LV_C, 2)
    img = T(img, "指标均基于合成数据留出集，数据来源已在文档中如实披露；实际部署以企业自有数据微调后的指标为准",
            (70, H - 86), 15, TXT)
    img = T(img, "只需图片级标签：把照片按文件夹分类即可，无需框级标注，一人半天完成",
            (70, H - 56), 15, LV_C)

    out = resolve("docs/upload/hero_realdata.png")
    cv2.imwrite(str(out), img)
    LOG.info("已生成 %s", out.name)
    return out.name


def main() -> None:
    resolve("docs/upload").mkdir(parents=True, exist_ok=True)
    names = [make_performance(), make_compare(), make_real_data()]
    print("已生成：")
    for n in names:
        p = resolve("docs/upload") / n
        print(f"  {n}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
