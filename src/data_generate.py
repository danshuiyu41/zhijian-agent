"""合成工业表面缺陷数据集生成器（冷启动 / 兜底数据方案）。

设计背景
--------
公开缺陷数据集（NEU-DET、GC10-DET 等）普遍把图片托管在百度网盘，自动化环境
无法获取；部分直链数据集（KolektorSDD2）服务器会中途掐断大文件下载。为保证
"作品在任何环境都能训练、演算、出指标"，本项目实现了一套**程序化合成带钢表面
缺陷**的生成器。

物理建模
--------
1. 基础纹理：轧制方向条纹 + 分形（多倍频）噪声 + 低频光照不均，模拟洁净金属表面；
2. 六类缺陷按形成机理分别建模：
   - 龟裂 crazing         ：晶界网状裂纹（多条随机折线交织，局部压暗）
   - 夹杂 inclusion       ：外来夹杂物压入（深色不规则块状，边界锐利 + 内部颗粒）
   - 斑块 patches         ：氧化/酸洗不良形成的片状明暗区（并抹除部分纹理）
   - 麻点 pitted_surface  ：密集小凹坑（随机散布的小圆点）
   - 压入氧化铁皮 rolled_in_scale：沿轧制方向的条带状压痕 + 伴随细裂纹
   - 划痕 scratches       ：细长直线/微弯线，方向偏轧制方向
3. 洁净样本 clean：只保留基础纹理，作为"合格品"负类。

工程价值
--------
- **可控可复现**：固定种子，可通过 `--per-class` 调节规模，便于消融实验；
- **可做鲁棒性压力测试**：缺陷尺度、对比度、密度均可参数化，用于评估模型在
  低对比度/小目标下的退化情况（设计说明书中的"压力测试"章节数据来源）；
- **合成预训练 + 真实微调**：与真实数据形成迁移学习路线，解决工业界"缺陷样本
  稀缺、冷启动无数据"的核心痛点。

用法：
    python -m src.data_generate --per-class 200
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .utils import class_codes, class_list, ensure_dirs, get_logger, load_config, resolve, set_seed

LOG = get_logger("data.gen")

DEFECT_CODES: tuple[str, ...] = (
    "crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches",
)


# ----------------------------------------------------------------------
# 1. 基础纹理合成
# ----------------------------------------------------------------------
def _fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """分形（多倍频）噪声：模拟金属表面微观起伏。

    每一倍频在更小的网格上采样随机场，再上采样叠加，得到自然的 1/f 频谱。
    """
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        scale = 2 ** o
        gh, gw = max(2, h // scale + 2), max(2, w // scale + 2)
        small = rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * big
        total += amp
        amp *= 0.55
    return out / max(total, 1e-6)


def synth_base_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    """合成洁净的轧制金属表面纹理（float32，值域约 0~255）。"""
    x = np.arange(size, dtype=np.float32)[None, :]
    line_freq = rng.uniform(0.15, 0.55)
    phase = rng.uniform(0, 2 * np.pi)
    stripes = np.tile(np.sin(2 * np.pi * line_freq * x / size * 8 + phase), (size, 1))

    noise = _fractal_noise(size, size, rng, octaves=5)
    grad = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), size / 6.0)
    grad = (grad - grad.min()) / max(float(grad.max() - grad.min()), 1e-6) - 0.5

    img = (128.0
           + 16.0 * stripes * rng.uniform(0.35, 1.0)
           + 20.0 * noise
           + 26.0 * grad)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.1))
    return np.clip(img, 0, 255).astype(np.float32)


# ----------------------------------------------------------------------
# 2. 六类缺陷叠加算子
# ----------------------------------------------------------------------
def _add_crazing(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """龟裂：晶界网状裂纹。"""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(6, 13))):
        pts = [np.array([rng.uniform(0, w), rng.uniform(0, h)], np.float32)]
        direction = rng.uniform(0, 2 * np.pi)
        for _step in range(int(rng.integers(3, 8))):
            direction += rng.normal(0, 0.7)
            step = rng.uniform(0.06, 0.16) * w
            pts.append(pts[-1] + np.array([np.cos(direction), np.sin(direction)], np.float32) * step)
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 1.0,
                      int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 0.7), 0, 1) ** 0.85
    return img - rng.uniform(28, 55) * mask


def _add_inclusion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """夹杂：深色不规则块状夹杂物。"""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
        ry, rx = rng.uniform(0.05, 0.20) * h, rng.uniform(0.05, 0.22) * w
        n = int(rng.integers(7, 12))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = 1.0 + rng.uniform(-0.45, 0.45, n)
        poly = np.stack([cx + rx * rad * np.cos(ang), cy + ry * rad * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1.0, lineType=cv2.LINE_AA)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.6, 1.6)), 0, 1)
    img = img - rng.uniform(45, 85) * mask
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.5)
    return img + 12.0 * grain * mask


def _add_patches(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """斑块：片状氧化/酸洗不良区域，并抹除部分纹理。"""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        ry, rx = rng.uniform(0.10, 0.26) * h, rng.uniform(0.10, 0.28) * w
        blob = np.zeros((h, w), np.float32)
        cv2.ellipse(blob, (int(cx), int(cy)), (int(rx), int(ry)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        mask += cv2.GaussianBlur(blob, (0, 0), rng.uniform(3.0, 9.0))
    mask = np.clip(mask, 0, 1)
    shift = float(rng.choice([-1.0, 1.0])) * rng.uniform(18, 40)
    img = img + shift * mask
    smooth = cv2.GaussianBlur(img, (0, 0), 4.0)
    return img * (1 - 0.6 * mask) + smooth * (0.6 * mask)


def _add_pitted_surface(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """麻点：密集小凹坑。"""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    count = int(rng.integers(60, 220))
    for y, x, r in zip(rng.integers(0, h, count), rng.integers(0, w, count),
                       rng.integers(1, max(2, int(0.012 * h)), count)):
        cv2.circle(mask, (int(x), int(y)), int(r), 1.0, -1, lineType=cv2.LINE_AA)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 0.8), 0, 1)
    return img - rng.uniform(30, 60) * mask


def _add_rolled_in_scale(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """压入氧化铁皮：沿轧制方向的条带状压痕 + 细密裂纹。"""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    y0 = rng.uniform(0.1, 0.9) * h
    thickness = rng.uniform(0.04, 0.17) * h
    length = rng.uniform(0.5, 1.0) * w
    x0 = rng.uniform(0.0, max(0.0, w - length))
    axis = np.linspace(x0, x0 + length, 64)
    wobble = y0 + rng.normal(0, thickness * 0.35, 64)
    cv2.polylines(mask, [np.stack([axis, wobble], 1).astype(np.int32)], False, 1.0,
                  max(2, int(thickness)), lineType=cv2.LINE_AA)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 1.4), 0, 1)
    sign = float(rng.choice([-1.0, 1.0]))
    img = img + sign * rng.uniform(22, 48) * mask

    fine = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 8))):
        yy = y0 + rng.normal(0, thickness * 0.5)
        xs = np.sort(rng.uniform(x0, x0 + length, 2))
        cv2.line(fine, (int(xs[0]), int(yy)), (int(xs[1]), int(yy + rng.normal(0, 3))), 1.0, 1, cv2.LINE_AA)
    return img - 20.0 * cv2.GaussianBlur(fine, (0, 0), 0.6) * mask


def _add_scratches(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """划痕：细长线状缺陷，方向偏轧制方向。"""
    h, w = img.shape
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 7))):
        ang = rng.normal(0, 0.35) if rng.random() < 0.7 else rng.uniform(0, np.pi)
        y0, x0 = rng.uniform(0, h), rng.uniform(0, w)
        length = rng.uniform(0.4, 1.1) * w
        x1, y1 = x0 + length * np.cos(ang), y0 + length * np.sin(ang)
        curve = rng.normal(0, 2.5)
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 1, cv2.LINE_AA)
        cv2.line(mask, (int(x0), int(y0 + curve)), (int(x1), int(y1 + curve)), 0.6, 1, cv2.LINE_AA)
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 0.55), 0, 1) ** 0.9
    return img + float(rng.choice([-1.0, 1.0])) * rng.uniform(35, 70) * mask


DEFECT_OPS = {
    "crazing": _add_crazing,
    "inclusion": _add_inclusion,
    "patches": _add_patches,
    "pitted_surface": _add_pitted_surface,
    "rolled_in_scale": _add_rolled_in_scale,
    "scratches": _add_scratches,
}


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def _apply_defect(code: str, img: np.ndarray, rng: np.random.Generator,
                  contrast: float = 1.0) -> np.ndarray:
    """叠加一次缺陷，并按系数衰减对比度。

    做法：对"原图 + 缺陷"与"原图"求差，按 contrast 缩放差值再加回原图。
    这样无需修改各缺陷算子即可统一控制缺陷强弱，用于构造低对比度压力测试集。
    """
    out = DEFECT_OPS[code](img, rng)
    if contrast >= 0.999:
        return out
    return img + contrast * (out - img)


def make_sample(code: str, size: int, rng: np.random.Generator,
                scale: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """生成单张样本；code 为 'clean' 时返回洁净表面。

    参数
    ----
    scale    : 缺陷几何尺度缩放系数。1.0 为正常训练数据；<1 生成更小缺陷，
               用于构造**鲁棒性压力测试集**（评估小目标下的性能退化）。
               实现方式：先在放大后的画布上叠加缺陷，再等比缩回目标尺寸。
    contrast : 缺陷对比度系数。<1 时缺陷更弱，用于评估低对比度条件的表现。
    """
    contrast = max(0.05, float(contrast))
    if code == "clean":
        img = synth_base_texture(size, rng)
    else:
        if code not in DEFECT_OPS:
            raise KeyError(f"未知缺陷类别：{code}")
        if abs(scale - 1.0) < 1e-6:
            img = synth_base_texture(size, rng)
        else:
            big = int(round(size / max(scale, 0.05)))
            big = min(max(big, size), size * 8)      # 限制上界，避免内存膨胀
            img = synth_base_texture(big, rng)
        for _ in range(int(rng.integers(1, 3))):
            img = _apply_defect(code, img, rng, contrast)
        if abs(scale - 1.0) >= 1e-6 and img.shape[0] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img * rng.uniform(0.92, 1.08) + rng.uniform(-8, 8)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_dataset(cfg: dict, per_class: int, clean_multiplier: float = 1.0) -> dict:
    """生成完整数据集并落盘。

    clean_multiplier：洁净样本相对于"每类缺陷张数"的倍数，默认 1.0（即与单类缺陷同量）。
    """
    out_root = resolve(cfg["paths"]["data_raw"])
    size = int(cfg["dataset"]["synthetic"].get("base_size", 256))
    seed = int(cfg["dataset"]["seed"])

    codes = class_codes(cfg)
    manifest: list[dict] = []
    counts: dict[str, int] = {}

    for ci, code in enumerate(codes):
        cls_dir = out_root / code
        cls_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed * 1000 + ci)
        for k in range(per_class):
            img = make_sample(code, size, rng)
            fp = cls_dir / f"{code}_{k:04d}.png"
            cv2.imwrite(str(fp), img)
            manifest.append({"path": str(fp.relative_to(resolve("."))), "label": code, "index": ci})
        counts[code] = per_class
        LOG.info("生成类别 %-16s %4d 张", code, per_class)

    clean_dir = out_root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    n_clean = max(per_class, int(round(per_class * clean_multiplier)))
    rng = np.random.default_rng(seed * 9999)
    for k in range(n_clean):
        img = make_sample("clean", size, rng)
        fp = clean_dir / f"clean_{k:04d}.png"
        cv2.imwrite(str(fp), img)
        manifest.append({"path": str(fp.relative_to(resolve("."))), "label": "clean", "index": -1})
    counts["clean"] = n_clean
    LOG.info("生成类别 %-16s %4d 张", "clean(洁净表面)", n_clean)

    meta = {
        "dataset": "Synthetic-Steel-Surface-Defect",
        "generator": "procedural-synthesis (src/data_generate.py)",
        "note": "程序化合成的带钢表面缺陷数据，用于冷启动训练与鲁棒性压力测试；"
                "真实数据接口见 src/data_real.py 与 src/data_github.py。",
        "image_size": size,
        "counts": counts,
        "total": sum(counts.values()),
        "seed": seed,
        "classes": codes,
        "class_zh": {c["code"]: c["zh"] for c in class_list(cfg)},
    }
    (out_root / "manifest.json").write_text(
        json.dumps({"meta": meta, "items": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("数据集生成完成：总计 %d 张 -> %s", meta["total"], out_root)
    return meta


def generate_stress_set(cfg: dict, per_class: int = 40) -> dict:
    """生成**鲁棒性压力测试集**（不参与训练，用于评估退化）。

    分三档难度，每档每类若干张：
      L1 正常      scale=1.0  contrast=1.0   （与训练分布一致，作为对照）
      L2 小尺度    scale=0.45 contrast=1.0   （缺陷几何尺寸约为正常的一半）
      L3 低对比度  scale=0.45 contrast=0.40  （既小又弱，模拟难例）

    输出到 `data/stress/<档位>/<类别>/`，由 `src/eval_robust.py` 读取评估。
    这组数据是"模型在什么条件下会失效"的量化依据，也是设计说明书中
    失效边界分析（知识库条目 TECH-LIMITS）的实测支撑。
    """
    out_root = resolve("data/stress")
    size = int(cfg["dataset"]["synthetic"].get("base_size", 256))
    seed = int(cfg["dataset"]["seed"])
    levels = {"L1_normal": (1.0, 1.0), "L2_small": (0.45, 1.0), "L3_small_lowcontrast": (0.45, 0.40)}
    codes = class_codes(cfg)
    summary: dict[str, dict] = {}

    for li, (level, (scale, contrast)) in enumerate(levels.items()):
        per_level: dict[str, int] = {}
        for ci, code in enumerate(codes + ["clean"]):
            d = out_root / level / code
            d.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed * 7717 + li * 131 + ci)
            n = per_class
            for k in range(n):
                img = make_sample(code, size, rng,
                                  scale=scale if code != "clean" else 1.0,
                                  contrast=contrast if code != "clean" else 1.0)
                cv2.imwrite(str(d / f"{code}_{k:04d}.png"), img)
            per_level[code] = n
        summary[level] = {"scale": scale, "contrast": contrast, "counts": per_level}
        LOG.info("压力集 %-22s scale=%.2f contrast=%.2f 共 %d 张",
                 level, scale, contrast, sum(per_level.values()))

    meta = {"dataset": "Synthetic-Steel-Surface-Defect-Stress",
            "purpose": "鲁棒性压力测试（不参与训练）",
            "image_size": size, "levels": summary,
            "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("压力测试集生成完成 -> %s", out_root)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="合成工业表面缺陷数据集")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--per-class", type=int, default=None, help="每类缺陷生成张数")
    ap.add_argument("--clean-multiplier", type=float, default=1.0, help="洁净样本相对单类缺陷的倍数")
    ap.add_argument("--stress", action="store_true", help="额外生成鲁棒性压力测试集")
    ap.add_argument("--stress-per-class", type=int, default=40, help="压力集每档每类张数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["dataset"]["seed"]))
    per_class = args.per_class or int(cfg["dataset"]["synthetic"]["per_class"])
    generate_dataset(cfg, per_class, args.clean_multiplier)
    if args.stress:
        generate_stress_set(cfg, args.stress_per_class)


if __name__ == "__main__":
    main()
