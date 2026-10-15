"""数据集与数据加载。

数据来源（三层策略，保证系统在任何环境都能训练与演示）
-----------------------------------------------------
1. **合成数据**：`src/data_generate.py` 程序化生成的 6 类带钢表面缺陷（冷启动）。
2. **真实公开数据**：GC10-DET（3570 张 / 10 类）、KolektorSDD2（3335 张）、
   DeepPCB 等，通过 `src/data_github.py` / `src/data_real.py` 下载整理。
3. **企业自有数据**：放入 `data/custom/<类别名>/` 即可被自动识别（部署阶段使用）。

统一约定
--------
- 分类任务的类别由 `config.yaml` 的 `dataset.classes` 定义（6 类缺陷）+ `clean`（洁净）；
- 目录结构：`<root>/<类别名>/*.{png,jpg}`；
- 训练/验证/测试按 `dataset.val_ratio`、`test_ratio` **分层抽样**，固定随机种子可复现；
- 归一化统计量从训练集与合成集联合估计，落盘到 `runs/norm_stats.json`，推理时复用。

用法：
    python -m src.data                  # 构建索引与划分，写出统计量
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .utils import ensure_dirs, get_logger, load_config, resolve, set_seed

LOG = get_logger("data")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------
# 索引与划分
# ----------------------------------------------------------------------
def scan_split_dir(root: Path, classes: list[str]) -> dict[str, list[Path]]:
    """扫描 `<root>/<class>/*` 结构，返回 {class: [paths]}。"""
    out: dict[str, list[Path]] = {}
    for c in classes:
        d = root / c
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if files:
            out[c] = files
    return out


def stratified_split(items: dict[str, list[Path]], val_ratio: float, test_ratio: float,
                     seed: int = 42) -> dict[str, dict[str, list[Path]]]:
    """按类别分层切分，返回 {split: {class: [paths]}}。"""
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[str, list[Path]]] = {"train": {}, "val": {}, "test": {}}
    for cls, files in items.items():
        idx = rng.permutation(len(files))
        n = len(files)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        # 每类至少留 1 张训练样本
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - 1 - n_test))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        for split, sel in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            if len(sel):
                splits[split][cls] = [files[i] for i in sel]
    return splits


def build_index(cfg: dict, use_real: bool = True, use_synth: bool = True) -> dict:
    """构建训练索引：合成数据 + 真实数据（可选）+ 自定义数据（可选）。

    返回结构：
        {
          "train": {"<class>": [相对路径, ...]},
          "val":   {...},
          "test":  {...},
          "sources": {...},
          "norm": {"mean": [...], "std": [...]}
        }
    """
    dcfg = cfg["dataset"]
    classes = [c["code"] for c in dcfg["classes"]]
    label_space = classes + ["clean"]

    pools: dict[str, list[Path]] = {c: [] for c in label_space}
    sources: dict[str, int] = {}

    if use_synth:
        synth_root = resolve(cfg["paths"]["data_raw"])
        got = scan_split_dir(synth_root, label_space)
        for k, v in got.items():
            pools[k].extend(v)
            sources[f"synthetic:{k}"] = len(v)

    if use_real:
        # GC10-DET 的 10 类中文目录名 -> 本项目 6 类英文 code 的映射（按语义近似对齐）
        gc10_map = {
            "夹杂": "inclusion",
            "轧入氧化皮": "rolled_in_scale",
            "划痕": "scratches",
            "折痕": "patches",
            "腰折": "patches",
            "冲孔": "pitted_surface",
            "水斑": "patches",
            "油斑": "patches",
            "丝斑": "scratches",
            "压痕": "pitted_surface",
            "焊缝": "crazing",
        }
        real_root = resolve("data/real/gc10det")
        if real_root.exists():
            for sub in sorted(p for p in real_root.iterdir() if p.is_dir()):
                target = gc10_map.get(sub.name)
                if not target:
                    continue
                files = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMG_EXTS)
                pools[target].extend(files)
                sources[f"gc10det:{sub.name}->{target}"] = len(files)

    custom_root = resolve("data/custom")
    if custom_root.exists():
        got = scan_split_dir(custom_root, label_space)
        for k, v in got.items():
            pools[k].extend(v)
            sources[f"custom:{k}"] = len(v)

    non_empty = {k: v for k, v in pools.items() if v}
    if not non_empty:
        raise SystemExit(
            "没有任何训练数据。请先执行：\n"
            "  python -m src.data_generate --per-class 180   # 生成合成数据（兜底，必做）\n"
            "  python -m src.data_github --dataset gc10det    # 下载真实数据（可选，加分）"
        )

    splits = stratified_split(non_empty, dcfg["val_ratio"], dcfg["test_ratio"], dcfg["seed"])
    root = resolve(".")
    index = {
        "splits": {
            s: {c: [str(p.relative_to(root)) for p in v] for c, v in d.items()}
            for s, d in splits.items()
        },
        "sources": sources,
        "classes": classes,
        "label_space": label_space,
        "counts": {s: {c: len(v) for c, v in d.items()} for s, d in splits.items()},
    }
    return index


def compute_norm_stats(index: dict, cfg: dict, max_images: int = 400) -> dict:
    """估计灰度归一化统计量（本项目输入统一按单通道灰度处理）。"""
    rng = np.random.default_rng(cfg["dataset"]["seed"])
    files: list[str] = []
    for c, v in index["splits"]["train"].items():
        files.extend(v)
    if not files:
        for s in ("val", "test"):
            for c, v in index["splits"][s].items():
                files.extend(v)
    rng.shuffle(files)
    files = files[:max_images]

    import cv2

    acc = []
    for rel in files:
        img = cv2.imread(str(resolve(rel)), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        acc.append(img.reshape(-1).astype(np.float32) / 255.0)
    if not acc:
        return {"mean": [0.5], "std": [0.25]}
    cat = np.concatenate(acc)
    mean = float(cat.mean())
    std = float(cat.std()) or 0.25
    return {"mean": [round(mean, 5)], "std": [round(std, 5)], "samples": len(acc), "pixels": int(cat.size)}


def load_index(path: str | Path = "runs/dataset_index.json") -> dict:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def save_index(index: dict, cfg: dict) -> Path:
    out = resolve(cfg["paths"]["runs"]) / "dataset_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------------------------
def make_transforms(cfg: dict, train: bool, norm: dict):
    """构造 torchvision 变换。输入统一转单通道灰度再复制为 3 通道（兼容 ImageNet 预训练主干）。"""
    from torchvision import transforms as T

    size = int(cfg["dataset"]["image_size"])
    mean, std = norm["mean"], norm["std"]
    # 3 通道归一化参数
    mean3, std3 = mean * 3, std * 3

    if train:
        a = cfg["train"]["augment"]
        ops = [T.Grayscale(num_output_channels=3)]
        if a.get("rotate90"):
            ops.append(T.RandomApply([T.RandomRotation((90, 90))], p=a["rotate90"]))
        if a.get("hflip"):
            ops.append(T.RandomHorizontalFlip(p=a["hflip"]))
        if a.get("vflip"):
            ops.append(T.RandomVerticalFlip(p=a["vflip"]))

        # 难度增强：把"小尺度 / 低对比度"缺陷引入训练分布。
        #
        # 动机（实测驱动）：初版训练集全部是 scale=1.0、contrast=1.0 的"标准难度"缺陷，
        # 结果验证集准确率直接到 100%，模型输出置信度高度集中于 0.92~0.99——既没有
        # 区分度，也无法用后处理温度缩放修正（该模型欠自信，缩放只会让分布更尖锐，
        # 详见 src/calibrate.py 的负面结论记录）。
        # 让模型在训练期就见到更小、更弱的缺陷，是唯一能从根源上拉开置信度分布的办法，
        # 同时直接改善鲁棒性（对应压力档 L3 的漏检问题）。
        dcfg = cfg["dataset"].get("difficulty_augment", {}) or {}
        if dcfg.get("enabled", False):
            ops.append(T.RandomApply(
                [T.RandomAffine(degrees=0,
                                scale=(float(dcfg.get("min_zoom", 1.0)),
                                       float(dcfg.get("max_zoom", 1.6))))],
                p=float(dcfg.get("zoom_p", 0.35))))
            ops.append(T.RandomApply(
                [T.ColorJitter(contrast=(float(dcfg.get("min_contrast", 0.55)), 1.0))],
                p=float(dcfg.get("contrast_p", 0.35))))

        ops.append(T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)))
        if a.get("color_jitter"):
            ops.append(T.ColorJitter(brightness=a["color_jitter"], contrast=a["color_jitter"]))
        ops += [T.ToTensor(), T.Normalize(mean3, std3)]
        if a.get("random_erasing"):
            ops.append(T.RandomErasing(p=a["random_erasing"], scale=(0.02, 0.08)))
        return T.Compose(ops)

    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean3, std3),
    ])


class DefectDataset:
    """轻量数据集：按 (路径, 标签) 列表读取，返回 (tensor, label)。

    说明：用 torchvision 的 ImageFolder 无法满足"多来源 + 自定义划分"的需要，
    故此处实现一个最小 Dataset（不继承 torch.utils.data.Dataset 也可以被
    DataLoader 使用，但继承后语义更清晰）。
    """

    def __init__(self, items: list[tuple[str, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        path, label = self.items[i]
        with Image.open(resolve(path)) as im:
            im = im.convert("L")
            x = self.transform(im)
        return x, label


def build_items(index: dict, split: str, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for cls, paths in index["splits"].get(split, {}).items():
        if cls not in class_to_idx:
            continue
        items.extend((p, class_to_idx[cls]) for p in paths)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="构建数据集索引与归一化统计量")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--no-real", action="store_true", help="不使用真实数据集")
    ap.add_argument("--no-synth", action="store_true", help="不使用合成数据集")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["dataset"]["seed"]))

    index = build_index(cfg, use_real=not args.no_real, use_synth=not args.no_synth)
    LOG.info("数据来源分布：")
    for k, v in sorted(index["sources"].items()):
        LOG.info("   %-34s %6d", k, v)
    for split in ("train", "val", "test"):
        LOG.info("%-5s 合计 %5d 张 | %s", split, sum(index["counts"][split].values()),
                 index["counts"][split])

    norm = compute_norm_stats(index, cfg)
    LOG.info("归一化统计量：%s", norm)
    index["norm"] = norm

    out = save_index(index, cfg)
    LOG.info("索引已写入 %s", out)


if __name__ == "__main__":
    main()
