"""GC10-DET / DeepPCB 等 GitHub 仓库型数据集下载与整理。

背景
----
本机网络实测：`github.com` 直连超时/连接被拒，但 `codeload.github.com`（GitHub 的
归档下载域）**可达**。因此本项目统一走 codeload 的 zip 归档地址获取公开数据集，
下载后按项目约定结构整理。

GC10-DET：3570 张真实钢板表面缺陷图，10 类，产线实采。
    - 来源：lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets
    - 论文：Deep Metallic Surface Defect Detection: the New Benchmark and Detection
      Network, Sensors 2020, 20(6):1562
    - 标注：PASCAL VOC XML（lable 目录）
    - 类别：10 类（冲孔/焊缝/油斑/折痕/水斑/腰折/丝斑/夹杂/轧入氧化皮/划痕）

DeepPCB：1500 对 640×640 PCB 图，6 类缺陷，含 txt 标注。
    - 来源：tangsanli5201/DeepPCB（仅限研究用途）

用法：
    python -m src.data_github --dataset gc10det
    python -m src.data_github --dataset deeppcb
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .data_real import download_with_progress, safe_extract
from .utils import ensure_dirs, get_logger, load_config, resolve

LOG = get_logger("data.github")

REPOS: dict[str, dict] = {
    "gc10det": {
        "url": "https://codeload.github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets/zip/refs/heads/master",
        "homepage": "https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets",
        "license": "学术引用（仓库未明示许可证，仅研究用途）",
        "domain": "钢板表面缺陷（真实产线）",
        "paper": "Lv et al., Deep Metallic Surface Defect Detection, Sensors 2020, 20(6):1562",
        "expect_min_bytes": 10_000_000,
    },
    "deeppcb": {
        "url": "https://codeload.github.com/tangsanli5201/DeepPCB/zip/refs/heads/master",
        "homepage": "https://github.com/tangsanli5201/DeepPCB",
        "license": "仅限研究用途（README 明示）",
        "domain": "PCB 线路缺陷（真实产线）",
        "paper": "Tang et al., PCB Defect Detection, 2019",
        "expect_min_bytes": 1_000_000,
    },
}

# GC10-DET 官方目录名到中文类名的映射（含常见拼写差异的兼容）
GC10_CLASS_ZH = {
    "1_chongkong": "冲孔",
    "2_hanfeng": "焊缝",
    "3_youdian": "油斑",
    "4_zhehen": "折痕",
    "5_shuiban": "水斑",
    "6_yaozhe": "腰折",
    "7_siban": "丝斑",
    "8_yiwu": "夹杂",
    "9_yaliang": "轧入氧化皮",
    "10_yahen": "压痕",
}


def _find_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def arrange_gc10det(extract_dir: Path, out_dir: Path) -> dict:
    """按官方目录名分到 10 个类别目录，并解析 VOC XML 统计缺陷框。"""
    images = _find_images(extract_dir)
    if not images:
        raise RuntimeError(f"{extract_dir} 下未找到图像")

    stats: dict[str, int] = {}
    boxes_total = 0
    labeled = 0
    index: list[dict] = []

    for img in images:
        # 类别目录：优先取官方中文映射，否则用原始目录名
        cls_dir = None
        for part in img.parts:
            if part.lower() in GC10_CLASS_ZH:
                cls_dir = GC10_CLASS_ZH[part.lower()]
                break
        if cls_dir is None:
            cls_dir = img.parent.name or "unknown"

        (out_dir / cls_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_dir / cls_dir / img.name)
        stats[cls_dir] = stats.get(cls_dir, 0) + 1

        # 找同名 XML
        xml = img.with_suffix(".xml")
        boxes = []
        if xml.exists():
            try:
                root = ET.parse(xml).getroot()
                for obj in root.findall("object"):
                    name = (obj.findtext("name") or "").strip()
                    bb = obj.find("bndbox")
                    if bb is None:
                        continue
                    boxes.append({
                        "name": name,
                        "xmin": int(float(bb.findtext("xmin", "0"))),
                        "ymin": int(float(bb.findtext("ymin", "0"))),
                        "xmax": int(float(bb.findtext("xmax", "0"))),
                        "ymax": int(float(bb.findtext("ymax", "0"))),
                    })
                if boxes:
                    labeled += 1
                    boxes_total += len(boxes)
            except Exception as e:
                LOG.debug("XML 解析失败 %s: %s", xml.name, e)

        index.append({
            "path": str((out_dir / cls_dir / img.name).relative_to(resolve("."))),
            "category": cls_dir,
            "boxes": boxes,
        })

    (out_dir / "index.json").write_text(
        json.dumps({"items": index, "classes": sorted(stats)}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"categories": len(stats), "images": sum(stats.values()),
            "with_boxes": labeled, "boxes": boxes_total, "per_class": stats}


def arrange_deeppcb(extract_dir: Path, out_dir: Path) -> dict:
    """DeepPCB：每张缺陷图配一个同名 txt 标注（x1,y1,x2,y2,type）。"""
    images = _find_images(extract_dir)
    if not images:
        raise RuntimeError(f"{extract_dir} 下未找到图像")
    # 模板图（无缺陷参考）与测试图混在同一目录，用文件名规则区分
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    index: list[dict] = []
    for img in images:
        shutil.copy2(img, out_dir / "images" / img.name)
        txt = img.with_suffix(".txt")
        boxes = []
        if txt.exists():
            shutil.copy2(txt, out_dir / "labels" / txt.name)
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.replace(";", ",").split(",")
                if len(parts) >= 5:
                    boxes.append({"x1": int(float(parts[0])), "y1": int(float(parts[1])),
                                  "x2": int(float(parts[2])), "y2": int(float(parts[3])),
                                  "type": parts[4].strip()})
            n_boxes += len(boxes)
        index.append({"path": img.name, "boxes": boxes})
    (out_dir / "index.json").write_text(
        json.dumps({"items": index}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"images": len(images), "boxes": n_boxes}


def get_github_dataset(cfg: dict, name: str, keep_zip: bool = False) -> dict:
    spec = REPOS[name]
    raw_dir = resolve(cfg["paths"]["data_raw"]) / "real"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = resolve("data/real") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = raw_dir / f"{name}.zip"
    if not download_with_progress(spec["url"], archive):
        raise SystemExit(f"下载失败：{spec['url']}")

    extract_dir = raw_dir / f"{name}_extract"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("解压中 ...")
        with zipfile.ZipFile(archive) as zf:
            safe_extract(zf, extract_dir)

    LOG.info("整理目录结构 ...")
    stats = arrange_gc10det(extract_dir, out_dir) if name == "gc10det" else arrange_deeppcb(extract_dir, out_dir)

    meta = {
        "dataset": name, "source_url": spec["url"], "homepage": spec["homepage"],
        "license": spec["license"], "domain": spec["domain"], "paper": spec["paper"],
        "archive_bytes": archive.stat().st_size if archive.exists() else None,
        "stats": stats, "arranged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("就绪：%s -> %s", name, out_dir)
    if not keep_zip:
        shutil.rmtree(extract_dir, ignore_errors=True)
        archive.unlink(missing_ok=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 GitHub 仓库型工业缺陷数据集")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dataset", choices=sorted(REPOS), default="gc10det")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    get_github_dataset(cfg, args.dataset, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
