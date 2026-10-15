"""真实数据集可达性实测（含断点续传）。

背景：真实照片无法凭空生成，只能从公开数据集获取。本脚本对已知候选源逐个实测，
并把可达的数据集下载到 data/real/。与 `src/fetch_data.py` 的区别：
  · fetch_data 面向"仓库归档型"数据集（zip 直链）；
  · 本脚本面向"大文件 + 会中途断流"的数据集，必须用断点续传。

候选源（按实测可达性排序）：
  1. KolektorSDD2 —— go.vicos.si，真实产线（电换向器）缺陷 3335 张，含像素掩膜
  2. VisA        —— AWS S3，12 类工业异常 10821 张，CC BY 4.0
  3. DAGM 2007   —— Zenodo 镜像，10 类弱监督工业光学检测

用法：
    python -m src.fetch_real              # 全部尝试
    python -m src.fetch_real --only kolektor
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .fetch_util import ResumableDownloader
from .utils import ensure_dirs, get_logger, load_config, resolve

LOG = get_logger("fetch.real")

TARGETS = {
    "kolektor": {
        "url": "https://go.vicos.si/kolektorsdd2",
        "desc": "KolektorSDD2 真实产线缺陷 3335 张（含像素掩膜）",
        "min_bytes": 30_000_000,
        "extract_to": "kolektorsdd2",
    },
    "visa": {
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "desc": "VisA 工业异常 10821 张（CC BY 4.0）",
        "min_bytes": 500_000_000,
        "extract_to": "visa",
    },
    "dagm": {
        "url": "https://zenodo.org/records/12750201/files/Class1.zip?download=1",
        "desc": "DAGM 2007 弱监督工业检测（第 1 类）",
        "min_bytes": 100_000_000,
        "extract_to": "dagm_class1",
    },
}


def try_download(key: str, spec: dict, raw_dir: Path) -> tuple[bool, str]:
    ext = ".tar" if spec["url"].endswith(".tar") else ".zip"
    dest = raw_dir / f"{key}{ext}"
    LOG.info("▶ %s", spec["desc"])
    d = ResumableDownloader(spec["url"], dest, chunk_mb=2, max_retries=6, timeout=45)
    # 断点续传：允许多轮，每轮失败后从已下载处继续
    for round_i in range(1, 4):
        if d.run():
            return True, f"{dest.name} ({dest.stat().st_size:,} B)"
        if not d.part.exists():
            break
        got = d.part.stat().st_size
        LOG.warning("第 %d 轮中断于 %s B，继续续传…", round_i, f"{got:,}")
        time.sleep(2)
    if d.part.exists():
        return False, f"多轮续传仍未完成（已获得 {d.part.stat().st_size:,} B）"
    return False, "不可达"


def main() -> None:
    ap = argparse.ArgumentParser(description="真实数据集可达性实测")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--only", choices=sorted(TARGETS), default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_dir = resolve(cfg["paths"]["data_raw"]) / "real"
    raw_dir.mkdir(parents=True, exist_ok=True)

    keys = [args.only] if args.only else sorted(TARGETS)
    results = {}
    print("=" * 76)
    print("真实数据集下载实测")
    print("=" * 76)
    for k in keys:
        ok, msg = try_download(k, TARGETS[k], raw_dir)
        results[k] = (ok, msg)
        print(f"  {'[成功]' if ok else '[失败]'} {k:<10} {msg}")
    print("-" * 76)
    good = [k for k, (ok, _) in results.items() if ok]
    if good:
        print(f"可用数据集：{'、'.join(good)}")
        print("下一步：python -m src.data_real --dataset " + good[0])
    else:
        print("本轮无可用数据集。已实现断点续传，可稍后重试；")
        print("若网络持续受限，改用自采照片：见 docs/真实数据采集指南.md")
    print("=" * 76)


if __name__ == "__main__":
    main()
