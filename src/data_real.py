"""真实工业缺陷数据接入：下载 + 解压 + 整理为统一目录结构。

支持的数据集（优先选择免注册直链，适配受限网络环境）：

1. KolektorSDD2  —— 真实产线（电换向器表面）缺陷检测数据集
   - 3335 张（356 缺陷 / 2979 正常），约 230×630，含像素级掩膜
   - 直链免注册，CC BY-NC-SA 4.0（非商用，竞赛/研究用途合规）
   - 已实测本机网络可达
2. VisA         —— Amazon 视觉异常检测数据集（含 4 个 PCB 子集）
   - 10821 张，CC BY 4.0，S3 直链免注册

设计说明
--------
公开数据集多为"异常检测"范式（正常 + 异常，带掩膜），而工业产线质检同时需要
"缺陷分类 + 定位"。本模块把下载到的真实数据统一整理成项目约定的目录结构：

    data/real/<dataset>/
        normal/       正常样本
        defect/       缺陷样本
        masks/        像素级掩膜（若有）
        meta.json     统计信息与来源（含许可证，供申报书引用）

用法：
    python -m src.data_real --dataset kolektorsdd2
    python -m src.data_real --dataset kolektorsdd2 --keep-zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from .fetch_data import UA, _ssl_ctx
from .utils import ensure_dirs, get_logger, load_config, resolve

LOG = get_logger("data.real")

SOURCES: dict[str, dict] = {
    "kolektorsdd2": {
        "url": "https://go.vicos.si/kolektorsdd2",
        "homepage": "https://www.vicos.si/resources/kolektorsdd2/",
        "license": "CC BY-NC-SA 4.0 (非商用)",
        "domain": "电换向器表面（真实产线）",
        "paper": "Božič et al., Mixed supervision for surface-defect detection, Computers in Industry 2021",
        "expect_min_bytes": 50_000_000,
    },
    "visa": {
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "homepage": "https://github.com/amazon-science/spot-diff",
        "license": "CC BY 4.0",
        "domain": "12 类物体（含 PCB）",
        "paper": "Zou et al., Spot-the-Difference, ECCV 2022",
        "expect_min_bytes": 1_000_000_000,
        "is_tar": True,
    },
}


def _safe_unlink(p: Path, attempts: int = 5) -> None:
    """Windows 下文件可能被索引/杀软短暂占用，重试删除。"""
    import gc
    import time as _t

    for i in range(attempts):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            _t.sleep(0.4 * (i + 1))
        except OSError:
            _t.sleep(0.4 * (i + 1))
    LOG.warning("删除失败（可能被占用）：%s", p)


def download_with_progress(url: str, dest: Path, timeout: int = 900,
                           min_valid_bytes: int = 1_000_000) -> bool:
    """流式下载，带进度日志与**完整性校验**。

    重要：早期版本只要文件存在就跳过下载，导致"半截包 / HTML 错误页"被当成有效
    缓存，后续解压报 BadZipFile。现在改为：存在但小于 min_valid_bytes 或文件头
    不是压缩包魔数时，一律删除重下。
    """
    if dest.exists() and dest.stat().st_size > 0:
        size = dest.stat().st_size
        with open(dest, "rb") as f:
            head = f.read(8)
        magic_ok = (head[:2] == b"PK") or head.startswith(b"Rar!") or head.startswith(b"7z\xbc")
        if size >= min_valid_bytes and magic_ok:
            LOG.info("已存在有效压缩包，跳过下载：%s (%s B)", dest.name, f"{size:,}")
            return True
        LOG.warning("缓存文件不可用（%s B, magic=%s），删除后重新下载", f"{size:,}", magic_ok)
        _safe_unlink(dest)

    tmp = dest.with_suffix(dest.suffix + ".part")
    _safe_unlink(tmp)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        resp = urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx())
        total = int(resp.headers.get("Content-Length") or 0)
        got, last = 0, 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got - last > 20 * (1 << 20):
                    last = got
                    pct = f"{100 * got / total:.1f}%" if total else "?"
                    LOG.info("  下载中 %s / %s (%s) %.1f MB/s", f"{got:,}", f"{total:,}" if total else "?",
                             pct, got / 1e6 / max(0.1, time.time() - t0))
        resp.close()   # 显式关闭，避免 Windows 句柄未释放导致后续 unlink 失败
    except Exception as e:
        LOG.error("下载失败：%s: %s", type(e).__name__, e)
        _safe_unlink(tmp)
        return False

    size = tmp.stat().st_size
    with open(tmp, "rb") as f:
        head = f.read(4)
    if size < min_valid_bytes:
        LOG.error("下载内容过小（%s B < %s B），判定失败", f"{size:,}", f"{min_valid_bytes:,}")
        _safe_unlink(tmp)
        return False
    if not (head[:2] == b"PK" or head.startswith(b"Rar!") or head.startswith(b"7z\xbc")):
        LOG.error("下载内容不是压缩包（文件头 %r），判定失败", head)
        _safe_unlink(tmp)
        return False
    try:
        if dest.exists():
            _safe_unlink(dest)
        tmp.replace(dest)
    except OSError as e:
        LOG.error("移动文件失败：%s", e)
        return False
    LOG.info("下载完成：%s (%s B) 用时 %.1fs", dest.name, f"{size:,}", time.time() - t0)
    return True


def safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    """防目录穿越的解压。"""
    target = target.resolve()
    for member in zf.namelist():
        dest = (target / member).resolve()
        if not str(dest).startswith(str(target)):
            raise RuntimeError(f"压缩包内含非法路径：{member}")
    zf.extractall(target)


def arrange_kolektorsdd2(extract_dir: Path, out_dir: Path) -> dict:
    """把 KolektorSDD2 原始结构整理为 normal/defect/masks。

    原始结构（vicos 官方 zip）通常为：
        KolektorSDD2/train/...  test/...  或  直接一堆 <id>.png + <id>_GT.png
    兼容处理：凡文件名含 `_GT` 视为掩膜；掩膜全黑 => 正常，否则缺陷。
    """
    import cv2
    import numpy as np

    pngs = sorted(p for p in extract_dir.rglob("*.png") if "_GT" not in p.name.upper())
    if not pngs:
        raise RuntimeError(f"未在 {extract_dir} 找到 PNG 图像，目录结构可能不同，请检查")

    for sub in ("normal", "defect", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    n_normal = n_defect = 0
    for p in pngs:
        stem = p.stem
        gt_candidates = [p.with_name(f"{stem}_GT.png"), p.with_name(f"{stem}_gt.png")]
        gt = next((g for g in gt_candidates if g.exists()), None)
        is_defect = False
        if gt is not None:
            mask = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.size and int(mask.max()) > 0:
                is_defect = True
                shutil.copy2(gt, out_dir / "masks" / gt.name)
        else:
            # 无掩膜时按目录名推断（如 train/defect/...）
            lower = str(p.parent).lower()
            is_defect = any(k in lower for k in ("defect", "bad", "anomal", "abnormal"))
        if is_defect:
            shutil.copy2(p, out_dir / "defect" / p.name)
            n_defect += 1
        else:
            shutil.copy2(p, out_dir / "normal" / p.name)
            n_normal += 1
    return {"normal": n_normal, "defect": n_defect,
            "masks": len(list((out_dir / "masks").glob("*.png")))}


def arrange_visa(extract_dir: Path, out_dir: Path) -> dict:
    """把 VisA 的 <class>/{Data,ImageSets,Masks} 结构整理为统一目录。"""
    for sub in ("normal", "defect", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    n_normal = n_defect = n_mask = 0
    # VisA 异常清单在 split_csv/1cls.csv
    anomaly: set[str] = set()
    for csv in extract_dir.rglob("*.csv"):
        try:
            for line in csv.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3 and parts[-1] == "anomaly":
                    anomaly.add(Path(parts[-2]).stem)
        except Exception:
            pass
    for img in extract_dir.rglob("*.JPG"):
        stem = img.stem
        if stem in anomaly:
            shutil.copy2(img, out_dir / "defect" / img.name)
            n_defect += 1
            m = list(extract_dir.rglob(f"{stem}.png"))
            if m:
                shutil.copy2(m[0], out_dir / "masks" / m[0].name)
                n_mask += 1
        else:
            shutil.copy2(img, out_dir / "normal" / img.name)
            n_normal += 1
    return {"normal": n_normal, "defect": n_defect, "masks": n_mask}


def get_real_dataset(cfg: dict, name: str, keep_zip: bool = False) -> dict:
    src = SOURCES[name]
    raw_dir = resolve(cfg["paths"]["data_raw"]) / "real"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = resolve("data/real") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".tar" if src.get("is_tar") else ".zip"
    archive = raw_dir / f"{name}{suffix}"
    if not download_with_progress(src["url"], archive):
        raise SystemExit(f"下载失败：{src['url']}")

    size = archive.stat().st_size
    if size < src["expect_min_bytes"]:
        LOG.warning("文件体积(%s B)小于预期(%s B)，可能不完整", f"{size:,}", f"{src['expect_min_bytes']:,}")

    extract_dir = raw_dir / f"{name}_extract"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("解压到 %s ...", extract_dir)
        if src.get("is_tar"):
            import tarfile

            with tarfile.open(archive) as tf:
                tf.extractall(extract_dir)
        else:
            with zipfile.ZipFile(archive) as zf:
                safe_extract(zf, extract_dir)
    else:
        LOG.info("已解压，跳过：%s", extract_dir)

    LOG.info("整理目录结构 ...")
    if name == "kolektorsdd2":
        stats = arrange_kolektorsdd2(extract_dir, out_dir)
    elif name == "visa":
        stats = arrange_visa(extract_dir, out_dir)
    else:
        raise SystemExit(f"未实现的数据集：{name}")

    meta = {
        "dataset": name,
        "source_url": src["url"],
        "homepage": src["homepage"],
        "license": src["license"],
        "domain": src["domain"],
        "paper": src["paper"],
        "archive_bytes": size,
        "stats": stats,
        "arranged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "usage_note": "公开真实工业数据，用于验证方法可行性；实际部署须用企业自有数据微调（许可证多为非商用）。",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("真实数据集就绪：%s -> %s | %s", name, out_dir, stats)

    if not keep_zip:
        archive.unlink(missing_ok=True)
        LOG.info("已删除压缩包（--keep-zip 可保留）")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并整理真实工业缺陷数据集")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dataset", choices=sorted(SOURCES), default="kolektorsdd2")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    get_real_dataset(cfg, args.dataset, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
