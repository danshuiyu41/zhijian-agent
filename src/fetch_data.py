"""多镜像数据集下载探针。

用途：本机处于受限网络环境，公开数据集的多数外网镜像不可达。
本脚本对一批候选直链做**带超时与内容校验**的下载测试，输出可达性报告，
并把成功的压缩包落到 data/raw/ 下。

判定标准（避免把错误页当数据）：
- HTTP 200；
- 响应体大于 --min-bytes（默认 1MB）；
- 文件头是 zip/rar/7z/tar 魔数之一，或 Content-Type 属于压缩包/八位字节流。

用法：
    python -m src.fetch_data --probe            # 只探连通性，不保存
    python -m src.fetch_data                    # 下载第一个成功的
    python -m src.fetch_data --all              # 全部成功的都下
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .utils import ensure_dirs, get_logger, load_config, resolve

LOG = get_logger("fetch")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

MAGIC = {
    b"PK\x03\x04": "zip",
    b"PK\x05\x06": "zip(empty)",
    b"Rar!": "rar",
    b"7z\xbc\xaf\x27\x1c": "7z",
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ": "xz",
}

# ----------------------------------------------------------------------
# 候选直链：按"国内可达性优先"排序
# ----------------------------------------------------------------------
CANDIDATES: list[dict] = [
    # ---------- NEU-DET（东北大学热轧带钢 6 类缺陷，1800 张） ----------
    {"name": "NEU-DET", "host": "github.com",
     "url": "https://github.com/Chasel-Tsui/NEU-DET/archive/refs/heads/main.zip"},
    {"name": "NEU-DET", "host": "github.com",
     "url": "https://github.com/kingqiuol/NEU-DET/archive/refs/heads/main.zip"},
    {"name": "NEU-DET", "host": "codeload.github.com",
     "url": "https://codeload.github.com/abin24/Surface-Inspection-by-using-Deep-Learning/zip/refs/heads/main"},
    {"name": "NEU-DET", "host": "gitee.com",
     "url": "https://gitee.com/mirrors_Chasel-Tsui/NEU-DET/repository/archive/main.zip"},
    {"name": "NEU-DET", "host": "huggingface.co",
     "url": "https://huggingface.co/datasets/JeffreyXiang/NEU-DET/resolve/main/NEU-DET.zip"},
    {"name": "NEU-DET", "host": "modelscope.cn",
     "url": "https://modelscope.cn/api/v1/datasets/AI-ModelScope/NEU-DET/repo?Revision=master&FilePath=NEU-DET.zip"},

    # ---------- GC10-DET（钢板 10 类缺陷，3570 张，真实产线） ----------
    {"name": "GC10-DET", "host": "github.com",
     "url": "https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets/archive/refs/heads/master.zip"},
    {"name": "GC10-DET", "host": "codeload.github.com",
     "url": "https://codeload.github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets/zip/refs/heads/master"},

    # ---------- Magnetic Tile Defects（磁瓦 6 类，1344 张） ----------
    {"name": "Magnetic-Tile", "host": "github.com",
     "url": "https://github.com/abin24/Magnetic-tile-defect-datasets./archive/refs/heads/main.zip"},
    {"name": "Magnetic-Tile", "host": "codeload.github.com",
     "url": "https://codeload.github.com/abin24/Magnetic-tile-defect-datasets./zip/refs/heads/main"},

    # ---------- DeepPCB（PCB 缺陷，1500 对） ----------
    {"name": "DeepPCB", "host": "github.com",
     "url": "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"},
    {"name": "DeepPCB", "host": "codeload.github.com",
     "url": "https://codeload.github.com/tangsanli5201/DeepPCB/zip/refs/heads/master"},

    # ---------- 通用连通性哨兵（用于判断哪些站点整体可达） ----------
    {"name": "_probe", "host": "pypi.org", "url": "https://pypi.org/simple/", "sentinel": True},
    {"name": "_probe", "host": "mirrors.aliyun.com",
     "url": "https://mirrors.aliyun.com/pypi/simple/", "sentinel": True},
    {"name": "_probe", "host": "mirrors.tuna.tsinghua.edu.cn",
     "url": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/", "sentinel": True},
    {"name": "_probe", "host": "modelscope.cn", "url": "https://www.modelscope.cn/", "sentinel": True},
    {"name": "_probe", "host": "tianchi.aliyun.com", "url": "https://tianchi.aliyun.com/", "sentinel": True},
    {"name": "_probe", "host": "aistudio.baidu.com", "url": "https://aistudio.baidu.com/", "sentinel": True},
    {"name": "_probe", "host": "www.kesci.com", "url": "https://www.kesci.com/", "sentinel": True},
]


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def probe(url: str, timeout: int = 20, min_bytes: int = 1_000_000,
          max_bytes: int = 400_000_000) -> dict:
    """探测单个 URL。返回 {ok, status, bytes, ctype, magic, msg, elapsed}。"""
    t0 = time.time()
    result = {"url": url, "ok": False, "status": None, "bytes": 0,
              "ctype": "", "magic": "", "msg": "", "elapsed": 0.0}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            result["status"] = resp.status
            result["ctype"] = resp.headers.get("Content-Type", "")
            head = resp.read(16)
            magic = next((v for k, v in MAGIC.items() if head.startswith(k)), "")
            result["magic"] = magic
            length = resp.headers.get("Content-Length")
            if length and int(length) < min_bytes:
                result["msg"] = f"内容过小({length}B)，判定为错误页"
                result["bytes"] = int(length)
                return result
            if magic:
                # 是压缩包：只校验大小是否达标，不整包读入（探测模式省流量）
                result["bytes"] = int(length) if length else -1
                result["ok"] = (result["bytes"] < 0) or (result["bytes"] >= min_bytes)
                result["msg"] = "压缩包可达"
                return result
            # 非压缩包：可能是哨兵页面
            body = head + resp.read(2048)
            text = body.decode("utf-8", "ignore").lower()
            is_html = "text/html" in result["ctype"] or text.lstrip().startswith(("<!doctype", "<html"))
            if is_html:
                result["ok"] = True
                result["msg"] = "HTML 可达（哨兵站点）"
            else:
                result["ok"] = False
                result["msg"] = f"非压缩包内容（{result['ctype'] or '未知类型'}）"
    except urllib.error.HTTPError as e:
        result["status"] = e.code
        result["msg"] = f"HTTP {e.code}"
    except urllib.error.URLError as e:
        result["msg"] = f"网络不可达: {getattr(e, 'reason', e)}"
    except Exception as e:  # 超时等
        result["msg"] = f"{type(e).__name__}: {e}"
    finally:
        result["elapsed"] = round(time.time() - t0, 1)
    return result


def download(url: str, dest: Path, timeout: int = 600) -> tuple[bool, str]:
    """真正下载到磁盘并做完整性校验。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        with open(tmp, "rb") as f:
            head = f.read(8)
        if not any(head.startswith(k) for k in MAGIC):
            tmp.unlink(missing_ok=True)
            return False, f"下载内容不是压缩包（{total}B）"
        tmp.replace(dest)
        return True, f"{total} B"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="缺陷数据集多镜像下载探针")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--probe", action="store_true", help="只探测连通性")
    ap.add_argument("--all", action="store_true", help="下载所有可达压缩包")
    ap.add_argument("--timeout", type=int, default=25, help="单个 URL 探测超时(秒)")
    ap.add_argument("--min-bytes", type=int, default=1_000_000)
    ap.add_argument("--report", default="data/raw/_fetch_report.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    out_dir = resolve(cfg["paths"]["data_raw"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    LOG.info("开始探测 %d 个候选地址（超时 %ds）...", len(CANDIDATES), args.timeout)
    for c in CANDIDATES:
        r = probe(c["url"], timeout=args.timeout, min_bytes=args.min_bytes)
        r.update({"name": c["name"], "host": c["host"], "sentinel": c.get("sentinel", False)})
        rows.append(r)
        flag = "OK  " if r["ok"] else "FAIL"
        LOG.info("%s %-14s %-28s %-8s %6.1fs %s", flag, c["name"], c["host"],
                 r["status"] or "-", r["elapsed"], r["msg"])

    report = {"tested_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": rows}
    rp = resolve(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    ok_datasets = [r for r in rows if r["ok"] and not r["sentinel"]]
    ok_probes = [r for r in rows if r["ok"] and r["sentinel"]]
    LOG.info("=" * 62)
    LOG.info("可达真实数据集镜像：%d 个", len(ok_datasets))
    for r in ok_datasets:
        LOG.info("   ✔ %s @ %s", r["name"], r["host"])
    LOG.info("可达站点哨兵：%d/%d", len(ok_probes), sum(1 for r in rows if r["sentinel"]))
    LOG.info("报告已写入 %s", rp)

    if args.probe or not ok_datasets:
        if not ok_datasets:
            LOG.warning("没有可达的真实数据集镜像 —— 请改用合成数据兜底，或在本机手动下载后放入 data/raw/")
        return

    seen: set[str] = set()
    for r in ok_datasets:
        if not args.all and r["name"] in seen:
            continue
        seen.add(r["name"])
        dest = out_dir / f"{r['name']}.zip"
        LOG.info("下载 %s -> %s", r["name"], dest.name)
        good, msg = download(r["url"], dest)
        LOG.info("   %s %s", "成功" if good else "失败", msg)
        if not args.all and good:
            LOG.info("已获取 %s，停止（--all 可全部下载）", r["name"])
            break


if __name__ == "__main__":
    main()
