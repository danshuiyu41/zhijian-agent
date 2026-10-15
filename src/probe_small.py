"""小体积真实数据集探针：批量探测"可达性 + 体积"，挑最小的下。

动机：KolektorSDD2 有 853 MB 且服务器限速 19 KB/s（约 10 小时），不适合。
本脚本对一批**预期体积较小**的真实缺陷数据集做 Range 探测，只读头部与
Content-Length，不整包下载，从而快速挑出可用的小数据集。

用法：
    python -m src.probe_small            # 只探测
    python -m src.probe_small --fetch    # 探测后下载最小的那个
"""
from __future__ import annotations

import argparse
import ssl
import time
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

CANDIDATES: list[tuple[str, str, str]] = [
    # (名称, URL, 说明)
    ("KolektorSDD(小)", "https://go.vicos.si/kolektorsdd",
     "电换向器 399 张，含掩膜，免注册"),
    ("KSDD-划分包", "https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip",
     "3-fold 划分，体积很小"),
    ("DeepPCB", "https://codeload.github.com/tangsanli5201/DeepPCB/zip/refs/heads/master",
     "PCB 缺陷 1500 对，6 类"),
    ("DAGM-Class1", "https://zenodo.org/records/12750201/files/Class1.zip?download=1",
     "DAGM 弱监督工业检测第 1 类"),
    ("GDXray-Castings", "https://www.dropbox.com/scl/fi/5e7m31ri5grvnl7kxt8mx/Castings.zip?rlkey=zetfg5g337d2ip265a16rcviw&st=gjub7hr3&dl=1",
     "铸件 X 光缺陷"),
    ("elpv-太阳能EL", "https://codeload.github.com/zae-bayern/elpv-dataset/zip/refs/heads/master",
     "太阳能电池 EL 缺陷 2624 张"),
    ("GC10-DET(镜像)", "https://codeload.github.com/alsaniie/GC10-DET-Metallic-Surface-Defect-Datasets/zip/refs/heads/main",
     "钢板 10 类缺陷（另一镜像）"),
    ("磁瓦缺陷", "https://codeload.github.com/Charmve/Surface-Defect-Detection/zip/refs/heads/master",
     "含 Magnetic-Tile 等数据集"),
]


def _ctx() -> ssl.SSLContext:
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def probe(url: str, timeout: int = 15) -> dict:
    """只读前 1KB + Content-Length，估算体积与可达性。"""
    out = {"ok": False, "size": 0, "ctype": "", "note": "", "sec": 0.0}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-1023"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
            cr = r.headers.get("Content-Range") or ""
            cl = r.headers.get("Content-Length") or "0"
            head = r.read(1024)
            if "/" in cr:
                out["size"] = int(cr.rsplit("/", 1)[-1])
            else:
                out["size"] = int(cl)
            out["ctype"] = r.headers.get("Content-Type") or ""
            out["ok"] = out["size"] > 0
            # 判断是不是压缩包
            magic = head[:4]
            out["note"] = ("zip" if magic[:2] == b"PK" else
                           "rar" if magic[:4] == b"Rar!" else
                           f"其它({magic!r})")
    except urllib.error.HTTPError as e:
        out["note"] = f"HTTP {e.code}"
    except Exception as e:
        out["note"] = f"{type(e).__name__}"
    out["sec"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="小体积真实数据集探针")
    ap.add_argument("--fetch", action="store_true", help="下载最小的可达数据集")
    args = ap.parse_args()

    print("=" * 88)
    print("小体积真实数据集可达性探测")
    print("=" * 88)
    print(f"  {'数据集':<18}{'大小':>10}{'类型':>10}{'耗时':>7}  说明")
    print("-" * 88)

    good: list[tuple[str, str, int]] = []
    for name, url, desc in CANDIDATES:
        r = probe(url)
        size_mb = r["size"] / 1e6
        size_txt = f"{size_mb:.1f} MB" if r["size"] else "-"
        flag = "OK" if r["ok"] else "FAIL"
        print(f"  [{flag}] {name:<14}{size_txt:>10}{r['note']:>10}{r['sec']:>6}s  {desc}")
        if r["ok"] and r["size"] > 100_000:
            good.append((name, url, r["size"]))

    print("-" * 88)
    if not good:
        print("  没有可达的小数据集。")
        return
    good.sort(key=lambda x: x[2])
    print("  按体积排序的可达数据集：")
    for name, _, sz in good:
        print(f"    {name:<18}{sz / 1e6:>8.1f} MB")

    if args.fetch:
        name, url, sz = good[0]
        print(f"\n  开始下载最小的：{name}（{sz / 1e6:.1f} MB）")
        from .fetch_util import ResumableDownloader
        from .utils import load_config, resolve
        cfg = load_config()
        dest = resolve(cfg["paths"]["data_raw"]) / "real" / (name.split("(")[0] + ".zip")
        d = ResumableDownloader(url, dest, chunk_mb=1, max_retries=5, timeout=40,
                                min_bytes=100_000)
        print("  下载结果:", "成功" if d.run() else "失败")
    print("=" * 88)


if __name__ == "__main__":
    main()
