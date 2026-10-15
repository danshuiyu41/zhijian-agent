"""仓库归档内容检查器。

用途：GitHub 上很多"数据集仓库"其实只放代码或说明，真正的图片托管在百度网盘 /
Kaggle。直接下载归档会浪费时间。本脚本用 **HTTP Range 只读 zip 的中央目录**，
在不整包下载的前提下快速判断归档里有多少图片、结构如何。

用法：
    python -m src.inspect_repo --url <zip-url> --token deeppcb
    python -m src.inspect_repo --all
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import time
import urllib.request
import zipfile
from collections import Counter

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

REPOS = {
    "deeppcb": "https://codeload.github.com/tangsanli5201/DeepPCB/zip/refs/heads/master",
    "neu_seg": "https://codeload.github.com/donghongwen/NEU_Seg/zip/refs/heads/main",
    "charmve": "https://codeload.github.com/Charmve/Surface-Defect-Detection/zip/refs/heads/master",
    "gc10det": "https://codeload.github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets/zip/refs/heads/master",
    "charmve_mtd": "https://codeload.github.com/Charmve/Surface-Defect-Detection/zip/refs/heads/master",
}


def _ctx() -> ssl.SSLContext:
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


class HttpFile(io.RawIOBase):
    """只读的 HTTP Range 文件对象，供 zipfile 随机访问中央目录。"""

    def __init__(self, url: str, timeout: int = 30):
        self.url = url
        self.timeout = timeout
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
            # Content-Length 可能缺失（分块传输），此时退化为用哨兵 HEAD 探测
            self.size = int(r.headers.get("Content-Length") or 0)
            self.accept_ranges = (r.headers.get("Accept-Ranges") or "").lower() == "bytes"
        if self.size <= 0:
            self.size = self._probe_size()

    def _probe_size(self) -> int:
        """通过 Range: bytes=0-0 读 Content-Range 推断总大小。"""
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=_ctx()) as r:
            cr = r.headers.get("Content-Range") or ""
            if "/" in cr:
                try:
                    return int(cr.rsplit("/", 1)[-1])
                except ValueError:
                    pass
            return int(r.headers.get("Content-Length") or 0)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        pos = self.tell()
        if n is None or n < 0:
            end = self.size - 1
        else:
            end = min(self.size - 1, pos + n - 1)
        if pos > end:
            return b""
        return self._range(pos, end)

    def _range(self, start: int, end: int) -> bytes:
        """发 Range 请求。部分服务器（或 CDN）会忽略 Range 直接返回 200 + 整包，
        此时必须自行切片，否则 zipfile 会读到错误偏移而报 BadZipFile。"""
        want = end - start + 1
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=_ctx()) as r:
                    status = r.status
                    data = r.read()
                if status == 206:
                    if len(data) > want:      # 极少见：多给了
                        data = data[:want]
                    return data
                # 200：服务器忽略了 Range，data 是整个文件
                if len(data) >= end + 1:
                    return data[start:end + 1]
                if start < len(data):
                    return data[start:]
                raise IOError(f"Range 忽略且数据不足（{len(data)} < {start}）")
            except Exception as e:  # noqa: PERF203
                last_err = e
                time.sleep(1.0 + attempt)
        raise IOError(f"Range 请求失败: {start}-{end} ({last_err})")


def inspect(url: str, token: str = "", timeout: int = 30) -> dict:
    out = {"token": token, "url": url, "ok": False, "size": 0, "entries": 0,
           "image_like": 0, "annotation_like": 0, "top_dirs": [], "samples": [], "error": ""}
    try:
        f = HttpFile(url, timeout=timeout)
        out["size"] = f.size
        with zipfile.ZipFile(f) as zf:
            names = zf.namelist()
        out["entries"] = len(names)
        img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ann_ext = {".xml", ".txt", ".json", ".csv"}
        imgs = [n for n in names if "." in n and ("." + n.rsplit(".", 1)[-1].lower()) in img_ext]
        anns = [n for n in names if "." in n and ("." + n.rsplit(".", 1)[-1].lower()) in ann_ext]
        out["image_like"] = len(imgs)
        out["annotation_like"] = len(anns)
        depth2 = Counter("/".join(n.split("/")[:2]) for n in names if "/" in n)
        out["top_dirs"] = depth2.most_common(12)
        out["samples"] = (imgs[:8] + anns[:4])
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="检查仓库归档中的真实图片数量")
    ap.add_argument("--url")
    ap.add_argument("--token", default="custom")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="data/raw/_repo_inspect.json")
    args = ap.parse_args()

    targets = REPOS if args.all else {args.token: args.url}
    if not args.url and not args.all:
        raise SystemExit("请给出 --url 或使用 --all")

    results = []
    for token, url in targets.items():
        r = inspect(url, token)
        results.append(r)
        print(f"\n=== {token} ===")
        print(f"  归档大小 : {r['size']:,} B")
        print(f"  条目数   : {r['entries']}")
        print(f"  图片数   : {r['image_like']}  标注数: {r['annotation_like']}")
        if r["top_dirs"]:
            print("  顶层目录 :")
            for d, c in r["top_dirs"][:8]:
                print(f"     {d:<46} {c}")
        if r["samples"]:
            print("  示例文件 :")
            for s in r["samples"][:8]:
                print(f"     {s}")
        if r["error"]:
            print(f"  错误     : {r['error']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tested_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results},
                  f, ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
