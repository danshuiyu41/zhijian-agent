"""归档内容检查（本地文件版）与断点续传下载器。

两个能力：
1. `inspect_local(zip_path)`：直接检查**已下载到本地**的 zip，列出图片/标注数量与
   目录结构。用于判断"这个仓库到底有没有内嵌数据集"。
2. `ResumableDownloader`：**支持断点续传**的下载器。本机实测部分数据集服务器
   （如 go.vicos.si）会在传输中途掐断连接，普通一次性下载拿不到完整文件。
   该下载器按固定块大小分块拉取，每块失败即重试，已完成的块写入磁盘，
   下次运行可继续，从而在 flaky 网络下也能拿到完整数据。

用法：
    python -m src.fetch_util --inspect-local data/raw/real/gc10det.zip
    python -m src.fetch_util --resume https://go.vicos.si/kolektorsdd2 --dest data/raw/real/kolektorsdd2.zip
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANN_EXT = {".xml", ".txt", ".json", ".csv"}


def _ctx() -> ssl.SSLContext:
    c = ssl.create_default_context()
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


# ----------------------------------------------------------------------
# 1. 本地归档检查
# ----------------------------------------------------------------------
def inspect_local(path: str | Path, max_list: int = 25) -> dict:
    p = Path(path)
    out: dict = {"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0,
                 "ok": False, "entries": 0, "images": 0, "annotations": 0,
                 "top_dirs": [], "samples": [], "error": ""}
    if not p.exists():
        out["error"] = "文件不存在"
        return out
    try:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
        out["ok"] = True
        out["entries"] = len(names)
        imgs = [n for n in names if "." in n and ("." + n.rsplit(".", 1)[-1].lower()) in IMG_EXT]
        anns = [n for n in names if "." in n and ("." + n.rsplit(".", 1)[-1].lower()) in ANN_EXT]
        out["images"] = len(imgs)
        out["annotations"] = len(anns)
        out["top_dirs"] = Counter("/".join(n.split("/")[:2]) for n in names if "/" in n).most_common(15)
        out["samples"] = imgs[:max_list]
        out["all_files"] = names[:60]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


# ----------------------------------------------------------------------
# 2. 断点续传下载器
# ----------------------------------------------------------------------
class ResumableDownloader:
    """分块下载 + 断点续传 + 自动重试。

    关键设计：
    - 以 `state_file` 记录已写入的字节数与目标总大小；
    - 每块用 Range 请求，若服务器忽略 Range（返回 200 整包）则自行切片；
    - 块级失败自动重试（指数退避），连续失败达上限才放弃；
    - 完成后校验 zip 中央目录可读，确保是完整可用的压缩包。
    """

    def __init__(self, url: str, dest: str | Path, chunk_mb: int = 2,
                 max_retries: int = 12, timeout: int = 60, min_bytes: int = 100_000):
        self.url = url
        self.dest = Path(dest)
        self.part = self.dest.with_suffix(self.dest.suffix + ".part")
        self.state_file = self.dest.with_suffix(self.dest.suffix + ".state.json")
        self.chunk = chunk_mb * 1024 * 1024
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_bytes = min_bytes

    # ------------------------------------------------------------------
    def _head(self) -> tuple[int, bool]:
        """返回 (总大小, 是否支持 Range)。"""
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=self.timeout, context=_ctx()) as r:
                if r.status == 206:
                    cr = r.headers.get("Content-Range", "")
                    total = int(cr.rsplit("/", 1)[-1]) if "/" in cr else 0
                    return total, True
                data = r.read()
                return len(data), False
        except Exception as e:
            print(f"  [head] 失败 {type(e).__name__}: {str(e)[:80]}")
            return 0, False

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
        last = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=_ctx()) as r:
                    status = r.status
                    data = r.read()
                want = end - start + 1
                if status == 206:
                    return data[:want] if len(data) > want else data
                # 200：忽略 Range，data 是整包，自行切片
                if len(data) >= end + 1:
                    return data[start:end + 1]
                # 包比预想短：直接用它剩余的
                return data[start:] if start < len(data) else b""
            except Exception as e:  # noqa: PERF203
                last = e
                wait = min(8.0, 0.8 * (attempt + 1))
                print(f"    块 {start}-{end} 第 {attempt + 1} 次失败：{type(e).__name__}，{wait:.1f}s 后重试")
                time.sleep(wait)
        raise IOError(f"块 {start}-{end} 重试 {self.max_retries} 次仍失败：{last}")

    # ------------------------------------------------------------------
    def run(self) -> bool:
        total, supports_range = self._head()
        print(f"  目标大小：{total:,} B | 支持 Range：{supports_range}")

        done = 0
        if self.state_file.exists() and self.part.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                if state.get("total") == total and self.part.stat().st_size == state.get("done"):
                    done = int(state["done"])
                    print(f"  从断点续传：已完成 {done:,} B ({100 * done / max(total, 1):.1f}%)")
            except Exception:
                done = 0

        if total and done >= total:
            print("  已下载完整，跳过")
            return self._finalize(total)

        mode = "ab" if done > 0 else "wb"
        t0 = time.time()
        stall = 0
        with open(self.part, mode) as f:
            while True:
                if total and done >= total:
                    break
                end = min(done + self.chunk - 1, total - 1) if total else done + self.chunk - 1
                before = done
                try:
                    chunk = self._fetch(done, end)
                except IOError as e:
                    print(f"  ✗ {e}")
                    self.state_file.write_text(json.dumps({"total": total, "done": done}), encoding="utf-8")
                    return False
                if not chunk:
                    # 没有更多数据：若已知总长且未下满，判为失败
                    if total and done < total:
                        stall += 1
                        print(f"  数据中断于 {done:,} B，尝试续传（第 {stall} 次）")
                        if stall > 20:
                            print("  ✗ 反复中断，放弃")
                            return False
                        time.sleep(1.5)
                        continue
                    break
                f.write(chunk)
                f.flush()
                done += len(chunk)
                self.state_file.write_text(json.dumps({"total": total, "done": done}), encoding="utf-8")
                if total:
                    pct = 100 * done / total
                    speed = done / 1e6 / max(0.1, time.time() - t0)
                    print(f"  {done:,}/{total:,} B ({pct:.1f}%) {speed:.2f} MB/s", end="\r")
                stall = 0

        print()
        if total and done < total:
            print(f"  ✗ 未下满：{done:,} / {total:,} B")
            return False
        return self._finalize(total)

    def _finalize(self, total: int) -> bool:
        size = self.part.stat().st_size
        if size < self.min_bytes:
            print(f"  ✗ 文件过小（{size:,} B），判定失败")
            return False
        try:
            with zipfile.ZipFile(self.part) as zf:
                n = len(zf.namelist())
            print(f"  ✓ 压缩包校验通过（{n} 个条目）")
        except Exception as e:
            print(f"  ✗ 压缩包损坏：{type(e).__name__}: {str(e)[:120]}")
            return False
        if self.dest.exists():
            self.dest.unlink()
        self.part.replace(self.dest)
        self.state_file.unlink(missing_ok=True)
        print(f"  ✓ 完成：{self.dest} ({size:,} B)")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description="归档检查 / 断点续传下载")
    ap.add_argument("--inspect-local", help="检查本地 zip")
    ap.add_argument("--resume", help="断点续传下载的 URL")
    ap.add_argument("--dest", help="下载目标路径")
    ap.add_argument("--chunk-mb", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    if args.inspect_local:
        r = inspect_local(args.inspect_local)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return

    if args.resume:
        if not args.dest:
            raise SystemExit("--resume 需要同时给出 --dest")
        d = ResumableDownloader(args.resume, args.dest, chunk_mb=args.chunk_mb, timeout=args.timeout)
        ok = d.run()
        raise SystemExit(0 if ok else 1)

    ap.print_help()


if __name__ == "__main__":
    main()
