"""网络与下载通道诊断脚本。

用途：在受限网络环境下快速判断"哪些站点可达、哪些直链真的能下"，
避免在主流程里反复试错浪费时间。所有请求都带超时与 UA，输出结构化结果。

用法：
    python -m src.diag_net            # 全量诊断
    python -m src.diag_net --only kolektor
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

TARGETS: dict[str, list[tuple[str, str]]] = {
    "kolektor": [
        ("go.vicos.si/kolektorsdd2", "https://go.vicos.si/kolektorsdd2"),
        ("go.vicos.si/kolektorsdd", "https://go.vicos.si/kolektorsdd"),
        ("data.vicos.si KSDD", "https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip"),
        ("vicos page", "https://www.vicos.si/resources/kolektorsdd2/"),
    ],
    "torch": [
        ("torchvision mobilenetv3_small", "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"),
        ("torchvision resnet18", "https://download.pytorch.org/models/resnet18-f37072fd.pth"),
    ],
    "github": [
        ("codeload gc10det", "https://codeload.github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets/zip/refs/heads/master"),
        ("codeload deeppcb", "https://codeload.github.com/tangsanli5201/DeepPCB/zip/refs/heads/master"),
        ("codeload NEU-donghongwen", "https://codeload.github.com/donghongwen/NEU_Seg/zip/refs/heads/main"),
        ("codeload charmve", "https://codeload.github.com/Charmve/Surface-Defect-Detection/zip/refs/heads/master"),
        ("raw.githubusercontent", "https://raw.githubusercontent.com/robotbitdev/metal-defect-datasets/main/README.md"),
    ],
    "cn_platform": [
        ("modelscope", "https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-7B-Instruct-AWQ"),
        ("hf-mirror", "https://hf-mirror.com/api/models/Qwen/Qwen2.5-7B-Instruct-AWQ"),
        ("tianchi", "https://tianchi.aliyun.com/dataset/148297"),
        ("openi", "https://openi.pcl.ac.cn/lwx/AITEX-Dataset/datasets"),
        ("aistudio", "https://aistudio.baidu.com/datasetoverview"),
    ],
    "cloud_api": [
        ("deepseek", "https://api.deepseek.com/v1/models"),
        ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1/models"),
    ],
}


def probe(name: str, url: str, timeout: int = 20, follow_bytes: int = 512) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    t0 = time.time()
    out = {"name": name, "url": url, "ok": False, "status": None, "content_length": None,
           "content_type": None, "elapsed": 0.0, "note": ""}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            out["status"] = r.status
            out["content_length"] = r.headers.get("Content-Length")
            out["content_type"] = r.headers.get("Content-Type")
            head = r.read(follow_bytes)
            out["ok"] = True
            out["note"] = f"head={head[:8]!r}"
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        out["elapsed"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="下载通道诊断")
    ap.add_argument("--only", default=None, help="只测某一组（kolektor/torch/github/cn_platform/cloud_api）")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", default="data/raw/_diag_net.json")
    args = ap.parse_args()

    groups = {args.only: TARGETS[args.only]} if args.only else TARGETS
    results = []
    print(f"{'结果':<6}{'分组':<13}{'名称':<30}{'状态':<7}{'耗时':<7}备注")
    print("-" * 110)
    for gname, items in groups.items():
        for name, url in items:
            r = probe(name, url, timeout=args.timeout)
            r["group"] = gname
            results.append(r)
            print(f"{'OK' if r['ok'] else 'FAIL':<6}{gname:<13}{name:<30}"
                  f"{str(r['status'] or '-'):<7}{r['elapsed']:<7}{r['note'][:60]}")
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"tested_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results},
                      f, ensure_ascii=False, indent=1)
        print(f"\n结果已写入 {args.out}")
    except Exception as e:
        print("写文件失败:", e)


if __name__ == "__main__":
    main()
