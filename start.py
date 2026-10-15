"""一键启动 Web 服务（含自检与友好提示）。

用法：
    python start.py                # 默认 127.0.0.1:8077
    python start.py --port 8080
    python start.py --lan          # 绑定 0.0.0.0，便于手机在同一局域网内访问
    python start.py --check        # 只做环境与模型自检，不启动服务
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_cache"))


def local_ip() -> str:
    """获取本机局域网 IP，用于手机访问提示。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check() -> bool:
    ok = True
    print("=" * 62)
    print("智检 Agent 启动自检")
    print("=" * 62)

    # 1) 依赖
    missing = []
    for mod, pkg in [("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "opencv-python"),
                     ("numpy", "numpy"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
                     ("multipart", "python-multipart"), ("yaml", "pyyaml")]:
        try:
            __import__(mod)
            print(f"  [OK]   {pkg}")
        except Exception:
            missing.append(pkg)
            print(f"  [缺失] {pkg}")
            ok = False
    if missing:
        print("\n请先安装缺失依赖：")
        print("  pip install " + " ".join(missing))

    # 2) 数据
    raw = ROOT / "data" / "raw"
    classes = [d for d in raw.iterdir() if d.is_dir()] if raw.exists() else []
    n_imgs = sum(len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg"))) for d in classes)
    if n_imgs > 0:
        print(f"  [OK]   数据集：{len(classes)} 个类别目录，共 {n_imgs} 张图像")
    else:
        print("  [缺失] 数据集为空，请运行：python -m src.data_generate --per-class 200")
        ok = False

    # 3) 索引
    idx = ROOT / "runs" / "dataset_index.json"
    print(f"  [{'OK' if idx.exists() else '缺失'}]   数据索引 runs/dataset_index.json"
          + ("" if idx.exists() else "（运行 python -m src.data）"))
    ok = ok and idx.exists()

    # 4) 模型
    ckpt = ROOT / "models" / "zhijian_best.pt"
    print(f"  [{'OK' if ckpt.exists() else '缺失'}]   模型权重 models/zhijian_best.pt"
          + (f"（{ckpt.stat().st_size/1e6:.1f} MB）" if ckpt.exists()
             else "（运行 python -m src.train）"))
    ok = ok and ckpt.exists()

    # 5) 知识库
    kb = ROOT / "knowledge" / "quality_kb.json"
    print(f"  [{'OK' if kb.exists() else '缺失'}]   知识库 knowledge/quality_kb.json")

    # 6) 前端
    web = ROOT / "web" / "index.html"
    print(f"  [{'OK' if web.exists() else '缺失'}]   前端 web/index.html")

    print("=" * 62)
    print("自检结果：" + ("全部通过，可启动服务" if ok else "存在缺失项，请按提示补齐"))
    print("=" * 62)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="启动智检 Agent Web 服务")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--lan", action="store_true", help="绑定 0.0.0.0，允许局域网/手机访问")
    ap.add_argument("--check", action="store_true", help="只自检，不启动")
    args = ap.parse_args()

    passed = check()
    if args.check:
        sys.exit(0 if passed else 1)

    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    host = args.host or ("0.0.0.0" if args.lan else cfg["server"]["host"])
    port = args.port or int(cfg["server"]["port"])

    print()
    print("服务地址：")
    print(f"  本机访问   http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"  手机/局域网 http://{local_ip()}:{port}  （手机与电脑需在同一 WiFi）")
    print("  接口文档   http://127.0.0.1:%d/docs" % port)
    print("按 Ctrl+C 停止服务")
    print()

    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
