"""打「软件安装包」——官方提交物之一。

产出：
    dist/zhijian-agent-v1.0.zip         完整安装包（评委本地可跑）
    dist/zhijian-agent-v1.0-src.zip     纯源码包（无权重，体积小）

包含：源码 + 模型权重 + 示例图片 + 文档 + 一键启动脚本 + 依赖清单
排除：数据集、训练中间产物、日志、缓存、数据库、Git 历史

用法：
    python -m src.make_installer
    python -m src.make_installer --light     # 不打包权重
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

# 顶层必带项
INCLUDE_DIRS = [
    "src",
    "web",
    "config",
    "knowledge",
    "models",
    "docs/upload",
    "docs/pdf",
    "runs/figures",
]
INCLUDE_FILES = [
    "start.py",
    "check_all.py",
    "deploy.sh",
    "README.md",
    "requirements.txt",
    ".gitignore",
]
INCLUDE_DOCS = [
    "docs/安装与部署说明.md",
    "docs/团队分工说明.md",
    "docs/AI使用与原创性声明.md",
    "docs/软件设计文档.md",
    "docs/作品核心资料.md",
    "docs/云服务器部署手册.md",
]

# 这些后缀一律不要
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".db", ".part", ".tmp", ".zip", ".onnx"}
SKIP_DIRNAME = {"__pycache__", ".git", ".vscode", ".idea", ".venv", "venv"}

# models 里只带这几个权重文件，其余（如 torch 预训练缓存）不要
MODEL_KEEP = {"zhijian_best.pt", "feature_bank.npz", "ood_config.json"}

# docs 下只带指定文件，避免把申报表等内部材料混进安装包
DOCS_DIR_KEEP = {"upload", "pdf"}


def _ignore(dirpath: str, names: list[str]) -> set[str]:
    drop: set[str] = set()
    p = Path(dirpath)
    for n in names:
        full = p / n
        if n in SKIP_DIRNAME:
            drop.add(n)
        elif full.is_file() and full.suffix.lower() in SKIP_SUFFIX:
            drop.add(n)
    return drop


def _stage(light: bool) -> tuple[Path, list[str]]:
    """把待打包内容拷到临时目录，返回（临时目录，文件清单）。"""
    stage = DIST / "_stage"
    if stage.exists():
        shutil.rmtree(stage)
    root = stage / "zhijian-agent"
    root.mkdir(parents=True, exist_ok=True)

    files: list[str] = []

    def copy_file(src: Path, rel: str) -> None:
        if not src.is_file():
            return
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append(rel)

    for d in INCLUDE_DIRS:
        src_dir = ROOT / d
        if not src_dir.is_dir():
            continue
        for f in src_dir.rglob("*"):
            if not f.is_file():
                continue
            rel_parts = f.relative_to(ROOT).parts
            if any(part in SKIP_DIRNAME for part in rel_parts):
                continue
            if f.suffix.lower() in SKIP_SUFFIX:
                continue
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            # models 白名单
            if rel.startswith("models/") and f.name not in MODEL_KEEP:
                continue
            if light and f.suffix == ".pt":
                continue
            copy_file(f, rel)

    for f in INCLUDE_FILES:
        copy_file(ROOT / f, f)

    for f in INCLUDE_DOCS:
        copy_file(ROOT / f, f)

    # 示例图片：让评委解压后立刻有东西可测
    sample_dir = ROOT / "data" / "samples"
    picked = 0
    if sample_dir.is_dir():
        for f in sorted(sample_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rel = f"data/samples/{f.name}"
                if (root / rel).exists():
                    continue
                copy_file(f, rel)
                picked += 1
            if picked >= 12:
                break

    # 一键启动脚本
    (root / "启动.bat").write_text(
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "echo ============================================\r\n"
        "echo   智检 Agent - 工业表面缺陷智能检测与处置系统\r\n"
        "echo ============================================\r\n"
        "echo.\r\n"
        "python -V >nul 2>&1\r\n"
        "if errorlevel 1 (\r\n"
        "    echo [错误] 未检测到 Python，请先安装 Python 3.10\r\n"
        "    echo        下载：https://www.python.org/downloads/\r\n"
        "    echo        安装时务必勾选 Add Python to PATH\r\n"
        "    pause\r\n"
        "    exit /b 1\r\n"
        ")\r\n"
        "python -c \"import torch\" >nul 2>&1\r\n"
        "if errorlevel 1 (\r\n"
        "    echo [首次运行] 正在安装依赖，约 3-5 分钟，请勿关闭窗口...\r\n"
        "    python -m pip install -r requirements.txt "
        "--index-url https://download.pytorch.org/whl/cpu\r\n"
        ")\r\n"
        "echo.\r\n"
        "echo 服务启动中，浏览器将自动打开 http://127.0.0.1:8077\r\n"
        "echo 关闭本窗口即停止服务。\r\n"
        "echo.\r\n"
        "python start.py\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    files.append("启动.bat")

    (root / "install.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# 智检 Agent 一键安装（Linux / macOS）\n"
        "set -e\n"
        "cd \"$(dirname \"$0\")\"\n"
        "python3 -m venv .venv\n"
        ". .venv/bin/activate\n"
        "python -m pip install --upgrade pip\n"
        "python -m pip install -r requirements.txt "
        "--index-url https://download.pytorch.org/whl/cpu\n"
        "echo '安装完成。启动：source .venv/bin/activate && python start.py'\n",
        encoding="utf-8",
    )
    files.append("install.sh")

    return root, sorted(files)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _zip(root: Path, out: Path, files: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for rel in list(files) + ["启动.bat", "install.sh"]:
        if rel in seen:
            continue
        seen.add(rel)
        ordered.append(rel)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in ordered:
            src = root / rel
            if src.is_file():
                z.write(src, f"zhijian-agent/{rel}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true", help="不打包模型权重")
    args = ap.parse_args()

    DIST.mkdir(exist_ok=True)
    root, files = _stage(light=args.light)

    tag = "v1.0-light" if args.light else "v1.0"
    out = DIST / f"zhijian-agent-{tag}.zip"
    _zip(root, out, files)

    total = sum((root / f).stat().st_size for f in files if (root / f).is_file())
    size_mb = out.stat().st_size / 1024 / 1024

    print("=" * 70)
    print("软件安装包已生成")
    print("=" * 70)
    print(f"  输出      : {out}")
    print(f"  文件数    : {len(files)}")
    print(f"  解压前    : {total/1024/1024:.2f} MB")
    print(f"  压缩后    : {size_mb:.2f} MB")
    print(f"  SHA256    : {_sha256(out)}")
    print()
    print("  目录结构：")
    top: dict[str, int] = {}
    for f in files:
        top[f.split("/")[0]] = top.get(f.split("/")[0], 0) + 1
    for k in sorted(top):
        print(f"    {k:<22} {top[k]:>4} 个文件")
    print()

    # 关键内容存在性检查
    need = [
        "start.py", "requirements.txt", "README.md",
        "src/api.py", "src/infer.py", "src/agent.py", "src/ood.py",
        "web/index.html", "web/design-system.css",
        "config/config.yaml", "knowledge/quality_kb.json",
        "docs/安装与部署说明.md", "docs/AI使用与原创性声明.md",
    ]
    if not args.light:
        need += ["models/zhijian_best.pt", "models/feature_bank.npz"]
    miss = [n for n in need if n not in files]
    if miss:
        print("  [警告] 缺少关键文件：")
        for m in miss:
            print(f"    - {m}")
        return 1

    print("  关键文件校验：全部存在 OK")

    shutil.rmtree(DIST / "_stage", ignore_errors=True)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
