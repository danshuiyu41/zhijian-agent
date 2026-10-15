"""一键流水线：从零到可演示的完整构建。

依次执行：
1. 生成合成缺陷数据集（冷启动，保证系统一定能跑）
2. 构建数据索引与归一化统计量（自动混入 data/real 与 data/custom 的真实数据）
3. 训练缺陷分类模型并导出评估指标与图表
4. 生成实测指标文档并回填设计说明书
5. 填充看板演示数据
6. 可选：启动 Web 服务

用法：
    python -m src.pipeline                    # 全流程（默认 200 张/类，26 轮）
    python -m src.pipeline --per-class 120 --epochs 12
    python -m src.pipeline --skip-train       # 已有模型时只做数据与文档
    python -m src.pipeline --serve            # 构建完成后启动服务
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .utils import ROOT, ensure_dirs, get_logger, load_config

LOG = get_logger("pipeline")
PY = sys.executable


def run(module: str, *args: str, check: bool = True) -> int:
    cmd = [PY, "-u", "-m", module, *args]
    LOG.info("▶ %s", " ".join(cmd[2:]))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    LOG.info("  %s（%.1f 秒）", "完成" if proc.returncode == 0 else f"失败 rc={proc.returncode}", dt)
    if check and proc.returncode != 0:
        raise SystemExit(f"步骤 {module} 失败")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="智检 Agent 一键构建流水线")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--per-class", type=int, default=200, help="每类缺陷合成张数")
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-data", action="store_true", help="跳过合成数据生成")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练（沿用已有权重）")
    ap.add_argument("--seed-demo", type=int, default=120, help="填充看板演示数据条数，0 表示不填充")
    ap.add_argument("--serve", action="store_true", help="构建完成后启动 Web 服务")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    LOG.info("=" * 68)
    LOG.info("智检 Agent 一键构建流水线")
    LOG.info("=" * 68)

    if not args.skip_data:
        run("src.data_generate", "--per-class", str(args.per_class))
    else:
        LOG.info("⏭ 跳过合成数据生成")

    run("src.data")
    run("src.visualize_data", "--per-class", "6")

    if not args.skip_train:
        run("src.train", "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--export-onnx")
    else:
        LOG.info("⏭ 跳过训练")

    metrics = ROOT / cfg["paths"]["runs"] / "metrics.json"
    if metrics.exists():
        run("src.make_metrics")
    else:
        LOG.warning("未找到 %s，跳过指标文档生成", metrics)

    if args.seed_demo > 0:
        try:
            from .db import Database

            n = Database.from_config().seed_demo_data(args.seed_demo)
            LOG.info("看板演示数据：新增 %d 条", n)
        except Exception as e:
            LOG.warning("填充演示数据失败：%s", e)

    LOG.info("=" * 68)
    LOG.info("构建完成。启动服务：python -m src.api")
    LOG.info("浏览器访问：http://%s:%s", cfg["server"]["host"], cfg["server"]["port"])
    LOG.info("=" * 68)

    if args.serve:
        run("src.api")


if __name__ == "__main__":
    main()
