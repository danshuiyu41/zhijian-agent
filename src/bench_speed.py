"""算清系统的真实节拍，用于判断"能不能上产线"。

问题背景：判断这套系统能不能用于实际产线，关键不是准确率，而是**节拍**。
因此需要量化三种运行模式的单件耗时，并对照典型产线速度给出结论。

三种模式：
  A 纯检测（不生成热力图）—— 生产可用形态
  B 检测 + 热力图（Grad-CAM 需反向传播）—— 复核/抽检形态
  C 检测 + 智能体处置建议 + 落库 —— 完整链路

用法：
    python -m src.bench_speed
    python -m src.bench_speed --n 12
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

from .agent import QualityAgent  # noqa: E402
from .db import Database  # noqa: E402
from .infer import DefectInspector  # noqa: E402
from .utils import get_logger, load_config  # noqa: E402

LOG = get_logger("bench")


def main() -> None:
    ap = argparse.ArgumentParser(description="系统节拍基准测试")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n", type=int, default=10, help="每种模式测多少张")
    args = ap.parse_args()

    cfg = load_config(args.config)
    insp = DefectInspector.from_config(args.config)
    agent = QualityAgent.from_config(args.config)
    db = Database.from_config(args.config)

    files = sorted(glob.glob("data/raw/*/*.png"))[: args.n]
    if not files:
        raise SystemExit("未找到测试图像")

    print("=" * 78)
    print(f"节拍基准测试（{len(files)} 张，CPU，主干 {insp.ckpt.get('backbone')}）")
    print("=" * 78)

    def bench(name: str, fn) -> float:
        ts = []
        for f in files:
            t0 = time.time()
            fn(f)
            ts.append((time.time() - t0) * 1000)
        med = statistics.median(ts)
        p90 = sorted(ts)[int(len(ts) * 0.9) - 1] if len(ts) > 1 else ts[0]
        print(f"  {name:<28} 中位 {med:>7.1f} ms   90 分位 {p90:>7.1f} ms   "
              f"≈ {1000 / max(med, 1e-6):>5.1f} 件/秒")
        return med

    # A 纯检测（生产形态）
    median_a = bench("A 纯检测", lambda f: insp.inspect(f, want_heatmap=False))

    # B 检测 + 热力图（复核形态）
    def with_heat(f: str):
        gray = __import__("cv2").imread(f, __import__("cv2").IMREAD_GRAYSCALE)
        r, _ = insp.inspect(f, want_heatmap=False)
        insp.heatmap(gray)

    median_b = bench("B 检测 + Grad-CAM 热力图", with_heat)

    # C 完整链路
    def full(f: str):
        r, _ = insp.inspect(f, want_heatmap=False)
        adv = agent.analyze(r.to_dict()).to_dict()
        db.save_inspection({**r.to_dict(), "advice": adv}, image_path=f)

    median_c = bench("C 检测+智能体+落库", full)

    print("-" * 78)
    print("对照典型产线速度所需能力：")
    for label, spm in (("低速产线 10 件/分", 10), ("中速 30 件/分", 30),
                       ("高速 60 件/分", 60), ("高速 120 件/分", 120)):
        need = 60000 / spm
        verdict = "可满足" if median_a <= need else "不满足"
        print(f"  {label:<20} 允许单件 {need:>6.0f} ms   A 模式 {median_a:>6.0f} ms → {verdict}")

    print("=" * 78)
    print(f"结论：A 模式 {median_a:.0f} ms（≈{1000/max(median_a,1e-6):.1f} 件/秒）为生产可用形态；")
    print(f"      B 模式 {median_b:.0f} ms 因需反向传播，只适合抽检/复核，不适合在线全检。")
    print("=" * 78)


if __name__ == "__main__":
    main()
