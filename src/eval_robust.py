"""鲁棒性评估：在不同难度档位上评估模型，量化性能退化。

为什么需要它
------------
在合成数据集上，模型测试准确率可达 100%——这说明该任务对模型而言偏易，
单一指标缺乏说服力。要证明"模型在什么条件下会失效"，必须做**受控退化实验**：
在保持类别不变的前提下，逐步缩小缺陷尺度、降低缺陷对比度，观察指标如何下降。

这正是知识库条目 `TECH-LIMITS`（系统适用边界与失效模式）的实测支撑，
也是答辩时回答"怎么保证不漏检"最有力的证据。

用法：
    python -m src.data_generate --stress          # 先生成压力测试集
    python -m src.eval_robust                      # 再评估
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .utils import class_list, get_logger, load_config, now_str, resolve, set_seed  # noqa: E402

LOG = get_logger("robust")

LEVEL_LABEL = {
    "L1_normal": "L1 正常（与训练同分布）",
    "L2_small": "L2 小尺度（缺陷尺寸约 45%）",
    "L3_small_lowcontrast": "L3 小尺度+低对比度（对比度约 40%）",
}


def evaluate_level(inspector, level_dir: Path, classes: list[str]) -> dict:
    """在单个难度档位上评估。按"是否检出缺陷"这一产线最关心的判据统计。"""
    y_true: list[int] = []          # 1 = 有缺陷, 0 = 洁净
    y_pred: list[int] = []
    y_cls_true: list[str] = []      # 缺陷类别（仅缺陷样本）
    y_cls_pred: list[str] = []
    elapsed: list[float] = []

    for cls in classes:
        d = level_dir / cls
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        for f in files:
            t0 = time.time()
            res, _ = inspector.inspect(f, want_heatmap=False)
            elapsed.append((time.time() - t0) * 1000)
            has_defect = len(res.detections) > 0
            y_true.append(0 if cls == "clean" else 1)
            y_pred.append(1 if has_defect else 0)
            if cls != "clean":
                y_cls_true.append(cls)
                y_cls_pred.append(res.detections[0].class_code if has_defect else "__miss__")

    yt, yp = np.array(y_true), np.array(y_pred)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    # 产线口径：漏检率（缺陷被判合格）与过杀率（合格被判不合格）
    miss_rate = fn / max(tp + fn, 1)
    overkill_rate = fp / max(fp + tn, 1)
    detection_rate = tp / max(tp + fn, 1)

    cls_acc = 0.0
    if y_cls_true:
        cls_acc = float(np.mean([a == b for a, b in zip(y_cls_true, y_cls_pred)]))

    return {
        "samples": len(y_true),
        "defect_samples": int((yt == 1).sum()),
        "clean_samples": int((yt == 0).sum()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "detection_rate": round(detection_rate, 4),
        "miss_rate": round(miss_rate, 4),
        "overkill_rate": round(overkill_rate, 4),
        "defect_class_accuracy": round(cls_acc, 4),
        "avg_ms": round(float(np.mean(elapsed)) if elapsed else 0.0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="鲁棒性评估（受控退化实验）")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--root", default="data/stress")
    ap.add_argument("--out", default="runs/robustness.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["dataset"]["seed"]))

    from .infer import DefectInspector

    inspector = DefectInspector.from_config(args.config)
    root = resolve(args.root)
    if not root.exists():
        raise SystemExit(f"未找到压力测试集 {root}，请先运行：python -m src.data_generate --stress")

    classes = [c["code"] for c in class_list(cfg)] + ["clean"]
    levels = sorted(d for d in root.iterdir() if d.is_dir())
    if not levels:
        raise SystemExit(f"{root} 下没有难度档位目录")

    results: dict[str, Any] = {}
    print("=" * 92)
    print(f"{'难度档位':<34}{'样本':<7}{'检出率':<10}{'漏检率':<10}{'过杀率':<10}{'分类准确':<10}{'耗时ms':<9}")
    print("-" * 92)
    for lv in levels:
        r = evaluate_level(inspector, lv, classes)
        results[lv.name] = r
        print(f"{LEVEL_LABEL.get(lv.name, lv.name):<34}{r['samples']:<7}"
              f"{r['detection_rate']:<10.4f}{r['miss_rate']:<10.4f}"
              f"{r['overkill_rate']:<10.4f}{r['defect_class_accuracy']:<10.4f}{r['avg_ms']:<9.1f}")
    print("=" * 92)

    payload = {
        "task": "鲁棒性评估（受控退化实验）",
        "model": {"ckpt": inspector.ckpt_path.name, "backbone": inspector.ckpt.get("backbone"),
                  "params_m": inspector.ckpt.get("num_params_m")},
        "levels": results,
        "level_labels": LEVEL_LABEL,
        "notes": [
            "检出率 = 缺陷样本中被判为不合格的比例（1 - 漏检率）。",
            "漏检率 = 缺陷被判为合格的比例，是产线最关注的指标。",
            "过杀率 = 洁净样本被判为不合格的比例（误报）。",
            "分类准确率 = 在正确检出的前提下，缺陷类别判定正确的比例。",
            "压力集不参与训练，仅用于量化模型在非理想成像条件下的退化。",
        ],
        "generated_at": now_str(),
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("鲁棒性结果已写入 %s", out)


if __name__ == "__main__":
    main()
