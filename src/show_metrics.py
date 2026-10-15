"""打印训练指标摘要（便于命令行快速查看）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/metrics.json")
if not p.exists():
    raise SystemExit(f"未找到 {p}")

m = json.loads(p.read_text(encoding="utf-8"))
t, v = m["test"], m["val"]
print("=" * 74)
print(f"主干 {m['backbone']} | 参数量 {m['num_params_m']} M | 设备 {m['device']}")
print(f"实际轮次 {m['epochs_run']} | 最佳第 {m['best_epoch']} 轮 val_acc={m['best_val_acc']} | 训练 {m['train_seconds']} s")
print("=" * 74)
print(f"{'指标':<26}{'验证集':<14}{'测试集':<14}")
print("-" * 74)
for k, label in [("accuracy", "准确率"), ("top2_accuracy", "Top-2 准确率"),
                 ("macro_precision", "宏平均精确率"), ("macro_recall", "宏平均召回率"),
                 ("macro_f1", "宏平均 F1"), ("weighted_f1", "加权 F1"), ("ece", "ECE 校准误差"),
                 ("infer_ms_per_image", "单图耗时(ms)"), ("samples", "样本数")]:
    print(f"{label:<26}{str(v.get(k)):<14}{str(t.get(k)):<14}")
print("-" * 74)
print(f"{'缺陷类别':<14}{'样本':<7}{'精确率':<11}{'召回率':<11}{'F1':<11}")
print("-" * 74)
for c in t["per_class"]:
    print(f"{c['name']:<14}{c['support']:<7}{c['precision']:<11.4f}{c['recall']:<11.4f}{c['f1']:<11.4f}")
print("-" * 74)
print("混淆矩阵（行=真实，列=预测）")
names = [c["name"] for c in t["per_class"]]
print("        " + "".join(f"{n[:6]:>8}" for n in names))
for n, row in zip(names, t["confusion_matrix"]):
    print(f"{n:<8}" + "".join(f"{x:>8}" for x in row))
print("=" * 74)
