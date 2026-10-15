"""训练与评估脚本。

功能
----
1. 读取 `runs/dataset_index.json`（由 `src/data.py` 生成）构建 DataLoader；
2. 迁移学习训练轻量 CNN（默认 MobileNetV3-Small），类别加权解决样本不均衡；
3. 每个 epoch 记录 train/val 的 loss 与 accuracy，支持早停与余弦退火；
4. 训练结束在 **测试集** 上计算：总体准确率、宏平均/加权平均 P-R-F1、
   混淆矩阵、各类别 P/R/F1，并落盘 `runs/metrics.json`；
5. 生成训练曲线与混淆矩阵图（`runs/figures/*.png`），可直接放进设计说明书；
6. 导出 ONNX（可选），用于无 GPU 的工控机部署。

用法：
    python -m src.data_generate --per-class 180
    python -m src.data --build
    python -m src.train --epochs 24
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import DefectDataset, build_items, load_index, make_transforms
from .model import build_model
from .utils import (ROOT, ensure_dirs, get_logger, load_config, resolve, set_seed)

# 受限环境下用户主目录不可写（WinError 5），把 torch 的权重缓存重定向到项目内，
# 既解决权限问题，也让"下载好的预训练权重"随项目一起交付，离线也能复现训练。
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_cache"))
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

LOG = get_logger("train")


# ----------------------------------------------------------------------
# 训练 / 验证
# ----------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    import torch

    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        correct += int((logits.argmax(1) == y).sum().item())
        seen += y.numel()
    return total_loss / max(seen, 1), correct / max(seen, 1)


def evaluate(model, loader, device, num_classes: int) -> dict:
    """在给定数据集上评估，返回混淆矩阵与各项指标。"""
    import torch

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    t0 = time.time()
    n = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if logits.requires_grad:           # 某些调用路径未关闭梯度，此处兜底
            logits = logits.detach()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(1)
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
        all_probs.append(probs)
        all_true.append(y.numpy())
        n += y.numel()
    elapsed = time.time() - t0

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, num_classes))
    true = np.concatenate(all_true) if all_true else np.zeros(0, dtype=int)

    per_class = []
    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"index": i, "support": support, "precision": round(precision, 4),
                          "recall": round(recall, 4), "f1": round(f1, 4)})

    supports = np.array([c["support"] for c in per_class], dtype=float)
    valid = supports > 0
    macro_f1 = float(np.mean([c["f1"] for c, v in zip(per_class, valid) if v])) if valid.any() else 0.0
    macro_p = float(np.mean([c["precision"] for c, v in zip(per_class, valid) if v])) if valid.any() else 0.0
    macro_r = float(np.mean([c["recall"] for c, v in zip(per_class, valid) if v])) if valid.any() else 0.0
    w = supports / supports.sum() if supports.sum() else np.zeros_like(supports)
    weighted_f1 = float(sum(c["f1"] * wi for c, wi in zip(per_class, w)))

    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    # Top-2 准确率（工业场景常用：允许"一次复检"）
    top2 = 0.0
    if len(true):
        k = min(2, num_classes)
        topk = np.argsort(-probs, axis=1)[:, :k]
        top2 = float(np.mean([t in row for t, row in zip(true, topk)]))

    ece = _expected_calibration_error(probs, true)

    return {
        "accuracy": round(acc, 4),
        "top2_accuracy": round(top2, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "ece": round(ece, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "samples": int(n),
        "infer_ms_per_image": round(1000 * elapsed / max(n, 1), 2),
    }


def _expected_calibration_error(probs: np.ndarray, true: np.ndarray, bins: int = 10) -> float:
    """期望校准误差（ECE）：衡量置信度是否可信，工业判废场景很关键。"""
    if len(probs) == 0:
        return 0.0
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------
def plot_curves(history: dict, out_path: Path, class_names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(epochs, history["train_loss"], label="训练损失", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"], label="验证损失", marker="s", ms=3)
    axes[0].set_xlabel("轮次 Epoch")
    axes[0].set_ylabel("损失 Loss")
    axes[0].set_title("训练 / 验证损失曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="训练准确率", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"], label="验证准确率", marker="s", ms=3)
    axes[1].set_xlabel("轮次 Epoch")
    axes[1].set_ylabel("准确率 Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("训练 / 验证准确率曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], class_names: list[str], out_path: Path, normalize: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    arr = np.array(cm, dtype=float)
    if normalize:
        denom = arr.sum(1, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6.5, n * 0.95), max(5.4, n * 0.85)))
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1 if normalize else arr.max())
    ax.set_xticks(range(n), class_names, rotation=45, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("缺陷分类混淆矩阵" + ("（归一化）" if normalize else ""))
    for i in range(n):
        for j in range(n):
            v = arr[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > (0.6 if normalize else arr.max() * 0.6) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="训练工业表面缺陷分类模型")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None, help="cpu / cuda / auto")
    ap.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    ap.add_argument("--export-onnx", action="store_true", help="训练后导出 ONNX")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，直接加载已有权重做评估与出图（训练脚本报错后可复用）")
    ap.add_argument("--tag", default="zhijian", help="权重文件前缀")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["dataset"]["seed"]))

    device = args.device or cfg["train"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        LOG.warning("请求 cuda 但当前 torch 不支持 CUDA，自动回退 CPU")
        device = "cpu"

    epochs = args.epochs or int(cfg["train"]["epochs"])
    batch_size = args.batch_size or int(cfg["train"]["batch_size"])
    lr = args.lr or float(cfg["train"]["lr"])

    index_path = resolve(cfg["paths"]["runs"]) / "dataset_index.json"
    if not index_path.exists():
        raise SystemExit("未找到数据索引，请先运行：python -m src.data --build")
    index = load_index(index_path)
    norm = index.get("norm") or {"mean": [0.5], "std": [0.25]}

    classes = index["classes"]          # 6 类缺陷
    label_space = index["label_space"]  # + clean
    class_to_idx = {c: i for i, c in enumerate(label_space)}
    num_classes = len(label_space)
    class_names_zh = {c["code"]: c["zh"] for c in cfg["dataset"]["classes"]}
    class_names_zh["clean"] = "洁净(合格)"
    display_names = [class_names_zh.get(c, c) for c in label_space]

    LOG.info("类别空间（%d 类）：%s", num_classes, " / ".join(display_names))
    LOG.info("设备：%s | 主干：%s | 轮次：%d | batch：%d | lr：%g",
             device, cfg["model"]["backbone"], epochs, batch_size, lr)

    train_items = build_items(index, "train", class_to_idx)
    val_items = build_items(index, "val", class_to_idx)
    test_items = build_items(index, "test", class_to_idx)
    LOG.info("样本数  train=%d  val=%d  test=%d", len(train_items), len(val_items), len(test_items))
    if not train_items:
        raise SystemExit("训练集为空，请检查 dataset_index.json")

    train_ds = DefectDataset(train_items, make_transforms(cfg, True, norm))
    val_ds = DefectDataset(val_items, make_transforms(cfg, False, norm))
    test_ds = DefectDataset(test_items or val_items, make_transforms(cfg, False, norm))

    nw = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    model = build_model(cfg, pretrained=not args.no_pretrained, num_classes=num_classes).to(device)
    params = model.num_params()
    LOG.info("模型参数量：%s M（可训练 %s M）", params["total_m"], params["trainable_m"])

    # 类别加权，缓解不均衡（洁净样本通常远多于某类缺陷）
    counts = np.zeros(num_classes, dtype=float)
    for _, y in train_items:
        counts[y] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * num_classes)
    LOG.info("类别权重：%s", [round(float(w), 3) for w in weights])
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = None
    if cfg["train"].get("cosine_schedule", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.02)

    runs_dir = resolve(cfg["paths"]["runs"])
    models_dir = resolve(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{args.tag}_best.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val = -1.0
    best_epoch = -1
    patience = int(cfg["train"].get("early_stop_patience", 8))
    bad = 0
    t_start = time.time()

    if args.eval_only:
        # 复用已有权重直接评估：训练因绘图/导出等后处理报错时，无需重新训练
        if not ckpt_path.exists():
            raise SystemExit(f"--eval-only 需要已存在的权重：{ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        best_val = float(ck.get("val_acc", -1.0))
        best_epoch = int(ck.get("epoch", -1))
        hist_path = runs_dir / "train_history.json"
        if hist_path.exists():
            try:
                history = json.loads(hist_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        LOG.info("--eval-only：已加载 %s（第 %d 轮，val_acc=%.4f），跳过训练",
                 ckpt_path.name, best_epoch, best_val)

    train_seconds = time.time() - t_start

    for epoch in range(1, epochs + 1) if not args.eval_only else []:
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(round(tr_loss, 5))
        history["val_loss"].append(round(va_loss, 5))
        history["train_acc"].append(round(tr_acc, 5))
        history["val_acc"].append(round(va_acc, 5))
        history["lr"].append(round(optimizer.param_groups[0]["lr"], 6))
        LOG.info("Epoch %2d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.5f",
                 epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, optimizer.param_groups[0]["lr"])

        if va_acc > best_val:
            best_val, best_epoch, bad = va_acc, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": label_space,
                "class_names_zh": display_names,
                "backbone": cfg["model"]["backbone"],
                "norm": norm,
                "image_size": int(cfg["dataset"]["image_size"]),
                "epoch": epoch,
                "val_acc": round(va_acc, 5),
                "num_params_m": params["total_m"],
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ckpt_path)
            LOG.info("   ↑ 保存最佳权重（val_acc=%.4f）-> %s", va_acc, ckpt_path.name)
        else:
            bad += 1
            if bad >= patience:
                LOG.info("验证准确率 %d 轮无提升，提前停止", patience)
                break

    if not args.eval_only:
        train_seconds = time.time() - t_start
        LOG.info("训练完成，用时 %.1f s；最佳验证准确率 %.4f（第 %d 轮）", train_seconds, best_val, best_epoch)
        (runs_dir / "train_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 最终评估 ----------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device, num_classes)
    val_metrics = evaluate(model, val_loader, device, num_classes)

    for i, name in enumerate(display_names):
        test_metrics["per_class"][i]["name"] = name
        val_metrics["per_class"][i]["name"] = name

    LOG.info("测试集：acc=%.4f  top2=%.4f  macro-F1=%.4f  weighted-F1=%.4f  ECE=%.4f  %.2f ms/图",
             test_metrics["accuracy"], test_metrics["top2_accuracy"], test_metrics["macro_f1"],
             test_metrics["weighted_f1"], test_metrics["ece"], test_metrics["infer_ms_per_image"])
    for c in test_metrics["per_class"]:
        LOG.info("   %-12s support=%-5d P=%.3f R=%.3f F1=%.3f",
                 c["name"], c["support"], c["precision"], c["recall"], c["f1"])

    report = {
        "task": "工业表面缺陷分类",
        "backbone": cfg["model"]["backbone"],
        "num_params_m": params["total_m"],
        "device": device,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 5),
        "train_seconds": round(train_seconds, 1),
        "classes": label_space,
        "class_names_zh": display_names,
        "data_counts": index["counts"],
        "data_sources": index["sources"],
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = runs_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    figs = runs_dir / "figures"
    plot_curves(history, figs / "training_curves.png", display_names)
    plot_confusion(test_metrics["confusion_matrix"], display_names, figs / "confusion_matrix.png")
    plot_confusion(test_metrics["confusion_matrix"], display_names, figs / "confusion_matrix_raw.png",
                   normalize=False)
    LOG.info("指标已写入 %s", metrics_path)
    LOG.info("图表已写入 %s", figs)

    if args.export_onnx:
        try:
            onnx_path = models_dir / f"{args.tag}_best.onnx"
            dummy = torch.randn(1, 3, int(cfg["dataset"]["image_size"]), int(cfg["dataset"]["image_size"]))
            torch.onnx.export(model.cpu(), dummy, str(onnx_path),
                              input_names=["input"], output_names=["logits"],
                              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                              opset_version=13)
            LOG.info("ONNX 已导出（供无 GPU 工控机部署）：%s", onnx_path)
        except Exception as e:
            LOG.warning("ONNX 导出失败：%s: %s", type(e).__name__, e)


if __name__ == "__main__":
    main()
