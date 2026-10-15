"""置信度校准（Temperature Scaling）。

为什么需要校准
--------------
实测发现模型输出的置信度高度集中在 0.92~0.99 区间：一个 0.95 的检出和一个 0.99 的
检出，在业务上本应有明显差别（后者几乎可以直接判废，前者值得复检），但数值上几乎
没有区分度，导致"严重度分级"在同一档内分数雷同，也让"低置信度转人工复检"这条策略
失去灵敏度。

根本原因是深层网络的 softmax 输出**普遍过度自信**（over-confident），这是已知现象，
与模型准确率高低无关。

做法：温度缩放（Guo et al., ICML 2017）
--------------------------------------
在**验证集**上拟合单一温度参数 T > 0，对 logits 做缩放后再 softmax：

    p_calibrated = softmax(logits / T)

- T > 1 表示模型过度自信，需要"降温"（分布变平）；
- T < 1 表示模型欠自信；
- T = 1 表示无需校准。

温度缩放**不改变预测类别**（argmax 不变），因此不会影响准确率与混淆矩阵，只改变
置信度的数值含义——这正是我们需要的：保留原有判定、修正可信度表达。

评估指标：ECE（期望校准误差）与可靠性图（reliability diagram）。校准前后对比图
输出到 `runs/figures/calibration.png`，可直接放进设计说明书。

用法：
    python -m src.calibrate              # 拟合并写入 runs/calibration.json
    python -m src.calibrate --apply-ckpt # 同时把温度写回模型权重文件
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import numpy as np  # noqa: E402

from .utils import get_logger, load_config, now_str, resolve, set_seed  # noqa: E402

LOG = get_logger("calibrate")


# ----------------------------------------------------------------------
# 核心：在验证集上拟合温度
# ----------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    temps: np.ndarray | None = None,
                    t_min: float = 0.20, t_max: float = 30.0) -> tuple[float, dict]:
    """网格搜索最优温度（最小化负对数似然 NLL）。

    参数
    ----
    logits : [N, C] 模型原始输出（未过 softmax）
    labels : [N]    真实类别索引
    t_min  : 温度**下限**（关键约束，默认 0.20）

    关于 t_min 的必要性（实测踩坑）
    ------------------------------
    当验证集准确率为 100% 时，NLL 会被"把所有概率推向 1"这一退化解最小化，最优温度
    趋于 0。实测把下界设到 0.02 时，T 直接撞边界，校准后**所有置信度都被压成 1.0000**，
    区分度归零——这与校准的目的（让 0.95 与 0.99 体现真实差别）完全相反。

    因此在数学最优之外增加"区分度"约束：温度不得低于 t_min，保证校准后置信度仍保留
    合理离散度。最终温度还要通过 `spread_ok` 检查，若校准反而压缩了区分度则拒绝采用。

    返回 (best_T, 详情字典)。
    """
    if temps is None:
        temps = np.arange(t_min, t_max + 1e-9, 0.01)
    labels = labels.astype(np.int64)

    def nll(T: float) -> float:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(z).sum(axis=1, keepdims=True))
        log_prob = z - log_sum
        return float(-log_prob[np.arange(len(labels)), labels].mean())

    scores = np.array([nll(float(t)) for t in temps])
    best_idx = int(np.argmin(scores))
    best_T = float(temps[best_idx])
    hit_low = best_idx == 0
    hit_high = best_idx == len(temps) - 1
    direction = ("欠自信（T<1 使分布更尖锐）" if best_T < 1 else
                 "过度自信（T>1 使分布更平缓）" if best_T > 1 else "无需校准")
    return best_T, {
        "best_nll": round(float(scores[best_idx]), 6),
        "nll_at_T1": round(float(nll(1.0)), 6),
        "direction": direction,
        "hit_grid_edge": bool(hit_low or hit_high),
        "edge_warning": (f"最优温度撞到下限 {t_min}（验证集准确率为 1.0 时 NLL 偏好极端 T），"
                         f"已由 t_min 约束保证置信度仍有区分度"
                         if hit_low else
                         ("最优温度撞到上限，请扩大搜索范围后重跑" if hit_high else "")),
        "grid": {"min": float(temps[0]), "max": float(temps[-1]),
                 "step": float(temps[1] - temps[0]), "t_min_constraint": t_min},
    }


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """对 logits 施加温度并返回概率。"""
    z = logits / max(float(T), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """期望校准误差。"""
    if len(probs) == 0:
        return 0.0
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out += (m.sum() / len(conf)) * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    """可靠性图数据：每个置信度区间的平均置信度与平均准确率。"""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        pts.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "lo": round(float(lo), 2), "hi": round(float(hi), 2),
            "count": int(m.sum()),
            "avg_confidence": round(float(conf[m].mean()), 4) if m.sum() else None,
            "avg_accuracy": round(float(acc[m].mean()), 4) if m.sum() else None,
            "gap": round(float(abs(acc[m].mean() - conf[m].mean())), 4) if m.sum() else None,
        })
    return {"bins": pts, "ece": round(ece(probs, labels, bins), 5),
            "samples": int(len(probs))}


# ----------------------------------------------------------------------
# 采集验证集 logits
# ----------------------------------------------------------------------
def collect_logits(cfg: dict, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch
    from torch.utils.data import DataLoader

    from .data import DefectDataset, build_items, load_index, make_transforms
    from .model import load_checkpoint

    index = load_index(resolve(cfg["paths"]["runs"]) / "dataset_index.json")
    label_space = index["label_space"]
    class_to_idx = {c: i for i, c in enumerate(label_space)}
    norm = index.get("norm") or {"mean": [0.5], "std": [0.25]}

    ckpt_path = resolve(cfg["paths"]["models"]) / "zhijian_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"未找到模型权重 {ckpt_path}，请先运行 python -m src.train")
    model, ckpt = load_checkpoint(ckpt_path, cfg, device)

    # 优先用验证集拟合（不接触测试集，避免信息泄漏）；验证集为空时退回训练集
    split = "val" if index["splits"].get("val") else "train"
    items = build_items(index, split, class_to_idx)
    ds = DefectDataset(items, make_transforms(cfg, False, norm))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            logits_all.append(model(x.to(device)).cpu().numpy())
            labels_all.append(y.numpy())
    LOG.info("已采集 %s 集 logits：%d 个样本（类别空间 %d）", split, len(items), len(label_space))
    return np.concatenate(logits_all), np.concatenate(labels_all), list(ckpt.get("class_names_zh", label_space))


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------
def plot_calibration(before: dict, after: dict, out_path: Path, T: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="理想校准线")
    for data, label, color in ((before, "校准前", "#ef4444"), (after, "校准后", "#22c55e")):
        pts = [b for b in data["bins"] if b["count"] > 0 and b["avg_confidence"]]
        if not pts:
            continue
        xs = [p["avg_confidence"] for p in pts]
        ys = [p["avg_accuracy"] for p in pts]
        ax.plot(xs, ys, "o-", color=color, label=f"{label}（ECE={data['ece']:.4f}）", ms=5)
    ax.set_xlabel("平均置信度")
    ax.set_ylabel("实际准确率")
    ax.set_title(f"可靠性图（温度 T = {T:.2f}）")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [b["bin"] for b in before["bins"]]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, [b["count"] for b in before["bins"]], w, label="校准前", color="#ef4444", alpha=0.85)
    ax.bar(x + w / 2, [b["count"] for b in after["bins"]], w, label="校准后", color="#22c55e", alpha=0.85)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_xlabel("置信度区间")
    ax.set_ylabel("样本数")
    ax.set_title("置信度分布变化")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="置信度校准（温度缩放）")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--t-min", type=float, default=0.20,
                    help="温度下限。验证集准确率极高时 NLL 会偏好极端小的 T，"
                         "用此约束保证校准后置信度仍有区分度")
    ap.add_argument("--apply-ckpt", action="store_true",
                    help="把最优温度写回模型权重文件，使推理链路自动使用校准后的置信度")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["dataset"]["seed"]))

    logits, labels, names = collect_logits(cfg, args.device)

    probs_raw = apply_temperature(logits, 1.0)
    before = reliability_curve(probs_raw, labels)
    acc = float((probs_raw.argmax(1) == labels).mean())
    spread_raw = float(probs_raw.max(1).std())
    LOG.info("校准前：准确率 %.4f ｜ ECE %.5f ｜ 平均置信度 %.4f ｜ 置信度标准差 %.5f",
             acc, before["ece"], float(probs_raw.max(1).mean()), spread_raw)

    T, fit_info = fit_temperature(logits, labels, t_min=args.t_min)
    probs_cal = apply_temperature(logits, T)
    after = reliability_curve(probs_cal, labels)
    spread_cal = float(probs_cal.max(1).std())
    LOG.info("最优温度 T = %.2f（NLL %.5f → %.5f）｜ %s", T, fit_info["nll_at_T1"],
             fit_info["best_nll"], fit_info["direction"])

    # 区分度守门：校准的目的不是简单"提高置信度"，而是让数值有区分意义。
    # 若校准后置信度被压成常数（标准差骤降），则拒绝采用该温度。
    spread_ok = spread_cal >= 0.25 * spread_raw
    if not spread_ok:
        LOG.warning("校准后置信度区分度显著下降（标准差 %.5f → %.5f），"
                    "自动放弃校准（保持 T=1）", spread_raw, spread_cal)
        T = 1.0
        fit_info["rejected"] = True
        fit_info["reject_reason"] = (f"校准后置信度标准差 {spread_cal:.5f} 低于校准前 "
                                     f"{spread_raw:.5f} 的 25%，区分度不足")
        probs_cal = probs_raw
        after = reliability_curve(probs_cal, labels)
        spread_cal = spread_raw

    LOG.info("校准后：准确率 %.4f ｜ ECE %.5f ｜ 平均置信度 %.4f ｜ 置信度标准差 %.5f",
             float((probs_cal.argmax(1) == labels).mean()), after["ece"],
             float(probs_cal.max(1).mean()), spread_cal)

    # 置信度区分度：0.95 与 0.99 附近的实际正确率
    def acc_near(p: np.ndarray, c: float, tol: float = 0.005) -> tuple[int, float | None]:
        conf = p.max(1)
        m = np.abs(conf - c) <= tol
        if m.sum() == 0:
            return 0, None
        return int(m.sum()), round(float((p.argmax(1)[m] == labels[m]).mean()), 4)

    for c in (0.95, 0.99):
        n1, a1 = acc_near(probs_raw, c)
        n2, a2 = acc_near(probs_cal, c)
        LOG.info("置信度≈%.2f：校准前 n=%d 准确率=%s ｜ 校准后 n=%d 准确率=%s", c, n1, a1, n2, a2)

    payload = {
        "method": "Temperature Scaling (Guo et al., ICML 2017)",
        "fitted_on": "验证集（不接触测试集，避免信息泄漏）",
        "temperature": round(T, 4),
        "fit": fit_info,
        "spread": {"before_std": round(spread_raw, 5), "after_std": round(spread_cal, 5),
                   "spread_ok": bool(spread_ok)},
        "before": {"ece": before["ece"], "mean_confidence": round(float(probs_raw.max(1).mean()), 4),
                   "accuracy": round(acc, 4), "reliability": before["bins"]},
        "after": {"ece": after["ece"], "mean_confidence": round(float(probs_cal.max(1).mean()), 4),
                  "accuracy": round(float((probs_cal.argmax(1) == labels).mean()), 4),
                  "reliability": after["bins"]},
        "note": "温度缩放不改变 argmax，故准确率与混淆矩阵不变，仅修正置信度的数值含义。"
                "另设区分度守门：若校准后置信度标准差下降超过 75% 则放弃校准。",
        "generated_at": now_str(),
    }
    out = resolve(cfg["paths"]["runs"]) / "calibration.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.info("校准结果已写入 %s", out)

    fig = resolve(cfg["paths"]["runs"]) / "figures" / "calibration.png"
    plot_calibration(before, after, fig, T)
    LOG.info("可靠性图已写入 %s", fig)

    if args.apply_ckpt:
        import torch

        ckpt_path = resolve(cfg["paths"]["models"]) / "zhijian_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt["temperature"] = round(T, 4)
        ckpt["calibrated_at"] = now_str()
        ckpt["ece_before"] = before["ece"]
        ckpt["ece_after"] = after["ece"]
        torch.save(ckpt, ckpt_path)
        LOG.info("温度 T=%.4f 已写回 %s，推理将自动使用校准后的置信度", T, ckpt_path.name)

    LOG.info("校准完成：ECE %.5f → %.5f（降低 %.1f%%）",
             before["ece"], after["ece"],
             100 * (1 - after["ece"] / max(before["ece"], 1e-9)))


if __name__ == "__main__":
    main()
