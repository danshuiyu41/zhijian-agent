"""未知缺陷识别（开集识别 / 分布外检测）。

为什么必须做这件事
------------------
当前模型是**闭集 7 分类**：无论输入什么，它一定会在 7 个已知类里挑一个，
并且可能给出很高的置信度。这在产线是危险的——

举例：产线推出一类全新缺陷（例如"边裂"），模型没见过，它会把它判成"最像"的
"划痕"或"龟裂"，并且信心十足。质检员看到高置信度就放行了，实际是全新问题被掩盖。

真正安全的系统必须具备"我识别不了"的能力，并把这个判断交给人。

实现方案：特征距离 + k 近邻（k-NN）分布外检测
--------------------------------------------
1. 用已训练模型提取**倒数第二层特征**（分类头之前的 256 维向量），该特征空间里
   同一类缺陷应聚成簇；
2. 在**训练集**上抽取每类的特征，建立特征库（memory bank）；
3. 推理时计算当前样本到特征库中第 k 近邻的距离：
   - 距离小 → 落在已知分布内，正常判定；
   - 距离大 → 与所有已知类都不像，判为「未知缺陷，转人工」。

为什么选这个方案，而不是训练时用开集损失
-----------------------------------------
- **不需重训**：直接复用已训练权重，几分钟就能上线；
- **零新依赖**：只用 numpy 做距离计算；
- **可解释**：可以直接展示"最近的三个已知样本有多远"，质检员能理解；
- **可标定**：阈值由正常样本的距离分布自动确定，无需人工拍脑袋。

用法：
    python -m src.ood --fit          # 建立特征库并标定阈值
    python -m src.ood --eval         # 在留出集上评估：已知类通过率 / 未知类拦截率
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

import numpy as np  # noqa: E402

from .utils import get_logger, load_config, now_str, resolve, set_seed  # noqa: E402

LOG = get_logger("ood")
BANK_FILE = "models/feature_bank.npz"
CFG_FILE = "models/ood_config.json"


# ----------------------------------------------------------------------
# 特征提取
# ----------------------------------------------------------------------
def extract_features(cfg: dict, paths: list[str], device: str = "cpu",
                     batch: int = 64) -> np.ndarray:
    """提取一批图像的特征向量（分类头之前的池化特征）。"""
    import cv2
    import torch

    from .infer import DefectInspector

    insp = DefectInspector.from_config(device=device)
    model, size = insp.model, insp.image_size

    feats: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            tensors = []
            for p in chunk:
                gray = cv2.imread(str(resolve(p)), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
                tensors.append(insp._preprocess(gray))
            if not tensors:
                continue
            x = torch.cat(tensors, 0).to(insp.device)
            f = model.forward_features(x)                 # [B, 256]
            f = torch.nn.functional.normalize(f, dim=1)    # 归一化，便于用余弦/欧氏距离
            feats.append(f.cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 256), np.float32)


# ----------------------------------------------------------------------
# 特征库
# ----------------------------------------------------------------------
class OODDetector:
    """基于 k-NN 距离的分布外检测器。"""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 threshold: float, k: int = 5, class_names: list[str] | None = None):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.threshold = float(threshold)
        self.k = int(k)
        self.class_names = class_names or []

    # ------------------------------------------------------------------
    @staticmethod
    def _dist_block(bank: np.ndarray, q: np.ndarray) -> np.ndarray:
        """计算 query 块到库中每条的欧氏距离矩阵 [nq, nb]。

        特征已做 L2 归一化，因此 ||a-b||² = 2 - 2·(a·b)，用矩阵乘法算，
        避免 `bank[None] - q[:,None]` 那种会构造 [nq, nb, dim] 三维数组的写法——
        库扩到上万条时那种写法会直接吃爆内存。
        """
        sim = q @ bank.T                       # 余弦相似度（已是归一化向量）
        d2 = np.maximum(2.0 - 2.0 * sim, 0.0)  # 数值稳定，防止出现负值
        return np.sqrt(d2)

    @classmethod
    def knn_distance(cls, bank: np.ndarray, query: np.ndarray, k: int = 5) -> np.ndarray:
        """query 到 bank 的第 k 近邻距离（升序第 k 个）。"""
        if len(query) == 0:
            return np.zeros(0, np.float32)
        out = np.empty(len(query), np.float32)
        step = 256
        for i in range(0, len(query), step):
            d = cls._dist_block(bank, query[i:i + step])       # [nq, nb]
            kk = min(k, d.shape[1])
            part = np.partition(d, kk - 1, axis=1)[:, :kk]
            out[i:i + step] = part[:, -1]
        return out

    # ------------------------------------------------------------------
    def score(self, features: np.ndarray) -> dict:
        """对一批特征打分：返回距离、是否在已知分布内、最近邻的多数类别。"""
        if len(features) == 0:
            return {"distance": [], "is_known": [], "nearest_class": []}
        dist = self.knn_distance(self.features, features, self.k)
        # 最近邻类别：只对需要的 k 个近邻做多数投票
        step = 256
        nn_cls: list = []
        for i in range(0, len(features), step):
            d = self._dist_block(self.features, features[i:i + step])
            kk = min(self.k, d.shape[1])
            idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            for row in idx:
                vals, counts = np.unique(self.labels[row], return_counts=True)
                ci = int(vals[int(np.argmax(counts))])
                nn_cls.append(self.class_names[ci] if self.class_names else ci)
        return {"distance": [round(float(x), 4) for x in dist],
                "is_known": (dist <= self.threshold).tolist(),
                "nearest_class": nn_cls}

    def is_known(self, features: np.ndarray) -> np.ndarray:
        return self.knn_distance(self.features, features, self.k) <= self.threshold

    # ------------------------------------------------------------------
    def save(self, cfg: dict) -> None:
        np.savez_compressed(resolve(BANK_FILE), features=self.features, labels=self.labels)
        resolve(CFG_FILE).write_text(json.dumps({
            "threshold": self.threshold, "k": self.k, "class_names": self.class_names,
            "bank_size": int(len(self.features)),
            "method": "k-NN distance on penultimate features (L2-normalized)",
            "generated_at": now_str(),
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, cfg: dict) -> "OODDetector | None":
        bank, cfile = resolve(BANK_FILE), resolve(CFG_FILE)
        if not bank.exists() or not cfile.exists():
            return None
        data = np.load(bank)
        meta = json.loads(cfile.read_text(encoding="utf-8"))
        return cls(data["features"], data["labels"], meta["threshold"],
                   meta.get("k", 5), meta.get("class_names"))


# ----------------------------------------------------------------------
# 建库与标定
# ----------------------------------------------------------------------
def build_bank(cfg: dict, per_class: int = 60) -> dict:
    """用训练集样本建立特征库，并用"留出的已知样本"标定阈值。

    阈值标定思路（不拍脑袋）：
      取已知类样本，计算其 k-NN 距离分布，阈值 = 均值 + 3×标准差。
      这样正常样本的通过率约 99.7%（正态假设下），同时能拦住明显不像的输入。
      为了更稳健，同时报告 95 / 99 分位数，便于按产线容忍度选择。
    """
    from .data import load_index

    index = load_index(resolve(cfg["paths"]["runs"]) / "dataset_index.json")
    label_space = index["label_space"]
    rng = np.random.default_rng(int(cfg["dataset"]["seed"]))

    # 从训练集按类抽样建库；从验证集抽样用于标定（与建库不重叠）
    bank_paths, bank_labels = [], []
    cal_paths, cal_labels = [], []
    for ci, cls in enumerate(label_space):
        tr = list(index["splits"].get("train", {}).get(cls, []))
        va = list(index["splits"].get("val", {}).get(cls, []))
        rng.shuffle(tr)
        rng.shuffle(va)
        bank_paths += tr[:per_class]
        bank_labels += [ci] * len(tr[:per_class])
        cal_paths += va[: max(1, per_class // 4)]
        cal_labels += [ci] * len(va[: max(1, per_class // 4)])

    LOG.info("抽取样本：建库 %d 张，标定 %d 张", len(bank_paths), len(cal_paths))
    fb = extract_features(cfg, bank_paths)
    fc = extract_features(cfg, cal_paths)
    if len(fb) == 0 or len(fc) == 0:
        raise SystemExit("特征提取为空，请检查数据集")

    det = OODDetector(fb, np.array(bank_labels), threshold=1e9, k=5, class_names=label_space)
    d = det.knn_distance(fb, fc, det.k)          # 用**标定集**对库算距离（更接近真实推理条件）
    stats = {
        "mean": float(d.mean()), "std": float(d.std()),
        "p50": float(np.percentile(d, 50)), "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)), "max": float(d.max()),
    }
    threshold = stats["mean"] + 3 * stats["std"]
    # 保证阈值不低于 99 分位，避免分布偏移时过于激进
    threshold = max(threshold, stats["p99"])
    det.threshold = threshold
    det.save(cfg)

    LOG.info("特征库：%d 条，%d 维", len(fb), fb.shape[1])
    LOG.info("已知样本 k-NN 距离分布：均值 %.4f 标准差 %.4f 99分位 %.4f 最大 %.4f",
             stats["mean"], stats["std"], stats["p99"], stats["max"])
    LOG.info("自动标定阈值 = %.4f（保证已知样本通过率 ≈ 100%%）", threshold)
    return {"bank_size": len(fb), "dim": int(fb.shape[1]), "threshold": round(threshold, 4),
            **{k: round(v, 4) for k, v in stats.items()}}


# ----------------------------------------------------------------------
# 评估：已知类通过率 + 未知类拦截率
# ----------------------------------------------------------------------
def evaluate(cfg: dict) -> dict:
    """在**与推理一致的条件**下评估 OOD 检测效果。

    关键修正：模型实际运行时是把图像切成 224 窗口逐块判定的，因此特征也必须
    在同一条件下提取。早期版本直接对整幅 256×256 图像提特征，评估条件与运行
    条件不一致，得出的拦截率偏低且不可采信。

    两类样本：
      - **已知类**（测试集，模型见过的类别）→ 应尽量判为 is_known
      - **未知类**（人为构造、模型没训过的形态）→ 应尽量判为 not known
    """
    import cv2

    from .data import load_index
    from .data_generate import DEFECT_OPS, synth_base_texture

    det = OODDetector.load(cfg)
    if det is None:
        raise SystemExit("未找到特征库，请先运行：python -m src.ood --fit")

    index = load_index(resolve(cfg["paths"]["runs"]) / "dataset_index.json")
    label_space = index["label_space"]

    # 已知类：测试集每类若干张
    known_paths, known_cls = [], []
    for cls in label_space:
        ps = list(index["splits"].get("test", {}).get(cls, []))[:10]
        known_paths += ps
        known_cls += [cls] * len(ps)

    # 未知类：程序合成"没训过"的形态
    rng = np.random.default_rng(20261014)
    tmp = resolve("data/stress/unknown")
    tmp.mkdir(parents=True, exist_ok=True)
    codes = [c for c in label_space if c != "clean"]
    unknown_paths = []
    size = int(cfg["dataset"]["synthetic"].get("base_size", 256))
    for i in range(60):
        img = synth_base_texture(size, rng)
        a, b = rng.choice(codes, 2, replace=False)     # 叠加两种不同缺陷
        img = DEFECT_OPS[a](img, rng)
        img = DEFECT_OPS[b](img, rng)
        img = np.clip(img * rng.uniform(0.5, 1.4), 0, 255).astype(np.uint8)
        p = tmp / f"unknown_{i:03d}.png"
        cv2.imwrite(str(p), img)
        unknown_paths.append(str(p.relative_to(resolve("."))))

    # 按运行时条件提特征：每张图取中心 224 窗口
    fk = extract_window_features(cfg, known_paths)
    fu = extract_window_features(cfg, unknown_paths)
    dk = det.knn_distance(det.features, fk, det.k)
    du = det.knn_distance(det.features, fu, det.k)

    # 阈值扫描：给出「误拦已知」与「拦住未知」的完整权衡，而不是只报一个点
    lo = float(min(dk.min(), du.min()))
    hi = float(max(dk.max(), du.max()))
    sweep = []
    for thr in np.linspace(lo, hi, 21):
        sweep.append({
            "threshold": round(float(thr), 4),
            "known_pass_rate": round(float((dk <= thr).mean()), 4),
            "unknown_reject_rate": round(float((du > thr).mean()), 4),
        })
    # 选平衡点：在"已知通过率 ≥ 95%"的前提下，最大化未知拦截率
    ok = [s for s in sweep if s["known_pass_rate"] >= 0.95]
    best = max(ok, key=lambda s: s["unknown_reject_rate"]) if ok else sweep[-1]

    res = {
        "known_samples": len(known_paths),
        "unknown_samples": len(unknown_paths),
        "calibrated_threshold": round(det.threshold, 4),
        "calibrated_known_pass_rate": round(float((dk <= det.threshold).mean()), 4),
        "calibrated_unknown_reject_rate": round(float((du > det.threshold).mean()), 4),
        "recommended_threshold": best["threshold"],
        "recommended_known_pass_rate": best["known_pass_rate"],
        "recommended_unknown_reject_rate": best["unknown_reject_rate"],
        "known_dist_mean": round(float(dk.mean()), 4),
        "unknown_dist_mean": round(float(du.mean()), 4),
        "separation": round(float(du.mean() / max(dk.mean(), 1e-9)), 2),
        "sweep": sweep,
    }
    out = resolve(cfg["paths"]["runs"]) / "ood_eval.json"
    out.write_text(json.dumps({**res, "generated_at": now_str()}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print("=" * 78)
    print("未知缺陷识别（开集识别）评估 —— 与推理同条件（中心 224 窗口）")
    print("=" * 78)
    print(f"  特征库 {len(det.features)} 条 / {det.features.shape[1]} 维")
    print(f"  已知类 {res['known_samples']} 张　距离均值 {res['known_dist_mean']:.4f}")
    print(f"  未知类 {res['unknown_samples']} 张　距离均值 {res['unknown_dist_mean']:.4f}"
          f"　分离度 {res['separation']} 倍")
    print("-" * 78)
    print("  阈值权衡（阈值越大越宽松，越不容易误拦，也越容易漏放未知）")
    print(f"  {'阈值':<10}{'已知通过率':<14}{'未知拦截率':<14}")
    for s in sweep[::4]:
        mark = "  ← 当前标定" if abs(s["threshold"] - det.threshold) < (hi - lo) / 21 else ""
        print(f"  {s['threshold']:<10.4f}{s['known_pass_rate'] * 100:<14.1f}"
              f"{s['unknown_reject_rate'] * 100:<14.1f}{mark}")
    print("-" * 78)
    print(f"  当前标定阈值 {det.threshold:.4f}：已知通过 {res['calibrated_known_pass_rate'] * 100:.1f}%"
          f"　未知拦截 {res['calibrated_unknown_reject_rate'] * 100:.1f}%")
    print(f"  推荐阈值     {best['threshold']:.4f}：已知通过 {best['known_pass_rate'] * 100:.1f}%"
          f"　未知拦截 {best['unknown_reject_rate'] * 100:.1f}%（在已知通过率≥95%前提下最优）")
    print("=" * 78)
    print(f"结果已写入 {out}")
    return res


def extract_window_features(cfg: dict, paths: list[str], device: str = "cpu") -> np.ndarray:
    """按**运行时条件**提特征：每张图取中心 224 窗口（与滑窗判定一致）。"""
    import cv2
    import torch

    from .infer import DefectInspector

    insp = DefectInspector.from_config(device=device)
    model, win = insp.model, insp.image_size
    feats: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for p in paths:
            gray = cv2.imread(str(resolve(p)), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h, w = gray.shape[:2]
            # 中心窗口，与检测时滑窗覆盖同一区域
            y0, x0 = max(0, (h - win) // 2), max(0, (w - win) // 2)
            patch = gray[y0:y0 + win, x0:x0 + win]
            patch = cv2.resize(patch, (win, win), interpolation=cv2.INTER_AREA)
            x = insp._preprocess(patch).to(insp.device)
            f = torch.nn.functional.normalize(model.forward_features(x), dim=1)
            feats.append(f.cpu().numpy()[0])
    return np.array(feats, np.float32) if feats else np.zeros((0, 576), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="未知缺陷识别（开集识别）")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--fit", action="store_true", help="建立特征库并标定阈值")
    ap.add_argument("--eval", action="store_true", help="评估已知类通过率与未知类拦截率")
    ap.add_argument("--per-class", type=int, default=60, help="每类用于建库的样本数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["dataset"]["seed"]))

    if args.fit or not (args.fit or args.eval):
        stats = build_bank(cfg, args.per_class)
        print("\n特征库已建立：")
        print(json.dumps(stats, ensure_ascii=False, indent=1))

    if args.eval:
        evaluate(cfg)


if __name__ == "__main__":
    main()
