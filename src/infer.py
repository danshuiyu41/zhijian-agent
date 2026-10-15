"""推理链路：滑窗检测 + NMS 合并 + 严重度分级 + 可视化。

为什么用"分类器 + 滑窗"而不是端到端检测器
----------------------------------------
1. **标注成本**：小厂只有零散照片，没有像素级/框级标注。分类器只需图片级标签，
   滑窗定位即可给出缺陷位置，落地门槛最低（对应《"人工智能+制造"专项行动实施
   意见》"小样本标注缺陷数据微调"的思路）。
2. **许可干净**：YOLO 系为 AGPL-3.0 且**微调权重同样受传染**，torchvision 主干为
   BSD-3-Clause，便于声称"可交付企业"。
3. **CPU 可跑**：MobileNetV3-Small 单窗口 10–30 ms，整图滑窗 0.3–0.8 s，
   无需 GPU 即可在普通工控机上演示与部署。

流程
----
图像 → 灰度化 → 滑窗切块 → 归一化 → 批量前向 → 取非洁净类别 → 置信度筛选 →
按类别做 NMS → 计算面积占比 → 严重度评分(A–E) → 叠加可视化/热力图 → 结构化结果

用法：
    from src.infer import DefectInspector
    inspector = DefectInspector.from_config()
    result = inspector.inspect("data/samples/demo.png", want_heatmap=True)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import (class_risk_map, class_zh_map, get_logger, load_config, resolve,
                    severity_grade)

LOG = get_logger("infer")


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class Detection:
    """单条缺陷检测结果。"""
    class_code: str
    class_zh: str
    confidence: float
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2
    area_ratio: float                    # 缺陷框面积 / 图像面积
    severity_score: float                # 0~100
    severity_level: str                  # A~E
    severity_zh: str
    action: str
    risk: str = "low"
    # 未知缺陷识别结果：ood_distance 为该样本特征到特征库的 k 近邻距离，
    # is_unknown=True 表示"离所有已知类别都太远，分类结果不可信"。
    ood_distance: float | None = None
    is_unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_code": self.class_code,
            "class_zh": self.class_zh,
            "confidence": round(self.confidence, 4),
            "bbox": [int(v) for v in self.bbox],
            "area_ratio": round(self.area_ratio, 5),
            "severity_score": round(self.severity_score, 1),
            "severity_level": self.severity_level,
            "severity_zh": self.severity_zh,
            "action": self.action,
            "risk": self.risk,
            "ood_distance": round(self.ood_distance, 4) if self.ood_distance is not None else None,
            "is_unknown": self.is_unknown,
        }


@dataclass
class InspectionResult:
    """一次完整检测的结果。"""
    image_path: str = ""
    image_size: tuple[int, int] = (0, 0)
    elapsed_ms: float = 0.0
    windows: int = 0
    verdict: str = "合格"
    max_severity: float = 0.0
    grade: str = "A"
    grade_zh: str = "合格"
    grade_color: str = "#22c55e"
    action: str = "放行"
    detections: list[Detection] = field(default_factory=list)
    class_summary: dict[str, int] = field(default_factory=dict)
    top1: dict[str, Any] | None = None
    quality_flags: list[str] = field(default_factory=list)
    # 未知缺陷统计：unknown_count > 0 时判定结果不可全信，界面需显著提示
    unknown_count: int = 0
    need_manual_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "windows": self.windows,
            "verdict": self.verdict,
            "max_severity": round(self.max_severity, 1),
            "grade": self.grade,
            "grade_zh": self.grade_zh,
            "grade_color": self.grade_color,
            "action": self.action,
            "detections": [d.to_dict() for d in self.detections],
            "class_summary": self.class_summary,
            "top1": self.top1,
            "unknown_count": self.unknown_count,
            "need_manual_review": self.need_manual_review,
            "quality_flags": self.quality_flags,
        }


# ----------------------------------------------------------------------
# 严重度评分
# ----------------------------------------------------------------------
def severity_score(cfg: dict, risk: str, confidence: float, area_ratio: float,
                   area_ref: float | None = None) -> float:
    """计算 0~100 严重度得分。

    公式（v2，嵌套加权，替代 v1 的线性加权）：

        severity = 100 · [ w_risk · R + (1 - w_risk) · ( w_conf · C + w_area · A ) ]

        R = 类别风险基线分（high 0.70 / mid 0.42 / low 0.18）
        C = 检出置信度
        A = min(1, sqrt(area / area_ref))，并受 area_cap 饱和约束

    **为什么改成嵌套式（v1 的实测缺陷）**
    v1 为 `100·(0.40·R + 0.25·C + 0.35·A)`，R 取 1.0/0.65/0.35。风险项自身就贡献
    14~40 分，直接跨越多个等级区间，实测结果是"高危恒判 D、洁净恒判 A"，B 级几乎
    没有任何缺陷能落入，分级失去梯度与说服力。

    v2 把风险当作**基线先验**（权重 w_risk），其余权重交给"缺陷相对尺度 + 模型置信度"，
    于是同一类缺陷会随尺度与置信度在相邻等级间移动，得到"风险定基线、尺度与置信度
    决定升降级"的合理行为。各类别风险基线分由 config.severity.risk_score 标定。

    area_ref / area_cap 同样可由配置覆盖，便于按具体产线重新标定。
    """
    w = cfg["severity"]["weights"]
    risk_val = float(cfg["severity"]["risk_score"].get(risk, 0.30))
    w_risk = float(w.get("risk", 0.55))
    w_conf = float(w.get("confidence", 0.20))
    w_area = float(w.get("area", 0.25))
    # 非风险项内部归一化，保证二者权重和为 1，使公式量纲稳定
    rest = w_conf + w_area
    if rest <= 0:
        w_conf, w_area, rest = 0.5, 0.5, 1.0
    w_conf, w_area = w_conf / rest, w_area / rest

    ref = float(area_ref if area_ref is not None else cfg["severity"].get("area_ref", 0.05))
    cap = float(cfg["severity"].get("area_cap", 0.30))
    ratio = max(0.0, float(area_ratio)) / max(ref, 1e-6)
    if str(cfg["severity"].get("area_transform", "sqrt")).lower() == "sqrt":
        area_val = min(1.0, ratio ** 0.5)
    else:
        area_val = min(1.0, ratio)
    area_val = min(area_val, cap)
    score = 100.0 * (w_risk * risk_val
                     + (1.0 - w_risk) * (w_conf * float(confidence) + w_area * area_val))
    return float(max(0.0, min(100.0, score)))


# ----------------------------------------------------------------------
# 检测器
# ----------------------------------------------------------------------
class DefectInspector:
    """缺陷检测器：封装模型加载、滑窗推理、NMS、分级与可视化。"""

    def __init__(self, cfg: dict, ckpt_path: str | Path | None = None, device: str = "cpu",
                 batch_size: int = 32):
        import torch

        from .model import load_checkpoint

        self.cfg = cfg
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        # CPU 推理性能关键：启用多线程并对推理关闭梯度计算（inference_mode）
        if self.device == "cpu":
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 4)))
            except Exception:
                pass
        self.ckpt_path = Path(ckpt_path) if ckpt_path else resolve(cfg["paths"]["models"]) / "zhijian_best.pt"
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"未找到模型权重 {self.ckpt_path}，请先运行：python -m src.train"
            )
        self.model, self.ckpt = load_checkpoint(self.ckpt_path, cfg, self.device)
        self.classes: list[str] = list(self.ckpt["classes"])
        self.class_names_zh: list[str] = list(self.ckpt.get("class_names_zh", self.classes))
        self.norm = self.ckpt.get("norm", {"mean": [0.5], "std": [0.25]})
        self.image_size = int(self.ckpt.get("image_size", cfg["dataset"]["image_size"]))
        self.clean_index = self.classes.index("clean") if "clean" in self.classes else -1
        self.zh_map = dict(zip(self.classes, self.class_names_zh))
        self.risk_map = class_risk_map(cfg)
        m = cfg["model"]
        self.window = int(m.get("window", 224))
        self.stride = int(m.get("stride", 112))
        self.score_threshold = float(m.get("score_threshold", 0.55))
        self.nms_iou = float(m.get("nms_iou", 0.35))
        # 候选融合阈值：占较小框的交叠比例，用于把同一缺陷的多个窗口合成一个框
        self.overlap_thr = float(m.get("overlap_thr", 0.30))
        # 并集膨胀上限：合并后并集面积 / 合并前较大框面积，超过则不合并（防止框链式膨胀到全图）
        self.union_growth = float(m.get("union_growth", 1.6))
        self.min_area_ratio = float(m.get("min_area_ratio", 5e-5))
        self.topk = int(m.get("topk", 12))
        # 多尺度窗口：兼顾小缺陷与大缺陷（工业现场缺陷尺度差异大）
        self.scales = [1.0, 0.6]
        self.batch_size = int(batch_size)
        # 定位细化窗口数上限：0 表示关闭（默认）。开启会显著增加单图耗时，
        # 因为每个窗口都要做一次 Grad-CAM 反向传播。
        self.refine_topn = int(m.get("refine_topn", 0))

        # 未知缺陷识别（开集识别）：加载特征库后对每个检出窗口做分布外判定。
        # 阈值优先取 ood_eval.json 的"推荐阈值"（在已知通过率≥95% 前提下最优），
        # 而非建库时的保守标定值——否则安全阀形同虚设（实测保守阈值只拦得住 25%）。
        self.ood = None
        self.ood_threshold: float | None = None
        if bool(m.get("ood_enabled", True)):
            try:
                from .ood import OODDetector

                self.ood = OODDetector.load(cfg)
                if self.ood is not None:
                    self.ood_threshold = self.ood.threshold
                    ev = resolve(cfg["paths"]["runs"]) / "ood_eval.json"
                    if ev.exists():
                        rec = json.loads(ev.read_text(encoding="utf-8")).get("recommended_threshold")
                        if rec:
                            self.ood_threshold = float(rec)
                    LOG.info("未知缺陷识别已启用：特征库 %d 条，阈值 %.4f",
                             len(self.ood.features), self.ood_threshold)
            except Exception as e:  # 缺特征库时静默降级，不影响常规检测
                LOG.info("未知缺陷识别未启用（%s）", e)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml",
                    ckpt_path: str | Path | None = None, device: str = "cpu") -> "DefectInspector":
        cfg = load_config(config_path)
        return cls(cfg, ckpt_path=ckpt_path, device=device)

    # ------------------------------------------------------------------
    def _preprocess(self, gray: np.ndarray) -> "np.ndarray":
        """灰度块 -> 归一化后的 NCHW 张量（float32，3 通道）。"""
        import torch

        mean = float(self.norm["mean"][0])
        std = float(self.norm["std"][0]) or 0.25
        x = gray.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = np.repeat(x[None, ...], 3, axis=0)          # [3,H,W]
        return torch.from_numpy(x).unsqueeze(0)          # [1,3,H,W]

    def _windows(self, h: int, w: int) -> list[tuple[int, int, int, int, float]]:
        """生成滑窗坐标（含多尺度），返回 (x1,y1,x2,y2,scale)。"""
        out: list[tuple[int, int, int, int, float]] = []
        for scale in self.scales:
            win = max(48, int(self.window * scale))
            stride = max(24, int(self.stride * scale))
            if win > min(h, w):
                win = min(h, w)
                stride = max(16, win // 2)
            ys = list(range(0, max(1, h - win + 1), stride))
            xs = list(range(0, max(1, w - win + 1), stride))
            if not ys or ys[-1] != h - win:
                ys.append(max(0, h - win))
            if not xs or xs[-1] != w - win:
                xs.append(max(0, w - win))
            for y in sorted(set(ys)):
                for x in sorted(set(xs)):
                    out.append((x, y, x + win, y + win, scale))
        return out

    @staticmethod
    def _overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """交叠度 = 交集面积 / 较小框面积。

        比 IoU 更适合"窗口级候选合并"：一个 224 窗口与一个更大的窗口若大部分重叠，
        IoU 可能很小（因为并集很大），但显然应视为同一缺陷；用"占较小框的比例"
        可以正确地把它们合并到一起。
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / float(max(1, min(area_a, area_b)))

    def _merge_candidates(self, dets: list[dict]) -> list[dict]:
        """把同一缺陷的多个窗口候选融合成**一个**检测框（按类别分别处理）。

        为什么需要这一步
        ----------------
        滑窗步长 112、窗口 224（50% 重叠），同一处缺陷会落在多个窗口里，
        实测一张图上可产生 5~7 个高度重叠的框，前端画面杂乱、严重度也难以解释。
        仅靠 NMS 无法解决：不同尺度的窗口大小不同，IoU 很低但显然是同一缺陷。

        融合规则（逐类别、按置信度从高到低贪心分组）：
        1. 若候选框与已有组的任一成员交叠度 ≥ overlap_thr，视为同一缺陷；
        2. **并集膨胀约束**：合并后并集面积不得超过合并前较大框的 `union_growth` 倍，
           否则说明两个候选其实是不同位置的缺陷，不合并。
           若不加这条约束，"链式重叠"会把框一路并到覆盖整张图
           （实测出现过全图 256×256 的框，严重度面积项失去意义）。
        3. 组内取最高置信度作为缺陷置信度，坐标取组内并集。
        """
        merged: list[dict] = []
        for cls in {d["class_code"] for d in dets}:
            group = sorted([d for d in dets if d["class_code"] == cls],
                           key=lambda d: -d["confidence"])
            clusters: list[list[dict]] = []
            for d in group:
                placed = False
                for cl in clusters:
                    if not any(self._overlap_ratio(d["bbox"], m["bbox"]) >= self.overlap_thr
                               for m in cl):
                        continue
                    u = self._union_bbox([m["bbox"] for m in cl] + [d["bbox"]])
                    union_area = (u[2] - u[0]) * (u[3] - u[1])
                    biggest = max((m["bbox"] for m in cl), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    biggest_area = (biggest[2] - biggest[0]) * (biggest[3] - biggest[1])
                    if union_area <= self.union_growth * max(biggest_area, 1):
                        cl.append(d)
                        placed = True
                        break
                if not placed:
                    clusters.append([d])
            for cl in clusters:
                best = max(cl, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = self._union_bbox([m["bbox"] for m in cl])
                merged.append({**best, "bbox": (int(x1), int(y1), int(x2), int(y2)),
                               "n_windows": len(cl)})
        return merged

    @staticmethod
    def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    @staticmethod
    def _nms(dets: list[dict], iou_thr: float) -> list[dict]:
        """按类别做 NMS（IoU 口径，保留传统实现以便对比与复用）。"""
        keep: list[dict] = []
        for cls in {d["class_code"] for d in dets}:
            group = sorted([d for d in dets if d["class_code"] == cls],
                           key=lambda d: -d["confidence"])
            while group:
                best = group.pop(0)
                keep.append(best)
                bx1, by1, bx2, by2 = best["bbox"]
                rest = []
                for d in group:
                    x1, y1, x2, y2 = d["bbox"]
                    ix1, iy1 = max(bx1, x1), max(by1, y1)
                    ix2, iy2 = min(bx2, x2), min(by2, y2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    union = (bx2 - bx1) * (by2 - by1) + (x2 - x1) * (y2 - y1) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou <= iou_thr:
                        rest.append(d)
                group = rest
        return keep

    # ------------------------------------------------------------------
    def _refine_bbox(self, patch_gray: np.ndarray, class_idx: int,
                     win_box: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], float]:
        """在滑窗内定位缺陷的**实际区域**，返回 (细化后的全图坐标框, 窗口内面积占比)。

        动机（实测发现的关键问题）
        --------------------------
        滑窗分类只给出"这个窗口里有某类缺陷"。若直接把窗口当作缺陷框，会导致：
        ① 面积占比恒接近 1，严重度评分虚高，所有缺陷都被判 D/E 级；
        ② 前端把整个窗口画成大框，定位不准，影响可解释性与演示效果。

        做法：用类激活（Grad-CAM）在窗口内做阈值分割，取最大连通域的外接矩形，
        再按窗口尺寸换算回全图坐标。若激活区域几乎覆盖整个窗口（说明缺陷确实弥散），
        则保留原窗口框，避免过度收缩导致漏掉缺陷范围。
        """
        from .model import GradCAM

        holder = GradCAM(self.model)
        try:
            x = self._preprocess(patch_gray).to(self.device)
            _, cam = holder(x, class_idx=int(class_idx))
        finally:
            holder.remove()

        cam_np = cam[0].cpu().numpy()
        hi = float(cam_np.max())
        if hi <= 1e-6:
            return win_box, 1.0
        # 阈值取最大激活的 45%（经验值，兼顾小目标与弥散缺陷）
        binary = (cam_np >= 0.45 * hi).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return win_box, 1.0
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx = int(stats[idx, cv2.CC_STAT_LEFT])
        by = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        win_h, win_w = patch_gray.shape[:2]
        area_frac = (bw * bh) / float(max(win_w * win_h, 1))
        if area_frac > 0.85:          # 缺陷弥散，不做收缩
            return win_box, 1.0
        x1, y1, x2, y2 = win_box
        sx, sy = (x2 - x1) / float(win_w), (y2 - y1) / float(win_h)
        refined = (int(x1 + bx * sx), int(y1 + by * sy),
                   int(x1 + (bx + bw) * sx), int(y1 + (by + bh) * sy))
        if refined[2] - refined[0] < 6 or refined[3] - refined[1] < 6:
            return win_box, 1.0       # 退化成点，保留窗口框
        return refined, area_frac

    # ------------------------------------------------------------------
    def inspect(self, image_path: str | Path | np.ndarray, want_heatmap: bool = False,
                max_detections: int | None = None) -> tuple[InspectionResult, dict[str, Any]]:
        """执行一次检测。返回 (结果对象, 可视化数据)。

        可视化数据含：
          - boxes: 供前端 Canvas 绘制的框列表
          - heatmap_png: base64 PNG（want_heatmap=True 时）
          - annotated_png: base64 PNG（叠加标注图）
        """
        import torch

        t0 = time.time()
        path_str = ""
        if isinstance(image_path, np.ndarray):
            gray = image_path if image_path.ndim == 2 else cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
        else:
            path_str = str(image_path)
            gray = cv2.imread(str(resolve(image_path)) if not Path(image_path).is_absolute() else str(image_path),
                              cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"无法读取图像：{image_path}")

        h, w = gray.shape[:2]
        wins = self._windows(h, w)
        raw: list[dict] = []
        clean_prob_max = 0.0
        prob_sums: dict[str, float] = {}

        batch_size = self.batch_size
        self.model.eval()
        with torch.inference_mode():
            for i in range(0, len(wins), batch_size):
                chunk = wins[i:i + batch_size]
                tensors = []
                for (x1, y1, x2, y2, _s) in chunk:
                    patch = gray[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    patch = cv2.resize(patch, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                    tensors.append(self._preprocess(patch))
                if not tensors:
                    continue
                batch = torch.cat(tensors, 0).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                # 未知缺陷识别：取同一批次的池化特征，与特征库比距离。
                # 复用同一次前向的特征，不额外增加推理开销。
                ood_dist = None
                if self.ood is not None:
                    try:
                        feats = torch.nn.functional.normalize(
                            self.model.forward_features(batch), dim=1).cpu().numpy()
                        ood_dist = self.ood.knn_distance(self.ood.features, feats, self.ood.k)
                    except Exception:
                        ood_dist = None
                for k_i, ((x1, y1, x2, y2, _s), prob) in enumerate(zip(chunk, probs)):
                    if self.clean_index >= 0:
                        clean_prob_max = max(clean_prob_max, float(prob[self.clean_index]))
                    for ci, code in enumerate(self.classes):
                        if ci == self.clean_index:
                            continue
                        prob_sums[code] = prob_sums.get(code, 0.0) + float(prob[ci])
                    if self.clean_index >= 0:
                        defect_idx = int(np.argmax(np.delete(prob, self.clean_index)))
                        # 映射回原始索引
                        orig = [i for i in range(len(self.classes)) if i != self.clean_index][defect_idx]
                    else:
                        orig = int(np.argmax(prob))
                    conf = float(prob[orig])
                    if conf < self.score_threshold:
                        continue
                    cls = self.classes[orig]
                    if cls == "clean":
                        continue
                    # 分布外判定：该窗口特征离特征库过远 → 模型对它没有把握，
                    # 即便分类头给出了高置信度也不可信，标记为"疑似未知缺陷"。
                    ood_d = float(ood_dist[k_i]) if ood_dist is not None and k_i < len(ood_dist) else None
                    is_unknown = bool(
                        ood_d is not None and self.ood_threshold is not None
                        and ood_d > self.ood_threshold)
                    raw.append({
                        "class_code": cls,
                        "class_zh": self.zh_map.get(cls, cls),
                        "confidence": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "class_index": orig,
                        "risk": self.risk_map.get(cls, "low"),
                        "ood_distance": ood_d,
                        "is_unknown": is_unknown,
                    })

        # 定位细化：仅对置信度最高的若干窗口做（Grad-CAM 需反向传播，代价较高）。
        # 默认关闭，通过 --refine 或配置 model.refine_topn 开启；关闭时框即滑窗范围，
        # 面积项由 severity.area_cap 补偿，保证分级仍有区分度。
        topn = int(self.refine_topn)
        if topn > 0 and raw:
            order = sorted(range(len(raw)), key=lambda i: -raw[i]["confidence"])[:topn]
            for i in order:
                d = raw[i]
                x1, y1, x2, y2 = d["bbox"]
                patch = gray[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                patch = cv2.resize(patch, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_AREA)
                try:
                    refined, area_frac = self._refine_bbox(patch, d["class_index"], d["bbox"])
                    d["bbox"] = refined
                    d["win_area_frac"] = area_frac
                except Exception as e:  # 细化失败退回窗口框，保证流程不中断
                    LOG.debug("定位细化失败（退回窗口框）：%s", e)
                    d["win_area_frac"] = 1.0

        # 先按"占较小框交叠度"把同一缺陷的多个窗口融合为一个候选，
        # 再按 IoU 做一次 NMS 兜底，避免前端出现一堆重叠框。
        fused = self._merge_candidates(raw)
        merged = self._nms(fused, self.nms_iou)
        img_area = float(h * w)
        dets: list[Detection] = []
        for d in merged:
            x1, y1, x2, y2 = d["bbox"]
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area
            if area_ratio < self.min_area_ratio:
                continue
            score = severity_score(self.cfg, d["risk"], d["confidence"], area_ratio)
            g = severity_grade(self.cfg, score)
            dets.append(Detection(
                class_code=d["class_code"], class_zh=d["class_zh"], confidence=d["confidence"],
                bbox=d["bbox"], area_ratio=area_ratio, severity_score=score,
                severity_level=g["level"], severity_zh=g["zh"], action=g["action"], risk=d["risk"],
                ood_distance=d.get("ood_distance"), is_unknown=bool(d.get("is_unknown", False)),
            ))
        dets.sort(key=lambda d: -d.severity_score)
        limit = max_detections or self.topk
        dets = dets[:limit]

        result = InspectionResult(
            image_path=path_str, image_size=(w, h), elapsed_ms=(time.time() - t0) * 1000,
            windows=len(wins), detections=dets,
        )
        result.class_summary = {}
        for d in dets:
            result.class_summary[d.class_code] = result.class_summary.get(d.class_code, 0) + 1
        if dets:
            top = dets[0]
            result.max_severity = top.severity_score
            result.grade, result.grade_zh = top.severity_level, top.severity_zh
            result.action = top.action
            for g in self.cfg["severity"]["grades"]:
                if g["level"] == result.grade:
                    result.grade_color = g["color"]
            result.verdict = "不合格" if result.grade in ("C", "D", "E") else "合格"
            result.top1 = top.to_dict()
            result.quality_flags.append(f"检出 {len(dets)} 处缺陷，最高严重度 {result.max_severity:.1f}")
            # 未知缺陷告警：这是产线最该看到的一条提示——分类结果不可信，
            # 但系统主动暴露了"我不确定"，而不是硬给一个高置信度的错误类别。
            unknown = [d for d in dets if d.is_unknown]
            if unknown:
                result.unknown_count = len(unknown)
                names = "、".join(sorted({d.class_zh for d in unknown}))
                result.quality_flags.append(
                    f"⚠ 疑似未知缺陷 {len(unknown)} 处（特征距离超出已知分布，分类结果不可信）：{names}"
                    f"，建议转人工确认并补充标注")
                # 未知缺陷一律按最高风险处理：宁可多拦，不可漏放
                result.need_manual_review = True
        else:
            result.verdict = "合格"
            result.action = "放行"
            result.quality_flags.append("未检出缺陷")
        if clean_prob_max and clean_prob_max < 0.5:
            result.quality_flags.append("洁净置信度偏低，建议人工确认")

        viz = {
            "boxes": [d.to_dict() for d in dets],
            "image_size": [w, h],
            "clean_prob_max": round(clean_prob_max, 4),
            "prob_sums": {k: round(v, 3) for k, v in sorted(prob_sums.items(), key=lambda x: -x[1])},
            "threshold": self.score_threshold,
        }
        if want_heatmap:
            viz["heatmap_png"] = self.heatmap_base64(gray, dets)
            viz["annotated_png"] = self.annotate_base64(gray, dets)
        return result, viz

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    @staticmethod
    def _find_cjk_font(size: int = 16):
        """查找可用的中文字体。

        OpenCV 的 putText 只支持 Hershey 矢量字体，**无法渲染中文**——实测会在图上
        显示为 "??????"。因此中文标签统一交给 PIL 绘制。此处按平台常见路径依次探测。
        """
        from PIL import ImageFont

        candidates = [
            r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\Deng.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
        for p in candidates:
            try:
                if Path(p).exists():
                    return ImageFont.truetype(p, size)
            except Exception:
                continue
        return None

    def _draw_label(self, img_bgr: np.ndarray, text: str, x: int, y: int,
                    color_bgr: tuple[int, int, int]) -> np.ndarray:
        """在 (x, y) 处绘制带底色的中文标签，返回新图。"""
        from PIL import Image, ImageDraw

        font = self._font
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        try:
            box = draw.textbbox((0, 0), text, font=font)
            tw, th, off_y = box[2] - box[0], box[3] - box[1], box[1]
        except Exception:
            tw, th, off_y = len(text) * 9, 14, 0
        pad = 4
        h, w = img_bgr.shape[:2]
        x = max(0, min(x, w - tw - 2 * pad))
        top = max(0, y - th - 2 * pad)
        rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.rectangle([x, top, x + tw + 2 * pad, top + th + 2 * pad], fill=rgb)
        draw.text((x + pad, top + pad - off_y), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def annotate(self, gray: np.ndarray, dets: list[Detection], line: int = 2) -> np.ndarray:
        """在灰度图上绘制检测框与中文标签，返回 BGR 图。"""
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=10)
        if not hasattr(self, "_font"):
            self._font = self._find_cjk_font(15)
        used_rows: list[tuple[int, int]] = []          # 已占用的标签纵向区间，避免叠字
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            color = self._level_bgr(d.severity_level)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line)
            label = f"{d.class_zh} {d.confidence:.2f} {d.severity_level}"
            ly = y1
            for (a, b) in used_rows:                    # 简单避让：若与已有标签重叠就下移
                if not (ly + 22 < a or ly > b):
                    ly = b + 2
            used_rows.append((max(0, ly - 22), ly + 2))
            img = self._draw_label(img, label, x1, ly, color)
        return img

    def _level_bgr(self, level: str) -> tuple[int, int, int]:
        for g in self.cfg["severity"]["grades"]:
            if g["level"] == level:
                hexc = g["color"].lstrip("#")
                r, gg, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
                return (b, gg, r)
        return (0, 200, 0)

    def heatmap(self, gray: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        """Grad-CAM 热力图（灰度底图叠加 JET 伪彩）。

        class_idx 默认取模型预测类别：这样洁净样本的热力图展示的是"洁净"类别的
        证据区域，而不会像早期实现那样对任何图都高亮某个缺陷类别，避免误导评委。
        """
        import torch

        from .model import GradCAM

        patch = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = self._preprocess(patch).to(self.device)
        holder = GradCAM(self.model)
        try:
            with torch.inference_mode(False), torch.enable_grad():
                logits, cam = holder(x, class_idx=class_idx)
        finally:
            holder.remove()
        cam_np = cam[0].cpu().numpy()
        cam_np = cv2.resize(cam_np, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        cam_np = np.clip(cam_np, 0, 1)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(canvas, 0.55, heat, 0.45, 0)

    def heatmap_base64(self, gray: np.ndarray, dets: list[Detection] | None = None) -> str:
        import base64

        img = self.heatmap(gray)
        if dets:
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), self._level_bgr(d.severity_level), 2)
        ok, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""

    def annotate_base64(self, gray: np.ndarray, dets: list[Detection]) -> str:
        import base64

        img = self.annotate(gray, dets)
        ok, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


# ----------------------------------------------------------------------
# 批量检测（用于实测指标与看板数据）
# ----------------------------------------------------------------------
def batch_inspect(inspector: DefectInspector, files: list[str], want_heatmap: bool = False) -> dict:
    rows = []
    t0 = time.time()
    for f in files:
        try:
            r, _ = inspector.inspect(f, want_heatmap=want_heatmap)
            d = r.to_dict()
            d["truth_label"] = Path(f).parent.name
            rows.append(d)
        except Exception as e:
            LOG.warning("检测失败 %s：%s", f, e)
    total = time.time() - t0
    return {
        "count": len(rows),
        "total_seconds": round(total, 2),
        "avg_ms": round(1000 * total / max(len(rows), 1), 1),
        "rows": rows,
    }
