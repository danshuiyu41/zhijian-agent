"""模型定义：轻量 CNN 缺陷分类器 + Grad-CAM 可解释性。

选型理由（面向"一台笔记本就能跑"的交付目标）
------------------------------------------
- 主干：**MobileNetV3-Small**（约 2.5M 参数），CPU 单张 224×224 推理约 10–30 ms，
  显存/内存占用极低。相比 YOLO 系（AGPL-3.0，微调权重同样受传染），
  torchvision 主干为 **BSD-3-Clause**，许可干净，便于声称"可交付企业"。
- 头部：自定义分类头 = 主干特征 → Dropout → Linear(hidden) → GELU → Dropout → Linear(类别数)。
- 输入：单通道灰度复制为 3 通道，兼容 ImageNet 预训练权重。
- 定位：分类器 + **滑窗扫描 + NMS**（见 `src/infer.py`），无需标注框即可训练，
  这正是小厂"没有标注数据"场景下的务实做法（也是《"人工智能+制造"专项行动
  实施意见》提到的"小样本标注缺陷数据微调"思路）。

Grad-CAM
--------
对主干最后一个卷积块做梯度加权，得到类激活热力图，用于：
1) 向质检员解释"模型在看哪里"，满足工业检测的可解释性要求；
2) 在报告中输出热力图叠加，提升可信度（GB/T 40659-2021 对在线检测系统
   的可追溯性有要求）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 主干
# ----------------------------------------------------------------------
def build_backbone(name: str, pretrained: bool):
    """返回 (backbone, feature_dim, target_layer)。

    重要坑点：torchvision 的 `mobilenet_v3_*` / `efficientnet_*` 直接 `net(x)` 会
    返回**经过 avgpool + classifier 之后的向量**（形状 [N, C]），而不是特征图。
    本函数统一把 `classifier` 置为 Identity 并返回 `net.features`，保证输出始终是
    空间特征图 [N, C, H, W]，这样才能做自适应池化与 Grad-CAM。
    """
    import torchvision.models as tvm

    if name == "mobilenetv3_small_100":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.mobilenet_v3_small(weights=weights)
        dim = net.classifier[0].in_features      # 576
        target = net.features[-1]
        net.classifier = nn.Identity()
        return net.features, dim, target

    if name == "mobilenetv3_large_100":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.mobilenet_v3_large(weights=weights)
        dim = net.classifier[0].in_features      # 960
        target = net.features[-1]
        net.classifier = nn.Identity()
        return net.features, dim, target

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)
        dim = net.fc.in_features
        target = net.layer4[-1]
        net.fc = nn.Identity()
        # 关键：ResNet.forward 里的 `torch.flatten(x, 1)` 依赖 avgpool 先做池化。
        # 若把 avgpool 设为 Identity，flatten 会把 [N,C,H,W] 拉平成 [N, C*H*W]，
        # 特征维度就不对了。这里保留 AdaptiveAvgPool2d，使 net(x) 输出 [N, C, 1, 1]，
        # 再经 flatten 得到 [N, C] —— 与其它主干语义一致。
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        return net, dim, target

    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.efficientnet_b0(weights=weights)
        dim = net.classifier[-1].in_features
        target = net.features[-1]
        net.classifier = nn.Identity()
        return net.features, dim, target

    raise ValueError(f"不支持的主干：{name}（可选 mobilenetv3_small_100 / mobilenetv3_large_100 / "
                     f"resnet18 / efficientnet_b0）")


class DefectClassifier(nn.Module):
    """轻量工业表面缺陷分类器。"""

    def __init__(self, num_classes: int, backbone: str = "mobilenetv3_small_100",
                 pretrained: bool = True, dropout: float = 0.2, head_hidden: int = 256):
        super().__init__()
        self.backbone_name = backbone
        self.body, dim, self.target_layer = build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        self.num_classes = num_classes
        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """输出 [B, C] 的池化特征。

        兼容两类主干：
        - 卷积特征图主干（mobilenetv3 / efficientnet 的 features）：输出 [B,C,H,W]，需池化；
        - 带内置 avgpool+flatten 的完整网络（resnet）：输出已是 [B,C]。
        """
        f = self.body(x)
        if f.dim() == 4:
            f = self.pool(f)
        return f.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    # ------------------------------------------------------------------
    # 参数量统计（申报书写"轻量化"要有数字支撑）
    # ------------------------------------------------------------------
    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "total_m": round(total / 1e6, 3), "trainable_m": round(trainable / 1e6, 3)}


# ----------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------
class GradCAM:
    """最小实现的 Grad-CAM（不依赖 pytorch-grad-cam，减少依赖）。"""

    def __init__(self, model: DefectClassifier, target_layer: nn.Module | None = None):
        self.model = model
        self.layer = target_layer or model.target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h1 = self.layer.register_forward_hook(self._save_activation)
        self._h2 = self.layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output):
        self.activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (logits, cam)。cam 形状 [B, H, W]，已按样本归一化到 0~1。

        class_idx 为 None 时对每个样本取各自的 argmax（逐样本解释），
        指定时对所有样本解释同一类别。
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)                                  # [B, C]
        if class_idx is None:
            idx = logits.argmax(1)                              # [B]
        else:
            idx = torch.full((logits.shape[0],), int(class_idx),
                             dtype=torch.long, device=logits.device)
        score = logits.gather(1, idx.view(-1, 1)).sum()
        score.backward()

        act, grad = self.activations, self.gradients
        if act is None or grad is None:
            raise RuntimeError("Grad-CAM 未捕获到激活/梯度，请检查 target_layer")
        weights = grad.mean(dim=(2, 3), keepdim=True)            # [B,C,1,1]
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))   # [B,1,H,W]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        b = cam.shape[0]
        flat = cam.view(b, -1)
        mn = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        mx = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return logits.detach(), cam.detach().squeeze(1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


# ----------------------------------------------------------------------
# 构建器
# ----------------------------------------------------------------------
def build_model(cfg: dict, pretrained: bool | None = None,
                num_classes: int | None = None) -> DefectClassifier:
    """构建模型。num_classes 未指定时用配置值（注意：实际标签空间为
    缺陷类别数 + clean，训练脚本应显式传入真实类别数以匹配标签索引）。"""
    m = cfg["model"]
    if pretrained is None:
        pretrained = bool(m.get("pretrained", True))
    return DefectClassifier(
        num_classes=int(num_classes if num_classes is not None else m["num_classes"]),
        backbone=m["backbone"],
        pretrained=pretrained,
        dropout=float(m.get("dropout", 0.2)),
        head_hidden=int(m.get("head_hidden", 256)),
    )


def load_checkpoint(path, cfg: dict, device: str = "cpu") -> tuple[DefectClassifier, dict]:
    """加载训练好的模型与类别映射。

    类别数优先从 `classes` 字段推断；若缺失，则从最后一层 Linear 的权重形状反推，
    避免"配置里的 num_classes 与实际标签空间不一致"导致的 shape mismatch。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    classes = ckpt.get("classes")
    if classes:
        num_classes = len(classes)
    else:
        # 从 head 最后一层 Linear 的 bias/weight 推断
        key = next((k for k in state if k.startswith("head.") and k.endswith(".bias")), None)
        num_classes = int(state[key].shape[0]) if key else int(cfg["model"]["num_classes"])
        ckpt["classes"] = [f"class_{i}" for i in range(num_classes)]
    model = DefectClassifier(
        num_classes=num_classes,
        backbone=ckpt.get("backbone", cfg["model"]["backbone"]),
        pretrained=False,
        dropout=float(cfg["model"].get("dropout", 0.2)),
        head_hidden=int(cfg["model"].get("head_hidden", 256)),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt
