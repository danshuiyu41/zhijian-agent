"""主干与 Grad-CAM 自检脚本。

对 config 中支持的所有主干做一次前向 + Grad-CAM 冒烟测试，输出参数量与形状。
用于在训练前快速确认环境与模型定义正确（本项目开发中曾发现 torchvision
mobilenet 系列 net(x) 返回池化后向量、resnet 的 flatten 依赖 avgpool 两个坑）。

用法：
    python -m src.selfcheck_model
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "torch_cache"))

import torch  # noqa: E402

from .model import GradCAM, build_model  # noqa: E402

BACKBONES = ["mobilenetv3_small_100", "mobilenetv3_large_100", "resnet18", "efficientnet_b0"]


def main() -> None:
    x = torch.randn(2, 3, 224, 224)
    print(f"{'主干':<26}{'logits':<14}{'cam':<20}{'参数量':<10}")
    print("-" * 72)
    bad = 0
    for bb in BACKBONES:
        cfg = {"model": {"backbone": bb, "num_classes": 7, "pretrained": False,
                         "dropout": 0.2, "head_hidden": 256}}
        try:
            m = build_model(cfg, num_classes=7)
            y = m(x)
            g = GradCAM(m)
            _, cam = g(x)
            g.remove()
            n = m.num_params()["total_m"]
            print(f"{bb:<26}{str(tuple(y.shape)):<14}{str(tuple(cam.shape)):<20}{n} M")
        except Exception as e:  # noqa: PERF203
            bad += 1
            print(f"{bb:<26}FAIL {type(e).__name__}: {str(e)[:60]}")
    print("-" * 72)
    print("结论：", "全部通过" if bad == 0 else f"{bad} 个主干失败")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
