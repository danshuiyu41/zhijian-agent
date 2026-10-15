"""公共工具：配置加载、随机种子、日志、路径管理。

作者：智检 Agent 项目组
说明：所有模块共用此处的配置与路径解析，避免硬编码相对路径问题。
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


def resolve(path_like: str | os.PathLike) -> Path:
    """把配置里的相对路径解析为相对项目根目录的绝对路径。"""
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(path: str | os.PathLike = "config/config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置（带缓存）。"""
    key = str(resolve(path))
    if key in _CONFIG_CACHE:
        return _CONFIG_CACHE[key]
    with open(resolve(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _CONFIG_CACHE[key] = cfg
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """按配置创建所有需要的目录。"""
    for _, v in cfg.get("paths", {}).items():
        target = Path(v)
        # db 这类是文件路径，取父目录
        target = target if target.suffix == "" else target.parent
        resolve(target).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 随机性
# ------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch 可选
        pass


# ------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------
_LOGGER_READY = False


def get_logger(name: str = "zhijian") -> logging.Logger:
    global _LOGGER_READY
    logger = logging.getLogger(name)
    if not _LOGGER_READY:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("zhijian")
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        root.propagate = False
        _LOGGER_READY = True
    return logger


# ------------------------------------------------------------------
# 类别辅助
# ------------------------------------------------------------------
def class_list(cfg: Dict[str, Any]) -> list[dict]:
    return list(cfg["dataset"]["classes"])


def class_codes(cfg: Dict[str, Any]) -> list[str]:
    return [c["code"] for c in class_list(cfg)]


def class_zh_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    return {c["code"]: c["zh"] for c in class_list(cfg)}


def class_risk_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    return {c["code"]: c["risk"] for c in class_list(cfg)}


def severity_grade(cfg: Dict[str, Any], score: float) -> Dict[str, Any]:
    """把 0-100 的严重度分数映射到 A-E 等级。"""
    for g in cfg["severity"]["grades"]:
        if score <= g["max"]:
            return dict(g)
    return dict(cfg["severity"]["grades"][-1])


def fmt_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    from datetime import datetime

    return datetime.now().strftime(fmt)
