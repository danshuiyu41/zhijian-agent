"""智检 Agent —— 面向中小制造企业的工业表面缺陷智能检测与处置智能体系统。

模块导航
--------
- data_generate : 程序化合成带钢表面缺陷数据集（冷启动兜底）
- data_github   : 下载整理 GitHub 归档型真实数据集（GC10-DET / DeepPCB）
- data_real     : 下载整理免注册直链真实数据集（KolektorSDD2 / VisA）
- data          : 数据索引、分层划分、归一化统计、PyTorch Dataset
- model         : 轻量 CNN 缺陷分类器 + Grad-CAM 可解释性
- train         : 训练与评估（混淆矩阵、各类 P/R/F1、训练曲线）
- infer         : 滑窗检测 + NMS + 严重度分级 + 可视化
- knowledge     : 质检标准知识库
- rag           : 极简中文检索（BM25 + 字符 n-gram 混合）
- agent         : 质检处置智能体（规则 / 本地大模型 / 云 API 三级降级）
- report        : 质检报告生成
- db            : SQLite 持久化（检测记录、工单）
- api           : FastAPI 后端（REST + 静态前端）
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
