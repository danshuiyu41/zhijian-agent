# AI 工业质检智能体系统 — 开源项目逐条验证调研

**验证方法**：`web_fetch` 优先打 `https://api.github.com/repos/<owner>/<repo>`（返回 JSON：stargazers_count / license / language / pushed_at），必要时打 `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/README.md` 或官方文档页；搜索类结论用 `https://api.github.com/search/repositories?...` 实测。
**数据抓取时间**：2026-10-12（UTC）。star 是动态值，会持续变化。
**验证标记**：`✅已验证(API)` = 本次成功拉取仓库元数据；`✅已验证(README/文档)` = 本次成功拉取正文；`⚠️部分验证` = 只验证到仓库存在，具体用法未复核；`❌未验证` = 本次未成功验证（不要直接采信）。
**关于重定向**：GitHub API 会自动跟随仓库改名，所以"URL 打不开但 API 有数据"的情况我会写清现名。

---

## 1. 轻量 RAG 框架

### 1.1 主流框架/组件总表

| 名称 | URL | star | 许可证 | 语言 | 最近推送 | 嵌入 FastAPI 的难度 | 验证状态 |
|---|---|---|---|---|---|---|---|
| LightRAG (HKUDS) | https://github.com/HKUDS/LightRAG | 40,085 | MIT | Python | 2026-10-12 | 中：可当库调用，但要图存储/额外 LLM 调用，比朴素 RAG 重 | ✅已验证(API) |
| LangChain | https://github.com/langchain-ai/langchain | 147,685 | MIT | Python | 2026-10-12 | 中：依赖树大、版本变动快，能嵌但"重" | ✅已验证(API) |
| LlamaIndex | https://github.com/run-llama/llama_index | 52,469 | MIT | Python | 2026-10-08 | 中：抽象层多，小项目偏重 | ✅已验证(API) |
| Haystack (deepset) | https://github.com/deepset-ai/haystack | 26,726 | Apache-2.0 | Python | 2026-10-12 | 中：Pipeline 显式、生产友好，代码量比 LangChain 可控 | ✅已验证(API) |
| txtai | https://github.com/neuml/txtai | 13,011 | Apache-2.0 | Python | 2026-10-11 | 低-中：一体化 embedding+索引+LLM，单进程嵌入式 | ✅已验证(API) |
| Chroma | https://github.com/chroma-core/chroma | 29,485 | Apache-2.0 | Rust（Python 客户端 `chromadb`） | 2026-10-12 | 低：pip 装、嵌入式持久化，最省事 | ⚠️部分验证（仓库已验，`PersistentClient` 用法页本次 404） |
| Qdrant（服务端） | https://github.com/qdrant/qdrant | 35,048 | Apache-2.0 | Rust | 2026-10-12 | 低（用 Python 客户端的本地模式，无需服务端） | ✅已验证(API+README) |
| Milvus（服务端） | https://github.com/milvus-io/milvus | 46,365 | Apache-2.0 | Go | 2026-10-12 | 高（要 Docker/集群，比赛项目不划算） | ✅已验证(API) |
| **Milvus Lite** | https://github.com/milvus-io/milvus-lite | 464 | Apache-2.0 | Python | 2026-10-09 | **低**：`pip` 装、本地 `.db` 文件、无 Docker | ✅已验证(API+README) |
| FAISS | https://github.com/facebookresearch/faiss | 41,134 | MIT | C++ | 2026-10-12 | 低：纯索引库，自己写 chunk/存储逻辑 | ✅已验证(API) |
| sqlite-vec | https://github.com/asg017/sqlite-vec | 8,181 | Apache-2.0 | C | 2026-05-18 | 低：SQLite 扩展，真·单文件 RAG | ✅已验证(API) |
| sentence-transformers | https://github.com/huggingface/sentence-transformers | 19,166 | Apache-2.0 | Python | 2026-10-12 | 低：就是 embedding 库本身 | ✅已验证(API)（原 `UKPLab/sentence-transformers` 已重定向到 huggingface 组织） |
| RAGFlow | https://github.com/infiniflow/ragflow | 92,061 | Apache-2.0 | Go | 2026-10-12 | 高：整套平台（Docker Compose），不适合"嵌入" | ✅已验证(API) |
| Dify | https://github.com/langgenius/dify | 158,238 | 自定义(NOASSERTION) | TypeScript | 2026-10-12 | 高：平台，只能当外部服务调 API | ✅已验证(API) |
| FastGPT | https://github.com/labring/FastGPT | 29,800 | 自定义(NOASSERTION) | TypeScript | 2026-10-12 | 高：平台，同样只能当外部服务 | ✅已验证(API) |
| Langchain-Chatchat | https://github.com/chatchat-space/Langchain-Chatchat | 38,681 | Apache-2.0 | Python | **2025-11-10（约一年无推送）** | 高：完整应用（Streamlit 前端），不适合嵌入 | ✅已验证(API) |

### 1.2 三个重点结论（都是核实过的）

**① Milvus Lite：可行，pip 装、无需 docker，Windows 有 CI 覆盖**
来自官方 README（本次已拉全文）：
- 安装：`pip install -U milvus-lite`，或（更常见）`pip install -U "pymilvus[milvus-lite]"`
- 用法：`from pymilvus import MilvusClient; client = MilvusClient("./demo.db")` —— 传本地 `.db` 路径即自动以嵌入式模式启动
- 这一代是**纯 Python 从零重写**：LSM 存储 + WAL + Parquet 段 + 段级 FAISS 索引 + 标量过滤 + BM25 全文检索 + 可选 gRPC 适配器
- 环境要求：Python ≥ 3.10；README 明确写 "macOS, Linux, or Windows where the Python dependencies are available"，CI 覆盖 **Linux 3.10–3.13 / macOS 3.12 / Windows 3.10**；Windows 其他 Python 版本取决于 `faiss-cpu`、`pyarrow` 是否有兼容 wheel
- 默认依赖已包含 `pyarrow`、`numpy`、`faiss-cpu`、`grpcio`
- **坑**：不支持旧版 C++ milvus-lite 生成的 `.db`（需重新导入）；单进程 per data_dir（文件锁）；写入需串行；无鉴权/无 TLS，别暴露到不可信网络
- 也能当独立本地服务：`milvus-lite server --data-dir ./data --port 19530`

**② Qdrant：服务端才需要 docker，Python 客户端有官方"本地模式"，同样不需要 docker**
来自 `qdrant-client` 官方 README（本次已拉全文）：
- `client = QdrantClient(":memory:")` 或 `client = QdrantClient(path="path/to/db")`（持久化到磁盘），README 原话是 "run same code in local mode **without running Qdrant server**"，定位是 development / prototyping / testing
- 需要服务端时才用 docker：`docker run -p 6333:6333 qdrant/qdrant:latest`
- 附带能力：客户端可选 `qdrant-client[fastembed]`（ONNX Runtime 本地 embedding，支持 `fastembed-gpu` 走 CUDA），正好和你们 onnxruntime 技术栈同源
- 限制：本地模式是单进程持锁，不适合当并发服务用（比赛演示单人用足够）

**③ Chroma：可用但本次没完全复核**
仓库元数据已核实（29,485 star / Apache-2.0 / 注意核心语言已变成 **Rust**，Python 侧是 `chromadb` 客户端）。`docs.trychroma.com/docs/run-chroma/persistent-client` 本次返回 404，所以 `chromadb.PersistentClient(path=...)` 这类具体 API **本次未逐条复核**。功能上它是"嵌入式向量库"最省心的选择之一，但对外部 API 稳定性有要求的场合，Qdrant 本地模式 / Milvus Lite 的文档证据更硬。

### 1.3 "不用框架"的最小 RAG 参考实现（star 较高的真实仓库）

| 名称 | URL | star | 说明 | 验证状态 |
|---|---|---|---|---|
| RAG_Techniques (NirDiamant) | https://github.com/NirDiamant/RAG_Techniques | 29,819 | 各类 RAG 技巧 Notebook 合集（切分/重排/上下文增强），适合抄思路 | ✅已验证(API) |
| rag-from-scratch (LangChain 官方) | https://github.com/langchain-ai/rag-from-scratch | 9,494 | 从零实现朴素 RAG 的 Notebook 系列（刻意"不用框架"讲原理），最贴近你要的"最小 RAG" | ✅已验证(API)（无 LICENSE 文件，商用注意） |
| openai-cookbook | https://github.com/openai/openai-cookbook | 76,575 | 官方 RAG / function calling 示例集，MIT | ✅已验证(API) |
| sqlite-vec | https://github.com/asg017/sqlite-vec | 8,181 | 纯 numpy + sqlite 路线的官方级方案：向量存 SQLite，KNN 用 SQL 查 | ✅已验证(API) |
| sentence-transformers | https://github.com/huggingface/sentence-transformers | 19,166 | 中文可选 `BAAI/bge-small-zh-v1.5`（384 维，CPU 够快） | ✅已验证(API) |

**最小可用组合（不用任何 RAG 框架）**：`sentence-transformers`(bge-small-zh) + `sqlite-vec`（或 `faiss.IndexFlatIP`）+ 自己在 app 启动时把标准文档切块入库，检索时 top-k + 拼 prompt。代码量约 100–150 行，最容易控场。

---

## 2. 中文 Agent 工具调用（function calling）

| 名称 | URL | star | 许可证 | 语言 | 最近推送 | 嵌入 FastAPI 难度 | 验证状态 |
|---|---|---|---|---|---|---|---|
| Qwen-Agent（Qwen 官方） | https://github.com/QwenLM/Qwen-Agent | 17,168 | Apache-2.0 | Python | **2026-03-04（约 7 个月未推送）** | 低：`get_chat_model({...})` 包一层就能用；自带 RAG / Code Interpreter / MCP | ✅已验证(API) |
| Qwen 官方 Function Calling 文档 | https://qwen.readthedocs.io/en/latest/framework/function_call.html | — | — | — | 本次成功拉全文 | 低：直接给 OpenAI 兼容写法 | ✅已验证(文档) |
| smolagents (HF) | https://github.com/huggingface/smolagents | 29,797 | Apache-2.0 | Python | 2026-10-06 | 低：极简（"think in code"），适合 2–3 个工具的演示 | ✅已验证(API) |
| pydantic-ai | https://github.com/pydantic/pydantic-ai | 20,576 | MIT | Python | 2026-10-12 | 低：类型化工具/依赖注入，和 FastAPI 的 pydantic 同源 | ✅已验证(API) |
| LangGraph | https://github.com/langchain-ai/langgraph | 43,135 | MIT | Python | 2026-10-12 | 中：要理解 StateGraph 心智模型，做多步质检流程很合适 | ✅已验证(API) |
| hello-agents（Datawhale 中文教程） | https://github.com/datawhalechina/hello-agents | 82,726 | 自定义(NOASSERTION) | Python | 2026-10-12 | 教程类：读+抄，不是依赖 | ✅已验证(API) |
| openai-cookbook（function calling 示例） | https://github.com/openai/openai-cookbook | 76,575 | MIT | Jupyter | 2026-10-09 | 示例代码 | ✅已验证(API) |

### 2.1 Qwen 官方 function calling 要点（已从文档正文核实）

- 文档明确：Qwen3 推荐 **Hermes-style tool use**；工具用 JSON Schema 描述（`type/function/name/description/parameters`）
- Qwen-Agent 是 Qwen3 function calling 的**canonical implementation**，可把**不支持 function calling 的 OpenAI 兼容 API** 包装出能力：
  ```python
  from qwen_agent.llm import get_chat_model
  llm = get_chat_model({"model": "Qwen/Qwen3-8B", "model_server": "http://localhost:8000/v1",
                        "api_key": "EMPTY"})
  for responses in llm.chat(messages=messages, functions=functions): ...
  ```
  返回的助手消息里用 `function_call` 字段（`name` + JSON 字符串 `arguments`），工具结果以 `{"role": "function", "name": ..., "content": ...}` 追加。
- **重要提醒（原文）**：推理模型不建议用基于 stopword 的 ReAct 工具模板（思考段里可能出现 stopword 导致工具调用异常）。
- vLLM 路线：`vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1`，然后用 `openai` SDK 的 `tools=` + `tool_call_id` 走标准 OpenAI 协议。

### 2.2 "OpenAI-compatible function calling 的最小 FastAPI 示例"有没有？

**结论：没有需要专门引用的仓库——标准做法就是 `openai` SDK 指向任何 OpenAI 兼容端点（vLLM / Ollama / DashScope 兼容模式 / 本地服务），在 FastAPI 路由里跑"模型要工具 → 本地执行 → 结果回灌 → 再问一次"的两轮循环。** Qwen 官方文档第 2.1 节给的代码就是这种最小实现（`client.chat.completions.create(..., tools=TOOLS)` → 处理 `tool_calls` → `role: "tool"` 回灌 → 再调一次）。**这是文档中已验证的写法；"包成 FastAPI 路由"那层属于常规工程包装，本次没有找到值得单独推荐的现成仓库（避免给你编链接）。**

### 2.3 中文 Agent 教程哪个 star 高？

`datawhalechina/hello-agents`（《从零开始构建智能体》）= **82,726 star**，是本次验证到的中文 Agent 教程里 star 最高的（Python，最近推送 2026-10-12，license 为 NOASSERTION）。⚠️ star 极高不等于代码质量全部可靠，建议只当"概念与术语对齐"的读物，生产代码还是走 Qwen-Agent / pydantic-ai 这类有测试的库。

---

## 3. 工业质检 / 缺陷检测开源项目

### 3.1 GitHub 搜索关键词实测命中量（本次真实搜索）

| 关键词 | 命中仓库总数 | 验证状态 |
|---|---|---|
| `surface defect detection` | 1,306 | ✅已验证(search API) |
| `industrial anomaly detection` | 1,914 | ✅已验证(search API) |
| `PCB defect detection` | 1,027 | ✅已验证(search API) |
| `yolov8 defect detection` | 986 | ✅已验证(search API) |

### 3.2 真实仓库清单（按 star 排序，全部本次实测）

| 名称 | URL | star | 用途 | 许可证 | 最近推送 | 验证状态 |
|---|---|---|---|---|---|---|
| Ultralytics YOLO (v8/11/26/27) | https://github.com/ultralytics/ultralytics | 62,419 | 检测/分割/分类训练与推理基座；`yolo11n.pt` 仅 5.6MB，4060 8G 绰绰有余 | AGPL-3.0（**商用需注意**） | 2026-10-12 | ✅已验证(API) |
| PaddleDetection | https://github.com/PaddlePaddle/PaddleDetection | 14,439 | 国产检测工具箱（PP-YOLOE / RT-DETR 等），中文文档全 | Apache-2.0 | 2026-05-28 | ✅已验证(API) |
| Anomalib | https://github.com/open-edge-platform/anomalib | 6,238 | **无监督异常检测/定位**（PaDiM、PatchCore、EfficientAD 等），正是"良品样本多、缺陷样本少"的工业场景解法 | Apache-2.0 | 2026-10-10 | ✅已验证(API)（`openvinotoolkit/anomalib` 已重定向到此新组织） |
| Charmve/Surface-Defect-Detection | https://github.com/Charmve/Surface-Defect-Detection | 4,131 | 中文"最大的工业缺陷检测数据库及论文集"（含 DeepPCB 等数据集索引） | MIT | 2024-05-27 | ✅已验证(API) |
| awesome-industrial-anomaly-detection | https://github.com/M-3LAB/awesome-industrial-anomaly-detection | 3,821 | 工业异常/瑕疵检测论文+数据集检索库（含 MVTec AD 等），中文说明 | 无 LICENSE 文件 | 2026-09-20 | ✅已验证(API) |
| Eatzhy/surface-defect-detection | https://github.com/Eatzhy/surface-defect-detection | 786 | 缺陷检测文献记录清单 | 无 | 2021-09-14 | ✅已验证(API) |
| ShuaiLYU/...Surface-Defect-Detection | https://github.com/ShuaiLYU/Deep-Learning-Approach-for-Surface-Defect-Detection | 752 | 论文 "Segmentation-Based Deep-Learning Approach for Surface-Defect Detection" 的 TF 实现（经典基线） | 无 | 2020-05-14 | ✅已验证(API) |
| tangsanli5201/DeepPCB | https://github.com/tangsanli5201/DeepPCB | 554 | **PCB 缺陷数据集**（6 类：开路/短路/鼠咬/毛刺/多余铜/缺孔） | MIT | 2018-12-19（老） | ✅已验证(API) |
| Tiny-Defect-Detection-for-PCB | https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB | 525 | 小目标 PCB 缺陷检测（含数据与代码） | 无 | 2026-03-24 | ✅已验证(API) |
| abin24/Surface-Inspection-defect-detection-dataset | https://github.com/abin24/Surface-Inspection-defect-detection-dataset | 399 | 多种表面缺陷数据集集合（分割/检测/显著性） | 无 | 2021-05-26 | ✅已验证(API) |
| GLASS (ECCV 2024) | https://github.com/cqylunlun/GLASS | 396 | 梯度上升异常合成，工业异常检测+定位 SOTA 之一 | MIT | 2026-03-30 | ✅已验证(API) |
| MMAD (ICLR'25) | https://github.com/jam-cc/MMAD | 278 | **多模态大模型做工业异常检测的 benchmark**（可用来论证"检测+大模型"路线合理） | 无 | 2026-01-14 | ✅已验证(API) |
| open-iad (M-3LAB) | https://github.com/M-3LAB/open-iad | 171 | 工业图像异常检测统一 benchmark 框架 | 无 | 2025-02-20 | ✅已验证(API) |
| ind_knn_ad | https://github.com/rvorias/ind_knn_ad | 166 | 纯 torch+timm 的 kNN 工业异常检测（**极轻，适合当嵌入式基线**） | MIT | 2026-09-05 | ✅已验证(API) |
| Marfbin/NEU-DET-with-yolov8 | https://github.com/Marfbin/NEU-DET-with-yolov8 | 152 | NEU-DET 钢材表面缺陷 + YOLOv8 改进（含坐标注意力/Swin） | AGPL-3.0 | 2023-12-07 | ✅已验证(API) |
| pcb-defect-detection-improved-yolov5-opencv | https://github.com/qunshansj/pcb-defect-detection-improved-yolov5-opencv | 91 | 中文：改进 YOLOv5+OpenCV 的 PCB 缺陷检测系统（源码+部署教程） | 无 | 2023-11-23 | ✅已验证(API) |
| yolov10-pcb-detect | https://github.com/Anionex/yolov10-pcb-detect | 78 | 中文：YOLOv10 电路板缺陷检测 | 无 | 2026-07-06 | ✅已验证(API) |
| JiaLim98/YOLO-PCB | https://github.com/JiaLim98/YOLO-PCB | 57 | PCB 缺陷检测 + 异常趋势报警系统（有配套论文） | MIT | 2023-03-14 | ✅已验证(API) |

### 3.3 数据集说明

- **MVTec AD** 是异常检测事实标准数据集。我本次尝试的 `abin24/Magnetic-Tile-Defect`、`abin24/Magnetic-tile-defect-datasets` 用 API 都是 **404 ❌未验证**（重名/改名情况多，不要直接贴链接）。请从上面两个 awesome 列表（Charmve 4,131★ / M-3LAB 3,821★）里的数据集索引页进入，或用官方站点 mvtec.com 的 AD 数据集。
- **PCB**：DeepPCB（554★，MIT）可直接用；铝材表面缺陷可用天池"铝型材表面缺陷"数据集（`YeahHuang/Al_surface_defect_detection` 253★ 是该比赛 top 0.3% 方案，✅已验证(API)）。

---

## 4. 报告生成 / Web 看板 / 可视化脚手架

### 4.1 报告生成（中文 PDF / Word）

| 名称 | URL | star | 许可证 | 最近推送 | 中文支持与嵌入难度 | 验证状态 |
|---|---|---|---|---|---|---|
| WeasyPrint | https://github.com/Kozea/WeasyPrint | 9,673 | BSD-3-Clause | 2026-10-12 | **推荐**：写 HTML/CSS → PDF，用系统已装 CJK 字体（Windows 自带微软雅黑/宋体）即可中文；pip 装；难点是 Windows 需 GTK/Pango 运行库 | ✅已验证(API)（"需 Pango 运行库"为已知常识，本次未逐条复核） |
| docxtpl (python-docx-template) | https://github.com/elapouya/python-docx-template | 2,720 | LGPL-2.1 | 2026-07-07 | **最省事**：模板就是 Word 文件，`{{ 变量 }}` 替换，中文零配置；生成 .docx，可再转 PDF | ✅已验证(API) |
| reportlab（官方 GitHub 镜像） | https://github.com/MrBitBucket/reportlab-mirror | 86 | BSD-3-Clause | 2026-08-19 | 纯 Python 生成 PDF，无系统依赖；中文需注册 CID 字体（如 `UnicodeCIDFont('STSong-Light')` 或 `TTFont` 加载 ttf） | ✅已验证(API)。⚠️注意：star 只有 86（官方主仓在 hg.reportlab.com，GitHub 只是镜像），别被 star 数误导 |
| wkhtmltopdf | https://github.com/wkhtmltopdf/wkhtmltopdf | 14,547 | LGPL-3.0 | **2022-11-22，仓库已归档(archived=true)** | **不推荐**：已停止维护 4 年 | ✅已验证(API) |

### 4.2 轻量 Web 看板

| 名称 | URL | star | 许可证 | 最近推送 | 与 FastAPI 集成 | 验证状态 |
|---|---|---|---|---|---|---|
| Gradio | https://github.com/gradio-app/gradio | 43,734 | Apache-2.0 | 2026-10-12 | **最低**：官方支持 `gr.mount_gradio_app(app, demo, path="/gradio")` 挂到现有 FastAPI 上（官方文档正文已含完整示例） | ✅已验证(API+文档) |
| Streamlit | https://github.com/streamlit/streamlit | 45,955 | Apache-2.0 | 2026-10-12 | **不能挂载**：独立进程/独立端口（`streamlit run`），要嵌只能 iframe | ✅已验证(API)（"不能挂载"为框架设计常识，本次未逐条复核） |
| NiceGUI | https://github.com/zauberzeug/nicegui | 16,281 | MIT | 2026-10-12 | 低-中：Python 写 UI，基于 FastAPI，可与之共存 | ⚠️部分验证（仓库已验；`ui.run_with(app)` 具体 API 本次未复核） |
| Reflex | https://github.com/reflex-dev/reflex | 28,952 | Apache-2.0 | 2026-10-12 | 中：纯 Python 全栈，自带前后端，不"挂"在 FastAPI 上 | ✅已验证(API) |

### 4.3 后台管理模板（FastAPI + Vue/React）

| 名称 | URL | star | 许可证 | 技术栈 | 最近推送 | 验证状态 |
|---|---|---|---|---|---|---|
| full-stack-fastapi-template（FastAPI 官方） | https://github.com/fastapi/full-stack-fastapi-template | 45,955 | MIT | FastAPI + **React** + SQLModel + Postgres + Vite + Tailwind + shadcn/ui + Docker Compose | 2026-10-08 | ✅已验证(API)。官方维护、最稳，但**前端是 React 不是 Vue**，且依赖 Postgres/Docker |
| sqladmin | https://github.com/smithyhq/sqladmin | 2,844 | BSD-3-Clause | FastAPI/Starlette + SQLAlchemy 的 Django-admin 式后台 | 2026-10-01 | ✅已验证(API)。⚠️原 `aminalaee/sqladmin` 已重定向到 `smithyhq/sqladmin`（现地址） |
| FastAPI-Amis-Admin | https://github.com/amisadmin/fastapi-amis-admin | 1,566 | Apache-2.0 | FastAPI + amis（JSON 驱动低代码后台） | 2025-12-15 | ✅已验证(API) |
| FastapiAdmin（中文） | https://github.com/fastapiadmin/FastapiAdmin | 1,199 | MIT | FastAPI + SQLAlchemy + **Vue3 + TS + Vite + Pinia + Element-Plus** | 2026-10-08 | ✅已验证(API)。中文文档、开箱即用中后台，最贴近"FastAPI+Vue 开箱即用"的需求 |
| fastapi_vue3_admin（同名 fork） | https://github.com/sanyuan0v0/fastapi_vue3_admin | 4 | MIT | 上面项目的 fork | 2025-08-19 | ✅已验证(API)。**star 仅 4，别用这个 fork，用上游 FastapiAdmin** |

**结论**：GitHub 上"FastAPI + Vue3 开箱即用后台"**没有 star 数很高（>1万）且维护活跃的公认项目**；官方模板是 React，Vue 阵营里 `FastapiAdmin`（1,199★，MIT，2026-10 活跃）是目前较合理的选择。比赛演示如果时间紧，**"单页 Gradio 挂载 + 一张自绘 ECharts 看板页" 成本远低于引入全套 admin 模板**。

---

## 5. OCR / 工单 / 视觉补充（可选）

| 名称 | URL | star | 许可证 | 语言 | 最近推送 | 是否依赖 paddle | 验证状态 |
|---|---|---|---|---|---|---|---|
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | 90,968 | Apache-2.0 | Python | 2026-09-16 | **是**：需装 `paddlepaddle`（https://github.com/PaddlePaddle/Paddle，24,121★，Apache-2.0）。Windows 装 paddle 常踩 wheel/CUDA 坑 | ✅已验证(API)（两仓均验） |
| RapidOCR | https://github.com/RapidAI/RapidOCR | 8,122 | Apache-2.0 | Python | 2026-10-12 | **否**：基于 **ONNX Runtime** / OpenVINO / MNN / TensorRT / PyTorch，模型即 onnx，与你们 onnxruntime 1.23 技术栈天然一致 | ✅已验证(API) |

**建议**：你们的栈里已经有 `onnxruntime 1.23`，**直接用 RapidOCR，不要为了 OCR 再引入 paddle 全家桶**（体积大、和 torch 一起装容易冲突）。PaddleOCR 的 star 高主要是历史积累与模型丰富度。

### 5.1 手机拍照 → 后端上传的最小方案（工程建议，非仓库验证）

- 后端：FastAPI 的 `UploadFile` / `File`，官方文档已核实（https://fastapi.tiangolo.com/tutorial/request-files/），要点：需装 `python-multipart`；`async def upload(file: UploadFile)` 即拿到文件流，落盘后交给检测模块。
- 前端：手机浏览器用 `<input type="file" accept="image/*" capture="environment">` 调后置摄像头，比 WebRTC 取流简单得多；Gradio 的 `gr.Image(type="filepath", sources=["upload","webcam"])` 也能直接给到本地路径。
- 备选：无文件域的场景（如微信内嵌/剪贴板）走 base64 JSON 提交。
- ⚠️ 以上是常规工程做法，本次**未找到值得单列的高 star 参考仓库**，不编造链接。

---

## 6. 最小可行组合建议（针对你们的比赛项目）

**技术栈前提**：Windows + RTX 4060 Laptop 8GB / torch 2.12.1 / opencv-python 4.11 / onnxruntime 1.23 / fastapi / gradio。

| 环节 | 推荐选型 | 理由 |
|---|---|---|
| 缺陷检测（有缺陷样本） | **Ultralytics YOLO11n/YOLOv8n**（62,419★，`yolo11n.pt` 5.6MB）+ 自标缺陷数据；导出 ONNX 用 onnxruntime 推理 | 8G 显存训练 `imgsz=640, batch=8` 完全够；676 万参数级模型在 4060 上 30+FPS；导出 onnx 后与你们已定的 onnxruntime 完全对齐。**注意 AGPL-3.0，比赛/演示无碍，商用要换许可或自研** |
| 缺陷检测（只有良品样本） | **Anomalib**（6,238★）的 PatchCore / EfficientAD，或用 **ind_knn_ad**（166★，纯 torch+timm，最轻） | 中小制造企业真实痛点就是"没有缺陷样本"，无监督异常检测是加分项，PatchCore 只需良品图 |
| 数据集 | **DeepPCB**（554★，MIT，6 类 PCB 缺陷）+ MVTec AD（经 M-3LAB 列表 3,821★ 索引进入） | 现成标注，避免从零标注；两者都是工业质检"标准答案"，评审认可度高 |
| 缺陷分级 | 规则引擎（缺陷类别 + 面积/置信度阈值 → 致命/严重/轻微） | 不要用模型做分级，规则可解释、可现场调参，比赛答辩更好讲 |
| RAG 知识库 | **Milvus Lite**（`pip install -U "pymilvus[milvus-lite]"`，本地 `.db`，无 Docker，纯 Python，Apache-2.0，CI 覆盖 Windows）**或** 更极简的 `sentence-transformers(bge-small-zh) + sqlite-vec`（8,181★） | 需求明确要"嵌入 FastAPI 的最小方案"：Milvus Lite 的 `MilvusClient("./qc.db")` 三行起库，还自带 BM25 混合检索和标量过滤（可按"标准类型/工序"过滤），日后要换 Milvus 服务端/云也是同一套 pymilvus API；若连 pip 依赖都想省，就 sqlite-vec 一把梭。**不要用 RAGFlow/Dify/FastGPT/Langchain-Chatchat**——它们是要 Docker 起平台的，和"嵌入"背道而驰 |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5`（384 维，CPU 毫秒级）或 Qwen/Qwen3-Embedding 小尺寸 | 中文标准文档检索够用，8G 显存的卡不用浪费在这 |
| Agent / 工具调用 | **Qwen-Agent**（17,168★，Apache-2.0）做 function calling 壳 + **openai SDK** 指向 DashScope/vLLM 兼容端点；若想更轻则 **pydantic-ai**（20,576★） | Qwen 官方文档明确 Qwen-Agent 是其 function calling 的 canonical implementation，中文工具描述友好；pydantic-ai 与 FastAPI 的 pydantic 体系同源、类型安全。工具就 3–4 个：`query_defect_stats` / `search_standard_kb` / `create_work_order` / `gen_report` |
| Agent 编排 | 先用 **单轮 function calling 循环**；要"多步根因分析"再上 **LangGraph**（43,135★） | 避免一开始就背 StateGraph 的复杂度；演示流程图用 LangGraph 画也更好讲 |
| 报告生成 | **docxtpl**（2,720★）生成 Word 质检报告（模板=Word，中文零配置）＋ **WeasyPrint**（9,673★）出 PDF 版 | 中文 PDF 最大的坑就是字体，docxtpl 直接绕开；WeasyPrint 用系统 CJK 字体渲染 HTML/CSS，样式好看。**不要用 wkhtmltopdf（已归档）**；reportlab（镜像仅 86★）留给"需要纯 Python 无外部依赖"的兜底 |
| Web 看板 | **Gradio `mount_gradio_app` 挂到 FastAPI**（43,734★，官方文档已验证）做"拍照上传→检测结果→Agent 建议"交互页；另加一个自绘 ECharts 静态页做看板 | 一个 FastAPI 进程、一个端口搞定全链路，最少运维；Streamlit 挂不进 FastAPI（要另起端口），Reflex/NiceGUI 成本更高。若必须"后台管理"，再评估 `FastapiAdmin`（1,199★，Vue3，MIT） |
| OCR（可选） | **RapidOCR**（8,122★，ONNX Runtime） | 与 onnxruntime 1.23 同源，**不用装 paddle**；用于识别工件号/批次码、工单号 |
| 拍照上传 | FastAPI `UploadFile`（官方文档已验证）+ 前端 `<input capture="environment">` | 最小实现，无额外依赖（只多一个 `python-multipart`） |

**一句话版本**：`YOLO11n(ONNX) + Anomalib(可选) 做检测分级 → 规则引擎定级 → bge-small-zh + Milvus Lite 做标准库 RAG → Qwen-Agent + openai SDK 做工具调用 Agent → docxtpl/WeasyPrint 出报告 → Gradio 挂载在 FastAPI 上做看板`。全链路**单进程、无 Docker、无重型平台**，一半依赖跑在 CPU、检测走 4060，符合你们既定技术栈。

**三个必须提前知道的坑**：
1. `ultralytics` 是 **AGPL-3.0**：比赛演示无碍，若声称"可商用"需说明或换模型。
2. **Milvus Lite 不兼容旧版 `.db`**，且单进程持锁、写入要串行——演示够用，别拿它做并发服务。
3. **wkhtmltopdf 已归档（archived，最后推送 2022-11）**，任何"HTML 转 PDF"教程里推荐它都该换成 WeasyPrint。

---

## 7. 未验证 / 需谨慎清单（诚实披露）

| 项目 | 状态 |
|---|---|
| Chroma 嵌入式 API（`chromadb.PersistentClient`） | ❌本次未验证（文档页 404）；仓库元数据已验 |
| `abin24/Magnetic-Tile-Defect(s)` 磁瓦数据集仓库 | ❌API 404，未验证；请从 awesome 列表进入 |
| MVTec AD 官方仓库链接 | ❌未验证（本次只验证了引用它的列表仓库，未拿到官方 GitHub 地址） |
| NiceGUI `ui.run_with(app)` 挂载 FastAPI | ⚠️仓库已验，具体 API 未复核 |
| Streamlit 无法挂载到 FastAPI | ⚠️框架设计常识，本次未逐条复核文档 |
| WeasyPrint Windows 需 Pango/GTK 运行库 | ⚠️常识级结论，本次未复核官方安装文档 |
| reportlab 中文 CID 字体用法（STSong-Light） | ⚠️通用做法，本次未复核官方文档正文 |
| "专用 FastAPI function calling 最小示例仓库" | ❌本次未找到值得推荐的仓库（未编造）；Qwen 官方文档已给出等价最小实现 |
| "手机拍照上传"参考仓库 | ❌未找到高 star 且值得推荐的仓库（未编造）；FastAPI 官方 `UploadFile` 文档已验证 |
