# 面向中小制造企业的 AI 工业质检智能体系统 — 开源资源清单

> 调研日期：**2026-10-12**（star 数、权重体积、模型版本均为此日实测值，会随时间变化）。开发机基线：Windows + RTX 4060 Laptop 8GB，已装 torch 2.12.1(cu)、opencv-python 4.11、onnxruntime 1.23、fastapi、gradio。
> 验证标记说明：**✅已验证** = 本次调研中 web_fetch / 官方 API 成功打开（HTTP 200）且标题/内容与描述一致；**⚠️部分验证** = 链接可打开但内容未能完全核对；**❌未验证** = 未能打开或未找到，仅作线索保留。
> 配套文件：`qc_agent_tech_research.md`（脚手架逐条验证，star 均为 GitHub API 实测）、`chinese-embedding-reranker-verification.md`（embedding/reranker 原始核验记录）。

---

## 1. 缺陷检测数据集

### 1.1 钢铁 / 带钢表面缺陷（重点）

| 名称 | URL | 关键参数 | 许可证 | 获取难度 |
|---|---|---|---|---|
| **NEU-DET 数据集（东北大学，官方页）** | 官方页 ✅ https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm | 1800 张 200×200 灰度；6 类各 300 张：rolled-in scale(RS)、patches(Pa)、crazing(Cr)、pitted surface(PS)、inclusion(In)、scratches(Sc)。NEU-CLS=仅图像（分类）；NEU-DET=含 PASCAL VOC XML 标注框（检测）；NEU-CLS-64=64×64 版 | 官方页未写明许可证，仅要求引用（Song & Yan, Applied Surface Science 285:858-864, 2013）。社区通行「仅供研究」 | **低**——官方页有百度网盘直链，国内可直连 |
| NEU 表面缺陷库 Kaggle 镜像 | ⚠️ https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database | 同 NEU 库 | 见 Kaggle 页 | 中（需 Kaggle 账号） |
| NEU-DET 百度网盘 | ✅（官方页内链接）https://pan.baidu.com/s/1nBbO-jWDm1_NHDQsc1dRkg 提取码 `pmqx` | 同上（含标注，检测用） | 同上 | 低 |
| NEU-CLS 百度网盘 | ✅（官方页内链接）https://pan.baidu.com/s/1l_RjTP7aTwr57ahcwelTpA | 1800 张仅图像 | 同上 | 低 |
| NEU-CLS-64 百度网盘 | ✅（官方页内链接）https://pan.baidu.com/s/1dcQ6pLVgh4L8Ti748fKrUw | 64×64 小图版，适合快速训练 | 同上 | 低 |
| NEU 原始库 百度网盘 | ✅（官方页内链接）https://pan.baidu.com/s/1B_SLhZNRPeKznDc78GY2QA | 原始全量 | 同上 | 低 |
| NEU-Seg（语义分割版） | ✅（官方页内链接）https://github.com/donghongwen/NEU_Seg | 像素级分割标注 | 见仓库 | 低 |
| NEU-DET 官方 Google Drive | ✅（官方页内链接）https://drive.google.com/open?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k | 同 NEU-DET | 同上 | 高（需梯子） |
| **X-SDD**（热轧带钢，7 类） | ✅ https://ieee-dataport.org/documents/x-sdd | 1360 张缺陷图，**7 类**（比 NEU-CLS 多一类）；DOI `10.21227/k1wr-rc37`；数据集文件 `X-SDD-A-New-benchmark-main.zip`（6.24 MB）；作者 Xinglong Feng / Xianwen Gao / Ling Luo，2023 | 见 IEEE DataPort 页面条款（未逐条核对） | **中**——需 IEEE DataPort 免费注册登录后才能下载 |
| **GC10-DET**（10 类金属表面缺陷） | ✅ https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets | **3570 张，10 类**金属表面缺陷；论文 *Deep Metallic Surface Defect Detection: the New Benchmark and Detection Network*, Sensors 2020, 20(6):1562 | 仓库要求引用，未明示许可证 | **低**——百度网盘：https://pan.baidu.com/s/1Zrd-gzfVhG6oKdVSa9zoPQ 提取码 `cdyt` |
| GC10-DET Kaggle 镜像 | ⚠️ https://www.kaggle.com/datasets/alex000kim/gc10det | 同上 | 见 Kaggle 页 | 中（需 Kaggle 账号） |
| GC10-DET Roboflow 即用版 | ⚠️ https://universe.roboflow.com/g-deepti-raj/gc10-det-latest/dataset/3 | COCO/YOLO 格式，已切分 train/valid/test 70/20/10 | 见 Roboflow 页 | 中（需 Roboflow 账号） |
| GC10-DET Dataset Ninja 直链 | ✅ https://raw.githubusercontent.com/dataset-ninja/gc10-det/main/DOWNLOAD.md | Supervisely JSON 格式直链 + `pip install dataset-tools` 下载法 | 见页面 | 低-中 |
| **Severstal Steel Defect Detection** | ⚠️ https://www.kaggle.com/c/severstal-steel-defect-detection | 约 18076 个文件，4 类钢板缺陷，像素级分割 | Competition Terms（仅比赛用途） | **中-高**——需 Kaggle 账号并接受竞赛规则 |
| **KolektorSDD**（电换向器表面） | ✅ https://www.vicos.si/resources/kolektorsdd/ | 399 张：52 张有缺陷 / 347 张无缺陷；尺寸 500×(1240~1270)；训练建议 resize 到 512×1408 | **CC BY-NC-SA 4.0**（非商用） | **低**——免注册直下：https://go.vicos.si/kolektorsdd （精细标注）/ https://go.vicos.si/kolektorsddboxes （框标注） |
| KolektorSDD 训练划分 | ✅（同页）https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip | 3-fold 交叉验证划分；另含 TensorFlow 预处理版 | 同上 | 低 |
| **KolektorSDD2** | ✅ https://www.vicos.si/resources/kolektorsdd2/ | **3335 张**：356 张有缺陷 + 2979 张无缺陷；尺寸约 230×630；官方划分 train 246 正 / 2085 负，test 110 正 / 894 负；多种缺陷类型（划痕、小斑点、表面瑕疵等） | **CC BY-NC-SA 4.0** | **低**——免注册直下：https://go.vicos.si/kolektorsdd2 |
| **RSDDs**（钢轨表面缺陷） | ❌未验证（原站 http://icn.bjtu.edu.cn/Visint/resources/RSDDs.aspx 可能已失效） | Type-I 67 张（快速车道）+ Type-II 128 张（重载轨道） | 未明示 | 中——百度网盘 https://pan.baidu.com/s/1svsnqL0r1kasVDNjppkEwg 提取码 `nanr` |
| **Magnetic Tile Defect (MTD)**（磁瓦） | ✅ 镜像 https://github.com/Charmve/Surface-Defect-Detection/tree/master/Magnetic-Tile-Defect | 6 类常见磁瓦缺陷，含**像素级 ground truth**；论文 DOI `10.1007/s00371-018-1588-5` | 要求引用 | 中——原仓 `abin24/Magnetic-tile-defect-datasets` 已 **404**，需用 Charmve 镜像或 Roboflow https://universe.roboflow.com/defectdatasets/magnatic-tile-defect/1 |
| **DAGM 2007**（弱监督工业光学检测） | ✅ **Zenodo 官方镜像** https://zenodo.org/records/12750201 （原 HCI 站 ❌ HTTP 500 故障） | **10 个类别**，每类独立 ZIP：dev 集 1000 张无缺陷 + 150 张有缺陷（competition 集为 2000+300）；8-bit PNG 灰度；弱标注为椭圆区域（`Label/` 目录 + `Labels.txt`）；单类 197–410 MB，合计 2.9 GB | **Open**（Zenodo 元数据标注 Open） | **低**——Zenodo 免注册直下（如 https://zenodo.org/records/12750201/files/Class1.zip?download=1 ） |
| DAGM 2007 Kaggle 转载 | ⚠️ https://www.kaggle.com/datasets/bassam165/dagm-2007-industrial-defect-detection-dataset ／ ⚠️ https://www.kaggle.com/datasets/amor000/reconstructed-dagm2007-dataset | 同上 | 见 Kaggle 页 | 中 |
| **NRSD-CR**（钢棒/圆钢裂纹） | ⚠️ 源码 https://github.com/zdfcvsn/MCnet | 见仓库；论文 DOI `10.1109/TIM.2020.3040890` | 见仓库 | 中——Roboflow https://universe.roboflow.com/defectdatasets/nrsd-cr/1 |
| **BSData**（表面损伤 + 磨损预测） | ⚠️ https://github.com/2Obe/BSData | 1104 张（通道 3）+ 394 张 pitting 标注（labelme JSON，可转 VOC/COCO）；另含 27 组磨损演化序列（每组 69 张） | CC BY-SA（见 awesome-industrial-datasets 表） | 低 |

> **中文镜像/Roboflow 即用版（强烈推荐，省去格式转换）**：`robotbitdev/metal-defect-datasets` ✅ https://raw.githubusercontent.com/robotbitdev/metal-defect-datasets/main/README.md
> 该仓库把 NEU-DET、KolektorSDD1/2、Magnetic-Tile-Defect、NRSD-CR、GC10-DET、Severstal、DAGM2007、RSDDs、BSData 全部整理了「源链接 + Roboflow 即用（COCO 格式，已按 70/20/10 切分）」两套链接。**NEU-DET 的 Roboflow 即用版**：https://universe.roboflow.com/defectdatasets/neu-det-fquva/1

### 1.2 PCB 缺陷

| 名称 | URL | 关键参数 | 许可证 | 获取难度 |
|---|---|---|---|---|
| **DeepPCB** | ✅ https://github.com/tangsanli5201/DeepPCB | 1500 对图像（每对 = 无缺陷模板 + 对齐后的缺陷测试图），640×640；**6 类**：open、short、mousebite、spur、copper(伪铜)、pin-hole；1000 训练 / 500 测试；约 3~12 个缺陷/图；标注格式 `x1,y1,x2,y2,type`；含 mAP/F-score 评测脚本与 `gt.zip` | **仅限研究用途**（README 明示） | **低**——仓库内提供数据集与评测代码 |
| **PKU-Market-PCB / HRIPCB** | ✅ 官方资源申请页 https://robotics.pkusz.edu.cn/resources/register/ | 1386 张高清 PCB 图（693 正向 + 693 随机旋转，用于配准研究）；分辨率 2282×2248 / 3034×1586 / 3056×2464 等；**6 类**：Missing hole、Mouse bite、Open circuit、Short、Spur、Spurious copper；标注为 PASCAL VOC XML；由 10 块无缺陷模板人工合成缺陷，属**微小目标检测** | **仅限学术研究，禁止商用**（官方页明示） | **中**——需下载并提交《PKU-HRI 数据集申请表及承诺书》(PDF: `/static/datasets/PKU-HRI数据集申请表.pdf`) |
| PKU-Market-PCB 论文 | ⚠️ https://robotics.pkusz.edu.cn/static/papers/ACAIT2019-huangweibo1.pdf | *HRIPCB: a challenging dataset for PCB defects detection and classification*（ACAIT 2019，北京大学深圳研究生院 HRI Lab） | — | 低 |
| **VisA**（含 4 个 PCB 子集） | ✅ https://github.com/amazon-science/spot-diff | 12 个子集共 **10821 张**（9621 正常 / 1200 异常）；其中 PCB1~PCB4 为复杂结构 PCB；含像素级 mask 与 `image_anno.csv`；提供 1-class / 2-class 划分脚本 | **CC BY 4.0** | **低-中**——S3 直链免注册：https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar （国内速度一般） |

> ⚠️ **避坑**：网上大量「PKU-Market-PCB / NEU-DET 完整数据集下载目录」实为 CSDN 付费资源或博客园的导流文章（例：`cnblogs.com/LimiiiingCV/articles/22872389`），**不是官方来源**。请走官方申请页或本节列出的官方/网盘链接。

### 1.3 铝型材 / 铸件 / 织物 / 通用异常检测

| 名称 | URL | 关键参数 | 许可证 | 获取难度 |
|---|---|---|---|---|
| **铝型材缺陷检测数据集 APDDD**（天池） | ⚠️ https://tianchi.aliyun.com/dataset/148297 | 天池数据集页（内容由 JS 渲染，需登录查看详情） | 见天池页 | 中（需阿里云/天池账号） |
| 铝型材表面缺陷（原天池赛题 231682） | ⚠️ https://tianchi.aliyun.com/competition/entrance/231682/information | 10000 张铝型材实际生产监控图，每图含 1 个或多个缺陷；竞赛描述称「样本图会明确标注所含缺陷类型」 | 见天池页 | **高**——本次访问该 URL 已跳转到天池通用大赛页，**疑似赛题下线**，建议改用上面的 APDDD 数据集 |
| **AITEX 织物疵点**（OpenI 启智镜像，国内直连） | ✅ https://openi.pcl.ac.cn/lwx/AITEX-Dataset/datasets | 245 张 4096×256；140 张无缺陷（7 种织物各 20 张）+ 105 张缺陷（12 类）；含分割 mask（白=缺陷）；文件：`NODefect_images.zip` 94 MiB、`Defect_images.zip` 74 MiB、`Mask_images.zip` 112 KiB | 未明示 | **中**——需 OpenI（启智）账号 |
| AITEX 百度网盘 | ⚠️ https://pan.baidu.com/s/1cfC4Ll5QlnwN5RTuSZ6b7w 提取码 `b9uy` | 同上 | — | 低（时效性未验证） |
| 织物疵点数据集（天池） | ⚠️ https://pan.baidu.com/s/1LMbujxvr5iB3SwjFGYHspA 提取码 `gat2` | 约 8000 张素色布（初赛）+ 约 12000 张花色布（复赛） | — | 低（时效性未验证） |
| **MVTec AD** | ✅ https://www.mvtec.com/research-teaching/datasets/mvtec-ad | **15 个类别，5000+ 高分辨率图**（每类：无缺陷训练集 + 含缺陷测试集 + 像素级异常标注） | **CC BY-NC-SA 4.0**（明确禁止商用） | **中**——需在页面填写表单提交后获取下载链接 |
| MVTec AD 评测代码 | ✅（同页链接）https://www.mydrive.ch/shares/150450/bb24b914a28ddd2b5e35bd53d23177cd/download/439517473-1665675012/mvtec_ad_evaluation.tar.xz | 官方评估脚本（含 readme） | 同上 | 低 |
| MVTec LOCO AD | ⚠️ https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad | 逻辑异常检测（非仅外观） | CC BY-NC-SA 4.0（推测） | 中 |
| MVTec 3D-AD | ⚠️ https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad | 3D 异常检测 | CC BY-NC-SA 4.0（推测） | 中 |
| **GDXray**（铸件 / 焊缝 X 光） | ✅ https://github.com/computervision-xray-testing/GDXray | **21100+ 张 X 光图**，5 组：Castings 314 MB、Welds 209 MB、Baggage 3.048 GB、Nature 192 MB、Settings 73 MB；每组含多个 series | **仅限研究与教育；禁止再分发与商用**（README 明示） | **低**——Dropbox 直链免注册（Castings: https://www.dropbox.com/scl/fi/5e7m31ri5grvnl7kxt8mx/Castings.zip?rlkey=zetfg5g337d2ip265a16rcviw&st=gjub7hr3 ） |
| 太阳能电池 EL 缺陷（elpv-dataset） | ⚠️ https://github.com/zae-bayern/elpv-dataset | 2624 张 300×300 8-bit 灰度，来自 44 个组件；功能正常/缺陷二分类 + 退化程度 | 见仓库 | 低 |
| 输电线路绝缘子 | ⚠️ https://github.com/InsulatorData/InsulatorDataSet | 600 张正常 + 248 张缺陷，含标注 | 见仓库 | 低 |
| 磁性瓦 / 齿轮等更多 | 见下方汇总仓库 | — | — | — |

### 1.4 数据集汇总仓库（一次找全，强烈建议先看这两个）

| 名称 | URL | 关键参数 | 验证状态 |
|---|---|---|---|
| **Charmve/Surface-Defect-Detection** | https://github.com/Charmve/Surface-Defect-Detection | 号称「目前最大的工业缺陷检测数据库及论文集」。收录 20+ 数据集（NEU-CLS、Severstal、elpv、KolektorSDD、DeepPCB、AITEX、天池织物、天池铝型材、DAGM2007、Magnetic Tile、RSDDs、Kylberg、KTH-TIPS、MVTec ITODD、BSData、GID、AeBAD、BTAD 等），大量条目附**百度网盘直链+提取码** | ✅（README 全文已验证） |
| **robotbitdev/metal-defect-datasets** | https://github.com/robotbitdev/metal-defect-datasets | 金属表面缺陷数据集集合，每条给「引用 DOI + 源链接 + Roboflow 即用 COCO 版（已 70/20/10 切分）」 | ✅（README 全文已验证） |
| **jonathanwvd/awesome-industrial-datasets** | https://github.com/jonathanwvd/awesome-industrial-datasets | **189 个工业数据集**的元数据总表：领域/模态/任务/标注类型/访问方式/规模/年份/许可证。含 3CAD(27039 图)、Real-IAD(150000+ 图, CC BY-NC-SA)、PIAD(11268 图/30 产品)、RobustAD(4442 图)、PCAD、CXR-AD、Eyecandies 等新数据集 | ✅（README 全文已验证） |
| Charmve 数据集打包（百度网盘） | https://pan.baidu.com/s/122WY8F5VKqm3qMirqebRQw 提取码 `i20n` | 仓库整体 ZIP | ⚠️（链接来自已验证 README，时效性未验证） |

---

## 2. 开源模型权重

> ⚠️ **显存基线**：RTX 4060 Laptop 标称 8 GB，Windows 下实际可用约 **7.2–7.5 GB**。下表的"磁盘尺寸"是权重文件字节总数（ModelScope API 实测），**加载到显存后还需加上 KV cache 与激活值**；经验值：4bit 权重 ≈ 磁盘尺寸，FP16 权重 ≈ 磁盘尺寸，KV cache 随上下文线性增长（7B 模型 4K 上下文约 0.5–1 GB）。

### 2.1 目标检测

| 名称 | URL | 关键参数 | 许可证 | 获取难度 | 验证状态 |
|---|---|---|---|---|---|
| **Ultralytics YOLO（v8 / 11 / 26）** | https://github.com/ultralytics/ultralytics | nano 级对比（官方文档实测，COCO 640）：YOLOv8n **3.2 M / 8.7 GFLOPs / 37.3 mAP / CPU-ONNX 80.4 ms**；YOLO11n **2.6 M / 6.5 G / 39.5 / 56.1 ms**；**YOLO26n 2.4 M / 5.5 G / 40.9 / 38.9 ms（全面最优，且 NMS-free）**。**YOLO27 标注 "Coming Soon"，模型尚未发布** | **AGPL-3.0**（已拉 LICENSE 原文确认） | 低——`pip install ultralytics` 后权重自动下载 | ✅README+LICENSE |
| Ultralytics 商用授权页 | https://www.ultralytics.com/license | 官方明确要求闭源 / SaaS / production 场景购买 **Enterprise** 许可 | — | — | ✅实测 200 |
| Ultralytics ONNX 导出文档 | https://docs.ultralytics.com/integrations/onnx/ | `model.export(format="onnx")` | — | 低 | ✅实测 200 |
| **RT-DETR**（AGPL 的许可干净替代） ⭐ | https://huggingface.co/PekingU/rtdetr_v2_r18vd | Apache-2.0，可直接 `transformers` 加载 | **Apache-2.0** | 低 | ✅实测 200 |
| PaddleDetection | https://github.com/PaddlePaddle/PaddleDetection | 14,439★，PP-YOLOE / RT-DETR 等，中文文档全 | Apache-2.0 | 低 | ✅API |

> ⚠️ **AGPL 的关键细节（很多人漏掉）**：Ultralytics 官方许可页**明确把 "custom-trained or fine-tuned YOLO models in a proprietary setting" 也纳入约束范围**——也就是说**你在 NEU-DET 上自己微调出来的权重同样受 AGPL 传染**，不是"我自训的就没事"。
> **缓解做法**：训练阶段用 ultralytics，**部署阶段只用 onnxruntime 加载 `.onnx`，代码与 ultralytics 物理隔离**；更保守的做法是换 **RT-DETR（Apache-2.0）**。申报书里建议明确写一句"演示阶段使用，商用需替换或购买许可"。

> ❗ **预期管理：GitHub 上基本没有"开箱即用的缺陷检测预训练权重"。** 逐仓库核实 `Marfbin/NEU-DET-with-yolov8`、`IamSunday/YOLOv8-Steel-Detection`、`YWT001-Cyber/steel-defect-yolo`、`LZY-233/yolov8_Imporved-Defect_detection` —— **全部只有训练代码，没有发布 NEU-DET 上训练好的权重**。`Marfbin` 仓库里的 `yolov8n.pt`(6,534,387 B)/`yolov8s.pt`(22,573,363 B) 体积与官方 COCO 检查点完全一致，**只是训练初始化权重，不是缺陷检测权重**。请预留自己训练的时间。

### 2.2 中文大模型（**本节数字均由 ModelScope `/api/v1/models/...` 实测**）

| 名称 | HuggingFace 镜像 | ModelScope（国内直连） | 参数量 | 精度 | 磁盘权重 | 许可证 | 8GB 可行性 | 验证状态 |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct | https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct | 7.61 B（非嵌入 6.53 B） | BF16 | **15.23 GB**（4 分片） | **Apache-2.0** | ❌ FP16 不可行（15.2 GB > 7.5 GB） | ✅API（下载量 7,294,195） |
| **Qwen2.5-7B-Instruct-AWQ** ⭐ | https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct-AWQ | https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct-AWQ | 7.61 B | **AWQ 4-bit**（tensor_type F16/I32） | **5.57 GB**（2 分片） | **Apache-2.0** | ✅**可跑**——5.57 GB 权重 + 短上下文 KV cache，7.5 GB 内可行；**上下文要压到 4–8K** | ✅API（下载量 44,468） |
| **Qwen2.5-VL-3B-Instruct** | https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct | https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct | 3.75 B | BF16 | **7.51 GB**（2 分片） | ⛔ **`qwen-research` 许可——仅供研究，不可商用** | ⚠️**临界**——7.51 GB 权重本身已接近 7.5 GB 上限，FP16 直载**基本不可行**；须用下面的 AWQ 版 | ✅API（下载量 1,870,648） |
| **Qwen2.5-VL-3B-Instruct-AWQ** ⭐ | https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct-AWQ | ⚠️ModelScope 未验证 | 3.75 B | **AWQ 4-bit**（`modules_to_not_convert: ["visual"]`，ViT 保持 BF16） | **3.41 GB**（单文件 `model.safetensors`） | ⛔ **`qwen-research`**（已在 HF `cardData.license_name` 确认） | ✅**显存可行**（3.41 GB），但**许可仅研究用途** | ✅hf-mirror API（下载量 471,844） |
| Qwen2.5-VL-3B-Instruct README 说明 | — | 同上 | 默认视觉 token 范围 **4–16384**；可用 `min_pixels`/`max_pixels` 控制（如 256–1280 token）以降显存 | — | — | — | ✅建议把 `max_pixels` 设为 `1280*28*28` 以内以省显存 | ✅README 全文 |
| **Qwen3-4B-Instruct-2507** ⭐ | https://hf-mirror.com/Qwen/Qwen3-4B-Instruct-2507 | https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507 | 4.0 B（非嵌入 3.6 B） | BF16 | **8.04 GB**（3 分片） | **Apache-2.0** | ⚠️**临界**——BF16 8.04 GB 超出 7.5 GB 上限，**须用 4bit 量化版**（或 `llama.cpp` Q4_K_M GGUF，约 2.5 GB） | ✅API（下载量 753,030） |
| Qwen3-4B-Instruct-2507 能力（README 已核实） | — | 同上 | 上下文**原生 262,144**（OOM 时官方建议降到 32,768）；仅 non-thinking 模式；**BFCL-v3 工具调用 61.9**、MMLU-Pro 69.6、ZhengLogic 80.2；官方明示 **Ollama / LM Studio / llama.cpp / KTransformers 均已支持**；推荐用 Qwen-Agent 做工具调用 | — | — | — | Apache-2.0 且工具调用强，**是同尺寸下许可最干净的候选** | ✅README 全文 |
| **GLM-4-9B-Chat** | https://hf-mirror.com/THUDM/glm-4-9b-chat | https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat | 9.4 B | BF16 | **18.62 GB**（10 分片） | ⚠️**License: "other"**（自定义 GLM-4 许可，非 Apache/MIT；权重使用遵循仓库 LICENSE） | ❌ FP16 完全不可行；4bit 量化后约 6–7 GB，**极端勉强**，不推荐在 8GB 上跑 | ✅API（下载量 438,290） |
| GLM-4-9B-Chat 能力（README 已核实） | — | 同上 | 支持 128K 上下文、网页浏览、代码执行、**自定义工具调用（Function Call）**；BFCL 工具调用 Overall Acc. **81.00**（对比 Llama-3-8B-Instruct 58.88、gpt-4-turbo 81.24） | — | — | — | — | ✅README 全文 |

**关键结论（基于实测字节数）**：

1. **8GB 显存下唯一"开箱即跑"的 7B 级方案是 `Qwen2.5-7B-Instruct-AWQ`**（5.57 GB 权重），且它是 **Apache-2.0**，许可最干净。**必须把上下文限制在 4–8K**，并关闭/限制并发。
2. ⛔ **重大许可警告：`Qwen2.5-VL-3B-Instruct` 及其 AWQ 版不是 Apache-2.0，而是 `qwen-research` 许可——仅供研究，不可商用。** 这一点极易踩坑（很多人以为 Qwen2.5 全系 Apache-2.0，实际 3B-VL 不是）。它显存友好（AWQ 版仅 3.41 GB），但若作品要宣称"可交付企业"就不能用它，建议改用 **Qwen3-4B-Instruct-2507（Apache-2.0）** 或云 API 的视觉模型。
3. **`Qwen2.5-VL-3B-Instruct` FP16 权重 7.51 GB、`Qwen3-4B-Instruct-2507` BF16 权重 8.04 GB，都超过 7.5 GB 可用显存上限**——不要以为"3B/4B 就一定轻"。要用它们必须走 4bit（AWQ/GPTQ）或 `llama.cpp` 的 Q4_K_M GGUF（Qwen3-4B 约 2.5 GB）。
4. **`GLM-4-9B-Chat` 在 8GB 上不实用**（FP16 18.6 GB），且**许可证是 "other" 不是 Apache-2.0**——申报书里不要写"Apache-2.0 开源"。
5. **KV cache 是隐藏杀手**：7B 模型即使 4bit，上下文从 4K 拉到 32K，KV cache 会从约 0.5 GB 涨到 4 GB 以上，直接 OOM。比赛演示请把上下文压在 4–8K。
6. **官方明示 Windows 可用的本地推理路径**：Qwen3-4B 的 README 明确列出 **Ollama、LM Studio、llama.cpp、KTransformers** 已支持（这些都是 Windows 友好的、不依赖 vLLM 的路径）。⚠️**vLLM 在 Windows 上支持有限**（官方主要面向 Linux），Windows 本地部署优先 llama.cpp / Ollama / LM Studio，或直接在 WSL2 里跑。
7. **最稳的工程做法**：检测模型走 GPU（占用很小），LLM 走**云端 API 兜底**（把 8 GB 全部留给视觉部分），或 LLM 走 4bit GGUF 的 CPU/GPU 混合推理。

**许可速查（申报书直接抄）**：

| 模型 | 许可 | 能否商用 |
|---|---|---|
| Qwen2.5-**7B**-Instruct / -AWQ | **Apache-2.0** | ✅ 可 |
| **Qwen2.5-3B-Instruct** | **Qwen Research License（LICENSE 原文 `FOR NON-COMMERCIAL PURPOSES ONLY`）** | ⛔ **不可** |
| Qwen3 全系（含 Qwen3-4B-Instruct-2507、Qwen3-4B-AWQ、Qwen3-8B-AWQ、Qwen3-VL-4B-Instruct） | **Apache-2.0** | ✅ 可 |
| **Qwen2.5-VL-3B-Instruct / -AWQ** | **qwen-research** | ⛔ **不可** |
| GLM-4-9B-Chat | **other**（自定义） | ⚠️ 需逐条读 LICENSE |
| **GLM-4-9B-0414**（新版） | **MIT** | ✅ 可 |
| **MiniCPM5-2B** | **Apache-2.0** | ✅ 可 |
| Ultralytics YOLO（含**你自己微调的权重**） | **AGPL-3.0** | ⚠️ 传染性，商用需说明 |
| BAAI bge-* embedder / bge-reranker-base | MIT | ✅ 可 |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | ✅ 可 |
| M3E 系列 | 非商用 | ⛔ 不可 |

> ⚠️ **反直觉但重要**：`Qwen2.5-7B-Instruct` 是 Apache-2.0（可商用），**而 `Qwen2.5-3B-Instruct` 和 `Qwen2.5-VL-3B-Instruct` 反而是非商用许可**。小 ≠ 许可宽松。要可商用的小模型请用 **Qwen3 全系（Apache-2.0）** 或 **GLM-4-9B-0414（MIT）**、**MiniCPM5-2B（Apache-2.0）**。

#### 2.2.1 官方显存基准（Qwen 官方 speed benchmark，比任何估算都可靠）

来源：https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html ✅（官方测试条件：H20 96 GB + FlashAttention 2，batch=1，单位 MB）

| 模型 | BF16 | AWQ-INT4 @1tok | AWQ-INT4 @6K | AWQ-INT4 @14K |
|---|---|---|---|---|
| Qwen3-0.6B | 1394 | — | — | — |
| Qwen3-1.7B | 3412 | （GPTQ-Int8 2229） | — | — |
| **Qwen3-4B** | **7973** ❌超预算 | **2915** ✅ | **3881** ✅ | **5151** ✅ |
| **Qwen3-8B** | **15947** ❌ | **6177** ⚠️ | **7113** ⚠️临界 | — |

**读法**：H20 的绝对值不能直接搬到 4060，但**同一模型不同精度的相对比例**是可靠的——4bit 大约是 BF16 的 **37–39%**。所以 `Qwen3-4B` 走 **AWQ-INT4（约 2.9 GB）在 8 GB 上非常舒服**（即使 14K 上下文也才 5.1 GB）；`Qwen3-8B` 走 AWQ-INT4 是 **6.2–7.1 GB，属于"能跑但零余量"**，上下文必须压到 ≤4K。

#### 2.2.2 已核验的额外模型（体积为实测权重字节数）

| 模型 | 获取链接 | 参数量 | 权重体积 | 许可 | 说明 / 验证状态 |
|---|---|---|---|---|---|
| **Qwen3-4B-AWQ** ⭐ | MS: https://modelscope.cn/models/Qwen/Qwen3-4B-AWQ ／ HF: https://hf-mirror.com/Qwen/Qwen3-4B-AWQ | 4.02 B | **2.48 GB** | Apache-2.0 | 8GB 上最舒服的中文 LLM，强烈推荐。✅两侧 |
| Qwen3-4B-Instruct-2507 | MS: https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507 ／ HF: https://hf-mirror.com/Qwen/Qwen3-4B-Instruct-2507 | 4.02 B | 7.49 GB（BF16） | Apache-2.0 | 建议走 AWQ / Ollama。✅两侧 |
| **Qwen3-8B-AWQ** | HF: https://hf-mirror.com/Qwen/Qwen3-8B-AWQ | 8.19 B | **5.69 GB** | Apache-2.0 | 能跑但零余量，上下文 ≤4K。✅HF，⚠️MS 未验证 |
| Qwen3-VL-4B-Instruct | MS: https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct ／ HF: https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct | 4.44 B | 8.27 GB（BF16）❌ | Apache-2.0 | 视觉版，BF16 爆显存；走 Ollama `qwen3-vl:4b`（3.3 GB）。✅两侧 |
| **GLM-4-9B-0414** ⭐ | MS: https://modelscope.cn/models/ZhipuAI/GLM-4-9B-0414 ／ HF: https://hf-mirror.com/zai-org/GLM-4-9B-0414 | ~9 B | 17.51 GB（BF16） | **MIT** | 比 GLM-4-9B-Chat 许可更干净（MIT），但仍需 4bit。✅两侧 |
| **MiniCPM5-2B** | MS: https://modelscope.cn/models/openbmb/MiniCPM5-2B ／ HF: https://hf-mirror.com/openbmb/MiniCPM5-2B | 2.52 B | 5.03 GB（BF16） | **Apache-2.0** | 小尺寸可商用备选；FP16 都能跑。✅两侧 |
| MiniCPM3-4B / -GPTQ-Int4 / -GGUF | HF: https://hf-mirror.com/openbmb/MiniCPM3-4B | ~4 B | GPTQ-Int4 ≈2.5 GB | Apache-2.0 | ✅HF |
| **MiniCPM-V-4.6** / -gguf（视觉） | HF: https://hf-mirror.com/openbmb/MiniCPM-V-4.6 | — | — | **Apache-2.0** | **可商用的视觉模型替代（避开 Qwen2.5-VL-3B 的许可问题）**。✅HF |
| **Qwen3.5-4B**（2026 新，多模态） | HF: https://hf-mirror.com/Qwen/Qwen3.5-4B | 4.66 B | — | **Apache-2.0** | ✅HF |
| Qwen3.5-9B（多模态） | HF: https://hf-mirror.com/Qwen/Qwen3.5-9B | — | — | Apache-2.0 | ✅HF |
| **GLM-4.1V-9B-Thinking**（视觉） | HF: https://hf-mirror.com/zai-org/GLM-4.1V-9B-Thinking | ~9 B | — | **MIT** | ✅HF |
| GLM-4-9B-chat | HF: https://hf-mirror.com/zai-org/glm-4-9b-chat | ~9 B | 18.62 GB（BF16） | ⚠️**other** | ✅HF |
| InternLM2.5-7B-chat | HF: https://hf-mirror.com/internlm/internlm2_5-7b-chat | 7.74 B | 15.5 GB（BF16） | ⚠️**other**（需 `trust_remote_code=True`） | ✅HF |
| Qwen3-4B / Qwen3-8B / Qwen3-4B-GGUF / Qwen3-8B-GGUF / Qwen3-VL-4B-Instruct-GGUF | HF: `https://hf-mirror.com/Qwen/<对应名>` | — | GGUF Q4 ≈2.5 GB | Apache-2.0 | ✅HF 侧 |

> **Anomalib 已改名**：`openvinotoolkit/anomalib` → **`open-edge-platform/anomalib`**（旧路径仅 302 重定向）。最新 tag `lib/v2.7.0`（2026-10-06）。

> ⚠️ **本节仍存在未验证条目**（不影响落地，但引用前请自行确认）：Qwen3 其余尺寸（8B / 30B-A3B）、InternLM2.5-7B、MiniCPM；`Qwen2.5-VL-3B-Instruct-AWQ` 在 **ModelScope** 上的对应仓库（HF 侧已确认存在，ModelScope 侧未验证）；阿里云百炼与智谱的具体价格数字。以上均已在对应行标注验证状态。

### 2.3 本地推理框架（Windows 上的可行路径）

| 名称 | URL | 关键参数 | 许可证 | Windows 可行性 | 验证状态 |
|---|---|---|---|---|---|
| **llama.cpp** ⭐ | https://github.com/ggml-org/llama.cpp | **MIT**；支持 1.5/2/3/4/5/6/8-bit 整数量化；**"CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity"**——正是 8GB 显存跑大模型的解法；内置 OpenAI 兼容 server；CUDA 后端完整支持 NVIDIA | **MIT** | ✅**首选**。Windows 一键安装：`irm https://llama.app/install.ps1 \| iex`；`llama serve -hf <repo>-GGUF` 启 OpenAI 兼容 API；也可从 [Releases](https://github.com/ggml-org/llama.cpp/releases) 下预编译二进制 | ✅README 全文 |
| **Ollama** ⭐ | https://github.com/ollama/ollama | 后端即 llama.cpp；REST API 默认 `http://localhost:11434`；官方 Python SDK `pip install ollama`；模型库 https://ollama.com/library | MIT | ✅**最省事**。Windows：`irm https://ollama.com/install.ps1 \| iex` 或直接下 [OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)；**RTX 4060 明确支持** | ✅README 全文 |
| **LM Studio** | https://lmstudio.ai/ | 图形界面，底层也是 llama.cpp；**专有软件**（个人/企业内部使用免费） | 专有 | ✅Windows 版 | ⚠️部分验证 |
| **bitsandbytes** ⭐ | https://github.com/bitsandbytes-foundation/bitsandbytes | **官方支持 Windows 11 + NVIDIA CUDA**；提供 `LLM.int8()` 与 **QLoRA 4-bit** 量化 | MIT | ✅**Windows 上做 4bit 量化的官方路径**（因为 AutoAWQ/GPTQ 官方基准是 Linux） | ✅README 全文 |
| vLLM | https://github.com/vllm-project/vllm | 吞吐最高 | Apache-2.0 | ❌**官方文档原文 "vLLM does not support Windows natively"，Requirements: OS = Linux**；只能走 WSL2 或社区 fork | ✅文档原文 |
| SGLang | https://github.com/sgl-project/sglang | Qwen3 README 官方推荐 | Apache-2.0 | ❌安装文档定位 **Linux + NVIDIA + CUDA 13** | ✅文档 |
| Xinference | https://github.com/xorbitsai/inference | 有 `install.ps1`，但 `vllm`/`sglang` extras 限 Linux | Apache-2.0 | ⚠️部分支持 | ⚠️部分验证 |

**Ollama 上现成可拉的模型体积（官方 library 实测）**——这是 8GB 最省事的落地路径：

| Ollama 模型 | 体积 | 上下文 | 视觉 | 备注 |
|---|---|---|---|---|
| `qwen3:4b` ⭐ | **2.5 GB** | 256K | ❌ | 官方称可对标 Qwen2.5-72B-Instruct |
| `qwen3-vl:4b` ⭐ | **3.3 GB** | 256K | ✅**支持图像** | 需 **Ollama ≥ 0.12.7** |
| `qwen3-vl:2b` | 1.9 GB | 256K | ✅ | 更省 |
| `qwen3:8b` | 5.2 GB | 40K | ❌ | 零余量 |
| `glm4:9b` | 5.5 GB | 128K | ❌ | — |
| `qwen2.5vl:3b` | 3.2 GB | — | ✅ | ⛔ 注意其上游许可为 qwen-research |
| `qwen2.5vl:7b` | 6.0 GB | — | ✅ | Apache-2.0 |

> **结论**：Windows + 8GB 显存，**本地 LLM 走 llama.cpp（`llama serve`）或 Ollama**，两者都提供 OpenAI 兼容端点（llama.cpp `llama serve`、Ollama `http://localhost:11434`），因此你的 FastAPI 代码可以统一用 `openai` SDK 指向本地端点，将来切到云 API 只改 `base_url`。**vLLM / SGLang 在 Windows 上不要考虑。**

### 2.4 云端 API 兜底（把 8GB 显存全部留给检测模型）

| 名称 | URL | 关键参数 | 验证状态 |
|---|---|---|---|
| **DeepSeek 开放平台 模型 & 价格** ⭐ | https://api-docs.deepseek.com/zh-cn/quick_start/pricing/ | 当前模型：`deepseek-flash`、`deepseek-v4-pro`；OpenAI 格式 BASE URL `https://api.deepseek.com`（Anthropic 格式 `/anthropic`）；上下文 1M、输出最大 384K；**`deepseek-flash` 同时支持「图像理解」+「Tool Calls」+「Json Output」+「Responses API」**；价格（元/百万 tokens）：flash 输入缓存命中空闲 0.02 / 高峰 0.04，缓存未命中空闲 **1** / 高峰 2，输出空闲 4 / 高峰 8；空闲时段为高峰半价（高峰＝北京时间周一至周五 9:00–12:00、14:00–18:00，节假日除外）；并发 flash 2500 | ✅页面全文 |
| 阿里云百炼 Model Studio 模型调用价格 ⭐ | https://help.aliyun.com/zh/model-studio/models （可 fetch 版：https://www.alibabacloud.com/help/zh/model-studio/model-pricing ） | **免费额度：每模型 1,000,000 tokens / 90 天**（国内仅华北2北京，国际仅新加坡）；参考价：qwen3.8-max $2/$6、qwen-plus $0.115 输入、qwen3.8-flash $0.15/$0.47；**qwen-turbo 已不在现行定价页** | ✅实测 200 |
| 智谱 AI 定价 ⭐ | https://docs.bigmodel.cn/cn/guide/start/pricing | **GLM-4.7-Flash 免费**；GLM-4.5-Flash 免费层 2026-01-30 到期；GLM-5.3 ¥8/¥28。⚠️ `bigmodel.cn/pricing` 是 JS 空壳页，不要引 | ✅实测 200 |
| 阿里云百炼 查询模型列表 | https://www.alibabacloud.com/help/zh/model-studio/list-models | OpenAI 兼容的模型列表接口文档 | ⚠️部分验证 |
| SiliconFlow 定价 | https://www.siliconflow.cn/pricing | ⚠️裸域名跨域重定向，抓取不可用 | ❌未验证 |

> ⭐ **重要建议**：`deepseek-flash` 同时具备**图像理解**与**工具调用**能力，价格极低（空闲时段缓存未命中输入 $0.15/百万 tokens、输出 $0.6/百万 tokens；谷时＝峰时半价），**非常适合作为"拍照 → VLM 看图 → Agent 工具调用"这条链路的云端兜底**，从而绕开 `Qwen2.5-VL-3B` 的 `qwen-research` 许可限制和 8GB 显存限制。若预算为零，可叠加**智谱 GLM-4.7-Flash（免费）**与**百炼每模型 100 万 tokens 免费额度**。由于上述平台均提供 OpenAI 兼容端点，本地/云端切换只需改 `base_url`。

### 2.5 中文 embedding 与 reranker（已逐条经 hf-mirror API + ModelScope API 核验）

> 核验方法：`hf-mirror.com/api/models/<org>/<name>` 原始 JSON + `raw/main/config.json`；ModelScope `/api/v1/models/<org>/<name>` 全量 JSON 与 `/repo/files` 文件树。尺寸取**单文件字节数**，不是 HF 的 `usedStorage`（后者含 bin+safetensors 重复与 git 历史，会虚高 2–4 倍）。完整明细见 `chinese-embedding-reranker-verification.md`。

| 名称 | HuggingFace 镜像 URL | ModelScope URL | 维度 | 权重尺寸 | 许可证 | 验证状态 |
|---|---|---|---|---|---|---|
| **BAAI/bge-small-zh-v1.5** ⭐最轻推荐 | https://hf-mirror.com/BAAI/bge-small-zh-v1.5 | https://modelscope.cn/models/BAAI/bge-small-zh-v1.5 | **512** | **95.8 MB**（fp32，约 24 M 参数） | **MIT** | ✅两侧 API 均 200 |
| BAAI/bge-base-zh-v1.5 | https://hf-mirror.com/BAAI/bge-base-zh-v1.5 | https://modelscope.cn/models/BAAI/bge-base-zh-v1.5 | 768 | 409 MB | MIT | ✅（⚠️ModelScope 侧**无 safetensors**，仅 `pytorch_model.bin`） |
| BAAI/bge-large-zh-v1.5 | https://hf-mirror.com/BAAI/bge-large-zh-v1.5 | https://modelscope.cn/models/BAAI/bge-large-zh-v1.5 | 1024 | 1.30 GB | MIT | ✅（⚠️BAAI 的 ModelScope 仓 `model.safetensors` **已被删除**；要 safetensors 请用 `AI-ModelScope/bge-large-zh-v1.5`） |
| BAAI/bge-m3 | https://hf-mirror.com/BAAI/bge-m3 | https://modelscope.cn/models/BAAI/bge-m3 | 1024（dense） | 2.27 GB fp32（≈1.2 GB fp16） | MIT | ✅**多向量确认**：含 `sparse_linear.pt` + `colbert_linear.pt`；`max_position_embeddings:8194` |
| BAAI/bge-reranker-base | https://hf-mirror.com/BAAI/bge-reranker-base | https://modelscope.cn/models/BAAI/bge-reranker-base | N/A（cross-encoder，标量分） | 1.11 GB（278 M 参数） | MIT | ✅ |
| **BAAI/bge-reranker-v2-m3** | https://hf-mirror.com/BAAI/bge-reranker-v2-m3 | https://modelscope.cn/models/BAAI/bge-reranker-v2-m3 | N/A（cross-encoder） | 2.27 GB（568 M 参数） | **Apache-2.0** | ✅ |
| shibing624/text2vec-base-chinese | https://hf-mirror.com/shibing624/text2vec-base-chinese | **❌无（ModelScope 404）** | 768 | 409 MB（102 M 参数） | Apache-2.0 | ⚠️仅 HF 侧可用 |
| shibing624/text2vec-large-chinese | — | ❌无 | — | — | — | ❌**不可用**：hf-mirror `/api/models`、`raw/main/config.json` 均 **HTTP 401**，模型页 404。**不要引用此链接** |
| **GanymedeNil/text2vec-large-chinese**（上者的可用替代） | https://hf-mirror.com/GanymedeNil/text2vec-large-chinese | ❌无 | 1024 | ~1.30 GB | Apache-2.0 | ✅（chinese-lert-large） |
| moka-ai/m3e-small | https://hf-mirror.com/moka-ai/m3e-small | https://modelscope.cn/models/AI-ModelScope/m3e-small | 512 | 95.8 MB | ⚠️**非商用，仅供研究** | ✅（⚠️ModelScope 上 `moka-ai/*` 404，须用 `AI-ModelScope/*`） |
| moka-ai/m3e-base | https://hf-mirror.com/moka-ai/m3e-base | https://modelscope.cn/models/AI-ModelScope/m3e-base | 768 | 409 MB | ⚠️**非商用，仅供研究** | ✅（同上） |
| Xorbits/bge-small-zh-v1.5（ModelScope 重传） | — | https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5 | 512 | 95.8 MB | Apache-2.0（MS 字段） | ✅（注意：上游 BAAI 声明 MIT，此处 MS 字段写 Apache-2.0） |

**关键结论**：

- **ModelScope 正确路径**：BGE 系列 embedder/reranker **直接用 `BAAI` 组织**（`BAAI/bge-small-zh-v1.5`、`BAAI/bge-base-zh-v1.5`、`BAAI/bge-large-zh-v1.5`、`BAAI/bge-m3`、`BAAI/bge-reranker-base`、`BAAI/bge-reranker-v2-m3`）全部实测 200，且是官方上游（下载量最高）。`AI-ModelScope` 可作 small/base/large-zh-v1.5 与 reranker-v2-m3 的备选，但 **`AI-ModelScope/bge-m3` 与 `AI-ModelScope/bge-reranker-base` 不存在（404）**。M3E 必须走 `AI-ModelScope/m3e-small` / `AI-ModelScope/m3e-base`。
- **许可证**：BGE embedder 与 `bge-reranker-base` 全部 **MIT**；**`bge-reranker-v2-m3` 是 Apache-2.0**；text2vec 是 Apache-2.0；**只有 M3E 系列是非商用**——HF `cardData` 无 license 字段，模型卡原文写明"包含大量非商用数据集，所以 M3E models 也是非商用的，仅供研究使用"，**商用 RAG 请排除 M3E**。
- **8GB 显存 / 纯 CPU 场景最省的选择**：`BAAI/bge-small-zh-v1.5`（512 维、95.8 MB fp32、量化后约 24 MB、512 token 上限），是 100 MB 以下中文模型里 C-MTEB 平均分最高的，也是本清单里**唯一同时具备已验证 HF 镜像链与两条已验证 ModelScope 路径**的轻量模型。m3e-small 尺寸维度相同但非商用，故优先 bge-small-zh-v1.5。
- **ModelScope 上的 404 清单（不要引用）**：`AI-ModelScope/bge-m3`、`AI-ModelScope/bge-reranker-base`、`moka-ai/m3e-small`、`moka-ai/m3e-base`、`shibing624/text2vec-base-chinese`、`shibing624/text2vec-large-chinese`、`AI-ModelScope/text2vec-base-chinese`、`AI-ModelScope/text2vec-large-chinese`。

---

## 3. 可复用的开源项目 / 脚手架

> star 数为 2026-10-12 UTC 经 GitHub API 实测值，会持续变化。完整明细见同目录 `qc_agent_tech_research.md`。

### 3.1 轻量 RAG 框架（重点：能嵌进 FastAPI 的最小方案）

| 名称 | URL | star | 许可证 | 语言 | 嵌入 FastAPI 难度 | 验证状态 |
|---|---|---|---|---|---|---|
| **Milvus Lite** | https://github.com/milvus-io/milvus-lite | 464 | Apache-2.0 | Python | **低**——`pip install -U "pymilvus[milvus-lite]"`，`MilvusClient("./demo.db")`，**无 Docker** | ✅API+README |
| **Qdrant（Python 客户端本地模式）** | https://github.com/qdrant/qdrant | 35,048 | Apache-2.0 | Rust | **低**——`QdrantClient(path="...")` / `(":memory:")`，官方 README 明示 "without running Qdrant server" | ✅API+README |
| Chroma | https://github.com/chroma-core/chroma | 29,485 | Apache-2.0 | Rust（Python 客户端 `chromadb`） | 低，但 `PersistentClient` 文档页本次 404 | ⚠️部分验证 |
| FAISS | https://github.com/facebookresearch/faiss | 41,134 | MIT | C++ | 低（纯索引库，chunk/存储自己写） | ✅API |
| sqlite-vec | https://github.com/asg017/sqlite-vec | 8,181 | Apache-2.0 | C | **最低**——SQLite 扩展，真·单文件 RAG | ✅API |
| sentence-transformers | https://github.com/huggingface/sentence-transformers | 19,166 | Apache-2.0 | Python | 低（就是 embedding 库） | ✅API || LightRAG | https://github.com/HKUDS/LightRAG | 40,085 | MIT | Python | 中（图存储+额外 LLM 调用，偏重） | ✅API |
| LangChain | https://github.com/langchain-ai/langchain | 147,685 | MIT | Python | 中（依赖树大、版本变动快） | ✅API |
| LlamaIndex | https://github.com/run-llama/llama_index | 52,469 | MIT | Python | 中（抽象层多） | ✅API |
| Haystack | https://github.com/deepset-ai/haystack | 26,726 | Apache-2.0 | Python | 中（Pipeline 显式，比 LangChain 可控） | ✅API |
| txtai | https://github.com/neuml/txtai | 13,011 | Apache-2.0 | Python | 低-中（一体化嵌入式） | ✅API |
| Milvus（服务端） | https://github.com/milvus-io/milvus | 46,365 | Apache-2.0 | Go | **高**（要 Docker/集群） | ✅API |
| ⛔ RAGFlow | https://github.com/infiniflow/ragflow | 92,061 | Apache-2.0 | Go | **高**——整套平台，不适合嵌入 | ✅API |
| ⛔ Dify | https://github.com/langgenius/dify | 158,238 | 自定义 | TypeScript | **高**——只能当外部服务 | ✅API |
| ⛔ FastGPT | https://github.com/labring/FastGPT | 29,800 | 自定义 | TypeScript | **高**——只能当外部服务 | ✅API |
| ⛔ Langchain-Chatchat | https://github.com/chatchat-space/Langchain-Chatchat | 38,681 | Apache-2.0 | Python | **高**，且**约一年无推送**（2025-11-10） | ✅API |

**Milvus Lite 已核实要点**：纯 Python 重写（LSM 存储 + WAL + Parquet 段 + 段级 FAISS + 标量过滤 + BM25 全文检索）；要求 Python ≥ 3.10，CI 覆盖 Windows 3.10（其他 Windows+Python 组合取决于 `faiss-cpu`/`pyarrow` 是否有 wheel）；默认依赖含 `pyarrow`、`numpy`、`faiss-cpu`、`grpcio`。**坑**：不兼容旧版 C++ milvus-lite 生成的 `.db`；单进程 per data_dir（文件锁）；写入需串行；无鉴权/TLS。

**不用框架的最小 RAG 参考实现**：`langchain-ai/rag-from-scratch`（9,494★，无 LICENSE 文件）、`NirDiamant/RAG_Techniques`（29,819★）、`openai/openai-cookbook`（76,575★，MIT）。
**最小可用组合**：`sentence-transformers(bge-small-zh-v1.5)` + `sqlite-vec`（或 `faiss.IndexFlatIP`）+ 启动时切块入库，检索 top-k 拼 prompt，约 100–150 行。

> ⚠️ **维度纠错**：`BAAI/bge-small-zh-v1.5` 是 **512 维**（不是 384——384 维的是英文 `bge-small-en-v1.5`）。建向量表时按 512 建。详见第 2.5 节。

### 3.2 中文 Agent 工具调用（function calling）

| 名称 | URL | star | 许可证 | 嵌入难度 | 验证状态 |
|---|---|---|---|---|---|
| **Qwen-Agent（Qwen 官方）** | https://github.com/QwenLM/Qwen-Agent | 17,168 | Apache-2.0 | 低；⚠️最近推送 2026-03-04，约 7 个月未更新 | ✅API |
| **Qwen 官方 Function Calling 文档** | https://qwen.readthedocs.io/en/latest/framework/function_call.html | — | — | 低（直接给 OpenAI 兼容写法） | ✅文档全文 |
| **pydantic-ai** | https://github.com/pydantic/pydantic-ai | 20,576 | MIT | 低（与 FastAPI 的 pydantic 同源，类型安全） | ✅API |
| smolagents | https://github.com/huggingface/smolagents | 29,797 | Apache-2.0 | 低（极简，适合 2–3 个工具） | ✅API |
| LangGraph | https://github.com/langchain-ai/langgraph | 43,135 | MIT | 中（多步根因分析合适） | ✅API |
| hello-agents（Datawhale 中文教程） | https://github.com/datawhalechina/hello-agents | 82,726 | NOASSERTION | 教程类，只读不依赖 | ✅API |
| openai-cookbook | https://github.com/openai/openai-cookbook | 76,575 | MIT | 示例代码 | ✅API |

**Qwen function calling 已核实要点**：Qwen3 推荐 **Hermes-style tool use**，工具用 JSON Schema 描述；Qwen-Agent 是 Qwen3 function calling 的 canonical implementation，可把不支持 function calling 的 OpenAI 兼容 API 包装出能力：
```python
from qwen_agent.llm import get_chat_model
llm = get_chat_model({"model": "Qwen/Qwen3-8B", "model_server": "http://localhost:8000/v1",
                      "api_key": "EMPTY"})
for responses in llm.chat(messages=messages, functions=functions): ...
```
返回助手消息用 `function_call` 字段（`name` + JSON 字符串 `arguments`），工具结果以 `{"role": "function", "name": ..., "content": ...}` 追加。
⚠️ **官方原文提醒**：推理模型不建议用基于 stopword 的 ReAct 工具模板（思考段可能出现 stopword 导致工具调用异常）。
vLLM 路线：`vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1`。

> **"OpenAI-compatible function calling 最小 FastAPI 示例"**：❌ 本次未找到值得单独引用的仓库（未编造）。标准做法是 `openai` SDK 指向任意 OpenAI 兼容端点，在 FastAPI 路由里跑「模型要工具 → 本地执行 → 结果回灌 → 再问一次」两轮循环；Qwen 官方文档已给等价最小实现。

### 3.3 工业质检 / 缺陷检测开源项目

> **Anomalib 已核实细节**（README 全文 + LICENSE，最新 tag `lib/v2.7.0` / 2026-10-06）：**Apache-2.0**；`pip install anomalib`，Windows + NVIDIA 可指定后端 `anomalib[cu126]` / `anomalib[cu130]`；要求 **Python ≥ 3.10、PyTorch ≥ 2.10**（你是 torch 2.12.1，满足）；内置 `anomalib.data.MVTecAD` 等数据集 datamodule，`anomalib.models.Patchcore` 三行起训：
> ```python
> from anomalib.data import MVTecAD
> from anomalib.models import Patchcore
> from anomalib.engine import Engine
> engine = Engine(); engine.fit(datamodule=MVTecAD(), model=Patchcore())
> ```
> CLI：`anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category transistor`；v2.7.0 新增 MH-PatchCore / RAD / FoundAD（few-shot）；**算法清单共 34 个图像 + 2 个视频模型**，含 PatchCore、PaDiM、EfficientAD、FastFlow、DRAEM、DFKDE、DFM、Reverse Distillation、STFPM、GLASS、SuperSimpleNet、WinCLIP、AnomalyDINO 等（❌**不存在 `KSigM`，勿引用**）；✅支持 Gradio 推理，另有低代码 Web 应用 **Anomalib Studio**（`application/` 目录，可 Docker 或独立应用部署，可直接接 USB/IP 相机、输出 ROS/MQTT 消息）。
> ⚠️**旧路径 `openvinotoolkit/anomalib` 已改名为 `open-edge-platform/anomalib`**（旧地址仅 302 重定向）。
> ❗ **重大预期管理：Anomalib 不附带任何预训练检测权重。** Release 资产只有 whl(1.1 MB) + tar.gz；每次推理都必须 `--ckpt_path`；`Patchcore(pre_trained=True)` 下载的只是 **ImageNet 主干**，**memory bank 必须用你自己的正常样本构建**。**导出**：v2.7.0 **移除了旧 ONNX 导出器**，须用 `dynamo=True` + `anomalib[openvino]`；CLI 见 https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/cli/index.html ✅，部署/导出见 https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/deploy/index.html ✅（`ExportType.ONNX/OPENVINO/TORCH`）。⚠️`how_to/export/*` 专页 404 未验证。
> **推荐 ONNX 路线**：Anomalib **PatchCore / PaDiM** → `anomalib export --export_type onnx` → onnxruntime。PatchCore 无梯度训练（memory bank + coreset），其距离算子被刻意写成不用 `torch.cdist` 以兼容 ONNX/OpenVINO。

> **唯一可直接下载的工业缺陷预训练模型**：`weld-porosity-detection-0001`（Open Model Zoo，**Apache-2.0**，ResNet18，焊缝气孔 3 类分类，准确率 97.14%，FP32 44.7 MB / INT8 11.2 MB，含 SHA512，IR 实测可下载）。⚠️**但它是 OpenVINO IR 不是 ONNX，onnxruntime 跑不了，需额外 `pip install openvino`**。⚠️注意 **OMZ 里没有 padim/patchcore**（已全文扫描确认），且 OMZ 自述处于 maintenance mode。

**GitHub 搜索关键词实测命中量**（search API 实测）：`surface defect detection` **1,306** / `industrial anomaly detection` **1,914** / `PCB defect detection` **1,027** / `yolov8 defect detection` **986**。

| 名称 | URL | star | 用途 | 许可证 |
|---|---|---|---|---|
| **Ultralytics YOLO** | https://github.com/ultralytics/ultralytics | 62,419 | 检测/分割/分类基座；`yolo11n.pt` 仅 5.6MB | **AGPL-3.0** ⚠️ |
| PaddleDetection | https://github.com/PaddlePaddle/PaddleDetection | 14,439 | 国产检测工具箱（PP-YOLOE/RT-DETR），中文文档全 | Apache-2.0 |
| **Anomalib** | https://github.com/open-edge-platform/anomalib | 6,238 | **无监督异常检测**（PaDiM/PatchCore/EfficientAD），解决「只有良品样本」 | Apache-2.0 || Charmve/Surface-Defect-Detection | https://github.com/Charmve/Surface-Defect-Detection | 4,131 | 中文最大工业缺陷数据集+论文索引 | MIT |
| M-3LAB/awesome-industrial-anomaly-detection | https://github.com/M-3LAB/awesome-industrial-anomaly-detection | 3,821 | 工业异常检测论文+数据集检索库，中文说明 | 无 LICENSE |
| Eatzhy/surface-defect-detection | https://github.com/Eatzhy/surface-defect-detection | 786 | 缺陷检测文献清单 | 无 |
| ShuaiLYU/Deep-Learning-Approach-for-Surface-Defect-Detection | https://github.com/ShuaiLYU/Deep-Learning-Approach-for-Surface-Defect-Detection | 752 | 经典分割基线（TF 实现） | 无 |
| **DeepPCB** | https://github.com/tangsanli5201/DeepPCB | 554 | PCB 缺陷数据集（6 类） | MIT |
| Tiny-Defect-Detection-for-PCB | https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB | 525 | 小目标 PCB 缺陷检测 | 无 |
| abin24/Surface-Inspection-defect-detection-dataset | https://github.com/abin24/Surface-Inspection-defect-detection-dataset | 399 | 多种表面缺陷数据集集合 | 无 |
| GLASS (ECCV 2024) | https://github.com/cqylunlun/GLASS | 396 | 梯度上升异常合成，SOTA 之一 | MIT |
| **MMAD (ICLR'25)** | https://github.com/jam-cc/MMAD | 278 | **多模态大模型做工业异常检测 benchmark**（可论证「检测+大模型」路线） | 无 |
| open-iad (M-3LAB) | https://github.com/M-3LAB/open-iad | 171 | 工业图像异常检测统一 benchmark | 无 |
| **ind_knn_ad** | https://github.com/rvorias/ind_knn_ad | 166 | 纯 torch+timm 的 kNN 异常检测（**极轻，适合嵌入式基线**） | MIT |
| Marfbin/NEU-DET-with-yolov8 | https://github.com/Marfbin/NEU-DET-with-yolov8 | 152 | NEU-DET + YOLOv8 改进 | AGPL-3.0 |
| YeahHuang/Al_surface_defect_detection | https://github.com/YeahHuang/Al_surface_defect_detection | 253 | 天池铝型材表面缺陷 top 0.3% 方案 | 见仓库 |
| qunshansj/pcb-defect-detection-improved-yolov5-opencv | https://github.com/qunshansj/pcb-defect-detection-improved-yolov5-opencv | 91 | 中文：改进 YOLOv5+OpenCV PCB 缺陷检测系统 | 无 |
| JiaLim98/YOLO-PCB | https://github.com/JiaLim98/YOLO-PCB | 57 | PCB 缺陷检测 + 异常趋势报警 | MIT |

### 3.4 报告生成 / Web 看板 / 后台模板

| 名称 | URL | star | 许可证 | 中文支持与集成 | 验证状态 |
|---|---|---|---|---|---|
| **docxtpl** | https://github.com/elapouya/python-docx-template | 2,720 | LGPL-2.1 | **最省事**：模板=Word 文件，`{{ 变量 }}` 替换，中文零配置 | ✅API |
| **WeasyPrint** | https://github.com/Kozea/WeasyPrint | 9,673 | BSD-3-Clause | HTML/CSS→PDF，用系统 CJK 字体；⚠️Windows 需 GTK/Pango 运行库 | ✅API |
| reportlab（GitHub 镜像） | https://github.com/MrBitBucket/reportlab-mirror | 86 | BSD-3-Clause | 纯 Python 无系统依赖；中文需注册 CID 字体 | ✅API（主仓在 hg.reportlab.com，勿被 star 误导） |
| ⛔ wkhtmltopdf | https://github.com/wkhtmltopdf/wkhtmltopdf | 14,547 | LGPL-3.0 | **已归档（archived，最后推送 2022-11）——不要用** | ✅API |
| **Gradio** | https://github.com/gradio-app/gradio | 43,734 | Apache-2.0 | **推荐**：官方 `gr.mount_gradio_app(app, demo, path="/gradio")` 挂到现有 FastAPI | ✅API+文档 |
| Streamlit | https://github.com/streamlit/streamlit | 45,955 | Apache-2.0 | **不能挂载**（独立进程/端口），只能 iframe | ✅API |
| NiceGUI | https://github.com/zauberzeug/nicegui | 16,281 | MIT | 基于 FastAPI 可共存；`ui.run_with(app)` API 未复核 | ⚠️部分验证 |
| Reflex | https://github.com/reflex-dev/reflex | 28,952 | Apache-2.0 | 纯 Python 全栈，不挂在 FastAPI 上 | ✅API |
| full-stack-fastapi-template（官方） | https://github.com/fastapi/full-stack-fastapi-template | 45,955 | MIT | 官方最稳，但前端是 **React** + Postgres/Docker | ✅API |
| **FastapiAdmin**（中文） | https://github.com/fastapiadmin/FastapiAdmin | 1,199 | MIT | FastAPI + SQLAlchemy + **Vue3+TS+Vite+Pinia+Element-Plus**，中文文档 | ✅API |
| sqladmin | https://github.com/smithyhq/sqladmin | 2,844 | BSD-3-Clause | ⚠️原 `aminalaee/sqladmin` 已重定向至此 | ✅API |
| FastAPI-Amis-Admin | https://github.com/amisadmin/fastapi-amis-admin | 1,566 | Apache-2.0 | FastAPI + amis 低代码后台 | ✅API |
| ⛔ fastapi_vue3_admin（同名 fork） | https://github.com/sanyuan0v0/fastapi_vue3_admin | 4 | MIT | **star 仅 4，别用，用上游 FastapiAdmin** | ✅API |

### 3.5 OCR / 拍照上传

| 名称 | URL | star | 许可证 | 是否依赖 paddle | 验证状态 |
|---|---|---|---|---|---|
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR | 90,968 | Apache-2.0 | **是**（需 `paddlepaddle`，24,121★，Windows 易踩 wheel/CUDA 坑） | ✅API |
| **RapidOCR** | https://github.com/RapidAI/RapidOCR | 8,122 | Apache-2.0 | **否**——基于 **ONNX Runtime**，与已有 onnxruntime 1.23 天然一致 | ✅API |

**手机拍照 → 后端上传**：后端用 FastAPI `UploadFile`/`File`（官方文档已验证 https://fastapi.tiangolo.com/tutorial/request-files/ ，需装 `python-multipart`）；前端用 `<input type="file" accept="image/*" capture="environment">` 调后置摄像头。❌ 未找到值得单列的高 star 参考仓库（未编造）。

---

## 4. 政策依据原文链接

> 全部链接均为 `.gov.cn` / `.miit.gov.cn` / `.mof.gov.cn` / `openstd.samr.gov.cn` 官方域名，经 web_fetch 实测 HTTP 200 并逐字比对标题、发文机关、文号、日期与正文。

### 4.1 "人工智能+"行动国家文件

| 文件全称 | 发布机构 | 文号 / 日期 | 官方 URL | 核心可引原句 | 验证状态 |
|---|---|---|---|---|---|
| **《国务院关于深入实施"人工智能+"行动的意见》** | 国务院 | **国发〔2025〕11号**；成文 2025-08-21，发布 2025-08-26（索引号 000014349/2025-00070） | https://www.gov.cn/zhengce/content/202508/content_7037861.htm （备选政策文件库：https://www.gov.cn/zhengce/zhengceku/202508/content_7037862.htm ；公报版：https://www.gov.cn/gongbao/2025/issue_12266/202509/content_7039598.html ） | "推进工业全要素智能化发展。推动工业全要素智能联动，加快人工智能在设计、中试、生产、服务、运营全环节落地应用。……深化人工智能与工业互联网融合应用，增强工业系统的智能感知与决策执行能力。" 另："到2027年，率先实现人工智能与6大重点领域广泛深度融合，新一代智能终端、智能体等应用普及率超70%"；"培育人工智能应用服务商，发展'模型即服务'、'智能体即服务'等" | ✅已验证 |
| **《"人工智能+制造"专项行动实施意见》**（本项目**最直接**的政策依据） | 工信部、中央网信办、国家发改委、教育部、商务部、国务院国资委、市场监管总局、国家数据局（八部门） | **工信部联科〔2025〕279号**；成文 2025-12-25，发布 2026-01-07 | 工信部：https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2026/art_01010414608a4226b30687773bb21bdf.html ；全文转载（含附件1/2）：http://xxzx.fj.gov.cn/jjxx/zcfg/202601/t20260126_7085338.htm | **①「质检技术」最贴题**："生产制造环节。深化人工智能技术在工业核心流程控制、工艺优化、排产调度等环节应用，促进生产过程分析、决策、执行智能化。**推广机器视觉、无人智能巡检等工业质检技术**，强化产线实时监测和预测性维护，提升设备故障识别准确性，实现安全生产风险预警与事件告警。" **②「小样本缺陷微调」直接对应技术路线**（附件2《制造业企业人工智能应用指南》）："质量检测与缺陷识别场景，重点开展基于预训练模型的小样本标注缺陷数据微调，强化模型对复杂微小特征提取能力" **③目标**："到2027年，……推出1000个高水平工业智能体，打造100个工业领域高质量数据集，推广500个典型应用场景。……选树1000家标杆企业。" **④中小企业**："深入实施中小企业数字化赋能专项行动，支持中小企业开展数字化、智能化改造，加快中小企业人工智能应用复制推广。" | ✅已验证 |

> ⚠️ **纠错**：未发现发改委单独发布的"人工智能+"实施方案（"人工智能+制造"专项即上面的工信部联科〔2025〕279号）；**未检索到名为《人工智能+制造业》的实施意见**——若申报书引用该名称会失实，请勿使用。

### 4.2 制造业数字化转型 / 中小企业数字化转型

| 文件全称 | 发布机构 | 文号 / 日期 | 官方 URL | 核心可引原句 | 验证状态 |
|---|---|---|---|---|---|
| 《国务院关于深化"互联网+先进制造业"发展工业互联网的指导意见》 | 国务院 | 无发文字号；成文 2017-11-19，发布 2017-11-27 | https://www.gov.cn/zhengce/content/2017-11/27/content_5242582.htm | "加快中小企业工业互联网应用普及。推动低成本、模块化工业互联网设备和系统在中小企业中的部署应用"；"形成一批……具有深度学习等人工智能技术的工业智能软件和解决方案。" | ✅已验证 |
| 《"十四五"智能制造发展规划》 | 工信部、发改委、教育部、科技部、财政部、人社部、市场监管总局、国资委（八部门） | 工信部联规〔2021〕207号；成文 2021-12-21 | https://www.gov.cn/zhengce/zhengceku/2021-12/28/content_5664996.htm （规划全文 PDF：https://www.gov.cn/zhengce/zhengceku/2021-12/28/5664996/files/a22270cdb0504e518a7630fa318dbcd8.pdf ） | 见 PDF 附件 | ✅已验证（正文为通知，全文在 PDF） |
| 《关于加快传统制造业转型升级的指导意见》 | 工信部等八部门 | 工信部联规〔2023〕258号；成文 2023-12-28 | https://www.gov.cn/zhengce/zhengceku/202312/content_6923270.htm | "大力推进企业智改数转网联。……加快人工智能、大数据、云计算、5G、物联网等信息技术与制造全过程、全要素深度融合。……加快推动中小企业数字化转型，推动智改数转网联在中小企业先行先试。" | ✅已验证 |
| **《中小企业数字化赋能专项行动方案（2025—2027年）》** | 工信部、财政部、中国人民银行、金融监管总局（四部门） | **工信部联企业〔2024〕239号**；成文 2024-12-12，发布 2024-12-13 | https://www.miit.gov.cn/zwgk/zcwj/wjfb/tz/art/2024/art_b286a153d2ff4494a6d8956964499d24.html | ★**"视觉质量检测"原句（重点任务9）**："鼓励各地组织开展中小企业人工智能应用案例征集遴选，培育挖掘**视觉质量检测**、客户画像与精准营销、财务管理自动化等一批典型场景，为中小企业提供借鉴参考。" 另："加强中小企业人工智能应用推广。……推动人工智能技术在研发设计、生产制造、**质量检测**、运行维护、经营管理等中小企业关键业务场景应用普及"；"建设一批适用于中小企业的垂直行业大模型" | ✅已验证 |
| 《中小企业数字化赋能专项行动方案》（2020 旧版，作历史沿革引用） | 工信部办公厅 | 工信厅企业〔2020〕10号；2020-03-18 | https://www.gov.cn/zhengce/zhengceku/2020-03/24/content_5494882.htm | — | ✅已验证 |
| 《制造业企业数字化转型实施指南》 | 工信部、国务院国资委、全国工商联 | 工信部联信发〔2024〕241号；成文 2024-12-17 | https://www.gov.cn/zhengce/zhengceku/202412/content_6994622.htm | — | ✅已验证 |
| 中小企业数字化转型城市试点（总方案） | 财政部、工信部 | **财建〔2023〕117号**；2023-06-12；2023—2025 分三批，首批 30 个左右城市 | https://www.mof.gov.cn/gkml/caizhengwengao/wg2023/wg202306/202311/t20231102_3914510.htm | — | ✅已验证 |
| **2025 年第三批城市试点通知** | 财政部办公厅、工信部办公厅 | **财办建〔2025〕20号**；2025-05-07（发布 05-09）；选择 34 个左右城市 | https://www.miit.gov.cn/xwfb/gxdt/sjdt/art/2025/art_2764fad98e3f45b5af7e1258a8cdbf8d.html | ★**极贴题原句**："因地制宜推进人工智能大模型、大数据、区块链等技术在研发设计、**视觉质检**、参数优化、能耗管理、智能分拣等场景中的应用。" | ✅已验证 |
| 2024 年第二批城市试点（35 个左右城市） | 财政部、工信部 | — | — | — | ⚠️**部分验证**——未定位到工信部/财政部原始通知页，只见新闻类转载（如 https://www.cnii.com.cn/gxxww/qy/202405/t20240506_564795.html ）。如需引用请注明"据 2024 年试点工作部署"或省略 |
| **《制造业数字化转型行动方案》** | 国务院常务会议审议通过 | 2024-05-11 国务院常务会议 | ⚠️**未见公开发布的全文/文号**。可核对的官方会议报道（新华社）：https://www.gov.cn/yaowen/liebiao/202405/content_6950510.htm | "会议指出，制造业数字化转型是推进新型工业化、建设现代化产业体系的重要举措。要根据制造业多样化个性化需求，分行业分领域挖掘典型场景。……要加大对中小企业数字化转型的支持，与开展大规模设备更新行动、实施技术改造升级工程等有机结合，完善公共服务平台，探索形成促进中小企业数字化转型长效机制。" | ⚠️**部分验证**（只能引"2024年5月11日国务院常务会议审议通过"，**不要编造文号**）。⛔ 注意 https://www.gov.cn/zhengce/202405/content_6950744.htm 是媒体解读稿，不是原文，不要作为政策依据引用 |

### 4.3 "十五五"规划中关于 AI 赋能的表述

| 文件全称 | 发布机构 | 日期 | 官方 URL | 验证状态 |
|---|---|---|---|---|
| **《中共中央关于制定国民经济和社会发展第十五个五年规划的建议》** | 中国共产党第二十届中央委员会第四次全体会议 | 2025-10-23 通过；新华社 2025-10-28 发布 | https://www.gov.cn/gongbao/2025/issue_12386/202511/content_7047415.html （国务院公报 2025 年第 31 号） | ✅已验证（全文可核对） |
| **《中华人民共和国国民经济和社会发展第十五个五年规划纲要》** | 全国人民代表大会 | 2026 年 3 月 | https://www.gov.cn/yaowen/liebiao/202603/content_7062633.htm | ✅已验证（全文） |
| **2026 年《政府工作报告》** | 国务院总理李强；十四届全国人大四次会议 | 2026-03-05 报告，新华社 03-13 播发 | https://www.gov.cn/yaowen/liebiao/202603/content_7062625.htm | ✅已验证 |

**《建议》可逐字引用的原句（已逐字核对）**：
1. （十四）深入推进数字中国建设："**全面实施'人工智能＋'行动，以人工智能引领科研范式变革，加强人工智能同产业发展、文化建设、民生保障、社会治理相结合，抢占人工智能产业应用制高点，全方位赋能千行百业。**"
2. （十四）："促进实体经济和数字经济深度融合，实施工业互联网创新发展工程。加快人工智能等数智技术创新，突破基础理论和核心技术，强化算力、算法、数据等高效供给。"
3. （七）优化提升传统产业："推动技术改造升级，促进制造业数智化转型，发展智能制造、绿色制造、服务型制造，加快产业模式和企业组织形态变革。"
4. （3）主要目标："新型工业化、信息化、城镇化、农业现代化取得重大进展，发展新质生产力、构建新发展格局、建设现代化经济体系取得重大突破。"
5. （12）："培育壮大科技领军企业，支持高新技术企业和科技型中小企业发展。"

**《纲要》可引用原句**：
1. 第四篇序言："把握数字化、网络化、智能化发展大势，充分发挥我国数据资源丰富、产业体系完备、应用场景广阔优势，激活数据要素潜能，加快数智技术创新，**深化拓展'人工智能+'**，赋能经济社会发展和治理能力提升，促进生产方式深层次变革和生产力革命性跃迁。"
2. 第十三章开篇："**全面实施'人工智能+'行动**，加强人工智能同科技创新、产业发展、文化建设、民生保障、社会治理相结合，抢占人工智能产业应用制高点，全方位赋能千行百业。"
3. 第十三章第一节："促进制造业'**智改数转网联**'，实施智能制造工程和工业互联网创新发展工程，一体推进网络、标识、平台、数据、安全体系建设和规模化应用。……建设数智化转型促进网络，**健全中小企业数智赋能服务体系**。"
4. 第十二章第三节："加快建设人工智能语料库，面向能源、交通、**制造**、教育、健康、金融等领域建设**高质量数据集**"

**2026 年《政府工作报告》可引用原句**：
1. （二）加紧培育壮大新动能 → 打造智能经济新形态："**深化拓展'人工智能+'**，促进新一代智能终端和智能体加快推广，推动重点行业领域人工智能商业化规模化应用，培育智能原生新业态新模式。支持人工智能开源社区建设，促进开源生态繁荣。"
2. 优化提升传统产业："推行普惠性'上云用数赋智'服务，**持续加大对中小企业数智化转型的支持**。拓展智能制造，新建设一批智能工厂和智慧供应链。"
3. 2025 年工作回顾："持续推进制造业数字化转型和'人工智能+'行动，行业应用加快落地，新型智能终端不断涌现。"

### 4.4 国家标准 / 行业标准（均已在 openstd.samr.gov.cn 实测核对）

| 标准号 | 名称 | 发布 / 实施 | 归口 | URL | 验证状态 |
|---|---|---|---|---|---|
| **GB/T 40659-2021** | 智能制造 机器视觉在线检测系统 通用要求 | 2021-10-11 / 2022-05-01，现行 | 工信部（电子）；CCS L67 / ICS 35.240.50 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=97379A05D29792BAEDC31E55E03A5499 （详情页含起草单位/起草人：https://std.samr.gov.cn/gb/search/gbDetailed?id=CE1E6A1DD42858F6E05397BE0A0A68DF ） | ✅已验证 |
| **GB/T 42980-2023** | 智能制造 机器视觉在线检测系统 测试方法（同系列） | 2023-09-07 / 2024-04-01，现行 | — | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=D0C759F050BA78E80A2F6C0DF476373A | ✅已验证 |
| **GB/T 45575-2025** | **工业产品表面缺陷自动检测系统技术要求**（与"表面缺陷检测"最对口） | 2025-04-25 / 2025-11-01，现行 | 中国机械工业联合会 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=A081A44880BE5D2FC012A992F17B4D56 | ✅已验证 |
| **GB/T 46608-2025** | 钢管无损检测 表面缺陷机器视觉检测技术通则 | 2025-10-31 / 2026-05-01，现行 | 中国钢铁工业协会 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=B5C3B77C4C22416A0A300CA80D98CC32 | ✅已验证 |
| **GB/T 41867-2022** | 信息技术 人工智能 术语 | 2022-10-14 / 2023-05-01，现行 | 国家标准委 | https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=195F522C14AD9A1A0094FF66D0B1EF1B | ✅已验证 |

> ⚠️ **重要纠错**：**GB/T 39578 与机器视觉/缺陷检测完全无关**——经核实为 GB/T 39578-2020《基于惯性导航的应急定位系统规范》。请从引用清单中删除。
> ⚠️ **GB/T 33863 系列**实为《OPC 统一架构》(OPC UA) 系列标准（如 GB/T 33863.1-2017），与机器视觉质检无关，建议不引用。
> ✅ **在研标准（可写"标准体系正在完善"，但须注明"在研"，不要写成已发布 GB/T 号）**：20263688-T-339《智能制造 机器视觉在线检测系统 互联互通要求》；20264529-T-339《智能制造 机器视觉 图像采集与标注要求》；20262111-T-605《钢铁生产过程机器视觉智能检测系统通用要求》。

### 4.5 建议的申报书引用链（最强支撑，3 层递进）

① **国家战略层**：国发〔2025〕11号《国务院关于深入实施"人工智能+"行动的意见》
→ ② **行业专项层**：工信部联科〔2025〕279号《"人工智能+制造"专项行动实施意见》（"推广机器视觉、无人智能巡检等工业质检技术"）
→ ③ **中小企业落地层**：工信部联企业〔2024〕239号（"视觉质量检测"典型场景）＋ 财办建〔2025〕20号（"视觉质检"应用场景）
＋ **标准依据**：GB/T 40659-2021 / GB/T 45575-2025 / GB/T 41867-2022
＋ **"十五五"层面**：《建议》第（十四）条 ＋《纲要》第十三章第一节（均为逐字可核对原句）

---

## 5. 坑与限制

### 5.1 国内下载障碍（按严重程度排序）

| 障碍 | 影响范围 | 应对 |
|---|---|---|
| **HuggingFace (huggingface.co) 国内不可直连** | 所有 HF 模型/数据集权重；`from_pretrained` 会卡住或超时 | ① 优先改用 **魔搭 ModelScope**（`modelscope.cn`，阿里，国内直连，官方镜像了大量 Qwen/GLM/BAAI 模型）；② 或设环境变量 `HF_ENDPOINT=https://hf-mirror.com` 走社区镜像（⚠️ 本次从本机未能打开 hf-mirror.com，可能是本机网络环境问题，站点本身在社区中被广泛使用）；③ 公司/学校若有 HF 企业代理则走代理 |
| **Google Drive 被墙** | NEU-DET / NEU-CLS 的官方 Google Drive 链接、部分数据集 | 全部改用同一官方页给出的**百度网盘**链接（见第 1 节，均已带提取码） |
| **百度网盘非会员限速 + 大文件需客户端** | NEU 系列、GC10-DET、AITEX、RSDDs、织物、铝型材等 | ① 小文件（<100 MB）网页可直接下；② 大文件需装百度网盘客户端，非会员限速明显；③ **优先找 GitHub/Dropbox/Zenodo 直链替代**——例如 DAGM 用 Zenodo、KolektorSDD 用 `go.vicos.si`、GDXray 用 Dropbox、DeepPCB 直接在 GitHub 仓库里 |
| **Kaggle 需登录且国内访问不稳** | Severstal、GC10-DET Kaggle 镜像、DAGM Kaggle 转载 | 需注册 Kaggle 账号并在竞赛页点「接受规则」；国内直连速度差，必要时用镜像或换源 |
| **IEEE DataPort 需免费注册才能下载** | X-SDD | 注册免费 IEEE 账号即可，无需付费；注意页面写的 "LOGIN TO ACCESS DATASET FILES" |
| **Roboflow 需注册 + 有免费额度** | NEU-DET/GC10-DET 等 Roboflow 即用版 | 免费账号可用，导出 COCO/YOLO 有次数限制；拿到格式转换后的数据集能省不少标注工作，值得注册 |
| **MVTec 系列需填表单** | MVTec AD / LOCO AD / 3D-AD | 网站表单提交后通过邮件/页面给下载链接；国内访问 mvtec.com 一般正常 |
| **北大 PKU-Market-PCB 需提交申请表** | PKU-Market-PCB / HRIPCB | 官方明示需下载《PKU-HRI 数据集申请表及承诺书》并申请；**不要**去买 CSDN/博客园上流传的"打包下载目录"（非官方且收费） |
| **OpenI 启智社区需注册** | AITEX 织物数据集镜像 | 注册 OpenI 账号即可，国内直连速度好，是 AITEX 在国内最顺的路径 |
| **部分官方站点已失效** | 天池铝型材赛题 231682（已跳转通用页）、DAGM 原 HCI 站（HTTP 500）、`abin24/Magnetic-tile-defect-datasets`（GitHub 404） | 见第 1 节各条目已给出的替代源（APDDD 数据集 / Zenodo / Charmve 镜像 / Roboflow） |
| **pip 下载慢** | torch / faiss-cpu / pyarrow / sentence-transformers 等 | 配国内镜像源，如清华 `https://pypi.tuna.tsinghua.edu.cn/simple` |

### 5.2 许可证陷阱（申报书里容易被评审追问）

| 项目 | 风险 |
|---|---|
| **Ultralytics YOLO = AGPL-3.0**（本次已拉取 `LICENSE` 原文确认为 GNU Affero GPL v3） | 比赛演示、学术用途无碍；但**若作品宣称"可商用/可交付给企业"，AGPL 的传染性要求你开源整个网络服务的源码**。申报书里应写清"演示阶段使用，商用需替换或购买 Ultralytics 商业许可"，或改用 Apache-2.0 的 PaddleDetection / RT-DETR |
| **多数数据集只允许研究用途** | DeepPCB（研究用途）、GDXray（研究+教育，禁止再分发与商用）、PKU-Market-PCB（仅学术研究）、MVTec AD（CC BY-NC-SA 4.0，禁商用）、KolektorSDD/2（CC BY-NC-SA 4.0）。**作品若声称"面向中小制造企业落地"，不要把训练数据说成可商用**，建议表述为"公开基准数据验证方法可行性，实际部署用企业自有数据微调" |
| **NEU-DET / GC10-DET 未明示许可证** | 两处官方页只写了"用于研究请引用"，无 SPDX 许可证。引用时按"仅研究用途"处理最稳妥 |
| **VisA = CC BY 4.0** | 相对宽松（可商用），但需署名。这是少见的许可友好的工业异常检测数据集 |
| **DAGM 2007 = Open** | Zenodo 标注 Open，相对友好 |
| **`langchain-ai/rag-from-scratch` 无 LICENSE 文件** | 参考学习可以，直接拷贝代码进作品有法律灰区 |
| **`abin24/Surface-Inspection-*`、`M-3LAB/*`、`DeepPCB` 等部分仓库无 LICENSE** | GitHub 上"无 LICENSE ≠ 可随意用"，默认保留所有权利。作品里若直接复用其代码需谨慎 |

### 5.3 硬件与工程限制（RTX 4060 Laptop 8GB）

- **可用显存只有约 7.2–7.5 GB**（8 GB 中一部分被 Windows 桌面/驱动占用）。任何"模型权重 + KV cache + 激活值"的估算都要按 7 GB 封顶来算。
- **不要同时把检测模型和大模型常驻显存**：YOLO11n 推理占用很小（几百 MB），但 7B 级 LLM 即使 4bit 也要 4–5 GB 权重 + KV cache。建议**串行调度**：先跑检测（GPU），再跑 LLM（GPU），或把 embedding/reranker 放 CPU。
- **Windows 上 vLLM 支持有限**（官方主要面向 Linux），本地部署 LLM 建议走 **llama.cpp / Ollama / LM Studio** 或直接在 WSL2 里跑；另一条路是**用云端 API 兜底**（阿里云百炼 DashScope、智谱、DeepSeek 官方 API），把 8 GB 显存全部留给检测模型。
- **本地向量库单进程持锁**：Milvus Lite 和 Qdrant 本地模式都是单进程 + 文件锁，写入需串行。演示够用，别拿它做并发服务（见第 3 节）。
- **ONNX 路线更稳**：已装 onnxruntime 1.23，把 YOLO 导出 ONNX 后推理可绕开 torch/CUDA 版本冲突，也便于日后交付到无 GPU 的产线工控机。

### 5.4 八个必避坑（全部经本次实证）

1. **`Qwen2.5-3B-Instruct` / `Qwen2.5-VL-3B-Instruct` 是非商用的**（Qwen Research License，LICENSE 原文 `FOR NON-COMMERCIAL PURPOSES ONLY`），而 **`Qwen2.5-7B-Instruct` 反而是 Apache-2.0**。想要小模型可商用 → **用 Qwen3 系列**。
2. **`moka-ai/m3e-*` 是非商用的**（模型卡明示"仅供研究使用"）。用 `BAAI/bge-small-zh-v1.5`（MIT）。
3. **`shibing624/text2vec-large-chinese` 无法访问**（HTTP 401 + 模型页 404）。改用 `GanymedeNil/text2vec-large-chinese`。
4. **GitHub 上所有 NEU-DET / 钢铁缺陷 YOLO 仓库都只有训练代码，没有已训练权重**（已逐仓库核实 4 个）。Marfbin 仓库里那两个 `.pt` 是**官方 COCO 检查点**，不是钢材权重。**必须自己训练**（1800 张图，8GB 足够）。
5. **Anomalib 不带任何预训练权重**，release 只有 1.1 MB 的 wheel。PatchCore 的 `pre_trained=True` 只下 ImageNet 主干，memory bank 要用你自己的良品图构建。
6. **OMZ 里没有 padim/patchcore**；唯一可直接下载的工业缺陷预训练模型是 `weld-porosity-detection-0001`（Apache-2.0，但**是 OpenVINO IR 不是 ONNX**，且只是 3 类焊缝分类）。
7. **Ultralytics AGPL-3.0 会传染到你自己微调的权重**（官方许可页明确包含 custom-trained/fine-tuned 场景）。商用请买 Enterprise License，或用 **RT-DETR（Apache-2.0）**。
8. **huggingface.co 在国内不可直连**（本环境实测 `fetch failed`），**一律用 hf-mirror.com 或魔搭**。

### 5.5 本次调研中"未能验证"的条目（诚实披露）

以下条目本次**没有成功打开或核实**，请勿直接引用为事实：

- **`huggingface.co` 直连** —— 本环境实测不可达（`TypeError: fetch failed`），所有 HF 链接改用 `hf-mirror.com` 等价核验。
- `shibing624/text2vec-large-chinese` —— **HTTP 401**（`Invalid username or password`）+ 模型页 404，**不可用**。
- Roboflow 的 NEU-DET 数据集页 —— **HTTP 403**（Cloudflare 挑战）。
- `github.com/ggml-org/llama.cpp/releases` —— fetch 失败，改由 `.github/workflows/release.yml` 佐证 Windows CUDA 产物。
- `api.github.com` —— 后期 **HTTP 403 限流**，故 Ultralytics release 资产的文件名/体积未能枚举。
- `bigmodel.cn/pricing` —— HTTP 200 但为 JS 空壳，改用 `docs.bigmodel.cn`。
- **YOLOX / DAMO-YOLO** 的许可与权重 —— 未逐条核验（RT-DETR 已核验为 Apache-2.0，可直接替代）。
- Anomalib `how_to/export/*` 专页 —— HTTP 404，官方确无此页。
- **约 10 条 ModelScope 路径未逐一核验**（Qwen3-4B / Qwen3-8B / Qwen3-8B-AWQ / Qwen3-4B-GGUF / Qwen3.5-4B / Qwen3.5-9B / glm-4-9b-chat / internlm2_5-7b-chat / MiniCPM3-4B / MiniCPM-V-4.6 / GLM-4.1V-9B-Thinking）—— 表中已逐条标 ⚠️未验证，命名规律与 HF 一致但**不作断言**。已核验的 ModelScope 路径共 9 条。
- `hf-mirror.com` 首页 —— 本次从本机 `web_fetch` 失败，但**其 API（`/api/models/...`）可正常访问**，故模型链接均已通过 API 核验。
- `www.modelscope.cn/docs/models/download`、`tianchi.aliyun.com/dataset/148297`（APDDD 铝型材）—— 返回 200 但正文由 JS 渲染，**内容未能核对**。
- 天池铝型材原赛题 `231682` —— 已跳转到通用大赛页，**疑似下线**。
- NEU-DET / GC10-DET 的 Roboflow 各镜像链接 —— 未逐一打开验证。
- AITEX / RSDDs / 织物 / Charmve 打包的**百度网盘链接时效性未验证**（提取码来自已验证的 README 正文）。
- 凡我写"未明示"的数据集许可证，均表示官方页面本次没有给出明确许可证文本。
- 阿里云百炼与智谱的**具体价格数字**已在对应行标注验证状态。
