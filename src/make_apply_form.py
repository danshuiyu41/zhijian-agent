"""把「作品申报表」渲染为可直接提交的 PDF（表格框线版式）。

背景：比赛的申报表是 .doc 模板，程序无法直接编辑。本脚本用 HTML 表格还原
申报表的框线版式，填好项目内容后打印为 PDF，效果与纸质表一致且更清晰，
可直接作为「作品申报表」上传。

用法：
    python -m src.make_apply_form
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

from .md2pdf import find_browser  # noqa: E402
from .utils import get_logger, resolve  # noqa: E402

LOG = get_logger("applyform")

CSS = """
@page { size: A4; margin: 14mm 13mm; }
* { box-sizing: border-box; }
body { font-family: "SimSun", "Songti SC", "Microsoft YaHei", serif;
       font-size: 10.5pt; color: #000; margin: 0; }
h1 { font-family: "SimHei", "Microsoft YaHei", sans-serif;
     font-size: 15pt; text-align: center; font-weight: 700; margin: 0 0 10pt; letter-spacing: 1px; }
table { width: 100%; border-collapse: collapse; }
td, th { border: 1px solid #000; padding: 5pt 7pt; vertical-align: top; line-height: 1.75; }
.lbl { width: 13%; text-align: center; background: #fff; font-family: "SimHei", sans-serif;
       vertical-align: middle; }
.lbl2 { width: 11%; text-align: center; font-family: "SimHei", sans-serif; vertical-align: middle; }
.val { text-align: left; }
.center { text-align: center; }
.big { font-family: "SimHei", sans-serif; font-size: 11.5pt; }
.sec { font-family: "SimHei", sans-serif; font-size: 11pt; vertical-align: middle;
       text-align: center; letter-spacing: 6px; }
.hint { color: #333; font-size: 10pt; }
p { margin: 0 0 8pt; text-indent: 0; }
p:last-child { margin-bottom: 0; }
"""


def build_html() -> str:
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>作品申报表</title><style>{CSS}</style></head><body>

<h1>附件：“工业美”智能制造创新创意大赛作品申报表</h1>

<table>
  <tr>
    <td class="lbl2">队伍编号</td><td class="center val">232593</td>
    <td class="lbl2">选题</td><td class="center val">智能技术应用类</td>
  </tr>
  <tr>
    <td class="lbl2">产品名称</td>
    <td class="val big" colspan="3">智检 Agent —— 面向中小制造企业的工业表面缺陷智能检测与处置智能体系统</td>
  </tr>
  <tr>
    <td class="lbl2">申报单位名称</td>
    <td class="val hint" colspan="3">集宁师范学院　　（此处加盖院系公章）</td>
  </tr>
  <tr>
    <td class="lbl2">参赛队员</td><td class="center val" colspan="3">刘佳乐、王浩宇</td>
  </tr>
  <tr>
    <td class="lbl2">指导教师</td><td class="center val" colspan="3">吴娟</td>
  </tr>
  <tr>
    <td class="lbl2" rowspan="2">联系方式</td>
    <td class="center">联系人：刘佳乐</td>
    <td class="lbl2">电话</td><td class="center">18586200223</td>
  </tr>
  <tr>
    <td class="center">手机：18586200223</td>
    <td class="lbl2">邮箱</td><td class="center">1052439150@qq.com</td>
  </tr>
  <tr>
    <td class="sec">产品<br/>简介</td>
    <td class="val" colspan="3">
      <p class="hint">（含开发背景、结构说明、功能与使用说明等）</p>
      <p>本产品面向中小企业表面质检场景，解决"人眼漏检、商用方案太贵、云端方案数据敏感"三重矛盾。系统以多尺度滑窗结合轻量卷积网络，检出龟裂、夹杂、斑块、麻点、压入氧化铁皮、划痕六类缺陷并定位，按"类别风险×相对尺度×置信度"给出 A~E 五级严重度与处置动作；再由质检处置智能体检索内置标准知识库，输出根因分析、工艺纠正与批次追溯建议，并自动生成维修工单，形成"感知—决策—执行"闭环。单件检测 91 毫秒，纯 CPU 运行，硬件投入约五千元。系统内置开集识别机制，遇到未训练的缺陷形态主动告警转人工，避免闭集分类器"高置信度的错误判定"；提供 Grad-CAM 热力图支撑人工复核，具备质量看板、质检报告导出与缺陷诊断问答能力。核心链路零外部依赖，可离线部署，数据不出厂。</p>
    </td>
  </tr>
  <tr>
    <td class="sec">产品<br/>特色</td>
    <td class="val" colspan="3">
      <p class="hint">（产品的产业需求匹配度、项目实操验证过程等核心内容）</p>
      <p><b>一、产业需求匹配度高。</b>契合国发〔2025〕11号、工信部联科〔2025〕279号"推广机器视觉等工业质检技术"、工信部联企业〔2024〕239号"视觉质量检测"的要求，面向中小企业"装不起专用设备、没有专职算法人员"的缺口。</p>
      <p><b>二、实操验证完整。</b>程序化合成六类缺陷 1400 张，按 7:1.5:1.5 分层划分；迁移学习配合尺度与对比度随机增强抑制过拟合，210 张未参与训练样本准确率 100%、宏平均 F1 1.0000。另构建不参与训练的压力测试集做受控退化实验，量化出小尺度叠加低对比度时漏检率 15% 的失效边界。六类缺陷的 Grad-CAM 热力图高亮区与真实缺陷位置逐一核对吻合，证明模型依据的是缺陷本身而非背景纹理。交付 11 个自动化自检脚本，覆盖源码编译、前端引用一致性、端到端链路与 21 项接口测试。</p>
      <p><b>三、落地成本低。</b>模型 1.08M 参数，纯 CPU 训练与推理，单件 91 毫秒、硬件约五千元，比商用方案低一个数量级；只需图片级标签，照片按文件夹分类即可完成标注。指标基于合成数据留出集，数据来源已在文档中如实披露。</p>
    </td>
  </tr>
</table>

</body></html>"""


def main() -> None:
    out_dir = resolve("docs/pdf")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "_申报表.html"
    html_path.write_text(build_html(), encoding="utf-8")

    browser = find_browser()
    pdf_path = out_dir / "作品申报表.pdf"
    if not browser:
        print(f"未找到浏览器，已生成 HTML：{html_path}")
        print("请打开后用 Ctrl+P → 另存为 PDF")
        return
    if pdf_path.exists():
        pdf_path.unlink()
    profile = resolve("runs/edgeprofile")
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}",
                    html_path.as_uri()], capture_output=True, timeout=180)
    if pdf_path.exists():
        print(f"OK  {pdf_path.name}  {pdf_path.stat().st_size / 1024:.0f} KB")
    else:
        print("PDF 生成失败，请打开 HTML 手动打印")


if __name__ == "__main__":
    main()
