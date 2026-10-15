"""质检报告生成：Markdown / HTML / JSON 三种格式。

设计目标
--------
产线质检员需要的不是一串坐标，而是一份能存档、能签字、能追溯的**报告**。
本模块把"检测结果 + 智能体建议 + 标准引用"渲染成规范报告，支持：

- **Markdown**：便于粘贴进设计说明书、工单系统；
- **HTML**：可直接在浏览器打印为 PDF（无需额外依赖，避开 wkhtmltopdf 已归档的问题）；
- **JSON**：供 MES/QMS 系统对接。

报告要素对齐 GB/T 40659-2021 的可追溯性要求：检测时间、批次、图像、缺陷类别、
位置、严重度、判定结论、处置动作、标准引用。

用法：
    from src.report import build_report
    report = build_report(payload, fmt="html")
"""
from __future__ import annotations

import html
import json
from typing import Any

from .utils import get_logger, load_config, now_str

LOG = get_logger("report")


def _grade_badge(grade: str, color: str = "#22c55e") -> str:
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
            f'background:{color};color:#fff;font-weight:600">{html.escape(grade)}</span>')


def build_markdown(payload: dict, advice: dict | None = None, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    rcfg = cfg.get("report", {})
    dets = payload.get("detections", [])
    insp_id = payload.get("inspection_id", "-")

    lines: list[str] = []
    lines.append(f"# 表面质量智能检测报告")
    lines.append("")
    lines.append(f"| 项目 | 内容 |")
    lines.append(f"|---|---|")
    lines.append(f"| 报告编号 | {insp_id} |")
    lines.append(f"| 检测时间 | {payload.get('created_at') or now_str()} |")
    lines.append(f"| 企业 / 车间 | {rcfg.get('factory_name', '')} / {rcfg.get('workshop', '')} |")
    lines.append(f"| 检测终端 | {payload.get('operator') or rcfg.get('inspector', '')} |")
    lines.append(f"| 产品批次 | {payload.get('batch_no') or '-'} |")
    lines.append(f"| 执行标准 | {rcfg.get('standard', '')} |")
    lines.append("")
    lines.append("## 一、检测结论")
    lines.append("")
    lines.append(f"- **判定结果**：{payload.get('verdict', '-')}")
    lines.append(f"- **严重度等级**：{payload.get('grade', '-')} 级（{payload.get('grade_zh', '-')}）")
    lines.append(f"- **最高严重度得分**：{payload.get('max_severity', 0)} / 100")
    lines.append(f"- **检出缺陷数**：{len(dets)}")
    lines.append(f"- **图像尺寸**：{payload.get('image_size')} ｜ **滑窗数**：{payload.get('windows')} ｜ **耗时**：{payload.get('elapsed_ms')} ms")
    lines.append("")

    if dets:
        lines.append("## 二、缺陷明细")
        lines.append("")
        lines.append("| # | 缺陷类型 | 置信度 | 位置(x1,y1,x2,y2) | 面积占比 | 严重度 | 等级 | 建议动作 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, d in enumerate(dets, 1):
            bbox = ",".join(str(int(v)) for v in d.get("bbox", []))
            lines.append(
                f"| {i} | {d.get('class_zh', '')} | {d.get('confidence', 0):.3f} | {bbox} | "
                f"{float(d.get('area_ratio', 0)) * 100:.2f}% | {d.get('severity_score', 0)} | "
                f"{d.get('severity_level', '')} | {d.get('action', '')} |")
        lines.append("")
        summary = payload.get("class_summary") or {}
        if summary:
            lines.append("**缺陷类型分布**：" + "，".join(f"{k}×{v}" for k, v in summary.items()))
            lines.append("")
    else:
        lines.append("## 二、缺陷明细")
        lines.append("")
        lines.append("未检出表面缺陷。")
        lines.append("")

    if advice:
        lines.append("## 三、智能处置建议")
        lines.append("")
        lines.append(f"> 推理后端：`{advice.get('backend')}` ｜ 置信度：{advice.get('confidence')} ｜ "
                     f"耗时：{advice.get('elapsed_ms')} ms"
                     + (f" ｜ 降级原因：{advice.get('degraded_reason')}" if advice.get('degraded_reason') else ""))
        lines.append("")
        lines.append(f"**1. 根因分析**：{advice.get('root_cause', '-')}")
        lines.append("")
        lines.append(f"**2. 立即处置**：{advice.get('immediate_action', '-')}")
        lines.append("")
        lines.append(f"**3. 工艺纠正措施**：{advice.get('process_action', '-')}")
        lines.append("")
        lines.append(f"**4. 批次追溯建议**：{advice.get('batch_trace', '-')}")
        lines.append("")
        if advice.get("need_manual_review"):
            lines.append("> ⚠️ **需人工复检**：按内控标准，C 级及以上缺陷须转人工复检确认。")
            lines.append("")
        wo = advice.get("work_order") or {}
        if wo:
            lines.append("## 四、维修/工艺工单")
            lines.append("")
            lines.append(f"- 工单号：`{wo.get('work_order_id')}`")
            lines.append(f"- 责任部门：{wo.get('department')} ｜ 优先级：{wo.get('priority')}")
            lines.append(f"- 处置动作：{wo.get('action')}")
            lines.append(f"- 检查项：{'；'.join(wo.get('check_items', []))}")
            lines.append(f"- 是否停机：{'是' if wo.get('stop_line') else '否'} ｜ "
                         f"是否隔离批次：{'是' if wo.get('quarantine_batch') else '否'}")
            lines.append("")
        if advice.get("citations"):
            lines.append("## 五、标准与知识依据")
            lines.append("")
            for c in advice["citations"]:
                src = f"（{c.get('source')}）" if c.get("source") else ""
                lines.append(f"- `{c.get('id')}` {c.get('title')}{src}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*本报告由「智检 Agent」表面缺陷智能检测与处置系统自动生成，"
                 "判定结论与处置建议仅供工艺参考，最终处置须由有资质人员确认。*")
    return "\n".join(lines)


def build_html(payload: dict, advice: dict | None = None, cfg: dict | None = None,
               image_data_url: str = "", heatmap_data_url: str = "") -> str:
    """生成可打印为 PDF 的独立 HTML 报告（无外部依赖，离线可用）。"""
    cfg = cfg or load_config()
    rcfg = cfg.get("report", {})
    dets = payload.get("detections", [])
    insp_id = payload.get("inspection_id", "-")
    grade_color = payload.get("grade_color", "#22c55e")

    rows = "".join(
        f"<tr><td>{i}</td><td>{html.escape(str(d.get('class_zh', '')))}</td>"
        f"<td>{float(d.get('confidence', 0)):.3f}</td>"
        f"<td class='mono'>{','.join(str(int(v)) for v in d.get('bbox', []))}</td>"
        f"<td>{float(d.get('area_ratio', 0)) * 100:.2f}%</td>"
        f"<td>{d.get('severity_score', 0)}</td>"
        f"<td><b>{html.escape(str(d.get('severity_level', '')))}</b></td>"
        f"<td>{html.escape(str(d.get('action', '')))}</td></tr>"
        for i, d in enumerate(dets, 1)
    ) or "<tr><td colspan='8' class='muted'>未检出表面缺陷</td></tr>"

    advice_html = ""
    if advice:
        wo = advice.get("work_order") or {}
        cites = "".join(
            f"<li><code>{html.escape(str(c.get('id')))}</code> {html.escape(str(c.get('title')))}"
            f"<span class='muted'>{('（' + html.escape(str(c.get('source'))) + '）') if c.get('source') else ''}</span></li>"
            for c in advice.get("citations", [])
        )
        advice_html = f"""
        <h2>三、智能处置建议</h2>
        <p class="muted">推理后端：<code>{html.escape(str(advice.get('backend')))}</code> ｜
           置信度：{advice.get('confidence')} ｜ 耗时：{advice.get('elapsed_ms')} ms
           {(' ｜ 降级原因：' + html.escape(str(advice.get('degraded_reason')))) if advice.get('degraded_reason') else ''}</p>
        <div class="kv"><b>1. 根因分析</b><p>{html.escape(str(advice.get('root_cause', '-')))}</p></div>
        <div class="kv"><b>2. 立即处置</b><p>{html.escape(str(advice.get('immediate_action', '-')))}</p></div>
        <div class="kv"><b>3. 工艺纠正措施</b><p>{html.escape(str(advice.get('process_action', '-')))}</p></div>
        <div class="kv"><b>4. 批次追溯建议</b><p>{html.escape(str(advice.get('batch_trace', '-')))}</p></div>
        {('<p class="warn">⚠️ 需人工复检：按内控标准，C 级及以上缺陷须转人工复检确认。</p>') if advice.get('need_manual_review') else ''}
        <h2>四、维修/工艺工单</h2>
        <table><tbody>
        <tr><th>工单号</th><td class="mono">{html.escape(str(wo.get('work_order_id', '-')))}</td>
            <th>责任部门</th><td>{html.escape(str(wo.get('department', '-')))}</td></tr>
        <tr><th>优先级</th><td>{html.escape(str(wo.get('priority', '-')))}</td>
            <th>停机 / 隔离批次</th><td>{'是' if wo.get('stop_line') else '否'} / {'是' if wo.get('quarantine_batch') else '否'}</td></tr>
        <tr><th>处置动作</th><td colspan="3">{html.escape(str(wo.get('action', '-')))}</td></tr>
        <tr><th>检查项</th><td colspan="3">{html.escape('；'.join(wo.get('check_items', [])))}</td></tr>
        </tbody></table>
        <h2>五、标准与知识依据</h2>
        <ul class="cites">{cites or '<li class="muted">无</li>'}</ul>
        """

    img_html = ""
    if image_data_url or heatmap_data_url:
        img_html = "<h2>六、检测图像</h2><div class='imgs'>"
        if image_data_url:
            img_html += f"<figure><img src='{image_data_url}' alt='检测图像'/><figcaption>原始图像与缺陷标注</figcaption></figure>"
        if heatmap_data_url:
            img_html += f"<figure><img src='{heatmap_data_url}' alt='Grad-CAM 热力图'/><figcaption>Grad-CAM 可解释热力图（模型关注区域）</figcaption></figure>"
        img_html += "</div>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>表面质量智能检测报告 {html.escape(str(insp_id))}</title>
<style>
  :root {{ --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif; color:var(--fg);
         background:#f3f4f6; margin:0; padding:24px; }}
  .sheet {{ max-width:920px; margin:0 auto; background:var(--bg); padding:36px 40px;
            border-radius:10px; box-shadow:0 2px 16px rgba(0,0,0,.08); }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:26px 0 10px; padding-left:10px; border-left:4px solid #2563eb; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
  th,td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }}
  th {{ background:#f9fafb; font-weight:600; white-space:nowrap; }}
  .mono {{ font-family:ui-monospace,Consolas,monospace; font-size:12px; }}
  .muted {{ color:var(--muted); }}
  .kv {{ margin:8px 0; font-size:13.5px; line-height:1.7; }}
  .kv p {{ margin:2px 0 0; }}
  .warn {{ background:#fef3c7; border-left:4px solid #f59e0b; padding:8px 12px;
           font-size:13px; margin:10px 0; border-radius:4px; }}
  .cites {{ font-size:13px; line-height:1.8; padding-left:20px; }}
  .imgs {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .imgs figure {{ margin:0; flex:1 1 320px; }}
  .imgs img {{ width:100%; border:1px solid var(--line); border-radius:6px; }}
  figcaption {{ font-size:12px; color:var(--muted); text-align:center; margin-top:6px; }}
  .hero {{ display:flex; align-items:center; gap:14px; padding:14px 16px; background:#f9fafb;
           border:1px solid var(--line); border-radius:8px; margin-bottom:6px; }}
  .hero .grade {{ font-size:30px; font-weight:700; }}
  .foot {{ margin-top:26px; font-size:12px; color:var(--muted); border-top:1px solid var(--line);
           padding-top:12px; line-height:1.7; }}
  @media print {{ body {{ background:#fff; padding:0; }} .sheet {{ box-shadow:none; border-radius:0; }} }}
</style></head>
<body><div class="sheet">
  <h1>表面质量智能检测报告</h1>
  <div class="sub">{html.escape(str(rcfg.get('factory_name', '')))} ｜ {html.escape(str(rcfg.get('workshop', '')))} ｜
      执行标准：{html.escape(str(rcfg.get('standard', '')))}</div>

  <div class="hero">
    <div class="grade" style="color:{grade_color}">{html.escape(str(payload.get('grade', '-')))}</div>
    <div>
      <div style="font-size:17px;font-weight:600">判定：{html.escape(str(payload.get('verdict', '-')))}　
        （{html.escape(str(payload.get('grade_zh', '-')))}）</div>
      <div class="muted">最高严重度 {payload.get('max_severity', 0)} / 100 ｜ 检出缺陷 {len(dets)} 处 ｜
        耗时 {payload.get('elapsed_ms', 0)} ms</div>
    </div>
  </div>

  <h2>一、基本信息</h2>
  <table><tbody>
    <tr><th>报告编号</th><td class="mono">{html.escape(str(insp_id))}</td>
        <th>检测时间</th><td>{html.escape(str(payload.get('created_at') or now_str()))}</td></tr>
    <tr><th>检测终端</th><td>{html.escape(str(payload.get('operator') or rcfg.get('inspector', '')))}</td>
        <th>产品批次</th><td>{html.escape(str(payload.get('batch_no') or '-'))}</td></tr>
    <tr><th>图像尺寸</th><td>{html.escape(str(payload.get('image_size')))}</td>
        <th>滑窗数量</th><td>{payload.get('windows', 0)}</td></tr>
  </tbody></table>

  <h2>二、缺陷明细</h2>
  <table><thead><tr>
    <th>#</th><th>缺陷类型</th><th>置信度</th><th>位置</th><th>面积占比</th>
    <th>严重度</th><th>等级</th><th>建议动作</th></tr></thead>
    <tbody>{rows}</tbody></table>
  {advice_html}
  {img_html}
  <div class="foot">
    本报告由「智检 Agent」面向中小制造企业的工业表面缺陷智能检测与处置智能体系统自动生成。<br/>
    判定结论与处置建议基于公开标准与工艺知识库推断，仅供工艺参考，最终处置须由有资质人员确认。<br/>
    报告生成时间：{now_str()}
  </div>
</div></body></html>"""


def build_report(payload: dict, advice: dict | None = None, fmt: str = "md",
                 cfg: dict | None = None, image_data_url: str = "",
                 heatmap_data_url: str = "") -> str:
    """统一入口。fmt ∈ {md, html, json}。"""
    if fmt == "json":
        return json.dumps({"inspection": payload, "advice": advice}, ensure_ascii=False, indent=1)
    if fmt == "html":
        return build_html(payload, advice, cfg, image_data_url, heatmap_data_url)
    return build_markdown(payload, advice, cfg)
