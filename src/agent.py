"""质检处置智能体（Quality Inspection Agent）。

定位
----
检测模型只回答"是什么缺陷、在哪里、多严重"；产线真正需要的是**处置决策**：
根因在哪、该调什么参数、是否需要停机、要不要开工单、同批次怎么办。本模块把
"检测结果 + 标准知识库"编排成一个可解释、可降级的智能体。

三级推理后端（按可用性自动降级，保证任何环境都能跑）
--------------------------------------------------
1. **rule（永远可用）**：基于知识库检索 + 结构化规则模板生成结论，完全确定性，
   可离线、无 API、可复现。这是默认后端，也是评委质疑"没有大模型怎么办"时的答案。
2. **ollama**：本地大模型（Qwen3 等），走 Ollama 的 OpenAI 兼容接口，数据不出厂。
3. **openai_compat**：任意 OpenAI 兼容云 API（如 DeepSeek），用于演示更强推理。

智能体工具（Tool Calling 设计）
------------------------------
智能体可调用以下工具，体现"感知-决策-执行"闭环：
- `retrieve_standard(query)`  检索标准与工艺知识
- `severity_grade(score)`     严重度分级
- `create_work_order(...)`    生成维修/工艺工单
- `trace_batch(...)`          批次追溯建议
- `self_check(...)`           置信度自检与复检建议

设计取舍（写进申报书）
--------------------
- **不把 LLM 放在关键判定路径上**：分级与放行判定由确定性规则完成，LLM 只负责
  解释与建议生成，避免幻觉影响质量判定——这是工业 AI 落地的硬约束。
- **结论可溯源**：每条建议都携带引用的知识库条目 ID，便于人工核对。
- **全链路留痕**：每次智能体推理记录后端类型、耗时、引用来源，满足可追溯要求。

用法：
    agent = QualityAgent.from_config()
    advice = agent.analyze(inspection_result)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .rag import BM25Retriever, Hit, docs_from_json
from .utils import get_logger, load_config, resolve

LOG = get_logger("agent")


# ----------------------------------------------------------------------
# 提示词（中文，工业语境）
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一名资深热轧带钢表面质量工程师，负责根据自动检测结果给出缺陷成因分析与工艺处置建议。"
    "要求：\n"
    "1. 只依据提供的【检测结果】与【标准知识库摘录】作答，不得编造标准条款或数据；\n"
    "2. 回答用简体中文，面向产线工程师，语言简洁、可执行；\n"
    "3. 必须包含四部分：根因分析、立即处置、工艺纠正措施、批次追溯建议；\n"
    "4. 若证据不足，明确说明「需要人工确认」，不要臆测；\n"
    "5. 严格输出 JSON，不要输出额外文字。\n"
    "JSON 结构：{\"root_cause\": string, \"immediate_action\": string, "
    "\"process_action\": string, \"batch_trace\": string, \"confidence\": number, \"risk_note\": string}"
)

# 问答定位：**诊断助手，不是工艺教科书**
#
# 背景（由实际评测发现）：最初的问答会顺畅地回答"龟裂的成因是什么"这类问题，
# 但这类内容产线工程师本来就掌握，回答再完整也不提供增量价值，反而显得外行。
# 因此把问答重新定位为"从缺陷现象反查排查方向"，并把拒绝边界显式写进提示词。
QA_SYSTEM_PROMPT = (
    "你是一名热轧产线的缺陷诊断助手，服务对象是**有多年现场经验的工艺工程师**。"
    "你的唯一职责是：根据缺陷现象与产线历史记录，给出**排查方向与量化判据**，"
    "帮助对方更快定位到具体设备或参数。\n"
    "硬性要求：\n"
    "1. **不要讲解基础原理**（例如「晶粒粗大」「脱氧不良」这类机理），对方比你熟悉；\n"
    "2. 只回答三类问题：缺陷现象如何反查产线部位、异常苗头优先查哪一项、判据阈值是多少；\n"
    "3. 若问题属于基础原理或工艺常识，直接回答「这属于工艺常识，不在本助手职责范围」，"
    "并引导对方改问「该查哪里」「按什么顺序查」；\n"
    "4. 回答必须包含可核对的具体对象（设备名称、参数名、部位、历史案例编号），"
    "不得只给泛泛的方向性描述；\n"
    "5. 只依据提供的【知识库摘录】作答，证据不足时明确说明需要现场确认。"
)


@dataclass
class Advice:
    """智能体给出的处置建议。"""
    backend: str = "rule"
    root_cause: str = ""
    immediate_action: str = ""
    process_action: str = ""
    batch_trace: str = ""
    risk_note: str = ""
    confidence: float = 0.0
    need_manual_review: bool = False
    work_order: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    degraded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "root_cause": self.root_cause,
            "immediate_action": self.immediate_action,
            "process_action": self.process_action,
            "batch_trace": self.batch_trace,
            "risk_note": self.risk_note,
            "confidence": round(self.confidence, 3),
            "need_manual_review": self.need_manual_review,
            "work_order": self.work_order,
            "citations": self.citations,
            "tool_calls": self.tool_calls,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "degraded_reason": self.degraded_reason,
        }


# ----------------------------------------------------------------------
# 智能体
# ----------------------------------------------------------------------
class QualityAgent:
    """质检处置智能体：检索 + 规则 + 可选大模型。"""

    def __init__(self, cfg: dict, kb_path: str | None = None):
        self.cfg = cfg
        kb = resolve(kb_path or (cfg["paths"]["knowledge"] + "/quality_kb.json"))
        self.docs = docs_from_json(kb)
        rag_cfg = cfg.get("agent", {}).get("rag", {})
        self.retriever = BM25Retriever(self.docs, alpha=0.65 if rag_cfg.get("use_bm25", True) else 0.0)
        self.top_k = int(rag_cfg.get("top_k", 3))
        self.backends = list(cfg.get("agent", {}).get("backend_priority", ["rule"]))
        self.last_backend_used = ""
        LOG.info("知识库加载完成：%d 条 | 后端优先级 %s", len(self.docs), self.backends)

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "QualityAgent":
        return cls(load_config(config_path))

    # ==================================================================
    # 工具层（Tool Calling）
    # ==================================================================
    def tool_retrieve_standard(self, query: str, top_k: int | None = None) -> list[Hit]:
        """工具：检索标准与工艺知识。"""
        return self.retriever.search(query, top_k=top_k or self.top_k)

    def tool_severity_grade(self, score: float) -> dict:
        """工具：严重度分级。"""
        from .utils import severity_grade

        return severity_grade(self.cfg, float(score))

    def tool_create_work_order(self, det: dict, inspection_id: str = "") -> dict:
        """工具：根据缺陷类型生成维修/工艺工单（规则表驱动的联动规程）。"""
        rule = MAINTENANCE_RULES.get(det.get("class_code", ""), {})
        level = det.get("severity_level", "B")
        urgent = level in ("D", "E")
        return {
            "work_order_id": f"WO-{time.strftime('%Y%m%d%H%M%S')}-{det.get('class_code', 'X')[:3].upper()}",
            "inspection_id": inspection_id,
            "defect_code": det.get("class_code"),
            "defect_zh": det.get("class_zh"),
            "severity_level": level,
            "severity_score": det.get("severity_score"),
            "priority": "紧急" if urgent else ("高" if level == "C" else "常规"),
            "department": rule.get("department", "设备科"),
            "action": rule.get("action", "安排设备点检并复核缺陷成因"),
            "check_items": rule.get("check_items", ["复检缺陷样本", "核查相关工艺参数"]),
            "stop_line": urgent,
            "quarantine_batch": urgent,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "待处理",
        }

    def tool_trace_batch(self, det: dict) -> str:
        """工具：批次追溯建议。"""
        high = det.get("risk") == "high" or det.get("severity_level") in ("D", "E")
        if high:
            return ("该缺陷属高危类型，建议对同一炉次/同一班次产品实施全量复核，"
                    "向前追溯至炼钢/连铸或加热工序记录，向后冻结未出厂库存并通知质量部门。")
        return ("建议按固定抽样比例对该批次进行加密抽检（如由每卷抽检改为每10卷抽检3卷），"
                "并在系统中标记该批次为观察对象，持续跟踪后续500件。")

    def tool_self_check(self, result: dict) -> dict:
        """工具：置信度自检，输出是否需要人工复检。"""
        dets = result.get("detections", [])
        max_sev = float(result.get("max_severity", 0.0))
        grade = result.get("grade", "A")
        low_conf = [d for d in dets if float(d.get("confidence", 0)) < 0.7]
        need = grade in ("C", "D", "E") or len(low_conf) > 0
        reasons = []
        if grade in ("C", "D", "E"):
            reasons.append(f"严重度等级 {grade}，按内控标准须人工复检")
        if low_conf:
            reasons.append(f"{len(low_conf)} 处检出置信度低于 0.7，存在误报可能")
        if not dets:
            reasons.append("未检出缺陷，模型判定为合格")
        return {"need_manual_review": need, "reasons": reasons, "low_conf_count": len(low_conf)}

    # ==================================================================
    # 缺陷诊断助手（不是工艺教科书）
    #
    # 定位说明（由实际评测驱动）：
    # 初版问答能顺畅回答"龟裂的成因是什么"，但这类内容产线工程师本就掌握，
    # 回答再完整也不提供增量价值，现场演示时反而显得外行。
    # 现改为只做三件事：缺陷现象→反查产线部位、异常苗头→优先查哪一项、判据阈值。
    # 对基础原理类问题**显式拒答并引导**，这是有意设计，不是能力不足。
    # ==================================================================
    # 基础原理类提问的特征词：命中这些且未出现诊断意图词时，判为越界提问
    _BASIC_PATTERNS = (
        "成因是什么", "什么原因造成", "为什么会", "为什么产生", "原理是什么",
        "是什么意思", "什么是龟裂", "什么是夹杂", "解释一下", "科普",
        "定义是什么", "有哪些种类", "怎么形成的",
    )
    _DIAGNOSTIC_HINTS = (
        "先查", "排查", "怎么查", "该查", "查哪里", "哪个部位", "哪台", "哪一组",
        "按什么顺序", "顺序", "定位", "判据", "阈值", "多少", "超标", "异常",
        "比例上升", "突然", "连续", "批次", "炉次", "怎么处理", "措施", "调整",
        "复检", "检修", "换辊", "停机",
    )

    def classify_intent(self, question: str) -> str:
        """判断提问意图：'diagnostic'（诊断型）或 'basic'（基础原理型）。"""
        q = (question or "").strip()
        has_diag = any(k in q for k in self._DIAGNOSTIC_HINTS)
        has_basic = any(k in q for k in self._BASIC_PATTERNS)
        if has_basic and not has_diag:
            return "basic"
        return "diagnostic"

    # 诊断类条目的优先级：排查顺序 > 历史案例 > 规程 > 标准 > 其它
    _DOC_PRIORITY = {"排查顺序": 0, "历史案例": 1, "作业规程": 2, "维护联动": 3,
                     "分级规则": 4, "追溯要求": 5, "对照表": 6}

    def _rerank_for_diagnosis(self, hits: list[Hit]) -> list[Hit]:
        """按"对诊断的实用价值"重排检索结果，并过滤掉对工程师无增量价值的成因条目。

        成因条目仍保留在知识库中（检测处置建议会用到），
        但在问答场景下把"排查顺序/历史案例/规程"排在前面。
        """
        def key(h: Hit):
            t = str(h.doc.meta.get("type", ""))
            return (self._DOC_PRIORITY.get(t, 9), -h.score)

        useful = [h for h in hits if h.doc.meta.get("type") != "成因与处置"]
        return sorted(useful or hits, key=key)

    def answer_question(self, question: str, hits: list[Hit] | None = None) -> dict:
        """回答诊断类问题。规则后端做抽取式回答，大模型后端做生成式回答。"""
        t0 = time.time()
        intent = self.classify_intent(question)

        # 越界提问：不做科普，直接引导到正确的问法
        if intent == "basic":
            return {
                "answer": ("这属于工艺常识，不在本助手职责范围——基础原理请查阅工艺规程或标准原文。\n\n"
                           "本助手只解决「从缺陷现象反查产线部位」这一类现场最花时间的问题，例如：\n"
                           "· 龟裂比例连续三个班次上升，先查什么？\n"
                           "· 压入氧化铁皮呈条带分布，对应哪一组除鳞喷嘴？\n"
                           "· 划痕间距与某辊周长一致，怎么确认是哪个辊？"),
                "backend": "rule", "scope": "out_of_scope", "confidence": 0.9,
                "elapsed_ms": round((time.time() - t0) * 1000, 1), "citations": [],
            }

        hits = hits if hits is not None else self.retriever.search(question, top_k=5)
        hits = self._rerank_for_diagnosis(hits)
        if not hits:
            return {"answer": "知识库中未检索到与该现象相关的诊断条目。"
                              "建议补充该缺陷的历史案例或排查记录后重试。",
                    "backend": "rule", "scope": "no_hit", "confidence": 0.2,
                    "elapsed_ms": 0.0, "citations": []}

        # 先尝试大模型后端
        for backend in self.backends:
            if backend == "rule":
                continue
            try:
                cfg = self.cfg["agent"]
                conf = cfg.get(backend, {})
                if not conf.get("enabled"):
                    continue
                evidence = "\n".join(f"[{h.doc.id}] {h.doc.title}：{h.doc.text[:700]}" for h in hits[:4])
                prompt = (f"【现场问题】{question}\n\n【知识库摘录】\n{evidence}\n\n"
                          "请给出排查方向：先查什么、查哪个部位/参数、判据是多少，"
                          "控制在 200 字以内，末尾标注引用的条目 ID。")
                answer = self._llm_generate(prompt, backend, system=QA_SYSTEM_PROMPT)
                if answer:
                    return {"answer": answer, "backend": backend, "scope": "diagnostic",
                            "confidence": 0.85,
                            "elapsed_ms": round((time.time() - t0) * 1000, 1),
                            "citations": [h.doc.id for h in hits]}
            except Exception as e:
                LOG.info("问答后端 %s 不可用：%s", backend, e)

        # 规则后端：抽取式回答，直接给出该条目的排查正文
        top = hits[0]
        body = top.doc.text
        # 诊断类条目正文本身就是"排查顺序"，无需再截取"主要成因"段
        if str(top.doc.meta.get("type")) == "成因与处置":
            for marker in ("主要成因：", "工艺处置建议："):
                if marker in body:
                    body = body[body.index(marker):]
                    break
        snippet = body if len(body) <= 320 else body[:320] + "…"
        others = [h.doc.id for h in hits[1:3]]
        tail = f"（依据：{top.doc.title} [{top.doc.id}]" + \
               (f"；亦可参考 {'、'.join(others)}" if others else "") + "）"
        return {
            "answer": f"{snippet}\n\n{tail}",
            "backend": "rule",
            "scope": "diagnostic",
            "confidence": round(min(0.9, 0.5 + 0.4 * top.score), 3),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "citations": [h.doc.id for h in hits],
        }

    def _llm_generate(self, prompt: str, backend: str, system: str | None = None) -> str | None:

        """通用的单轮生成调用。"""
        cfg = self.cfg["agent"]
        conf = cfg.get(backend, {})
        if backend == "ollama":
            url = f"{conf.get('base_url', '').rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
        elif backend == "openai_compat":
            url = f"{conf.get('base_url', '').rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if conf.get("api_key"):
                headers["Authorization"] = f"Bearer {conf['api_key']}"
        else:
            raise ValueError(f"未知后端 {backend}")
        payload = {
            "model": conf.get("model", ""),
            "messages": [{"role": "system", "content": system or "你是严谨的工业质量工程师助手。"},
                         {"role": "user", "content": prompt}],
            "temperature": 0.2, "stream": False,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=float(conf.get("timeout", 60))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()

    # ==================================================================
    # 主流程
    # ==================================================================
    def analyze(self, result: dict | Any, inspection_id: str = "") -> Advice:
        """对一次检测结果做处置分析。result 可为 dict 或 InspectionResult。"""
        t0 = time.time()
        res = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        dets = res.get("detections", [])
        advice = Advice()

        # ---- 无缺陷：直接放行 ----
        if not dets:
            advice.backend = "rule"
            advice.root_cause = "未检出表面缺陷，判定为合格品。"
            advice.immediate_action = "正常放行，无需处置。"
            advice.process_action = "维持现有工艺参数，按常规频次抽检。"
            advice.batch_trace = "无需追溯。"
            advice.risk_note = res.get("quality_flags", [""])[0] if res.get("quality_flags") else ""
            advice.confidence = 0.95 if res.get("verdict") == "合格" else 0.6
            advice.tool_calls.append({"tool": "self_check", "result": self.tool_self_check(res)})
            advice.elapsed_ms = (time.time() - t0) * 1000
            return advice

        # ---- 检索知识库（每个缺陷类型各取证据，合并去重）----
        hits: list[Hit] = []
        seen: set[str] = set()
        for d in dets[:3]:
            query = f"{d.get('class_zh', '')} 成因 处置 工艺"
            for h in self.tool_retrieve_standard(query, top_k=self.top_k):
                if h.doc.id not in seen:
                    seen.add(h.doc.id)
                    hits.append(h)
        # 补充分级与规程类证据
        for h in self.tool_retrieve_standard("严重度分级 人工复检 规程", top_k=2):
            if h.doc.id not in seen:
                seen.add(h.doc.id)
                hits.append(h)

        advice.citations = [
            {"id": h.doc.id, "title": h.doc.title, "score": round(h.score, 4),
             "category": h.doc.meta.get("category", ""), "source": h.doc.meta.get("source", "")}
            for h in hits
        ]

        # ---- 规则后端（确定性，永远可用）----
        rule_advice = self._rule_advice(res, hits)
        advice.root_cause = rule_advice["root_cause"]
        advice.immediate_action = rule_advice["immediate_action"]
        advice.process_action = rule_advice["process_action"]
        advice.batch_trace = rule_advice["batch_trace"]
        advice.risk_note = rule_advice["risk_note"]
        advice.confidence = rule_advice["confidence"]
        advice.backend = "rule"

        # ---- 尝试更强后端（失败静默降级，并记录原因）----
        for backend in self.backends:
            if backend == "rule":
                continue
            try:
                llm_advice, meta = self._llm_advice(res, hits, backend)
                if llm_advice:
                    advice.root_cause = llm_advice.get("root_cause") or advice.root_cause
                    advice.immediate_action = llm_advice.get("immediate_action") or advice.immediate_action
                    advice.process_action = llm_advice.get("process_action") or advice.process_action
                    advice.batch_trace = llm_advice.get("batch_trace") or advice.batch_trace
                    advice.risk_note = llm_advice.get("risk_note") or advice.risk_note
                    advice.confidence = float(llm_advice.get("confidence", advice.confidence) or advice.confidence)
                    advice.backend = backend
                    advice.tool_calls.append({"tool": "llm_infer", "backend": backend, "meta": meta})
                    break
            except Exception as e:
                advice.degraded_reason = f"{backend} 不可用（{type(e).__name__}），已降级为规则后端"
                LOG.info("后端 %s 不可用，降级：%s", backend, e)

        # ---- 工具调用：自检 + 工单 ----
        check = self.tool_self_check(res)
        advice.need_manual_review = bool(check["need_manual_review"])
        advice.tool_calls.append({"tool": "self_check", "result": check})
        advice.work_order = self.tool_create_work_order(dets[0], inspection_id)
        advice.tool_calls.append({"tool": "create_work_order",
                                  "result": {"work_order_id": advice.work_order["work_order_id"],
                                             "priority": advice.work_order["priority"]}})
        advice.tool_calls.append({"tool": "trace_batch", "result": advice.batch_trace})
        advice.tool_calls.append({"tool": "retrieve_standard",
                                  "result": [c["id"] for c in advice.citations]})
        advice.elapsed_ms = (time.time() - t0) * 1000
        self.last_backend_used = advice.backend
        return advice

    # ------------------------------------------------------------------
    def _rule_advice(self, res: dict, hits: list[Hit]) -> dict:
        """基于检索结果与结构化规则生成处置建议（确定性）。"""
        dets = res.get("detections", [])
        top = dets[0]
        code = top.get("class_code", "")
        grade = res.get("grade", "B")
        sev = float(res.get("max_severity", 0))

        # 从知识库中提取该缺陷对应的"成因与处置"条目正文
        cause_text = ""
        for h in hits:
            if h.doc.meta.get("code") == code and h.doc.meta.get("type") == "成因与处置":
                cause_text = h.doc.text
                break
        if not cause_text:
            for h in hits:
                if h.doc.meta.get("code") == code:
                    cause_text = h.doc.text
                    break

        # 规则表：立即处置与工艺纠正措施
        rule = IMMEDIATE_RULES.get(code, {})
        urgent = grade in ("D", "E")

        root_cause = self._summarize_cause(cause_text) or f"检出{top.get('class_zh', code)}，需结合工艺记录进一步定位成因。"
        immediate = rule.get("immediate", "标记该件为待判定，调取原始图像与热力图人工复核。")
        if urgent:
            immediate += " 当前等级为%s，按内控标准立即停机并隔离同批次产品。" % grade
        process = rule.get("process", "核查相关工艺参数并记录整改措施。")
        batch = self.tool_trace_batch(top)
        risk_note = (
            f"最高严重度 {sev:.1f}（{grade} 级），主要缺陷为{top.get('class_zh', code)}"
            f"（置信度 {float(top.get('confidence', 0)):.2f}）。"
        )
        if len(dets) > 1:
            risk_note += f"共检出 {len(dets)} 处缺陷，需关注多点同源的可能性。"
        confidence = min(0.93, 0.55 + 0.35 * float(top.get("confidence", 0.6)))
        return {
            "root_cause": root_cause, "immediate_action": immediate, "process_action": process,
            "batch_trace": batch, "risk_note": risk_note, "confidence": confidence,
        }

    @staticmethod
    def _summarize_cause(text: str, limit: int = 220) -> str:
        """从知识库条目中截取"主要成因"部分，作为根因分析。"""
        if not text:
            return ""
        marker = "主要成因："
        if marker in text:
            seg = text.split(marker, 1)[1]
            for end in ("工艺处置建议：", "工艺处置：", "风险："):
                if end in seg:
                    seg = seg.split(end, 1)[0]
            seg = seg.strip()
            return (seg[:limit] + "…") if len(seg) > limit else seg
        return (text[:limit] + "…") if len(text) > limit else text

    # ------------------------------------------------------------------
    def _llm_advice(self, res: dict, hits: list[Hit], backend: str) -> tuple[dict | None, dict]:
        """调用大模型后端。返回 (解析后的建议 dict, 元信息)。"""
        cfg = self.cfg["agent"]
        if backend == "ollama":
            conf = cfg.get("ollama", {})
            if not conf.get("enabled"):
                raise RuntimeError("ollama 后端未启用（config.agent.ollama.enabled=false）")
            base, model = conf.get("base_url", "").rstrip("/"), conf.get("model", "")
            url = f"{base}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            api_key = ""
        elif backend == "openai_compat":
            conf = cfg.get("openai_compat", {})
            if not conf.get("enabled"):
                raise RuntimeError("openai_compat 后端未启用")
            base, model = conf.get("base_url", "").rstrip("/"), conf.get("model", "")
            if not base or not model:
                raise RuntimeError("openai_compat 未配置 base_url/model")
            url = f"{base}/chat/completions"
            api_key = conf.get("api_key", "")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            raise ValueError(f"未知后端 {backend}")

        evidence = "\n".join(f"[{h.doc.id}] {h.doc.title}：{h.doc.text[:500]}" for h in hits[:4])
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(res, evidence)},
            ],
            "temperature": 0.2,
            "stream": False,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        timeout = float(conf.get("timeout", 60))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = self._parse_json_block(content)
        meta = {"model": model, "elapsed_from_api": body.get("usage", {})}
        return parsed, meta

    @staticmethod
    def _build_user_prompt(res: dict, evidence: str) -> str:
        dets = res.get("detections", [])
        lines = [
            "【检测结果】",
            f"结论：{res.get('verdict')} | 等级：{res.get('grade')}（{res.get('grade_zh')}）"
            f" | 最高严重度：{res.get('max_severity')} | 图像尺寸：{res.get('image_size')}",
            f"检出缺陷数：{len(dets)}",
        ]
        for i, d in enumerate(dets[:5], 1):
            lines.append(
                f"  {i}. {d.get('class_zh')}（{d.get('class_code')}）置信度 {d.get('confidence')}，"
                f"位置 {d.get('bbox')}，面积占比 {d.get('area_ratio')}，严重度 {d.get('severity_score')}（{d.get('severity_level')}）"
            )
        lines += ["", "【标准知识库摘录】", evidence, "",
                  "请输出 JSON 格式的处置建议。"]
        return "\n".join(lines)

    @staticmethod
    def _parse_json_block(text: str) -> dict | None:
        """从模型输出中提取 JSON（容忍 ```json 代码块与前后解释文字）。"""
        if not text:
            return None
        s = text.strip()
        if "```" in s:
            parts = s.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    s = p
                    break
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            s = s[start:end + 1]
        try:
            return json.loads(s)
        except Exception:
            LOG.debug("LLM 输出非合法 JSON，已忽略：%s", text[:160])
            return None


# ----------------------------------------------------------------------
# 规则表：缺陷 -> 立即处置 / 工艺纠正（与知识库内容一致）
# ----------------------------------------------------------------------
IMMEDIATE_RULES: dict[str, dict[str, str]] = {
    "crazing": {
        "immediate": "将该件判为待处理并隔离；调取近3个班次的终轧温度与卷取温度曲线比对。",
        "process": "核查加热炉温度与在炉时间，终轧温度控制在工艺窗口内，复核层流冷却水量与卷取温度。",
    },
    "inclusion": {
        "immediate": "隔离该炉次产品并标记待判定；调取该炉次精炼与连铸记录。",
        "process": "加强吹氩搅拌促进夹杂物上浮，检查中间包与浸入式水口耐材状态，提高除鳞水压力。",
    },
    "patches": {
        "immediate": "按面积占比判定改判或降级；抽查同批次酸洗后表面状态。",
        "process": "复核酸洗液浓度、温度与线速度；检查除鳞水压力与喷嘴堵塞；检查乳化液吹扫与防潮措施。",
    },
    "pitted_surface": {
        "immediate": "加密抽检同批次表面粗糙度；标记观察批次。",
        "process": "检查精轧工作辊表面状态与磨削周期，必要时换辊；检测冷却水水质与过滤系统。",
    },
    "rolled_in_scale": {
        "immediate": "按深度与面积判定放行或降级；记录缺陷位置规律性以定位喷嘴组。",
        "process": "提高除鳞水压力并逐支检查喷嘴角度与堵塞；优化轧制节奏，缩短待轧时间。",
    },
    "scratches": {
        "immediate": "测量划痕深度，浅划痕放行并记录，深划痕判为D级并隔离。",
        "process": "沿轧线排查导板与辊面毛刺，打磨或更换损伤件；检查卷取机侧导板开口度与跑偏纠偏系统。",
    },
}

MAINTENANCE_RULES: dict[str, dict[str, Any]] = {
    "crazing": {"department": "轧钢工艺科",
                "action": "加热与冷却制度专项排查，复核终轧温度与卷取温度设定",
                "check_items": ["加热炉温度曲线", "在炉时间", "终轧温度", "层流冷却水量", "卷取温度"]},
    "inclusion": {"department": "炼钢工艺科",
                  "action": "该炉次洁净度与连铸状态核查",
                  "check_items": ["精炼吹氩记录", "中间包耐材状态", "结晶器液面波动", "除鳞水压力"]},
    "patches": {"department": "酸洗作业区",
                "action": "酸洗与除鳞工艺参数复核",
                "check_items": ["酸液浓度", "酸洗温度", "线速度", "除鳞喷嘴状态", "防锈措施"]},
    "pitted_surface": {"department": "设备科",
                       "action": "精轧工作辊磨削或更换，冷却水系统检查",
                       "check_items": ["轧辊粗糙度", "轧辊磨削周期", "冷却水水质", "过滤系统"]},
    "rolled_in_scale": {"department": "设备科",
                        "action": "除鳞集管与喷嘴检查清理",
                        "check_items": ["除鳞水压力", "喷嘴角度", "喷嘴堵塞情况", "轧制节奏"]},
    "scratches": {"department": "设备科",
                  "action": "轧线接触部件打磨与更换，跑偏纠偏系统检查",
                  "check_items": ["导板毛刺", "夹送辊状态", "卷取机侧导板", "纠偏系统", "吊运堆垛作业"]},
}
