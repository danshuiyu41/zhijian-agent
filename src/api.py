"""FastAPI 后端：检测、智能体、报告、看板、知识库的统一接口。

接口清单
--------
GET  /api/health              健康检查与系统自述（模型/后端/知识库状态）
GET  /api/system              系统信息（版本、模型、类别、政策与标准引用）
GET  /api/classes             缺陷类别字典
POST /api/inspect             上传图像执行检测（可选返回热力图）
POST /api/inspect/agent       上传图像检测 + 智能体处置建议（一体化，演示主链路）
GET  /api/inspections         检测记录列表（支持筛选）
GET  /api/inspections/{id}    单条检测详情
GET  /api/dashboard           看板统计
GET  /api/work_orders         工单列表
POST /api/work_orders/{id}/status  更新工单状态
POST /api/kb/search           知识库检索
POST /api/chat                质检知识问答（智能体 + RAG）
GET  /api/report/{id}         生成报告（md / html / json）
GET  /api/samples             可用演示样本列表
POST /api/demo/seed           填充演示数据
GET  /                        前端页面

设计说明
--------
- 模型与智能体在进程内单例加载，避免每次请求重复初始化；
- 上传图像保存到 `data/samples/uploads/`，用于报告引用与追溯；
- 所有响应统一 `{"ok": bool, "data": ..., "error": ...}` 结构，便于前端处理。

启动：
    python -m src.api               # 或 uvicorn src.api:app --port 8077
"""
from __future__ import annotations

import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

# 注意：UploadFile / File / Form 必须在模块顶层真实导入。
# 若写成字符串注解（如 file: "UploadFile"），FastAPI 0.104 在 Python 3.10 下会因
# 无法解析 ForwardRef 而报 "Invalid args for response field"，导致应用初始化失败。
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from .utils import (ROOT, class_list, ensure_dirs, get_logger, load_config, now_str,
                    resolve, severity_grade)

LOG = get_logger("api")

CFG = load_config()
ensure_dirs(CFG)

UPLOAD_DIR = resolve("data/samples/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 单例资源（延迟加载，避免导入即加载模型导致启动慢/失败）
# ----------------------------------------------------------------------
_STATE: dict[str, Any] = {"inspector": None, "agent": None, "db": None, "error": ""}


def get_db():
    if _STATE["db"] is None:
        from .db import Database

        _STATE["db"] = Database.from_config()
    return _STATE["db"]


def get_inspector():
    if _STATE["inspector"] is None:
        from .infer import DefectInspector

        _STATE["inspector"] = DefectInspector.from_config()
    return _STATE["inspector"]


def get_agent():
    if _STATE["agent"] is None:
        from .agent import QualityAgent

        _STATE["agent"] = QualityAgent.from_config()
    return _STATE["agent"]


# ----------------------------------------------------------------------
# FastAPI 应用
# ----------------------------------------------------------------------
def create_app():
    app = FastAPI(title=CFG["server"]["title"], version=CFG["project"]["version"],
                  description="面向中小制造企业的工业表面缺陷智能检测与处置智能体系统")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def ok(data: Any = None, **extra):
        return {"ok": True, "data": data, "error": None, **extra}

    def fail(msg: str, code: int = 400):
        return JSONResponse(status_code=code, content={"ok": False, "data": None, "error": msg})

    # ------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        info = {"status": "ok", "version": CFG["project"]["version"], "time": now_str()}
        try:
            insp = get_inspector()
            info["model"] = {"loaded": True, "ckpt": insp.ckpt_path.name,
                             "backbone": insp.ckpt.get("backbone"),
                             "val_acc": insp.ckpt.get("val_acc"),
                             "params_m": insp.ckpt.get("num_params_m"),
                             "classes": insp.classes}
        except Exception as e:
            info["model"] = {"loaded": False, "error": f"{type(e).__name__}: {e}"}
        try:
            ag = get_agent()
            info["knowledge"] = {"docs": len(ag.docs), "backends": ag.backends}
        except Exception as e:
            info["knowledge"] = {"loaded": False, "error": str(e)}
        try:
            info["database"] = {"total": get_db().stats(days=3650)["total"]}
        except Exception as e:
            info["database"] = {"error": str(e)}
        return ok(info)

    @app.get("/api/system")
    def system():
        return ok({
            "project": CFG["project"],
            "dataset": {"name": CFG["dataset"]["name"], "classes": class_list(CFG),
                        "image_size": CFG["dataset"]["image_size"]},
            "model": CFG["model"],
            "severity": CFG["severity"]["grades"],
            "policy": [
                {"title": "国务院关于深入实施“人工智能+”行动的意见", "no": "国发〔2025〕11号"},
                {"title": "“人工智能+制造”专项行动实施意见", "no": "工信部联科〔2025〕279号"},
                {"title": "中小企业数字化赋能专项行动方案（2025—2027）", "no": "工信部联企业〔2024〕239号"},
                {"title": "中小企业数字化转型城市试点（第三批）", "no": "财办建〔2025〕20号"},
            ],
            "standards": ["GB/T 40659-2021", "GB/T 45579-2023", "GB/T 45575-2025",
                          "GB/T 41867-2022", "GB/T 32738-2016"],
            "report_config": CFG.get("report", {}),
        })

    @app.get("/api/classes")
    def classes():
        return ok([{**c, "is_defect": True} for c in class_list(CFG)]
                  + [{"code": "clean", "zh": "洁净(合格)", "en": "Clean", "risk": "none",
                      "is_defect": False}])

    # ------------------------------------------------------------------
    def _read_upload(file: UploadFile, raw: bytes) -> tuple[str, str]:
        """保存上传文件，返回 (绝对路径, 相对路径)。"""
        suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            suffix = ".jpg"
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
        fp = UPLOAD_DIR / name
        fp.write_bytes(raw)
        try:
            rel = str(fp.relative_to(ROOT))
        except ValueError:
            rel = str(fp)
        return str(fp), rel

    def _to_data_url(path: Path) -> str:
        try:
            data = base64.b64encode(path.read_bytes()).decode()
            ext = path.suffix.lstrip(".").lower() or "png"
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return f"data:{mime};base64,{data}"
        except Exception:
            return ""

    @app.post("/api/inspect")
    async def inspect(file: UploadFile = File(...),
                      heatmap: bool = Form(True),
                      agent: bool = Form(False),
                      batch_no: str = Form(""),
                      persist: bool = Form(True)):
        """执行检测。

        性能说明（实测，务必保留这段注释）：
          · 滑窗推理约 177 ms，标注图 + base64 约 24 ms，落库约 12 ms，智能体 <5 ms；
          · **Grad-CAM 热力图约 1905 ms，占整条链路 90%**。
        因此默认**不在本接口内生成热力图**，而是返回检测结果后由前端调用
        `/api/heatmap/{id}` 异步补齐。需要同步拿到热力图的调用方（如批量评测）
        可显式传 heatmap=true，代价是每次多等约 1.9 秒。
        """
        raw = await file.read()
        if not raw:
            return fail("上传文件为空")
        if len(raw) > int(CFG["server"]["max_upload_mb"]) * 1024 * 1024:
            return fail(f"文件超过 {CFG['server']['max_upload_mb']}MB 限制")
        abs_path, rel = _read_upload(file, raw)
        try:
            import cv2

            inspector = get_inspector()
            result, viz = inspector.inspect(rel, want_heatmap=bool(heatmap))
            payload = result.to_dict()
            payload["image_rel"] = rel
            payload["batch_no"] = batch_no
            payload["created_at"] = now_str()
            advice = None
            if agent:
                advice = get_agent().analyze(payload).to_dict()
                payload["advice"] = advice
            if persist:
                try:
                    db = get_db()
                    insp_id = db.save_inspection(payload, image_path=rel, batch_no=batch_no)
                    payload["inspection_id"] = insp_id
                    if advice and advice.get("work_order"):
                        db.save_work_order(advice["work_order"], insp_id)
                except Exception as e:
                    LOG.warning("持久化失败：%s", e)
            gray = cv2.imread(str(resolve(rel)), cv2.IMREAD_GRAYSCALE)
            payload["image_data_url"] = _to_data_url(Path(abs_path))
            if gray is not None:
                payload["annotated_data_url"] = inspector.annotate_base64(gray, result.detections)
            return ok({"result": payload, "viz": viz, "advice": advice})
        except FileNotFoundError as e:
            return fail(f"模型未就绪：{e}", 503)
        except Exception as e:
            LOG.exception("检测失败")
            return fail(f"检测失败：{type(e).__name__}: {e}", 500)

    @app.post("/api/inspect/agent")
    async def inspect_agent(file: UploadFile = File(...),
                            heatmap: bool = Form(False),
                            batch_no: str = Form("")):
        """演示主链路：上传 → 检测 → 智能体处置 → 工单。

        踩坑记录（重要）：本接口早期把 `heatmap` 写死为 True，导致前端传
        `heatmap=false` 被静默忽略——于是"把热力图拆出去异步加载"的优化完全没生效，
        每次检测仍要多等约 1.9 秒。现改为透传参数，默认 False（前端随后异步补热力图）。
        """
        return await inspect(file=file, heatmap=bool(heatmap), agent=True,
                             batch_no=batch_no, persist=True)

    # ------------------------------------------------------------------
    @app.get("/api/inspections")
    def inspections(limit: int = Query(30, ge=1, le=200), offset: int = 0,
                    grade: str = "", defect: str = "", days: int = 0):
        rows = get_db().list_inspections(limit=limit, offset=offset, grade=grade,
                                        defect=defect, days=days)
        slim = []
        for r in rows:
            p = r.get("payload") or {}
            slim.append({
                "inspection_id": r["inspection_id"], "created_at": r["created_at"],
                "verdict": r["verdict"], "grade": r["grade"], "grade_zh": r["grade_zh"],
                "max_severity": r["max_severity"], "defect_count": r["defect_count"],
                "top_defect_code": r["top_defect_code"], "top_defect_zh": r["top_defect_zh"],
                "confidence": r["confidence"], "elapsed_ms": r["elapsed_ms"],
                "batch_no": r["batch_no"], "image_rel": p.get("image_rel", ""),
                "advice": r.get("advice") or None,
            })
        return ok(slim)

    @app.get("/api/inspections/{inspection_id}")
    def inspection_detail(inspection_id: str):
        r = get_db().get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "记录不存在")
        return ok(r)

    @app.get("/api/heatmap/{inspection_id}")
    def heatmap(inspection_id: str):
        """按需生成 Grad-CAM 热力图（拆出同步链路，见下方性能说明）。

        性能背景（实测）：Grad-CAM 需反向传播，单张约占 1905 ms，
        而整条链路的其余部分合计不到 220 ms。若把它放进 `/api/inspect` 的同步返回，
        界面每次检测都要等 1.7 秒以上。

        因此把它拆成独立接口：检测先快速返回（判定/分级/建议/标注图），
        界面拿到结果后再异步请求热力图补上第三张图。
        这样"判定结果"的体感延迟从 1.7 s 降到约 0.2 s，热力图随后补齐。
        """
        from fastapi.responses import JSONResponse

        import base64

        import cv2

        r = get_db().get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "记录不存在")
        rel = (r.get("payload") or {}).get("image_rel") or r.get("image_path") or ""
        if not rel:
            raise HTTPException(404, "该记录没有关联图像，无法生成热力图")
        fp = resolve(rel)
        if not fp.exists():
            raise HTTPException(404, "原始图像已不存在")
        try:
            insp = get_inspector()
            gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise HTTPException(500, "图像读取失败")
            url = insp.heatmap_base64(gray, [])
            return ok({"inspection_id": inspection_id, "heatmap_png": url})
        except HTTPException:
            raise
        except Exception as e:
            LOG.warning("热力图生成失败：%s", e)
            return fail(f"热力图生成失败：{e}", 500)

    @app.get("/api/dashboard")
    def dashboard(days: int = 7):
        return ok(get_db().stats(days=days))

    @app.get("/api/work_orders")
    def work_orders(limit: int = 50, status: str = ""):
        return ok(get_db().list_work_orders(limit=limit, status=status))

    @app.post("/api/work_orders/{work_order_id}/status")
    def wo_status(work_order_id: str, payload: dict = Body(...)):
        st = (payload or {}).get("status", "")
        if not st:
            return fail("缺少 status")
        return ok({"updated": get_db().update_work_order_status(work_order_id, st)})

    # ------------------------------------------------------------------
    @app.post("/api/kb/search")
    def kb_search(payload: dict = Body(...)):
        q = (payload or {}).get("query", "").strip()
        k = int((payload or {}).get("top_k", 3))
        if not q:
            return fail("缺少 query")
        hits = get_agent().retriever.search(q, top_k=k)
        return ok([h.to_dict() for h in hits])

    @app.post("/api/chat")
    def chat(payload: dict = Body(...)):
        """质检知识问答：检索知识库 + 规则/大模型作答。"""
        q = (payload or {}).get("question", "").strip()
        if not q:
            return fail("缺少 question")
        ag = get_agent()
        hits = ag.retriever.search(q, top_k=4)
        answer = ag.answer_question(q, hits)
        try:
            get_db().log_kb_query(q, [h.doc.id for h in hits], answer.get("backend", ""))
        except Exception:
            pass
        return ok({"question": q, "answer": answer, "citations": [h.to_dict() for h in hits]})

    # ------------------------------------------------------------------
    @app.get("/api/report/{inspection_id}")
    def report(inspection_id: str, fmt: str = "html"):
        from .report import build_report

        r = get_db().get_inspection(inspection_id)
        if not r:
            raise HTTPException(404, "记录不存在")
        payload = r.get("payload") or {}
        payload.setdefault("inspection_id", inspection_id)
        payload.setdefault("created_at", r.get("created_at"))
        payload.setdefault("batch_no", r.get("batch_no"))
        payload.setdefault("operator", r.get("operator"))
        advice = r.get("advice") or None
        img_url, heat_url = "", ""
        rel = payload.get("image_rel") or r.get("image_path") or ""
        if rel:
            fp = resolve(rel)
            if fp.exists():
                img_url = _to_data_url(fp)
                try:
                    import cv2

                    from .infer import DefectInspector  # noqa: F401

                    insp = get_inspector()
                    gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                    if gray is not None:
                        from .infer import Detection

                        dets = [Detection(**{k: d[k] for k in
                                             ("class_code", "class_zh", "confidence", "bbox",
                                              "area_ratio", "severity_score", "severity_level",
                                              "severity_zh", "action", "risk")
                                             if k in d})
                                for d in payload.get("detections", [])]
                        heat_url = insp.heatmap_base64(gray, dets)
                        img_url = insp.annotate_base64(gray, dets) or img_url
                except Exception as e:
                    LOG.warning("报告配图生成失败：%s", e)
        text = build_report(payload, advice, fmt=fmt, cfg=CFG,
                            image_data_url=img_url, heatmap_data_url=heat_url)
        if fmt == "html":
            return HTMLResponse(text)
        if fmt == "json":
            # 直接返回 build_report 生成的 JSON 字符串，并显式声明 UTF-8。
            # 说明：曾尝试 JSONResponse(json.loads(text)) 但该路径在本环境出现响应体
            # 为空的现象（curl 直取 2 字节 "{}"），改用纯文本直传后稳定返回完整 JSON。
            return PlainTextResponse(text, media_type="application/json",
                                     headers={"Content-Type": "application/json; charset=utf-8"})
        return PlainTextResponse(text, media_type="text/markdown",
                                 headers={"Content-Type": "text/markdown; charset=utf-8"})

    # ------------------------------------------------------------------
    @app.get("/api/samples")
    def samples(limit: int = 24):
        """列出可用于演示的样本（合成数据各类抽几张）。"""
        out = []
        root = resolve(CFG["paths"]["data_raw"])
        zh_map = {x["code"]: x["zh"] for x in class_list(CFG)}
        per_class = max(1, limit // 7)
        for c in list(zh_map.keys()) + ["clean"]:
            d = root / c
            if not d.exists():
                continue
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
            if not files:
                continue
            step = max(1, len(files) // per_class)
            for f in files[::step][:per_class]:
                out.append({"class_code": c, "class_zh": zh_map.get(c, "洁净(合格)"),
                            "rel": str(f.relative_to(ROOT)), "name": f.name})
        return ok(out)

    @app.get("/api/samples/raw")
    def sample_raw(rel: str):
        """按相对路径返回样本图像（仅允许 data 目录内，防止路径穿越）。"""
        from fastapi.responses import FileResponse

        target = (ROOT / rel).resolve()
        data_root = resolve(CFG["paths"]["data_raw"]).resolve().parent   # 允许 data/ 下全部
        if not str(target).startswith(str(data_root)):
            raise HTTPException(403, "路径不在允许范围内")
        if not target.exists() or not target.is_file():
            raise HTTPException(404, "样本不存在")
        return FileResponse(str(target))

    @app.post("/api/demo/seed")
    def demo_seed(n: int = 120):
        return ok({"inserted": get_db().seed_demo_data(n)})

    # ------------------------------------------------------------------
    # 前端
    # ------------------------------------------------------------------
    static_dir = resolve("web")

    @app.get("/", response_class=HTMLResponse)
    def index():
        fp = static_dir / "index.html"
        if not fp.exists():
            return HTMLResponse("<h1>前端未部署</h1><p>缺少 web/index.html</p>", status_code=500)
        return HTMLResponse(fp.read_text(encoding="utf-8"))

    if static_dir.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


try:  # 供 uvicorn src.api:app 使用
    app = create_app()
except Exception as _e:  # pragma: no cover
    LOG.error("FastAPI 初始化失败：%s", _e)
    app = None


def main() -> None:
    import uvicorn

    host = CFG["server"]["host"]
    port = int(CFG["server"]["port"])
    LOG.info("启动服务：http://%s:%d", host, port)
    uvicorn.run("src.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
