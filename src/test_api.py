"""API 端到端测试：启动服务后逐个验证接口。

覆盖：健康检查、系统信息、类别、样本、检测（含智能体）、记录、详情、看板、
工单、工单状态更新、知识库检索、问答、报告三种格式。

用法：
    python -m src.test_api
    python -m src.test_api --base http://127.0.0.1:8077
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = "http://127.0.0.1:8077"
PASS, FAIL = "  [通过]", "  [失败]"


def get(path: str, timeout: int = 120):
    """GET 请求。

    返回 (status, payload)：
      - Content-Type 含 application/json 时，payload 为**已解析的 Python 对象**；
      - 其余情况为解码后的字符串。

    踩坑记录：早期版本在 `with` 块外访问 `r.status`（恒为 None），且第 11 项测试
    对"已解析的 dict"误用 `len()` 判断长度（dict 只有 2 个键，被误读成响应体 2 字节），
    导致 JSON 报告接口被长期误报为失败。此处显式区分类型，调用方需按类型处理。
    """
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "zhijian-selftest"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        ctype = r.headers.get("Content-Type", "") or ""
        raw = r.read()
    if "application/json" in ctype:
        try:
            return status, json.loads(raw.decode("utf-8"))
        except Exception:
            return status, raw.decode("utf-8", "ignore")
    return status, raw.decode("utf-8", "ignore")


def post_json(path: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post_file(path: str, file_path: str, fields: dict | None = None, timeout: int = 300):
    """构造 multipart/form-data 上传。"""
    boundary = "----zhijian" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in (fields or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    name = file_path.replace("\\", "/").split("/")[-1]
    with open(file_path, "rb") as f:
        content = f.read()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main() -> None:
    global BASE
    ap = argparse.ArgumentParser(description="API 端到端测试")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--sample", default=None, help="用于检测的图片路径，默认自动取一个龟裂样本")
    args = ap.parse_args()
    BASE = args.base.rstrip("/")

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{PASS if cond else FAIL} {name}" + (f"　{detail}" if detail else ""))
        if not cond:
            failures.append(name)

    sample = args.sample
    if not sample:
        import glob

        files = sorted(glob.glob("data/raw/crazing/*.png"))
        if not files:
            raise SystemExit("未找到测试样本，请先生成数据集")
        sample = files[0]

    print("=" * 78)
    print(f"API 端到端测试　目标：{BASE}")
    print(f"测试样本：{sample}")
    print("=" * 78)

    # 1 健康检查
    try:
        st, j = get("/api/health")
        d = j.get("data", {})
        check("GET /api/health", j.get("ok") and d.get("model", {}).get("loaded"),
              f"版本 {d.get('version')} ｜ 模型 {d.get('model', {}).get('ckpt')} "
              f"｜ 知识库 {d.get('knowledge', {}).get('docs')} 条 "
              f"｜ 历史 {d.get('database', {}).get('total')} 条")
    except Exception as e:
        check("GET /api/health", False, str(e))

    # 2 系统信息
    try:
        st, j = get("/api/system")
        d = j.get("data", {})
        check("GET /api/system", j.get("ok") and len(d.get("policy", [])) >= 4,
              f"赛道 {d.get('project', {}).get('track', '')[:24]} ｜ 标准 {len(d.get('standards', []))} 项")
    except Exception as e:
        check("GET /api/system", False, str(e))

    # 3 类别
    try:
        st, j = get("/api/classes")
        check("GET /api/classes", j.get("ok") and len(j["data"]) == 7, f"{len(j.get('data', []))} 类")
    except Exception as e:
        check("GET /api/classes", False, str(e))

    # 4 样本列表与原图
    try:
        st, j = get("/api/samples")
        items = j.get("data", [])
        check("GET /api/samples", j.get("ok") and len(items) > 0, f"{len(items)} 个样本")
        if items:
            rel = urllib.parse.quote(items[0]["rel"])
            st2, raw = get(f"/api/samples/raw?rel={rel}")
            check("GET /api/samples/raw", st2 == 200 and len(raw) > 100, f"{len(raw)} 字符")
    except Exception as e:
        check("GET /api/samples", False, str(e))

    # 5 检测 + 智能体
    insp_id = None
    try:
        st, j = post_file("/api/inspect/agent", sample,
                          fields={"batch_no": "API-TEST-001"})
        d = j.get("data", {})
        res = d.get("result", {})
        adv = d.get("advice") or {}
        viz = d.get("viz") or {}
        insp_id = res.get("inspection_id")
        check("POST /api/inspect/agent", j.get("ok") and res.get("grade"),
              f"编号 {insp_id} ｜ 等级 {res.get('grade')} ｜ 缺陷 {len(res.get('detections', []))} 处 "
              f"｜ {res.get('elapsed_ms')} ms")
        check("　└ 返回标注图", bool(res.get("annotated_data_url")),
              f"标注图 {len(res.get('annotated_data_url') or '')} 字符")
        # 热力图已拆到独立接口异步加载（Grad-CAM 实测约 1.9 s，占整条链路 90%），
        # 因此同步响应里**不应**再有 heatmap_png —— 这是有意的性能取舍，不是缺陷。
        check("　└ 同步响应不含热力图（已拆为异步）", not viz.get("heatmap_png"),
              "符合预期：热力图改走 /api/heatmap/{id}")
        if insp_id:
            try:
                st, hj = get(f"/api/heatmap/{insp_id}", timeout=120)
                payload_h = (hj or {}).get("data") or {}
                has = bool(payload_h.get("heatmap_png"))
                check("GET /api/heatmap/{id}", st == 200 and has,
                      f"热力图 {len(payload_h.get('heatmap_png') or '')} 字符")
            except Exception as e:
                check("GET /api/heatmap/{id}", False, str(e))
        check("　└ 智能体建议", bool(adv.get("root_cause")) and bool(adv.get("process_action")),
              f"后端 {adv.get('backend')} ｜ 引用 {len(adv.get('citations', []))} 条")
        check("　└ 自动工单", bool((adv.get("work_order") or {}).get("work_order_id")),
              (adv.get("work_order") or {}).get("work_order_id", ""))
    except Exception as e:
        check("POST /api/inspect/agent", False, str(e))

    # 6 记录与详情
    try:
        st, j = get("/api/inspections?limit=5")
        rows = j.get("data", [])
        check("GET /api/inspections", j.get("ok") and len(rows) > 0, f"{len(rows)} 条")
        if insp_id:
            st, j = get(f"/api/inspections/{insp_id}")
            check("GET /api/inspections/{id}", j.get("ok") and j["data"]["inspection_id"] == insp_id)
    except Exception as e:
        check("GET /api/inspections", False, str(e))

    # 7 看板
    try:
        st, j = get("/api/dashboard?days=365")
        d = j.get("data", {})
        check("GET /api/dashboard", j.get("ok") and "pass_rate" in d,
              f"总量 {d.get('total')} ｜ 合格率 {d.get('pass_rate')} ｜ 平均节拍 {d.get('avg_elapsed_ms')} ms")
    except Exception as e:
        check("GET /api/dashboard", False, str(e))

    # 8 工单
    wo_id = None
    try:
        st, j = get("/api/work_orders?limit=5")
        rows = j.get("data", [])
        check("GET /api/work_orders", j.get("ok"), f"{len(rows)} 条")
        if rows:
            wo_id = rows[0]["work_order_id"]
            st, j = post_json(f"/api/work_orders/{wo_id}/status", {"status": "已完成"})
            check("POST /api/work_orders/{id}/status", j.get("ok") and j["data"]["updated"], wo_id)
    except Exception as e:
        check("GET /api/work_orders", False, str(e))

    # 9 知识库检索
    try:
        st, j = post_json("/api/kb/search", {"query": "龟裂 成因 处置", "top_k": 3})
        hits = j.get("data", [])
        check("POST /api/kb/search", j.get("ok") and hits and hits[0]["id"] == "DEF-CRAZING",
              f"Top1 {hits[0]['id'] if hits else '无'} ｜ 分数 {hits[0]['score'] if hits else '-'}")
    except Exception as e:
        check("POST /api/kb/search", False, str(e))

    # 10 问答
    try:
        st, j = post_json("/api/chat", {"question": "C级缺陷需要人工复检吗？"})
        d = j.get("data", {})
        a = d.get("answer", {})
        check("POST /api/chat", j.get("ok") and len(a.get("answer", "")) > 20,
              f"后端 {a.get('backend')} ｜ 引用 {len(d.get('citations', []))} 条 ｜ {a.get('elapsed_ms')} ms")
    except Exception as e:
        check("POST /api/chat", False, str(e))

    # 11 报告三种格式
    if insp_id:
        for fmt, marker in (("html", "<!DOCTYPE html"), ("md", "检测报告"), ("json", "inspection")):
            try:
                st, raw = get(f"/api/report/{insp_id}?fmt={fmt}")
                if fmt == "json":
                    # JSON 会被解析为 dict，按对象判定
                    ok = st == 200 and isinstance(raw, dict) and "inspection" in raw and "advice" in raw
                    detail = f"对象键 {list(raw)[:3]}" if isinstance(raw, dict) else f"类型 {type(raw).__name__}"
                else:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "ignore")
                    ok = st == 200 and marker in raw and len(raw) > 200
                    detail = f"{len(raw)} 字符"
                check(f"GET /api/report/{insp_id}?fmt={fmt}", ok, detail)
            except Exception as e:
                check(f"GET /api/report?fmt={fmt}", False, str(e))

    # 12 前端页面
    try:
        st, raw = get("/")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "ignore")
        check("GET / （前端页面）", st == 200 and "智检 Agent" in raw, f"{len(raw)} 字节")
    except Exception as e:
        check("GET /", False, str(e))

    print("=" * 78)
    if failures:
        print(f"未通过 {len(failures)} 项：" + "、".join(failures))
        sys.exit(1)
    print("全部接口测试通过 —— Web 端可演示。浏览器打开：" + BASE)
    print("=" * 78)


if __name__ == "__main__":
    main()
