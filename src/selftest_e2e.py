"""端到端自检：检测 → 分级 → 智能体 → 报告 → 数据库。

在不启动 Web 服务的情况下，用真实权重跑一遍完整链路，验证各模块协同正常。
这是提交前的最后一道自动检查，也是答辩前确认"现场能不能跑"的手段。

用法：
    python -m src.selftest_e2e
    python -m src.selftest_e2e --per-class 3      # 每类抽 3 张做指标复核
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

from .utils import class_list, get_logger, load_config, resolve  # noqa: E402

LOG = get_logger("e2e")
PASS, FAIL = "  [通过]", "  [失败]"


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端链路自检")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--per-class", type=int, default=2, help="每类抽检张数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{PASS if cond else FAIL} {name}" + (f"　{detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("=" * 70)
    print("智检 Agent 端到端链路自检")
    print("=" * 70)

    # ---------------- 1. 模型加载 ----------------
    t0 = time.time()
    from .infer import DefectInspector

    inspector = DefectInspector.from_config(args.config)
    check("模型加载", True,
          f"{inspector.ckpt_path.name} ｜ {inspector.ckpt.get('backbone')} ｜ "
          f"类别 {len(inspector.classes)} ｜ {time.time() - t0:.1f}s")

    # ---------------- 2. 知识库与智能体 ----------------
    from .agent import QualityAgent

    agent = QualityAgent.from_config(args.config)
    check("知识库加载", len(agent.docs) > 0, f"{len(agent.docs)} 条")
    hits = agent.retriever.search("龟裂 成因 处置", top_k=3)
    check("知识库检索", len(hits) > 0 and hits[0].doc.id == "DEF-CRAZING",
          f"Top1 = {hits[0].doc.id}（{hits[0].score:.3f}）" if hits else "无结果")

    # ---------------- 3. 逐类检测 ----------------
    print("-" * 70)
    print(f"{'真实类别':<16}{'检出等级':<8}{'主要缺陷':<14}{'置信度':<9}{'缺陷数':<7}{'耗时(ms)':<9}")
    print("-" * 70)
    root = resolve(cfg["paths"]["data_raw"])
    zh_map = {c["code"]: c["zh"] for c in class_list(cfg)}
    zh_map["clean"] = "洁净(合格)"

    n_ok = n_total = 0
    for code in [c["code"] for c in class_list(cfg)] + ["clean"]:
        d = root / code
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        if not files:
            continue
        step = max(1, len(files) // args.per_class)
        for f in files[::step][:args.per_class]:
            t = time.time()
            res, viz = inspector.inspect(f, want_heatmap=True)
            ms = (time.time() - t) * 1000
            top = res.detections[0] if res.detections else None
            print(f"{zh_map.get(code, code):<16}{res.grade:<8}"
                  f"{(top.class_zh if top else '—'):<14}"
                  f"{(f'{top.confidence:.3f}' if top else '—'):<9}"
                  f"{len(res.detections):<7}{ms:<9.0f}")
            n_total += 1
            expect_defect = code != "clean"
            got_defect = len(res.detections) > 0
            if expect_defect == got_defect:
                n_ok += 1
            # 顺带验证可视化产物
            if not viz.get("heatmap_png") or not viz.get("annotated_png"):
                check(f"可视化产物（{f.name}）", False, "热力图或标注图为空")
    print("-" * 70)
    check("逐类检出方向一致", n_ok == n_total, f"{n_ok}/{n_total} 张（是否有缺陷的判断与真实类别一致）")

    # ---------------- 4. 智能体端到端 ----------------
    print("-" * 70)
    defect_dir = root / "crazing"
    sample = sorted(defect_dir.iterdir())[0]
    res, viz = inspector.inspect(sample, want_heatmap=True)
    payload = res.to_dict()
    advice = agent.analyze(payload).to_dict()
    check("智能体生成建议", bool(advice.get("root_cause")) and bool(advice.get("process_action")),
          f"后端={advice.get('backend')} ｜ 引用 {len(advice.get('citations', []))} 条")
    check("智能体生成工单", bool((advice.get("work_order") or {}).get("work_order_id")),
          (advice.get("work_order") or {}).get("work_order_id", ""))
    check("工具调用留痕", len(advice.get("tool_calls", [])) >= 4,
          "、".join(t.get("tool", "") for t in advice.get("tool_calls", [])))

    # ---------------- 5. 报告生成 ----------------
    from .report import build_report

    payload["inspection_id"] = "SELFTEST001"
    md = build_report(payload, advice, fmt="md", cfg=cfg)
    html = build_report(payload, advice, fmt="html", cfg=cfg,
                        image_data_url=viz.get("annotated_png", ""),
                        heatmap_data_url=viz.get("heatmap_png", ""))
    js = build_report(payload, advice, fmt="json", cfg=cfg)
    check("Markdown 报告", "表面质量智能检测报告" in md and len(md) > 800, f"{len(md)} 字符")
    check("HTML 报告", html.startswith("<!DOCTYPE html>") and len(html) > 2000, f"{len(html)} 字符")
    check("JSON 报告", '"inspection"' in js and '"advice"' in js, f"{len(js)} 字符")

    # ---------------- 6. 数据库与看板 ----------------
    from .db import Database

    db = Database.from_config(args.config)
    insp_id = db.save_inspection(payload, image_path=str(sample), batch_no="SELFTEST")
    if advice.get("work_order"):
        db.save_work_order(advice["work_order"], insp_id)
    got = db.get_inspection(insp_id)
    check("检测记录入库", got is not None and got["inspection_id"] == insp_id, f"编号 {insp_id}")
    st = db.stats(days=3650)
    check("看板统计可用", st["total"] > 0, f"累计 {st['total']} 条 ｜ 合格率 {st['pass_rate']:.3f}")

    # ---------------- 7. 报告落盘（便于人工查看） ----------------
    out_dir = resolve("runs/selftest")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    (out_dir / "result.json").write_text(js, encoding="utf-8")
    import base64

    for key, name in (("annotated_png", "annotated.png"), ("heatmap_png", "heatmap.png")):
        url = viz.get(key, "")
        if url and "," in url:
            (out_dir / name).write_bytes(base64.b64decode(url.split(",", 1)[1]))
    print(f"{PASS} 自检产物已写入 {out_dir}")

    # ---------------- 汇总 ----------------
    print("=" * 70)
    if failures:
        print(f"自检未通过，{len(failures)} 项失败：" + "、".join(failures))
        sys.exit(1)
    print("自检全部通过 —— 系统可演示。启动服务：python -m src.api")
    print("=" * 70)


if __name__ == "__main__":
    main()
