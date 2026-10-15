"""报告接口深度诊断：对比新旧记录、打印完整响应元信息。

排查目标：/api/report/{id}?fmt=json 在部分记录上返回 2 字节（疑似 "{}"），
而旧记录返回完整 9975 字节。本脚本并排输出两种记录的响应详情与库内数据规模。
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8077"


def fetch(path: str):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=90) as r:
        status = r.status
        ctype = r.headers.get("Content-Type")
        clen = r.headers.get("Content-Length")
        body = r.read()
    return status, ctype, clen, body


def main() -> None:
    with urllib.request.urlopen(BASE + "/api/inspections?limit=8", timeout=30) as r:
        rows = json.loads(r.read().decode("utf-8"))["data"]
    print(f"库中最近 {len(rows)} 条记录：")
    for i, row in enumerate(rows):
        print(f"  [{i}] {row['inspection_id']}  grade={row['grade']}  缺陷数={row['defect_count']}")

    for i, row in enumerate(rows[:4]):
        iid = row["inspection_id"]
        print("\n" + "=" * 78)
        print(f"记录 {iid}")
        # 先看详情接口返回的 payload 规模
        st, ctype, clen, body = fetch(f"/api/inspections/{iid}")
        try:
            detail = json.loads(body.decode("utf-8"))
            payload = (detail.get("data") or {}).get("payload") or {}
            print(f"  详情接口: {st} {len(body)} B | payload 键数={len(payload)} "
                  f"detections={len(payload.get('detections', []))} "
                  f"advice={'有' if (detail.get('data') or {}).get('advice') else '无'}")
        except Exception as e:
            print(f"  详情接口解析失败: {e}")

        for fmt in ("md", "json"):
            st, ctype, clen, body = fetch(f"/api/report/{iid}?fmt={fmt}")
            head = body[:70]
            print(f"  fmt={fmt:<5} status={st} ctype={ctype} clen={clen} bytes={len(body)}")
            print(f"          head={head!r}")


if __name__ == "__main__":
    main()
