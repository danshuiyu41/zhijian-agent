"""诊断界面链路的剩余耗时与响应体体积。

背景：进程内分解显示"检测+标注+落库"合计不到 220 ms，但同一接口的 HTTP
往返却要约 2000 ms，差额至今没有解释清楚。本脚本把**服务内部上报耗时**与
**响应体体积**同时打出来，用于判断瓶颈在计算、序列化还是传输。

用法：
    python -m src.diag_payload
"""
from __future__ import annotations

import glob
import json
import time
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8077"


def post(fp: str, fields: dict) -> tuple[bytes, float]:
    boundary = "----zj" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    name = Path(fp).name
    data = Path(fp).read_bytes()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + "/api/inspect/agent", data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    return raw, (time.time() - t0) * 1000


def kb(s) -> float:
    return len(s or "") / 1024


def main() -> None:
    files = sorted(glob.glob("data/raw/crazing/*.png"))
    if not files:
        raise SystemExit("未找到测试图像")
    f = files[0]
    print("=" * 82)
    print("界面链路剩余耗时诊断（服务内部耗时 vs 响应体体积）")
    print("=" * 82)
    for label, flag in (("heatmap=false", "false"), ("heatmap=true", "true")):
        raw, ms = post(f, {"heatmap": flag, "agent": "true", "persist": "true", "batch_no": ""})
        try:
            res = json.loads(raw.decode("utf-8"))["data"]["result"]
        except Exception as e:
            print(f"  {label}: 解析失败 {e}")
            continue
        print(f"\n  [{label}]")
        print(f"    HTTP 往返总耗时      {ms:>8.0f} ms")
        print(f"    服务内部上报 elapsed {res.get('elapsed_ms', 0):>8.0f} ms")
        print(f"    差值（传输+序列化）  {ms - float(res.get('elapsed_ms') or 0):>8.0f} ms")
        print(f"    响应体总体积         {len(raw) / 1024:>8.1f} KB")
        print(f"      标注图 base64      {kb(res.get('annotated_data_url')):>8.1f} KB")
        print(f"      原图 base64        {kb(res.get('image_data_url')):>8.1f} KB")
        print(f"      热力图 base64      {kb((res.get('advice') or {}).get('x')):>8.1f} KB")
    print("\n" + "=" * 82)
    print("判读：若差值远大于服务内部 elapsed，则瓶颈在**响应体传输/序列化**；")
    print("      若两者接近，则瓶颈在**服务端计算**。")
    print("=" * 82)


if __name__ == "__main__":
    main()
