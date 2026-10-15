"""分级效果核验：通过 HTTP API 逐类检测，打印各缺陷类别的判定等级。

用途：确认 v2 分级公式在**运行中的服务**里已生效（避免"改了代码但服务仍跑旧逻辑"）。
这是本项目实际踩过的坑：进程内加载的是启动时刻的代码对象，改完公式不重启服务，
接口仍返回旧等级。

用法：
    python -m src.verify_grades
"""
from __future__ import annotations

import glob
import json
import os
import urllib.request
import uuid
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

BASE = "http://127.0.0.1:8077"

ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled_in_scale": "压入氧化铁皮",
    "scratches": "划痕",
}


def post_file(path: str, fp: str, fields: dict, timeout: int = 300) -> dict:
    boundary = "----zj" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    name = fp.replace("\\", "/").split("/")[-1]
    data = Path(fp).read_bytes()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + path, data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    print("=" * 70)
    print("分级效果核验（经 HTTP API，反映服务进程实际使用的公式）")
    print("=" * 70)
    print(f"{'类别':<16}{'等级':<7}{'严重度':<9}{'检出':<8}{'处置建议'}")
    print("-" * 70)
    levels: list[str] = []
    for code, name in ZH.items():
        files = sorted(glob.glob(f"data/raw/{code}/*.png"))
        if not files:
            continue
        d = post_file("/api/inspect/agent", files[0], {"batch_no": "GRADE-VERIFY"})
        res = d["data"]["result"]
        levels.append(res["grade"])
        action = (res.get("top1") or {}).get("action", "-")
        print(f"{name:<16}{res['grade']:<7}{res['max_severity']:<9}"
              f"{len(res['detections']):<8}{action}")

    clean = sorted(glob.glob("data/raw/clean/*.png"))
    if clean:
        d = post_file("/api/inspect/agent", clean[0], {"batch_no": "GRADE-VERIFY"})
        res = d["data"]["result"]
        levels.append(res["grade"])
        print(f"{'洁净(合格)':<14}{res['grade']:<7}{res['max_severity']:<9}{len(res['detections']):<8}放行")

    print("-" * 70)
    uniq = sorted(set(levels))
    print(f"出现的等级: {'、'.join(uniq)}  （共 {len(uniq)} 档）")
    if len(uniq) >= 3:
        print("结论：分级具备梯度，未退化为单一档位。")
    else:
        print("警告：等级过于集中，请检查分级参数或确认服务已重启加载最新代码。")
    print("=" * 70)


if __name__ == "__main__":
    main()
