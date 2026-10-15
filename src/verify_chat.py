"""验证 /api/chat 的越界拒答与诊断回答行为。

用法：
    python -m src.verify_chat
"""
from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8077"

CASES = [
    "龟裂的成因是什么？",
    "为什么会龟裂？",
    "龟裂比例连续三个班次从0.3%升到4.1%，按什么顺序排查？",
    "夹杂缺陷集中在同一炉次，该查哪里？",
    "C级缺陷需要人工复检吗？",
]


def ask(q: str) -> dict:
    req = urllib.request.Request(
        BASE + "/api/chat",
        data=json.dumps({"question": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))["data"]


def main() -> None:
    print("=" * 80)
    print("诊断接口行为验证")
    print("=" * 80)
    for q in CASES:
        d = ask(q)
        a = d.get("answer", {})
        scope = a.get("scope", "-")
        tag = "拒答" if scope == "out_of_scope" else "作答"
        body = str(a.get("answer", "")).replace("\n", " ")[:150]
        print(f"\n[{tag}] {q}")
        print(f"  scope={scope}  backend={a.get('backend')}  引用={d.get('citations', [])[:3]}")
        print(f"  {body}…")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
