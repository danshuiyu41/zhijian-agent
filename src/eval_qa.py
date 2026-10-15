"""评估知识问答的实际回答质量（自查用）。

动机：问答模块最容易沦为"教科书复述"——工程师本来就知道的东西，
答得再顺也没价值。本脚本把知识库对典型问题的实际回答打出来，
便于判断它是在"教基础"还是在"给决策依据"。

用法：
    python -m src.eval_qa
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_cache"))

from .agent import QualityAgent  # noqa: E402

# 三类问题，用来区分"复述常识"与"提供增量信息"
CASES = [
    ("常识型（工程师本就知道）", "龟裂缺陷的成因是什么？"),
    ("常识型（工程师本就知道）", "C级缺陷需要人工复检吗？"),
    ("决策型（有增量价值）", "连续三个班次龟裂比例从0.3%升到4.1%，先查什么？"),
    ("决策型（有增量价值）", "除鳞喷嘴堵塞会表现成什么缺陷？"),
    ("判据型（可核对）", "哪些缺陷属于高危、通常不允许放行？"),
]


def main() -> None:
    agent = QualityAgent.from_config()
    print("=" * 78)
    print("知识问答实际回答质量自查")
    print("=" * 78)
    for kind, q in CASES:
        hits = agent.retriever.search(q, top_k=3)
        ans = agent.answer_question(q, hits)
        print(f"\n【{kind}】")
        print(f"  问：{q}")
        print(f"  答（{ans.get('backend')}，置信度 {ans.get('confidence')}）：")
        body = str(ans.get("answer", "")).strip().replace("\n", "\n     ")
        print(f"    {body[:420]}{'…' if len(body) > 420 else ''}")
        print(f"  引用：{'、'.join(ans.get('citations', []))}")
    print("\n" + "=" * 78)
    print("判断标准：若回答与工艺手册/常识高度重合，则本模块缺乏增量价值。")
    print("=" * 78)


if __name__ == "__main__":
    main()
