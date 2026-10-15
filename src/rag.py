"""极简中文检索（BM25 + 字符 n-gram 混合），零外部依赖。

为什么自己实现而不用 LangChain / rank_bm25
----------------------------------------
1. **交付友好**：比赛演示机不保证能联网装包，本项目坚持"核心功能零额外依赖"。
   实测本环境的 pip 通道不稳定（timm/jieba/rank_bm25 均安装失败），自研更稳。
2. **中文场景更合适**：工业标准文本专业术语多、无空格分词边界，字符 2-gram/3-gram
   的召回往往优于词典分词；再叠加 BM25 的词频饱和与长度归一化，效果足够。
3. **可解释**：检索打分过程完全透明，方便在设计说明书中说明，也便于答辩时解释。

实现要点
--------
- 分词：中文字符 bigram/trigram + 英文/数字按单词切分（正则）；
- 打分：BM25（k1=1.5, b=0.75）逐项累加，IDF 用平滑公式避免除零；
- 混合：`score = α * bm25_norm + (1-α) * overlap_ratio`，其中 overlap 为查询 n-gram
  在文档中的覆盖率，用于弥补短查询下 BM25 失效的问题；
- 支持字段加权（标题权重高于正文），贴合"标准条款检索"的直觉。

用法：
    from src.rag import BM25Retriever
    r = BM25Retriever(docs)          # docs: [{"id","title","text","meta"}]
    hits = r.search("龟裂 成因", top_k=3)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


# ----------------------------------------------------------------------
# 分词
# ----------------------------------------------------------------------
def tokenize(text: str, ngrams: tuple[int, ...] = (2, 3)) -> list[str]:
    """中英混合分词：英文/数字按词，中文按字符 n-gram。"""
    text = (text or "").lower()
    units = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    # 中文连续片段单独抽取做 n-gram
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        for n in ngrams:
            if len(seg) >= n:
                tokens.extend(seg[i:i + n] for i in range(len(seg) - n + 1))
        if len(seg) == 1:
            tokens.append(seg)
    tokens.extend(u for u in units if not re.fullmatch(r"[\u4e00-\u9fff]", u))
    return tokens


# ----------------------------------------------------------------------
# 文档
# ----------------------------------------------------------------------
@dataclass
class Doc:
    id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title}。{self.text}"

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text, "meta": self.meta}


@dataclass
class Hit:
    doc: Doc
    score: float
    matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.doc.id, "title": self.doc.title, "text": self.doc.text,
                "meta": self.doc.meta, "score": round(self.score, 4),
                "matched_terms": self.matched[:12]}


# ----------------------------------------------------------------------
# BM25 检索器
# ----------------------------------------------------------------------
class BM25Retriever:
    """BM25 + n-gram 覆盖率的混合检索器。"""

    def __init__(self, docs: Iterable[Doc | dict], k1: float = 1.5, b: float = 0.75,
                 alpha: float = 0.65, title_weight: float = 2.0):
        self.k1, self.b, self.alpha, self.title_weight = k1, b, alpha, title_weight
        self.docs: list[Doc] = []
        for d in docs:
            self.docs.append(d if isinstance(d, Doc) else Doc(**d))
        self._index()

    def _index(self) -> None:
        self.doc_tokens: list[Counter] = []
        self.doc_len: list[float] = []
        df: Counter = Counter()
        for d in self.docs:
            body = tokenize(d.text)
            title = tokenize(d.title)
            c = Counter(body)
            for t in title:
                c[t] += self.title_weight
            self.doc_tokens.append(c)
            self.doc_len.append(float(sum(c.values())) or 1.0)
            for t in c:
                df[t] += 1
        n = max(len(self.docs), 1)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
        self.avg_len = sum(self.doc_len) / n if self.doc_len else 1.0

    # ------------------------------------------------------------------
    def _bm25(self, q_tokens: list[str], i: int) -> tuple[float, list[str]]:
        score = 0.0
        matched: list[str] = []
        c = self.doc_tokens[i]
        dl = self.doc_len[i]
        for t in set(q_tokens):
            f = c.get(t, 0)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
            score += idf * (f * (self.k1 + 1)) / denom
            matched.append(t)
        return score, matched

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[Hit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        raw: list[tuple[float, float, list[str], int]] = []
        for i in range(len(self.docs)):
            bm, matched = self._bm25(q_tokens, i)
            # 覆盖率：查询 n-gram 被文档命中的比例
            cover = len(q_set & set(self.doc_tokens[i].keys())) / max(len(q_set), 1)
            raw.append((bm, cover, matched, i))
        max_bm = max((r[0] for r in raw), default=0.0) or 1.0
        hits: list[Hit] = []
        for bm, cover, matched, i in raw:
            score = self.alpha * (bm / max_bm) + (1 - self.alpha) * cover
            if score > min_score:
                hits.append(Hit(self.docs[i], score, matched))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]


# ----------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------
def docs_from_json(path: str | Any) -> list[Doc]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["docs"] if isinstance(data, dict) else data
    return [Doc(id=str(d.get("id", i)), title=d.get("title", ""), text=d.get("text", ""),
                meta=d.get("meta", {})) for i, d in enumerate(items)]
