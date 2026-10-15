"""SQLite 持久化层：检测记录、工单、批次统计。

为什么用标准库 sqlite3 而不是 ORM
--------------------------------
演示机不保证能装 SQLAlchemy 等依赖；sqlite3 是 Python 标准库，零安装、单文件、
便于随作品一起打包交付，也方便评委直接用 DB Browser 打开查看数据。

表结构
------
inspections : 每次检测一条记录（含全部结构化结果 JSON、缩略图路径、耗时）
work_orders : 智能体生成的工单
kb_queries  : 知识库检索留痕（可选，用于展示可追溯性）

用法：
    db = Database.from_config()
    db.save_inspection(payload)
    db.stats(days=7)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .utils import get_logger, load_config, resolve

LOG = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id   TEXT UNIQUE NOT NULL,
    created_at      TEXT NOT NULL,
    image_path      TEXT,
    image_name      TEXT,
    image_size      TEXT,
    verdict         TEXT,
    grade           TEXT,
    grade_zh        TEXT,
    max_severity    REAL,
    defect_count    INTEGER,
    top_defect_code TEXT,
    top_defect_zh   TEXT,
    confidence      REAL,
    elapsed_ms      REAL,
    windows         INTEGER,
    payload         TEXT,
    advice          TEXT,
    batch_no        TEXT,
    operator        TEXT
);
CREATE INDEX IF NOT EXISTS idx_insp_created ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_insp_grade   ON inspections(grade);
CREATE INDEX IF NOT EXISTS idx_insp_defect  ON inspections(top_defect_code);

CREATE TABLE IF NOT EXISTS work_orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    work_order_id TEXT UNIQUE NOT NULL,
    inspection_id TEXT,
    created_at    TEXT NOT NULL,
    defect_code   TEXT,
    defect_zh     TEXT,
    severity_level TEXT,
    severity_score REAL,
    priority      TEXT,
    department    TEXT,
    action        TEXT,
    check_items   TEXT,
    stop_line     INTEGER DEFAULT 0,
    quarantine_batch INTEGER DEFAULT 0,
    status        TEXT DEFAULT '待处理',
    payload       TEXT
);
CREATE INDEX IF NOT EXISTS idx_wo_status ON work_orders(status);
CREATE INDEX IF NOT EXISTS idx_wo_created ON work_orders(created_at);

CREATE TABLE IF NOT EXISTS kb_queries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query      TEXT,
    hit_ids    TEXT,
    backend    TEXT
);
"""


class Database:
    """线程安全的轻量 SQLite 封装（FastAPI 多线程环境安全）。"""

    def __init__(self, path: str | Path = "data/app.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init()

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "Database":
        cfg = load_config(config_path)
        return cls(resolve(cfg["paths"]["db"]))

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(SCHEMA)
        LOG.info("数据库就绪：%s", self.path)

    # ------------------------------------------------------------------
    def save_inspection(self, payload: dict, image_path: str = "", batch_no: str = "",
                        operator: str = "智能质检终端 QC-02") -> str:
        """保存一次检测。payload 为 InspectionResult.to_dict() 的输出；可附加 advice。"""
        insp_id = payload.get("inspection_id") or f"QC{time.strftime('%Y%m%d%H%M%S')}{int(time.time() * 1000) % 1000:03d}"
        payload["inspection_id"] = insp_id
        dets = payload.get("detections", [])
        top = dets[0] if dets else {}
        advice = payload.get("advice") or {}
        row = (
            insp_id, time.strftime("%Y-%m-%d %H:%M:%S"), image_path,
            Path(image_path).name if image_path else "",
            json.dumps(payload.get("image_size", []), ensure_ascii=False),
            payload.get("verdict", ""), payload.get("grade", ""), payload.get("grade_zh", ""),
            float(payload.get("max_severity", 0.0)), len(dets),
            top.get("class_code", ""), top.get("class_zh", ""), float(top.get("confidence", 0.0)),
            float(payload.get("elapsed_ms", 0.0)), int(payload.get("windows", 0)),
            json.dumps(payload, ensure_ascii=False),
            json.dumps(advice, ensure_ascii=False) if advice else "",
            batch_no, operator,
        )
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO inspections
                   (inspection_id, created_at, image_path, image_name, image_size, verdict, grade,
                    grade_zh, max_severity, defect_count, top_defect_code, top_defect_zh, confidence,
                    elapsed_ms, windows, payload, advice, batch_no, operator)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
        return insp_id

    def save_work_order(self, wo: dict, inspection_id: str = "") -> str:
        wo_id = wo.get("work_order_id") or f"WO-{int(time.time())}"
        row = (
            wo_id, inspection_id or wo.get("inspection_id", ""), wo.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
            wo.get("defect_code", ""), wo.get("defect_zh", ""), wo.get("severity_level", ""),
            float(wo.get("severity_score", 0.0)), wo.get("priority", ""), wo.get("department", ""),
            wo.get("action", ""), json.dumps(wo.get("check_items", []), ensure_ascii=False),
            1 if wo.get("stop_line") else 0, 1 if wo.get("quarantine_batch") else 0,
            wo.get("status", "待处理"), json.dumps(wo, ensure_ascii=False),
        )
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO work_orders
                   (work_order_id, inspection_id, created_at, defect_code, defect_zh, severity_level,
                    severity_score, priority, department, action, check_items, stop_line,
                    quarantine_batch, status, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
        return wo_id

    def log_kb_query(self, query: str, hit_ids: list[str], backend: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("INSERT INTO kb_queries (created_at, query, hit_ids, backend) VALUES (?,?,?,?)",
                         (time.strftime("%Y-%m-%d %H:%M:%S"), query,
                          json.dumps(hit_ids, ensure_ascii=False), backend))

    # ------------------------------------------------------------------
    def list_inspections(self, limit: int = 50, offset: int = 0, grade: str = "",
                         defect: str = "", days: int = 0) -> list[dict]:
        sql = "SELECT * FROM inspections WHERE 1=1"
        args: list[Any] = []
        if grade:
            sql += " AND grade = ?"
            args.append(grade)
        if defect:
            sql += " AND top_defect_code = ?"
            args.append(defect)
        if days:
            sql += " AND created_at >= datetime('now','localtime',?)"
            args.append(f"-{int(days)} days")
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_inspection(r) for r in rows]

    def get_inspection(self, inspection_id: str) -> dict | None:
        with self._lock, self._conn() as conn:
            r = conn.execute("SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)).fetchone()
        return self._row_to_inspection(r) if r else None

    @staticmethod
    def _row_to_inspection(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("payload", "advice"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        if isinstance(d.get("image_size"), str):
            try:
                d["image_size"] = json.loads(d["image_size"])
            except Exception:
                pass
        return d

    def list_work_orders(self, limit: int = 50, status: str = "") -> list[dict]:
        sql = "SELECT * FROM work_orders WHERE 1=1"
        args: list[Any] = []
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["check_items"] = json.loads(d.get("check_items") or "[]")
            except Exception:
                pass
            out.append(d)
        return out

    def update_work_order_status(self, work_order_id: str, status: str) -> bool:
        with self._lock, self._conn() as conn:
            cur = conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ?",
                               (status, work_order_id))
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    def stats(self, days: int = 7) -> dict:
        """看板统计：产量、合格率、等级分布、缺陷类型分布、小时趋势、节拍。"""
        since = f"-{int(days)} days"
        with self._lock, self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) c FROM inspections WHERE created_at >= datetime('now','localtime',?)",
                (since,)).fetchone()["c"]
            ok = conn.execute(
                "SELECT COUNT(*) c FROM inspections WHERE created_at >= datetime('now','localtime',?) "
                "AND verdict = '合格'", (since,)).fetchone()["c"]
            by_grade = conn.execute(
                "SELECT grade, grade_zh, COUNT(*) c FROM inspections "
                "WHERE created_at >= datetime('now','localtime',?) GROUP BY grade ORDER BY grade",
                (since,)).fetchall()
            by_defect = conn.execute(
                "SELECT top_defect_code code, top_defect_zh zh, COUNT(*) c FROM inspections "
                "WHERE created_at >= datetime('now','localtime',?) AND top_defect_code <> '' "
                "GROUP BY top_defect_code ORDER BY c DESC", (since,)).fetchall()
            by_hour = conn.execute(
                "SELECT substr(created_at,12,2) h, COUNT(*) c, "
                "SUM(CASE WHEN verdict='合格' THEN 1 ELSE 0 END) ok FROM inspections "
                "WHERE created_at >= datetime('now','localtime',?) GROUP BY h ORDER BY h",
                (since,)).fetchall()
            perf = conn.execute(
                "SELECT AVG(elapsed_ms) avg_ms, AVG(defect_count) avg_defects, COUNT(*) n "
                "FROM inspections WHERE created_at >= datetime('now','localtime',?)",
                (since,)).fetchone()
            wo_open = conn.execute("SELECT COUNT(*) c FROM work_orders WHERE status <> '已完成'").fetchone()["c"]
            wo_total = conn.execute("SELECT COUNT(*) c FROM work_orders").fetchone()["c"]

        return {
            "days": days,
            "total": total,
            "qualified": ok,
            "unqualified": total - ok,
            "pass_rate": round(ok / total, 4) if total else 0.0,
            "by_grade": [dict(r) for r in by_grade],
            "by_defect": [dict(r) for r in by_defect],
            "by_hour": [dict(r) for r in by_hour],
            "avg_elapsed_ms": round(float(perf["avg_ms"] or 0.0), 1),
            "avg_defect_count": round(float(perf["avg_defects"] or 0.0), 2),
            "work_orders_total": wo_total,
            "work_orders_open": wo_open,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def seed_demo_data(self, n: int = 120) -> int:
        """生成演示用的历史数据，让看板与趋势图在首次打开时就有内容。"""
        import random

        from .utils import class_zh_map, severity_grade
        from .utils import load_config as _lc

        cfg = _lc()
        zh = class_zh_map(cfg)
        codes = list(zh.keys())
        weights = [0.12, 0.08, 0.2, 0.18, 0.26, 0.16]
        inserted = 0
        base = time.time() - n * 60
        rng = random.Random(20261012)
        exist = self.stats(days=365)["total"]
        if exist >= n:
            LOG.info("已有 %d 条记录，跳过演示数据填充", exist)
            return 0
        for i in range(n):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(base + i * 60))
            if rng.random() < 0.62:
                grade, verdict, code, conf, sev = "A", "合格", "", 0.0, 0.0
            else:
                code = rng.choices(codes, weights=weights, k=1)[0]
                conf = round(rng.uniform(0.56, 0.97), 3)
                risk = {"crazing": 1.0, "inclusion": 1.0, "patches": 0.65,
                        "pitted_surface": 0.65, "rolled_in_scale": 0.35, "scratches": 0.65}[code]
                area = rng.uniform(0.002, 0.08)
                w = cfg["severity"]["weights"]
                sev = round(100 * (w["risk"] * risk + w["confidence"] * conf
                                   + w["area"] * min(1.0, area / 0.05)), 1)
                g = severity_grade(cfg, sev)
                grade, verdict = g["level"], ("合格" if g["level"] in ("A", "B") else "不合格")
            payload = {
                "inspection_id": f"DEMO{1000 + i}",
                "verdict": verdict, "grade": grade,
                "grade_zh": severity_grade(cfg, sev)["zh"] if sev else "合格",
                "max_severity": sev,
                "detections": ([{"class_code": code, "class_zh": zh[code], "confidence": conf,
                                 "bbox": [40, 40, 180, 180], "area_ratio": 0.01,
                                 "severity_score": sev, "severity_level": grade,
                                 "severity_zh": severity_grade(cfg, sev)["zh"],
                                 "action": severity_grade(cfg, sev)["action"], "risk": "mid"}]
                               if code else []),
                "elapsed_ms": round(rng.uniform(320, 780), 1), "windows": rng.randint(12, 40),
                "image_size": [256, 256], "quality_flags": [],
            }
            payload["inspection_id"] = f"DEMO{1000 + i}"
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO inspections
                       (inspection_id, created_at, image_path, image_name, image_size, verdict, grade,
                        grade_zh, max_severity, defect_count, top_defect_code, top_defect_zh, confidence,
                        elapsed_ms, windows, payload, advice, batch_no, operator)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f"DEMO{1000 + i}", ts, "", f"demo_{i:04d}.png", "[256, 256]", verdict, grade,
                     payload["grade_zh"], sev, len(payload["detections"]), code,
                     zh.get(code, ""), conf, payload["elapsed_ms"], payload["windows"],
                     json.dumps(payload, ensure_ascii=False), "",
                     f"B{time.strftime('%Y%m%d', time.localtime(base + i * 60))}", "演示数据"))
            inserted += 1
        LOG.info("已填充演示数据 %d 条", inserted)
        return inserted
