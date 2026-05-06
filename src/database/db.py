# -*- coding: utf-8 -*-
"""SQLite 数据库封装。"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import DB_PATH, ensure_directories


def _connect() -> sqlite3.Connection:
    ensure_directories()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化表结构。"""
    ensure_directories()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS image_info (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                image_width INTEGER,
                image_height INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_result (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                model_name TEXT,
                model_version TEXT,
                detect_time TEXT NOT NULL,
                mask_path TEXT,
                overlay_path TEXT,
                defect_area INTEGER,
                defect_area_ratio REAL,
                connected_components INTEGER,
                max_defect_area INTEGER,
                confidence_score REAL,
                severity_level TEXT,
                suggestion TEXT,
                FOREIGN KEY (image_id) REFERENCES image_info(image_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS report_info (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER NOT NULL,
                report_path TEXT NOT NULL,
                generate_time TEXT NOT NULL,
                FOREIGN KEY (result_id) REFERENCES detection_result(result_id)
            )
            """
        )
        conn.commit()


def insert_image_info(
    image_name: str,
    image_path: str,
    image_width: int,
    image_height: int,
    upload_time: str | None = None,
) -> int:
    if upload_time is None:
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO image_info (image_name, image_path, upload_time, image_width, image_height)
            VALUES (?, ?, ?, ?, ?)
            """,
            (image_name, image_path, upload_time, image_width, image_height),
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_detection_result(
    image_id: int,
    model_name: str,
    model_version: str,
    mask_path: str,
    overlay_path: str,
    defect_area: int,
    defect_area_ratio: float,
    connected_components: int,
    max_defect_area: int,
    confidence_score: float,
    severity_level: str,
    suggestion: str,
    detect_time: str | None = None,
) -> int:
    if detect_time is None:
        detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO detection_result (
                image_id, model_name, model_version, detect_time,
                mask_path, overlay_path, defect_area, defect_area_ratio,
                connected_components, max_defect_area, confidence_score,
                severity_level, suggestion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                model_name,
                model_version,
                detect_time,
                mask_path,
                overlay_path,
                defect_area,
                defect_area_ratio,
                connected_components,
                max_defect_area,
                confidence_score,
                severity_level,
                suggestion,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_report_info(result_id: int, report_path: str, generate_time: str | None = None) -> int:
    if generate_time is None:
        generate_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO report_info (result_id, report_path, generate_time)
            VALUES (?, ?, ?)
            """,
            (result_id, report_path, generate_time),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_all_records(severity_filter: str | None = None) -> list[dict[str, Any]]:
    """联合查询历史记录。"""
    init_db()
    sql = """
        SELECT
            dr.result_id,
            ii.image_name,
            ii.image_path,
            ii.upload_time,
            dr.detect_time,
            dr.mask_path,
            dr.overlay_path,
            dr.defect_area,
            dr.defect_area_ratio,
            dr.connected_components,
            dr.max_defect_area,
            dr.confidence_score,
            dr.severity_level,
            dr.suggestion,
            dr.model_name,
            dr.model_version,
            ri.report_path,
            ri.generate_time AS report_generate_time
        FROM detection_result dr
        JOIN image_info ii ON dr.image_id = ii.image_id
        LEFT JOIN report_info ri ON ri.result_id = dr.result_id
    """
    params: tuple[Any, ...] = ()
    if severity_filter and severity_filter != "全部":
        sql += " WHERE dr.severity_level = ?"
        params = (severity_filter,)
    sql += " ORDER BY dr.detect_time DESC"
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def delete_record(result_id: int) -> bool:
    """按 result_id 删除检测记录及关联报告。"""
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT image_id FROM detection_result WHERE result_id = ?", (result_id,))
        row = cur.fetchone()
        if not row:
            return False
        image_id = int(row["image_id"])
        cur.execute("DELETE FROM report_info WHERE result_id = ?", (result_id,))
        cur.execute("DELETE FROM detection_result WHERE result_id = ?", (result_id,))
        cur.execute(
            "SELECT COUNT(*) FROM detection_result WHERE image_id = ?",
            (image_id,),
        )
        cnt = int(cur.fetchone()[0])
        if cnt == 0:
            cur.execute("DELETE FROM image_info WHERE image_id = ?", (image_id,))
        conn.commit()
    return True


def resolve_project_path(rel_or_abs: str, project_root: Path) -> Path:
    """将库存路径转为绝对路径。"""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return project_root / p
