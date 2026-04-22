import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

DB_FILE = os.path.join(PROJECT_ROOT, "sqlite", "data_app.db")


class MetadataDB:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
        return [dict(row) for row in rows]

    # ─────────────────────────────────────────────
    # 1. Lấy toàn bộ dữ liệu
    # ─────────────────────────────────────────────
    def get_all(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("SELECT * FROM metadata_nodes ORDER BY id DESC;")
        rows = cur.fetchall()

        conn.close()
        return self._rows_to_dicts(rows)

    # ─────────────────────────────────────────────
    # 2. Lấy theo node_id
    # ─────────────────────────────────────────────
    def get_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM metadata_nodes WHERE node_id = ? LIMIT 1;",
            (node_id,)
        )
        row = cur.fetchone()

        conn.close()
        return dict(row) if row else None

    # ─────────────────────────────────────────────
    # 3. Lấy theo course
    # ─────────────────────────────────────────────
    def get_by_course(self, course: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM metadata_nodes WHERE course = ? ORDER BY id DESC;",
            (course,)
        )
        rows = cur.fetchall()

        conn.close()
        return self._rows_to_dicts(rows)

    # ─────────────────────────────────────────────
    # 4. Lấy theo category
    # ─────────────────────────────────────────────
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM metadata_nodes WHERE category = ? ORDER BY id DESC;",
            (category,)
        )
        rows = cur.fetchall()

        conn.close()
        return self._rows_to_dicts(rows)
