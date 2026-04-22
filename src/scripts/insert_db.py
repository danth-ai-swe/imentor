import json
import sqlite3
import os

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
JSON_FILE = r"D:\Deverlopment\huudan.com\PythonProject\data\metadata_node.json"
DB_FILE   = r"D:\Deverlopment\huudan.com\PythonProject\sqlite\data_app.db"

# ── SQL tạo bảng ──────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metadata_nodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id       TEXT,
    node_name     TEXT,
    definition    TEXT,
    category      TEXT,
    domain_tags   TEXT,
    related_nodes TEXT,
    source        TEXT,
    summary       TEXT,
    course        TEXT,
    module        TEXT,
    lesson        TEXT,
    path          TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# ── SQL chèn dữ liệu ──────────────────────────────────────────────────────────
INSERT_SQL = """
INSERT INTO metadata_nodes (
    node_id, node_name, definition, category, domain_tags,
    related_nodes, source, summary, course, module, lesson, path
) VALUES (
    :node_id, :node_name, :definition, :category, :domain_tags,
    :related_nodes, :source, :summary, :course, :module, :lesson, :path
);
"""

def map_record(record: dict) -> dict:
    """Chuyển key gốc (có khoảng trắng) sang key chuẩn cho INSERT."""
    return {
        "node_id":       record.get("Node ID", ""),
        "node_name":     record.get("Node Name", ""),
        "definition":    record.get("Definition", ""),
        "category":      record.get("Category", ""),
        "domain_tags":   record.get("Domain Tags", ""),
        "related_nodes": record.get("Related Nodes", ""),
        "source":        record.get("Source", ""),
        "summary":       record.get("Summary", ""),
        "course":        record.get("Course", ""),
        "module":        record.get("Module", ""),
        "lesson":        record.get("Lesson", ""),
        "path":          record.get("path", ""),
    }

def main():
    # 1. Đọc file JSON
    if not os.path.exists(JSON_FILE):
        print(f"[LỖI] Không tìm thấy file JSON: {JSON_FILE}")
        return

    with open(JSON_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ cả dạng list và dạng dict đơn
    if isinstance(data, dict):
        data = [data]

    print(f"[INFO] Đọc được {len(data)} bản ghi từ JSON.")

    # 2. Kết nối SQLite (tự tạo file nếu chưa có)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # 3. Tạo bảng
    cur.executescript(CREATE_TABLE_SQL)
    conn.commit()
    print("[INFO] Bảng metadata_nodes đã sẵn sàng.")

    # 4. Chèn dữ liệu
    inserted = 0
    errors   = 0
    for i, record in enumerate(data, start=1):
        try:
            cur.execute(INSERT_SQL, map_record(record))
            inserted += 1
        except sqlite3.Error as e:
            print(f"[CẢNH BÁO] Bản ghi #{i} lỗi: {e}")
            errors += 1

    conn.commit()
    conn.close()

    # 5. Báo kết quả
    print(f"\n✅ Hoàn thành!")
    print(f"   • Chèn thành công : {inserted} bản ghi")
    print(f"   • Lỗi             : {errors} bản ghi")
    print(f"   • Database        : {DB_FILE}")

if __name__ == "__main__":
    main()