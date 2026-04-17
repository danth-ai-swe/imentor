import json
import hashlib
from pathlib import Path

FILE_PATH = r"/data/quiz_LOMA291_backup.json"


def hash_object(obj: dict) -> str:
    """Tạo hash từ toàn bộ nội dung object để so sánh."""
    serialized = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def remove_duplicates(file_path: str) -> None:
    path = Path(file_path)

    if not path.exists():
        print(f"❌ File không tìm thấy: {file_path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ File JSON không phải dạng danh sách (list). Script này chỉ xử lý list.")
        return

    total_before = len(data)
    print(f"📂 File: {file_path}")
    print(f"📊 Tổng số câu hỏi trước khi xử lý: {total_before}")

    seen_hashes = {}       # hash -> index đầu tiên xuất hiện
    duplicates_info = []   # lưu thông tin các bản trùng
    unique_data = []

    for i, item in enumerate(data):
        h = hash_object(item)
        if h in seen_hashes:
            duplicates_info.append({
                "duplicate_index": i,
                "original_index": seen_hashes[h],
                "question": item.get("question", "(no question field)")
            })
        else:
            seen_hashes[h] = i
            unique_data.append(item)

    total_after = len(unique_data)
    removed = total_before - total_after

    if removed == 0:
        print("✅ Không tìm thấy bản trùng nào. File giữ nguyên.")
        return

    print(f"\n🔍 Tìm thấy {removed} bản trùng:")
    for d in duplicates_info:
        print(f"   - Index [{d['duplicate_index']}] trùng với index [{d['original_index']}]: \"{d['question'][:80]}...\""
              if len(d['question']) > 80 else
              f"   - Index [{d['duplicate_index']}] trùng với index [{d['original_index']}]: \"{d['question']}\"")

    # Ghi lại file đã lọc (ghi đè)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã xóa {removed} bản trùng.")
    print(f"📊 Tổng số câu hỏi sau khi xử lý: {total_after}")
    print(f"💾 File đã được lưu lại: {file_path}")


if __name__ == "__main__":
    remove_duplicates(FILE_PATH)