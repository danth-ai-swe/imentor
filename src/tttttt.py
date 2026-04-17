import json
import os

INPUT_FILE = r"C:\imt-ai-brain\data\quiz_LOMA291.json"
OUTPUT_FILE = r"C:\imt-ai-brain\data\quiz_LOMA291.json"  # ghi đè file gốc

# Nếu muốn backup trước khi ghi đè, đổi OUTPUT_FILE thành đường dẫn khác
# OUTPUT_FILE = r"D:\Deverlopment\huudan.com\PythonProject\data\quiz_LOMA281_transformed.json"

def transform_sources(sources: list) -> list:
    transformed = []
    for source in sources:
        course  = source.get("course", "").strip()
        module  = source.get("module", "").strip()
        lesson  = source.get("lesson", "").strip()
        name    = source.get("name", "").strip()

        # Ghép name mới: course | module | lesson | name_cũ
        parts = [p for p in [course, module, lesson, name] if p]
        new_name = " | ".join(parts)

        new_source = {k: v for k, v in source.items()
                      if k not in ("course", "module", "lesson")}
        new_source["name"] = new_name

        transformed.append(new_source)
    return transformed


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy file: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ File JSON phải là một mảng (list) các câu hỏi ở cấp cao nhất.")
        return

    total_questions   = len(data)
    total_sources_mod = 0

    for question in data:
        if "sources" in question and isinstance(question["sources"], list):
            question["sources"] = transform_sources(question["sources"])
            total_sources_mod += len(question["sources"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Hoàn thành!")
    print(f"   • Tổng câu hỏi xử lý : {total_questions}")
    print(f"   • Tổng sources cập nhật: {total_sources_mod}")
    print(f"   • File đầu ra         : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()