import json

# Đường dẫn file
file_path = r"/data/quiz_LOMA281_backup.json"

# Domain cũ và mới
old_domain = "http://10.98.36.83:8080"
new_domain = "https://api.fpt-apps.com/imt-ai-brain"

# Đọc file JSON
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Lặp qua toàn bộ câu hỏi
for item in data:
    sources = item.get("sources", [])

    for source in sources:
        url = source.get("url")

        if url and old_domain in url:
            source["url"] = url.replace(old_domain, new_domain)

# Ghi lại file (ghi đè)
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Đã update toàn bộ URL thành công!")