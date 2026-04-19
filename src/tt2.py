import os

# Thư mục gốc
root_dir = (
    r"/mnt/d/Deverlopment/java/java-core/java-multithread-concurrency/src/main/java"
)

# File output
output_file = r"./merged_output.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)

                # Lấy folder con đầu tiên sau "concurrency"
                relative_path = os.path.relpath(file_path, root_dir)
                parts = relative_path.split(os.sep)

                folder_name = parts[0] if len(parts) > 1 else "root"

                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        content = infile.read()

                        # Ghi ra file
                        outfile.write(f"[{folder_name}]\n")
                        outfile.write(content)
                        outfile.write("\n\n")  # xuống dòng giữa các file

                except Exception as e:
                    print(f"Lỗi đọc file {file_path}: {e}")

print("Hoàn thành! File đã được ghi tại:", output_file)
