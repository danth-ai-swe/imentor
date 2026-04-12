import pandas as pd

# Đường dẫn file
file_path = r"D:\Deverlopment\huudan.com\PythonProject\data\metadata_node.xlsx"

# Đọc file Excel
df = pd.read_excel(file_path)

# Đảm bảo đúng tên cột
# (strip để tránh lỗi do khoảng trắng)
df.columns = df.columns.str.strip()

# Group theo Category và gom Node Name
grouped = df.groupby('Category')['Node Name'].apply(list)

# In kết quả
for category, nodes in grouped.items():
    print(f"{category}: {', '.join(nodes)}")