def find_lines_with_xef(file_path):
    with open(file_path, 'rb') as f:  # 打开文件时使用二进制模式
        lines_with_xef = []
        for line_num, line in enumerate(f, start=1):
            if b'\xef' in line:  # 使用字节串来检查
                try:
                    line_str = line.decode('latin-1')  # 尝试使用 UTF-8 解码
                except UnicodeDecodeError:
                    break
                lines_with_xef.append((line_num, line_str.strip()))  # 添加解码后的字符串
    return lines_with_xef

# 示例用法
file_path = "data_folder/new_data5000_1.pkl"
xef_lines = find_lines_with_xef(file_path)

if xef_lines:
    print("Lines containing '\xef' in file:")
    for line_num, line in xef_lines:
        print(f"Line {line_num}: {line}")
else:
    print("No lines containing '\xef' found in the file.")

