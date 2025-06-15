import os
import pandas as pd

# 指定目录路径
folder_path = './base_line_results'

# 遍历文件夹下的所有文件
for filename in os.listdir(folder_path):
    # 只处理 CSV 文件
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # 读取 CSV 文件
        try:
            df = pd.read_csv(file_path)

            # 假设第二列的列名是 'Attempts'，可以根据实际情况调整
            if 'Attempt' in df.columns:
                second_column = df['Attempt']
                # 计算均值
                average_attempts = second_column.mean()
                print(f"文件: {filename} 的第二列尝试次数均值是: {average_attempts}")
            else:
                print(f"文件: {filename} 没有名为 'Attempt' 的列")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

        # 等待用户按回车键继续
        input("按回车键继续...")
