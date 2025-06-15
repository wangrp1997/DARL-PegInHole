import pandas as pd
import os

# 固定的文件夹名称
base_folder = "baseline_results"

while 1:
    # 提示用户输入文件名（而不是完整路径）
    file_name = input("请输入CSV文件的名称（例如 file.csv）: ")

    # 拼接完整路径
    file_path = os.path.join(base_folder, file_name)

    try:
        # 读取CSV文件
        data = pd.read_csv(file_path)
        # print(data['Success'])
        # 筛选成功回合（Success=1）的数据
        success_data = data[data['Success'] == 1]

        # 计算成功回合的尝试次数均值
        mean_attempts = success_data['Attempt'].mean()
        # 获取最后一行的成功率
        last_row_success_rate = data.iloc[-1]['Cumulative Success Rate (%)']

        # 输出结果
        print(f"成功回合 (Success=1) 的尝试次数均值是: {mean_attempts:.2f}")
        print(f"成功率: {last_row_success_rate:.2f}%")
    except FileNotFoundError:
        print("文件路径无效，请检查后重新输入！")
    except Exception as e:
        print(f"发生错误: {e}")
