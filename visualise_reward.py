import os
import matplotlib.pyplot as plt

# 文件夹路径列表
folders = [
    "05-14-2024-17-14-18-rr10",
    "05-17-2024-10-37-09-rr11",
    "05-14-2024-10-53-42-rr20",
    "05-16-2024-18-41-16-rr21",
    "05-13-2024-10-20-13-rr30",
    "05-16-2024-09-46-01-rr31",
    "05-13-2024-21-59-01-rr40",
    "05-16-2024-00-20-54-rr41",
]

# 存储所有文件夹的数据
all_mean_len = {}
all_mean_reward = {}

# 读取每个文件夹中的 logs.txt 文件
for folder in folders:
    file_path = os.path.join("./trained_models", folder, "logs.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            mean_len = []
            mean_reward = []
            for line in lines:
                if "mean(len)" in line and "mean(reward)" in line:
                    parts = line.split(", ")
                    mean_len.append(float(parts[4].split(" = ")[1]))
                    mean_reward.append(float(parts[7].split(" = ")[1]))
            if mean_len and mean_reward:
                all_mean_len[folder[-2] + "-" + folder[-1]] = mean_len
                all_mean_reward[folder[-2] + "-" + folder[-1]] = mean_reward

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 16,        # 全局字体大小
    'axes.titlesize': 20,   # 标题字体大小
    'axes.labelsize': 18,   # 坐标轴标签字体大小
    'xtick.labelsize': 14,  # x轴刻度字体大小
    'ytick.labelsize': 14,  # y轴刻度字体大小
    'legend.fontsize': 16   # 图例字体大小
})

# 检查是否有数据
if not all_mean_len or not all_mean_reward:
    print("没有数据可供绘制。")
else:
    # 绘制 mean(len) 图
    plt.figure(figsize=(7, 8))
    for folder, data in all_mean_len.items():
        plt.plot(data, label=folder)
    plt.xlabel("Episode")
    plt.ylabel("mean(len)")
    plt.legend(loc='lower right')  # 图例放置在右下角
    plt.title("Mean Length Over Episodes")
    plt.show()

    # 绘制 mean(reward) 图
    plt.figure(figsize=(7, 8))
    for folder, data in all_mean_reward.items():
        plt.plot(data, label=folder)
    plt.xlabel("Episode")
    plt.ylabel("mean(reward)")
    plt.legend(loc='lower right')  # 图例放置在右下角
    plt.title("Mean Reward Over Episodes")
    plt.show()
