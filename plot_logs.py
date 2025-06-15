import re
import matplotlib.pyplot as plt

# 从文件中读取数据
file_path = "/home/rw/文档/convlstm_insertion_torque-SL/trained_models/01-21-2024-23-30-57/logs.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化存储数据的列表
num_steps_list = []
curriculum_stages = []
mean_rewards = []
success_rates = []

# 提取数据
for line in lines:
    match = re.search(r'curriculum stage = (\d+),.*num_steps = (\d+),.*mean\(reward\) = ([\d.-]+),.*success_rate = ([\d.]+)', line)
    if match:
        num_steps_list.append(int(match.group(2)))
        curriculum_stages.append(int(match.group(1)))
        mean_rewards.append(float(match.group(3)))
        success_rates.append(float(match.group(4)))

# 绘制图表
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 绘制curriculum stage图
axs[0].plot(num_steps_list, curriculum_stages, label='Curriculum Stage',  linestyle='-')
axs[0].set_title('Curriculum Stage')
axs[0].set_xlabel('num_steps')
axs[0].set_ylabel('Stage')

# 绘制mean reward图
axs[1].plot(num_steps_list, mean_rewards, label='Mean Reward', linestyle='-')
axs[1].set_title('Mean Reward')
axs[1].set_xlabel('num_steps')
axs[1].set_ylabel('Reward')

# 绘制success rate图
axs[2].plot(num_steps_list, success_rates, label='Success Rate', linestyle='-')
axs[2].set_title('Success Rate')
axs[2].set_xlabel('num_steps')
axs[2].set_ylabel('Rate')

# 调整图表布局
plt.tight_layout()
plt.show()
