import os
import csv
from datetime import datetime
import numpy as np
# import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RobotSimulation:
    def __init__(self, seed=None):
        # 配置初始位置和最大误差
        self.initial_target_pos = np.array([-0.1345, 0.5095, 0.157])
        self.max_init_error = 0.005  # 误差范围 ±0.005
        self.max_insert_error = 0.0015
        self.np_random = np.random.default_rng(seed)  # 随机数生成器

        # 定义结果保存的文件夹路径
        self.log_dir = './base_line_results'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)  # 如果文件夹不存在，则创建

    def reset(self):
        # 添加噪声，注意z轴误差是固定范围±0.0002
        while True:
            position_noise = self.np_random.uniform(
                low=[-self.max_init_error, -self.max_init_error, -0.0002],
                high=[self.max_init_error, self.max_init_error, 0.0002],
                size=3
            )
            # 初始位置加上噪声
            init_object_pose = self.initial_target_pos + position_noise

            # 检查机器人是否在目标的误差范围内
            if (abs(init_object_pose[0] - self.initial_target_pos[0]) > self.max_insert_error or
                    abs(init_object_pose[1] - self.initial_target_pos[1]) > self.max_insert_error):
                break  # 退出循环，表示机器人不在误差范围内，初始化成功
            else:
                # 如果机器人在误差范围内，重新初始化
                print(f"Initial pose {init_object_pose} is within the error range. Reinitializing...")

        # 打印并返回初始位置
        return init_object_pose

    def archimedes_spiral_search(self, init_pose, target_pose, max_steps=10, step_size=0.0005, initial_radius=0.0005):
        # Archimedes螺旋搜索
        radius = initial_radius  # 初始半径设为较大的值
        angle = 0.0
        current_pose = np.copy(init_pose)

        # 存储轨迹点
        trajectory_x = [current_pose[0]]  # 初始位置
        trajectory_y = [current_pose[1]]  # 初始位置

        # 从第一步开始，步数不计入初始位置
        step_count = 0

        for step in range(max_steps):
            # 计算当前的螺旋路径点
            delta_x = radius * np.cos(angle)
            delta_y = radius * np.sin(angle)
            # 模拟机器人的新位置
            current_pose = init_pose + np.array([-delta_x, -delta_y, 0.0])

            # 存储轨迹点
            trajectory_x.append(current_pose[0])
            trajectory_y.append(current_pose[1])

            # 计算当前位置与目标位置的误差
            self.curr_error = current_pose - target_pose
            # 使用新的成功条件
            success = (abs(self.curr_error[0]) <= self.max_insert_error and abs(self.curr_error[1]) <= self.max_insert_error)

            if success:  # 检查是否到达目标位置
                print("current error：", self.curr_error[0], self.curr_error[1])
                return True, step_count + 1, trajectory_x, trajectory_y  # 返回成功、步数和轨迹点

            # 更新螺旋的半径和角度
            radius += step_size
            angle += np.pi / 3
            step_count += 1

        return False, step_count, trajectory_x, trajectory_y  # 超过最大步骤数，未能到达目标

    def square_spiral_search(self, init_pose, target_pose, max_steps=10, step_size=0.002):
        """方形螺旋搜索"""
        current_pose = np.copy(init_pose)
        trajectory_x, trajectory_y = [current_pose[0]], [current_pose[1]]
        step_count = 0
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0), ]  # 顺时针：右、上、左、下
        direction_index = 0
        flag = 1
        flag_1 = 0

        for i in range(max_steps):
            dx, dy = directions[direction_index]
            for i in range(flag):
                current_pose += np.array([dx * step_size, dy * step_size, 0.0])
                trajectory_x.append(current_pose[0])
                trajectory_y.append(current_pose[1])
                step_count += 1
                self.curr_error = current_pose - target_pose
                if (abs(self.curr_error[0]) <= self.max_insert_error and
                    abs(self.curr_error[1]) <= self.max_insert_error):
                    return True, step_count, trajectory_x, trajectory_y
                if step_count == max_steps:
                    flag_1 = 1
                    break
            if flag_1:
                break

            direction_index = (direction_index + 1) % 4  # 顺时针切换方向
            if direction_index % 2 == 0:  # 每完成半圈增加步长
                flag += 1

        return False, step_count, trajectory_x, trajectory_y

    def windmill_search(self, init_pose, target_pose, max_steps=10, step_size=0.003):
        """
        风车搜索算法：严格按照序号顺序执行
        - init_pose: 初始位置
        - target_pose: 目标位置
        - max_steps: 最大步数
        - step_size: 每次步长的增量
        """
        current_pose = np.copy(init_pose)
        trajectory_x, trajectory_y = [current_pose[0]], [current_pose[1]]
        step_count = 0

        # 定义风车轨迹的方向，按图中序号顺序
        directions = [
            (-1, -1),  # 1：向下
            (0, 1),  # 2：向左
            (1, 0),  # 3：向右
            (1, 0),  # 4：向上
            (0, 1),  # 5：向右
            (-1, -1),  # 6：向下
            (0, 1),  # 7：向左
            (-1, 0),  # 8：向上
            (1, -1),  # 9：向左
            (1, -1),  # 10：向下
        ]

        # 执行搜索
        for i in range(len(directions)):
            if step_count >= max_steps:
                break

            # 当前方向
            direction = directions[i]

            # 计算当前方向的移动
            delta_x = direction[0] * step_size
            delta_y = direction[1] * step_size

            # 更新当前位置
            current_pose = current_pose + np.array([delta_x, delta_y, 0.0])
            trajectory_x.append(current_pose[0])
            trajectory_y.append(current_pose[1])

            # 检查当前位置与目标位置的误差
            self.curr_error = current_pose - target_pose
            if (abs(self.curr_error[0]) <= self.max_insert_error and
                    abs(self.curr_error[1]) <= self.max_insert_error):
                return True, step_count + 1, trajectory_x, trajectory_y

            # 增加步数
            step_count += 1

            # 每次按序号方向走一步
            # step_size += 0.05  # 每次增大步长

        return False, step_count, trajectory_x, trajectory_y

    def raster_search(self, init_pose, target_pose, max_steps=10, n=2, step_size=0.002, width=0.001):
        """栅格搜索"""
        current_pose = np.copy(init_pose)
        trajectory_x, trajectory_y = [current_pose[0]], [current_pose[1]]
        step_count = 0
        moving_reverse = True  # 标记当前方向
        flag = 1

        if target_pose[0] > init_pose[0] and target_pose[1] < init_pose[1]:  # 目标在下方
            diretion = 0  # 优先向下
        elif target_pose[0] > init_pose[0] and target_pose[1] > init_pose[1]:  # 目标在右侧
            diretion = 1  # 优先向右
        elif target_pose[0] < init_pose[0] and target_pose[1] > init_pose[1]:  # 目标在上方
            diretion = 2  # 优先向上
        elif target_pose[0] < init_pose[0] and target_pose[1] < init_pose[1]:   # 目标在左侧
            diretion = 3  # 优先向左


        if diretion==1:
            width = width
            step_size = -step_size
        if diretion == 2:
            width = -width
            step_size = -step_size
        if diretion==3:
            width = -width
            step_size = step_size
        for i in range(0,max_steps):
            if step_count % n == 0 and step_count != 0 and flag:  # 换行时垂直移动
                current_pose[0] += width
                moving_reverse = not moving_reverse  # 方向反转
                flag = 0
            elif (step_count-n) % (n+1) == 0 and step_count != 0 and step_count != n+1:
                current_pose[0] += width
                moving_reverse = not moving_reverse  # 方向反转
            else:  # 横向移动
                delta_y = -step_size if moving_reverse else step_size
                current_pose[1] += delta_y
            trajectory_x.append(current_pose[0])
            trajectory_y.append(current_pose[1])

            self.curr_error = current_pose - target_pose
            if (abs(self.curr_error[0]) <= self.max_insert_error and
                abs(self.curr_error[1]) <= self.max_insert_error):
                return True, step_count + 1, trajectory_x, trajectory_y

            step_count += 1

        return False, step_count, trajectory_x, trajectory_y

    def plot_trajectory(self, trajectory_x, trajectory_y, target_pos, round_num):
        # 绘制轨迹图
        plt.figure(figsize=(6, 6))
        plt.plot(trajectory_x, trajectory_y, marker='o', label=f"Round {round_num} Path")  # 从第二个点开始绘制路径
        plt.scatter(target_pos[0], target_pos[1], color='red', marker='*', s=100, label="Target Position")

        # 绘制初始位置，使用绿色标记
        plt.scatter(trajectory_x[0], trajectory_y[0], color='green', marker='o', s=100, label="Initial Position")

        # 定义矩形的宽度和高度为 max_insert_error
        rect_width = self.max_insert_error * 2  # 宽度是误差范围的两倍
        rect_height = self.max_insert_error * 2  # 高度是误差范围的两倍
        # 创建矩形（从中心位置出发）
        rectangle = patches.Rectangle((target_pos[0] - self.max_insert_error, target_pos[1] - self.max_insert_error),
                                      rect_width, rect_height, linewidth=1.5, edgecolor='r', facecolor='none',
                                      linestyle='--')

        # 在当前的图形中添加矩形
        plt.gca().add_patch(rectangle)

        # 自动调整坐标轴范围，确保显示完整的轨迹和误差圆
        margin = 0.005  # 增加一个小的边距，确保误差圆和轨迹完全可见
        x_min = min(trajectory_x) - margin - self.max_init_error  # 增加误差圆的半径
        x_max = max(trajectory_x) + margin + self.max_init_error  # 增加误差圆的半径
        y_min = min(trajectory_y) - margin - self.max_init_error  # 增加误差圆的半径
        y_max = max(trajectory_y) + margin + self.max_init_error  # 增加误差圆的半径

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # 添加图例和标签
        plt.legend()
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"Trajectory and Target for Round {round_num}")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')  # 确保X、Y轴等比例
        plt.show()

    def run_tests(self, num_tests, search_method="archimedes"):
        search_methods = {
            "archimedes": self.archimedes_spiral_search,
            "square_spiral": self.square_spiral_search,
            "windmill": self.windmill_search,
            "raster": self.raster_search,
        }

        if search_method not in search_methods:
            raise ValueError(f"Invalid search method: {search_method}")

        success_count = 0
        # 创建一个时间戳用于文件命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 使用 datetime.now()
        log_filename = os.path.join(self.log_dir, f"{timestamp}_{search_method}.csv")

        # 创建并初始化CSV文件
        with open(log_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 写入表头
            csv_writer.writerow(["Test Round", "Attempt", "Success", "Cumulative Success Rate (%)"])

        # 执行多个测试
        total_steps = 0  # 累计成功回合的步数
        for round_num in range(1, num_tests + 1):
            initial_pose = self.reset()
            # 测试是否能在螺旋搜索中到达目标位置，并记录尝试次数
            success, steps, trajectory_x, trajectory_y = search_methods[search_method](initial_pose, self.initial_target_pos)
            total_steps += steps  # 只在成功的回合中累计步数

            if success:
                success_count += 1

            # 计算当前的成功率
            success_rate = (success_count / round_num) * 100

            # 追加写入每次测试的结果（成功用1表示，失败用0表示）
            with open(log_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([round_num, steps, 1 if success else 0, success_rate])

            # 绘制轨迹和目标位置
            self.plot_trajectory(trajectory_x, trajectory_y, self.initial_target_pos, round_num)

        print(f"Results saved to: {log_filename}")
        print(f"Success rate: {success_count / num_tests * 100}%")
        print(f"Average steps to reach target: {total_steps / num_tests:.2f}")


# 示例运行
sim = RobotSimulation(seed=42)  # 传入固定的随机数种子
# search_methods = {
#     "archimedes": self.archimedes_spiral_search,
#     "square_spiral": self.square_spiral_search,
#     "windmill": self.windmill_search,
#     "raster": self.raster_search,
# }
sim.run_tests(20, search_method="square_spiral")
