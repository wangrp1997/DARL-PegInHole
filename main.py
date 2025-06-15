"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import glfw
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from robot_control import UR5Control
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from abr_control.utils import transformations
import time
import threading

# initialize our robot config for the jaco2
robot_config = arm("square_insertion.xml", folder='./ur5', use_sim_state=False)

maximum_range = 0.001  # 传感器最大量程，根据实际情况进行调整
sensor_data = np.zeros((4, 4, 2))  # 2D vector field (xy-direction forces)
# # 创建一个图形窗口并设置子图
# plt.figure()
# plt.suptitle('Sensor Data Visualization')
# plt.axis('on')  # 隐藏坐标轴
# sensor_subplot = plt.subplot(111)

# 初始化传感器数据和颜色范围
vmin, vmax = 0, 4

# 创建初始的colorbar
# im = sensor_subplot.imshow(sensor_data, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 4, 0, 4])  # 设置x轴和y轴范围
# quiver = sensor_subplot.quiver(np.arange(4), np.arange(4), sensor_data[:, :, 0], sensor_data[:, :, 1], scale=20, color='black')
# sensor_subplot.set_title('Sensor Data')

# cbar = plt.colorbar(im, ax=sensor_subplot)
# cbar.set_label('Sensor Value')  # 颜色条标题
# sensor_subplot.set_xlim(-1, 4)
# sensor_subplot.set_ylim(-1, 4)
# sensor_subplot.set_title('Vector Field')

# create our path planner
n_timesteps = 2000
# create our interface
dt = 0.002
interface = Mujoco(robot_config, dt=dt, im=None, sensor_visual=False)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)
interface.set_target_ctrl(robot_config.START_ANGLES)
interface.close_gripper()


flag = True
target = None
ur5_control = UR5Control(interface, robot_config, n_timesteps)
# ur5_control.move_ee1(det_pos=[0, 0, 0.003], det_ori=[0, 0, 0])
# feedback = interface.get_feedback()
# print("当前关节角度为：",feedback['q'])
# 位移的上限是正负5mm，角度的上下限是正负0.1弧度
low = np.array([-0.005, -0.005, -0.005, -0.1, -0.1, -0.1], dtype=np.float32)
high = np.array([0.005, 0.005, 0.005, 0.1, 0.1, 0.1], dtype=np.float32)

pos_low = np.array([-0.185, 0.45, -0.05, -0.05, -0.5], dtype=np.float32)
pos_high = np.array([-0.075, 0.57, 0.05, 0.05, 0.5], dtype=np.float32)
pos_space = spaces.Box(low=pos_low, high=pos_high, dtype=np.float32)

sampled_pos = pos_space.sample()
feedback = interface.get_feedback()
R = robot_config.R("EE", q=feedback["q"])
ori_orientation = transformations.euler_from_matrix(R, "sxyz")
target_orientation = list(map(lambda x, y: x + y, ori_orientation, sampled_pos[2:]))
ur5_control.move_ee2(pos=[-0.126, 0.529, 0.157],ori=ori_orientation)  # x: -0.185~c y:0.45 0.57
# ur5_control.move_ee1(det_pos=[0., 0.002, 0], det_ori=[0, 0, 0])

# 创建6维连续动作空间
action_space = spaces.Box(low=low, high=high, dtype=np.float32)



try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")
    # print(robot_config.data.body('hole').xipos)
    p1 = ur5_control.get_current_pose("hole")
    while not glfw.window_should_close(interface.viewer.window):
        # 尝试装
        if flag:
            pass
            # while True:
            ur5_control.move_ee1(det_pos=[0, 0, -0.02], det_ori=[0, 0, 0])
            flag = False
            #     ur5_control.move_ee1(det_pos=[0, 0, -0.005], det_ori=[0, 0, 0])
            #     ur5_control.move_ee1(det_pos=[0, 0, -0.003], det_ori=[0, 0, 0])
            # # C_Tx = robot_config.Tx("EE", q=feedback["q"]).copy()
            #     current_obj_pose = ur5_control.get_current_pose("peg")
            #
            #     print(current_obj_pose[3].copy())

            # ur5_control.move_ee1(det_pos=[0, 0, 0.02], det_ori=[0, 0, 0])
            # p2 = ur5_control.get_current_pose("hole")
            # print(p1)
            # print(p2)
        #
        # for t in range(1000):
        #     # print(robot_config.model.joint("slide_joint").damping[0])
        #     # 采样一个动作
        #     sampled_action = action_space.sample()
        #     # print("Sampled Action:", sampled_action)
        #     # ur5_control.move_ee1(det_pos=[-0.01,0,0],det_ori=[0,0,0])
        #
        #     # ur5_control.joint_control([0,0,0,0,0,1])
        #     interface.viewer.render()
        # ur5_control.move_ee1(det_pos=[0, 0, -0.002], det_ori=[0, 0, 0])
        # feedback = interface.get_feedback()
        # C_Tx = robot_config.Tx("EE", q=feedback["q"]).copy()
        # print("当前末端高度：", C_Tx[2])
        interface.viewer.render()

finally:
    # stop and reset the simulation
    p3 = ur5_control.get_current_pose("hole")
    print(p3)
    interface.disconnect()
    # plot_process.terminate()
    print("Simulation terminated...")


