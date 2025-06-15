import math
import time

from abr_control.utils import transformations
from abr_control.controllers import path_planners
import numpy as np
import torch
import mujoco
import threading
from utils.torch_utils import *
import matplotlib.pyplot as plt


class UR5Control:
    def __init__(self, interface, robot_config, n_timesteps):
        self.path_planner = path_planners.InverseKinematics(robot_config)
        self.interface = interface
        self.robot_config = robot_config
        self.n_timesteps = n_timesteps
        self.tolerance = 1
        self.gripper_ctrl_LIMIT = self.robot_config.model.actuator("fingers_actuator").ctrlrange[1]
        self.init = True
        self.start = True
        self.sensor_data = np.zeros((4, 4, 2))  # 2D vector field (xy-direction forces)

    def get_current_pose(self, name, object_type="body"):
        Tx = self.robot_config.Tx(name, object_type=object_type)
        R = self.robot_config.R(name)
        euler_ori = transformations.euler_from_matrix(R, "sxyz")
        return np.concatenate((Tx,euler_ori))

    def render_move(self,render_interval=2):
        render_counter = 0
        while self.path_planner.n < self.path_planner.n_timesteps/2:
            target = self.path_planner.next()[0]
            # use position control
            self.interface.set_target_ctrl(target[: self.robot_config.N_JOINTS])
            # 使用位置控制
            # 每隔一定步数执行一次渲染
            # if render_counter == render_interval:
            #     self.interface.viewer.render()
            #     render_counter = 0
            # else:
            #     mujoco.mj_step(self.robot_config.model, self.robot_config.data)
            #     render_counter += 1
        self.interface.viewer.render()

    def move_ee1(self,det_pos, det_ori):
        feedback = self.interface.get_feedback()
        Tx = self.robot_config.Tx("EE", q=feedback["q"], object_type="body")
        # target_xyz = np.array([Tx[0]+det_pos[0], Tx[1]+det_pos[1], Tx[2]+det_pos[2]])
        target_xyz = list(map(lambda x,y: x+y, Tx, det_pos))
        R = self.robot_config.R("EE", q=feedback["q"])
        # quat = self.robot_config.quaternion("EE", q=feedback["q"])
        if self.init:
            self.target_orientation = transformations.euler_from_matrix(R, "sxyz")
            self.init = False

        self.target_orientation = list(map(lambda x,y:x+y, self.target_orientation, det_ori))
        # target_orientation = np.clip(target_orientation, -np.pi, np.pi)
        # update the position of the target
        # self.interface.set_mocap_xyz("target", target_xyz)
        # can use 3 different methods to calculate inverse kinematics
        # see inverse_kinematics.py file for details
        self.path_planner.generate_path(
            position=feedback["q"],
            target_position=np.hstack([target_xyz, self.target_orientation]),
            method=2,
            dt=0.002,
            n_timesteps=1800,
            plot=False, )

        self.start = True
        sensor_thread = threading.Thread(target=self.sensor_thread,
                                         args=(self.robot_config.model, self.robot_config.data))
        sensor_thread.start()
        self.render_move(render_interval=0)
        self.start = False
        if sensor_thread:
            sensor_thread.join()  # 等待线程结束
        print("采集结束！")
        """
        plt.plot(list(range(0, len(self.collected_data))), self.collected_data,linestyle='-', color='b', label='Total Magnitude')
        # Adding labels and title
        plt.xlabel('Time')
        plt.ylabel('Total Magnitude')
        plt.title('Total Magnitude Over Iterations')
        # Displaying legend
        plt.legend()
        # Display the plot
        plt.show()
        """
        try:
            collected_touch_data = torch.stack(self.touch_data, dim=0)
            collected_torque_data = torch.stack(self.torque_data, dim=0)
            # collected_force_data = torch.stack(self.force_data, dim=0)
            # return collected_touch_data, collected_torque_data, collected_force_data
            return collected_touch_data, collected_torque_data

        except:
            return None, None, None



    def move_ee2(self, pos,ori):
        feedback = self.interface.get_feedback()
        target_xyz = np.array([pos[0], pos[1], pos[2]])
        R = self.robot_config.R("EE", q=feedback["q"])
        # target_orientation = transformations.euler_from_matrix(R, "sxyz")
        target_orientation = ori
        if self.init:
            self.target_orientation = target_orientation
            self.init = False
        # update the position of the target
        # self.interface.set_mocap_xyz("target", target_xyz)
        # can use 3 different methods to calculate inverse kinematics
        # see inverse_kinematics.py file for details
        self.path_planner.generate_path(
            position=feedback["q"],
            target_position=np.hstack([target_xyz, target_orientation]),
            method=2,
            dt=0.002,
            n_timesteps=2000,
            plot=False, )

        self.render_move(render_interval=1)

    def sensor_thread(self, model, data):
        # global sensor_data
        print("采集开始")
        self.touch_data = []
        self.torque_data = []
        # self.force_data = []
        counter = 0
        while self.start:
            counter+=1
            # total_magnitude = 0.0
            torque_data = data.sensor('torque0').data.copy()
            # force_data = data.sensor('force').data.copy()
            for i in range(16):
                sensor_name = f'touch{i}'
                # data_value = data.sensor(sensor_name).data[2].copy()
                # self.sensor_data[i // 4, i % 4] = abs(data_value)
                x_value = data.sensor(sensor_name).data[0].copy()  # Assuming index 0 is the x-component
                y_value = data.sensor(sensor_name).data[2].copy()  # Assuming index 2 is the y-component
                self.sensor_data[i // 4, i % 4] = [x_value, y_value]
                # magnitude = math.sqrt(x_value ** 2 + y_value ** 2)
                # total_magnitude += magnitude
            print(counter)
            if counter % 15 == 0:
                sd = self.sensor_data.copy()
                self.touch_data.append(to_torch(sd))
                self.torque_data.append(to_torch(torque_data))
                # self.force_data.append(to_torch(force_data))
                # self.collected_data.append(total_magnitude)

    def joint_control(self, detal_array):
        current_q = self.interface.get_feedback()['q'].copy()
        new_q = current_q + detal_array
        t0 = time.time()
        while 1:
            self.interface.set_target_ctrl(new_q)
            self.interface.viewer.render()
            error = abs(self.interface.get_feedback()['q'].copy() - new_q)
            # print(max(abs(error)))
            if max(error) < 0.012 or time.time()-t0 > 10:
                break

    def close_gripper(self, current_target_joint_values):
        # self.model.actuator("fingers_actuator").forcerange[1] += 1
        # self.robot_config.data.qpos[6] = 1
        # 完全松开 [0. 0. 0. 0. 0. 0. 0. 0.]
        # 完全闭合 [0.79642269  0.0043969   0.79481524 - 0.78996885  0.79642246  0.00437121
        #  0.79478977 - 0.78995321]
        target_ctrl = math.ceil(current_target_joint_values*self.gripper_ctrl_LIMIT/0.8)
        prev_time = time.time()
        while 1:
            current_joint_values = self.robot_config.data.qpos[6]
            deltas = abs(current_target_joint_values - current_joint_values)
            if deltas < self.tolerance or time.time() - prev_time > 4:
                self.robot_config.data.ctrl[6] = math.ceil(current_joint_values*self.gripper_ctrl_LIMIT/0.796)+5
                self.interface.viewer.render()
                break
            # self.robot_config.data.actuator("fingers_actuator").ctrl += 1
            self.robot_config.data.ctrl[6] = target_ctrl
            mujoco.mj_forward(self.robot_config.model, self.robot_config.data)
            self.interface.viewer.render()
