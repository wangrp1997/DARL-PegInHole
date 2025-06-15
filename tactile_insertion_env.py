import os
from re import S
import sys
import hashlib
import glfw
import torch

from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from abr_control.utils import transformations
from robot_control import UR5Control

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from utils import math, torch_utils
from utils.common import *
from abr_control.arms.mujoco_config import MujocoConfig as arm

import shutil
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class TactileInsertionEnv(gym.Env):
    def __init__(self, use_torch = True, observation_type = "tactile_map", observation_noise = True, \
                normalize_tactile_obs = True, allow_translation = True, allow_rotation = True, \
                num_tac_obs_frames =5, tor_length = 32, action_xy_scale = 0.01, action_rot_scale=np.pi / 48., \
                action_type = 'relative', reward_type="delta", domain_randomization =False, \
                seed = 12, render_tactile = True, verbose=True, visual_enable=False,env_name=None):

        # asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        # model_path = os.path.join(asset_folder, 'tactile_insertion/tactile_insertion.xml')
        self.visual_enable = visual_enable
        self.allow_translation = allow_translation
        self.allow_rotation = allow_rotation

        self.working_space_boundary = torch.tensor([0.005, 0.005])
        self.working_rotation_boundary = np.pi / 24.
        
        self.max_error = np.array([0.005, 0.005, np.pi / 24.])

        self.grasp_force_range = np.array([1. / 8., 0.8])

        self.relative_translation_limit = action_xy_scale
        self.relative_translation_xy = None
        self.relative_rotation_angle_limit = action_rot_scale
        self.relative_rotation_angle = None
        self.action_type = action_type
        self.reward_type = reward_type

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.observation_noise = observation_noise
        self.normalize_tactile_obs = normalize_tactile_obs
        self.domain_randomization = domain_randomization
        self.verbose = verbose
        self.insertion = 0.005

        self.execution_num_steps = 45
        if self.observation_type == "privilege":
            self.ndof_obs = 2
            self.tac_obs_shape = (2,)
        elif self.observation_type == "tactile_flatten":
            self.tactile_rows = 4
            self.tactile_cols = 4
            self.tactile_initial_frame = 15
            self.tactile_samples = num_tac_obs_frames
            self.obs_frequency = (self.execution_num_steps - self.tactile_initial_frame) // self.tactile_samples
            self.ndof_obs = self.tactile_rows * self.tactile_cols * 2 * 2 * self.tactile_samples
            self.tac_obs_shape = (self.tactile_rows * self.tactile_cols * 2 * 2 * self.tactile_samples, )
        elif self.observation_type == "tactile_map":
            self.tactile_rows = 4
            self.tactile_cols = 4
            self.tactile_initial_frame = 12
            self.torque_samples = 32
            self.tactile_samples = num_tac_obs_frames
            self.obs_frequency = (self.execution_num_steps - self.tactile_initial_frame) // self.tactile_samples
            self.tac_obs_shape = (self.tactile_samples, 2, self.tactile_rows, self.tactile_cols)
        else:
            raise NotImplementedError
        self.tor_obs_shape = (tor_length, 3)
        self.tactile_masks = torch.zeros(self.execution_num_steps, dtype = bool)
        self.tactile_masks[self.tactile_initial_frame + self.obs_frequency - 1::self.obs_frequency] = True # observation frames
        self.tactile_masks[6] = True # reference frame
        
        self.mask_prob = torch.full((self.tactile_samples, 2, 1, 1, 1), 0.8)
        self.scale_mask = torch.ones((self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2))
        center_1 = np.array([8., 6.])
        center_2 = np.array([2.5, 5.5])
        for row in range(self.tactile_rows):
            for col in range(self.tactile_cols):
                if np.linalg.norm(np.array([row, col], dtype = float) - center_1) < 5. + 1e-5:
                    self.scale_mask[:, 0, row, col, :] = 0.4
                if np.linalg.norm(np.array([row, col], dtype = float) - center_2) < 3. + 1e-5:
                    self.scale_mask[:, 1, row, col, :] = 0.6
                
        self.scale_mask[:, 0, ]
        self.tactile_force_buf = torch.zeros(self.tac_obs_shape)
        self.tac_obs_buf = torch.zeros(self.tac_obs_shape)
        self.tor_obs_buf = torch.zeros(self.tor_obs_shape)

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.initial_sensor_locations = None
        self.grasp_force = 1.

        # self.ndof_q = self.sim.ndof_r
        # self.ndof_var = self.sim.ndof_var
        # self.ndof_u = self.sim.ndof_u
        # self.action_space = spaces.Box(low=np.full(self.ndof_u, -1.), high=np.full(self.ndof_u, 1.), dtype=np.float32)
        pos_low = np.array([-0.185, 0.45, -0.05, -0.05, -0.5], dtype=np.float32)
        pos_high = np.array([-0.075, 0.57, 0.05, 0.05, 0.5], dtype=np.float32)
        self.pos_space = spaces.Box(low=pos_low, high=pos_high, dtype=np.float32)


        tac_obs_tmp, tor_obs_tmp = self._get_obs()
        self.tac_observation_space = self.convert_observation_to_space(tac_obs_tmp)
        self.tor_observation_space = self.convert_observation_to_space(tor_obs_tmp)

        self.record_folder = os.path.join('./record/', get_time_stamp())
        self.record_idx = 0
        self.record_episode_idx = 0
        self.seed(seed=seed)

        self.render_mode = 'episode'
        self.render_tactile = render_tactile
        self.env_name = env_name
        # print(self.env_name)
        self.robot_config = arm(self.env_name, folder='./ur5', use_sim_state=False)
        self.h_threshold_dict = {"square_insertion.xml": 0.0985, "round_insertion.xml": 0.098,
                                 "triangle_insertion.xml": 0.098, "pentagon_insertion.xml":0.098,
                                 "hexagon_insertion.xml":0.098}
        self.h_threshold = self.h_threshold_dict[self.env_name]

        self.target_pos = [-0.126, 0.529, -3.14]

        self.generate_initial_pose()

        self.ndof_u = 0
        if self.allow_translation:
            if self.allow_rotation:
                self.ndof_u = 3
                self.action_scale = torch.tensor([self.relative_translation_limit, self.relative_translation_limit, self.relative_rotation_angle_limit])
            else:
                self.ndof_u = 2
                self.action_scale = torch.tensor([self.relative_translation_limit, self.relative_translation_limit])
        else:
            assert self.allow_rotation
            self.ndof_u = 1
            self.action_scale = torch.tensor([self.relative_rotation_angle_limit])

        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32)

        self.prev_object_pose = None
        self.only_allow_y = False
        self.stage = 2
        self.attempt_time = 0

    def convert_observation_to_space(self, observation):
        if hasattr(observation, 'shape'):
            if len(observation.shape) == 1:
                low = np.full(observation.shape, -float('inf'), dtype=np.float32)
                high = np.full(observation.shape, float('inf'), dtype=np.float32)
                space = spaces.Box(low, high, dtype=np.float32)
            else:
                space = spaces.Box(low=-np.inf, high=np.inf, shape=observation.shape, dtype=np.float32)
        else:
            return None
        return space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_initial_pose(self):
        dt = 0.002
        self.interface = Mujoco(self.robot_config, dt=dt, im=None, sensor_visual=False)
        self.interface.connect()
        self.ur5_control = UR5Control(self.interface, self.robot_config, n_timesteps=2000)
        self.interface.send_target_angles(self.robot_config.START_ANGLES)
        self.interface.set_target_ctrl(self.robot_config.START_ANGLES)
        self.interface.close_gripper()

        # sampled_pos = self.pos_space.sample()
        feedback = self.interface.get_feedback()
        R = self.robot_config.R("EE", q=feedback["q"])
        ori_orientation = transformations.euler_from_matrix(R, "sxyz")
        # target_orientation = list(map(lambda x, y: x + y, ori_orientation, sampled_pos[2:]))
        self.ur5_control.move_ee2(pos=[-0.1345,0.5095, 0.157], # 0.512 0.508
                             ori=ori_orientation)  # x: -0.185~c y:0.45 0.5
        # self.execute_insertion(100,-1,-1)
        # return initial_q, initial_qdot
        # cop = self.ur5_control.get_current_pose("peg")  # [x,y,theta]-0.126 0.529 -3.14
        # print(cop)
        # square： -0.12580494  0.52970698  0.10251228 -0.00359857 -0.02002258 -3.13873392
        # triangle：-0.12633942  0.52912557  0.10131655  0.01254291 -0.02828068 -3.1386516
        # round：-0.12641164  0.5291969   0.10115299  0.01326721 -0.02936699 -3.13861926
        # pentagon: -0.12637812  0.52915673  0.10122809  0.01303611 -0.02885713 -3.13861639
        # hexagon: -0.12637789  0.52915102  0.10123907  0.01301274 -0.02885775 -3.1386155
        # s = input()
        # self.ur5_control.move_ee1(det_pos=[0,0.0, -0.005], # 0.130最低
        #                      det_ori=[0,0,0])  # x: -0.185~c y:0.45 0.5
        # print(self.ur5_control.get_current_pose("peg")[2])
        # s = input()
        # hexagon 0.0960 pentagon 0.0960 round 0.0966 square 0.0974 triangle 0.0962

    def _get_obs(self):
        if not self.use_torch:
            return self.tac_obs_buf.detach().cpu().numpy(), self.tor_obs_buf.detach().cpu().numpy()
        else:
            return self.tac_obs_buf, self.tor_obs_buf

    '''
    reset by randomizing the initial gripper position and rotation (no rotation for now)
    '''
    def reset(self, num_updates, position_noise = None, rotation_noise = None, grasp_height_noise = None):
        self.attempt_time=0
        if self.stage == 0:
            self.allow_rotation, self.only_allow_y = False, True
        elif self.stage == 1:
            self.allow_rotation, self.only_allow_y = False, False
        else:
            self.allow_rotation, self.only_allow_y = True, False
        if position_noise is None:
            if self.allow_translation:
                position_noise = self.np_random.uniform(low = [-self.max_error[0], -self.max_error[1], -0.0002], high = [self.max_error[0], self.max_error[1], 0.0002], size = 3)
            else:
                position_noise = np.zeros(2)
            if self.only_allow_y:
                position_noise[0] = 0
        if rotation_noise is None:
            if self.allow_rotation:
                rotation_noise = self.np_random.uniform(low=-self.max_error[2], high=self.max_error[2])
            else:
                rotation_noise = 0.

        if grasp_height_noise is None:
            # grasp_height_noise = self.np_random.uniform(low = 0, high = 0.005)
            grasp_height_noise = 0


        # feedback = self.interface.get_feedback()
        # R = self.robot_config.R("EE", q=feedback["q"])
        # ori_orientation = transformations.euler_from_matrix(R, "sxyz")
        # print("ori before", ori_orientation) #ori before (3.140374707047834, -0.003278030630421441, -1.3718923644167694e-05)

        position_noise[2] = position_noise[2]+grasp_height_noise

        self.ur5_control.move_ee1(det_pos=position_noise, det_ori=[0,0,rotation_noise])

        # feedback = self.interface.get_feedback()
        # R = self.robot_config.R("EE", q=feedback["q"])
        # ori_orientation = transformations.euler_from_matrix(R, "sxyz")
        # print("ori before", ori_orientation) #ori before (3.1404654110045422, -0.003275287452662525, -0.12223026337794116)

        init_object_pose = self.ur5_control.get_current_pose("peg")

        self.prev_object_pose = torch.tensor([init_object_pose[0], init_object_pose[1], init_object_pose[5]])


        self.prev_error = torch.tensor([self.prev_object_pose[0]-self.target_pos[0],
                                        self.prev_object_pose[1]-self.target_pos[1],
                                       abs(self.prev_object_pose[2])-abs(self.target_pos[2])])  # -0.13515704  0.51026058  0.14965744

        prev_error = self.prev_error.clone()
        # domain randomization
        if self.domain_randomization:
            self.do_domain_randomization()

        self.execute_insertion(num_updates,-1,-1)
        
        return self.obs_buf, prev_error
    
    def do_domain_randomization(self):
        contact_kn_range = [2e3, 14e3]
        contact_kt_range = [20., 140.]
        contact_mu_range = [0.5, 2.5]
        contact_damping_range = [1e3, 1e3]

        tactile_kn_range = [50, 450]
        tactile_kt_range = [0.2, 2.3]
        tactile_mu_range = [0.5, 2.5]
        tactile_damping_range = [0, 100]

        contact_kn = self.np_random.uniform(*contact_kn_range)
        contact_kt = self.np_random.uniform(*contact_kt_range)
        contact_mu = self.np_random.uniform(*contact_mu_range)
        contact_damping = self.np_random.uniform(*contact_damping_range)

        self.sim.update_contact_parameters('tactile_pad_left', 'box', \
                                            kn = contact_kn,\
                                            kt = contact_kt,\
                                            mu = contact_mu,\
                                            damping = contact_damping)
        self.sim.update_contact_parameters('tactile_pad_right', 'box', \
                                            kn = contact_kn,\
                                            kt = contact_kt,\
                                            mu = contact_mu,\
                                            damping = contact_damping)
        
        tactile_kn = self.np_random.uniform(*tactile_kn_range)
        tactile_kt = self.np_random.uniform(*tactile_kt_range)
        tactile_mu = self.np_random.uniform(*tactile_mu_range)
        tactile_damping = self.np_random.uniform(*tactile_damping_range)

        self.sim.update_tactile_parameters('tactile_pad_left', \
                                            kn = tactile_kn,\
                                            kt = tactile_kt,\
                                            mu = tactile_mu,\
                                            damping = tactile_damping)
        self.sim.update_tactile_parameters('tactile_pad_right', \
                                            kn = tactile_kn,\
                                            kt = tactile_kt,\
                                            mu = tactile_mu,\
                                            damping = tactile_damping)     

        self.grasp_force = self.np_random.uniform(self.grasp_force_range[0], self.grasp_force_range[1])

    '''
    action is a 2-d vector specify the relative location change for the pre-grasp location
    '''
    def step(self, u, num_updates, epoch, step_num):


        if self.use_torch:
            action = torch.clip(u, -1., 1.)
        else:
            action = torch.clip(torch.tensor(u), -1., 1.)

        action_unnorm = action * self.action_scale
        print("归一化后的action",action_unnorm)

        if self.action_type == 'relative':
            # base_idx = 0
            clip_hx, clip_lx, clip_hy, clip_ly = self.get_clip_xy()
            if self.allow_translation:  # 第二个课程允许xy轴的平移
                relative_translation_xy = torch.clip(action_unnorm[0:2],torch.tensor([clip_lx,clip_ly]),
                                                     torch.tensor([clip_hx,clip_hy]))
                base_idx = 2
            else:
                relative_translation_xy = torch.zeros(2)
                base_idx = 0

            if self.only_allow_y:
                relative_translation_xy[0] = torch.tensor(0.)  # 第一个课程只允许y轴的平移

            if self.allow_rotation:  # 第三个课程允许xy轴的平移及z轴的旋转
                clip_hr, clip_lr = self.get_clip_r()
                relative_rotation_angle = torch.clip(action_unnorm[base_idx],clip_lr,clip_hr)
            else:
                relative_rotation_angle = torch.tensor(0.)

            print("当前xy相对移动为：",relative_translation_xy)
            self.ur5_control.move_ee1(det_pos=[relative_translation_xy[0],relative_translation_xy[1],0],
                                      det_ori=[0, 0, relative_rotation_angle])



        elif self.action_type == 'accumulative':
            base_idx = 0
            if self.allow_translation:
                relative_translation_xy = action_unnorm[0:2]
                base_idx = 2
            else:
                relative_translation_xy = torch.zeros(2)
                base_idx = 0

            if self.allow_rotation:
                relative_rotation_angle = action_unnorm[base_idx]
            else:
                relative_rotation_angle = torch.tensor(0.)
            feedback = self.interface.get_feedback()
            R = self.robot_config.R("EE", q=feedback["q"])
            Tx = self.robot_config.Tx("EE", q=feedback["q"])
            ori_orientation = transformations.euler_from_matrix(R, "sxyz")
            self.ur5_control.move_ee2(pos=[relative_translation_xy[0],relative_translation_xy[1], Tx[2]],
                                      ori=[ori_orientation[0],ori_orientation[1],relative_rotation_angle])  # x: -0.185~c y:0.45 0.57
            # self.current_q_init = self.apply_relative_motion(self.original_q_init, relative_translation_xy, relative_rotation_angle)
        else:
            raise NotImplementedError

        # real_action = torch.tensor([relative_translation_xy[0],relative_translation_xy[1],relative_rotation_angle])
        # self.info_buf['real_action'] = real_action.clone()
        self.execute_insertion(num_updates, epoch, step_num)
        self.info_buf['real_action'] = self.curr_error.clone()
        self.attempt_time += 1
        reward, done = self.reward_buf, self.done_buf
        
        if not self.use_torch:
            reward = reward.detach().cpu().item()

        obs = self.obs_buf
        
        return obs, reward, done, self.info_buf.copy()

    '''
    execute an insertion by starting from self.current_location_xy
    '''
    def execute_insertion(self,num_updates,epoch, step_num):
        in_count = 0
        done1 = False
        while True:
            print("执行插入")
            while True:
                origin_tactiles, origin_torques = self.ur5_control.move_ee1(det_pos=[0,0,-self.insertion],det_ori=[0,0,0])
                if origin_tactiles is not None:
                    break
            first_index = 0
            last_index = len(origin_tactiles)-1
            origin_tactiles_np = origin_tactiles.numpy()
            tactile_sum = np.sum(origin_tactiles_np, axis=(1, 2, 3))
            # tactile_sum = np.sum(np.linalg.norm(origin_tactiles[:, :, :, 0:2], axis=-1), axis=(1, 2))
            for i in range(len(origin_tactiles)):
                if tactile_sum[i] > 0:
                    if first_index == 0:
                        first_index = i
                    last_index = i
            selected_index = first_index

            if len(origin_tactiles[selected_index:last_index+1]) >= 6 and len(origin_torques[selected_index:last_index+1]) >= 33:
                selected_tactiles, selected_torques = self.select_data(origin_tactiles, origin_torques,
                                                                       selected_index, last_index)
                break
            else:
                cop = self.ur5_control.get_current_pose("peg")
                curr_error = torch.tensor([cop[0] - self.target_pos[0],
                                            cop[1] - self.target_pos[1],
                                            np.sign(cop[5]) * (abs(cop[5]) - abs(self.target_pos[2]))])  # -0.13515704  0.51026058   0.14965744
                # success = False  # target： [-1.34918169e-01  5.09271600e-01 5.39001626e-06]
                success = self.success_check(curr_error)

                if success:
                    selected_index = 0
                    last_index = len(origin_tactiles) - 1
                    selected_tactiles, selected_torques = self.select_data(origin_tactiles,origin_torques,
                                                                           selected_index,last_index)
                    break

                else:
                    print(f"Original data length is not sufficient. Re-collect！")
                    in_count += 1
                if in_count == 3:
                    done1 = True
                    selected_index = 0
                    last_index = len(origin_tactiles) - 1
                    selected_tactiles, selected_torques = self.select_data(origin_tactiles,origin_torques,
                                                                           selected_index,last_index)
                    break
                cop = self.ur5_control.get_current_pose("peg")
                if cop[2]<0.099:  # 高度也要调整
                    _ = self.ur5_control.move_ee1(det_pos=[0, 0, self.insertion], det_ori=[0, 0, 0])
                else:
                    _ = self.ur5_control.move_ee1(det_pos=[0, 0, self.insertion-0.002], det_ori=[0, 0, 0])
                # """

        feedback = self.interface.get_feedback()
        C_Tx = self.robot_config.Tx("EE", q=feedback["q"])
        self.info_buf['EE_Z'] = C_Tx[2].copy()
        # 转换为 PyTorch tensor
        selected_tactiles = torch.stack(selected_tactiles)
        # use the relative tactile forces
        selected_tactiles = selected_tactiles - selected_tactiles[0]
        tactiles = selected_tactiles[1:]

        selected_torques = torch.stack(selected_torques)
        # use the relative tactile forces
        selected_torques = selected_torques - selected_torques[0]
        torques = selected_torques[1:]

        # origin_force = origin_force - origin_force[0]
        # force = origin_force[1:]

        self.tactile_force_buf = tactiles.reshape(self.tactile_samples, self.tactile_rows, self.tactile_cols, 2).clone()
        self.torque_buf = torques.reshape(self.torque_samples, 3).clone()

        if self.observation_noise:
            noise_std = 0.00001
            noise0= self.np_random.randn(*self.tactile_force_buf.shape) * noise_std
            noise1= self.np_random.randn(*self.torque_buf.shape) * noise_std
            self.tactile_force_buf += torch.from_numpy(noise0)
            self.torque_buf += torch.from_numpy(noise1)

        if self.normalize_tactile_obs:
            self.tactile_force_buf = self.normalize_tactile(self.tactile_force_buf)
            self.tor_obs_buf = self.normalize_torque(self.torque_buf)

        if self.visual_enable:
            self.plot_torques_and_tactiles_with_highlight(origin_tactiles,origin_torques, selected_index, last_index)
            self.visualize_tactiles(self.tactile_force_buf)

        if self.observation_type == "tactile_flatten":
            self.tactile_force_buf = self.tactile_force_buf.reshape(-1)
        elif self.observation_type == "tactile_map":
            # self.obs_buf = np.linalg.norm(self.obs_buf, axis=-1)
            self.tac_obs_buf = self.tactile_force_buf \
                            .permute(0, 3, 1, 2)
                            # .reshape(-1, self.tactile_rows, self.tactile_cols)
        self.obs_buf = [self.tac_obs_buf, self.tor_obs_buf]

        current_obj_pose = self.ur5_control.get_current_pose("peg")

        self.info_buf['obj_rx'] = current_obj_pose[3].copy()
        self.current_object_pose = torch.tensor([current_obj_pose[0], current_obj_pose[1], current_obj_pose[5]])
        self.curr_error = torch.tensor([self.current_object_pose[0]-self.target_pos[0],
                                        self.current_object_pose[1]-self.target_pos[1],
                                        np.sign(current_obj_pose[5])*(abs(self.current_object_pose[2])-abs(self.target_pos[2]))])  # -0.13515704  0.51026058 0.14965744

        # success = False # target： [-1.34918169e-01  5.09271600e-01 5.39001626e-06]

        success = self.success_check(self.curr_error)
        if success:
            print("插入成功！")

        if done1:
            self.done_buf = True
        pose_normed_prev = self.prev_error / self.max_error
        pose_normed_now = self.curr_error / self.max_error

        improve = torch.norm(pose_normed_prev) > torch.norm(pose_normed_now)
        self.info_buf['prev_object_pose'] = self.prev_object_pose.detach().cpu().numpy()
        self.info_buf['new_object_pose'] = self.current_object_pose.detach().cpu().numpy()
        self.info_buf['improve'] = improve

        if self.reward_type == "absolute":
            self.reward_buf = (-torch.sum(self.curr_error[0:2] ** 2) * 10000 - torch.sum(self.curr_error[2] ** 2) * 20.
                               - 0.1*self.attempt_time)
        elif self.reward_type == "delta":
            self.reward_buf = ((torch.norm(self.prev_error / self.max_error) - torch.norm(self.curr_error / self.max_error)) * 10.
                               - 0.1*self.attempt_time)
            if success:
                if self.stage==0:
                    self.reward_buf += 10
                elif self.stage==1:
                    self.reward_buf += 20
                else:
                    self.reward_buf += 30.
            else:
                self.reward_buf -= 1.
        else:
            raise NotImplementedError
        self.prev_object_pose = self.current_object_pose
        self.prev_error = self.curr_error

        if success:
            if self.verbose:
                print_info('[Epoch {}/{} - Step {}] [Curriculum stage {}] [Success] dx {:.5f}, dy {:.5f}'.format(epoch,num_updates,step_num,self.stage,self.curr_error[0].detach().cpu().item(), self.curr_error[1].detach().cpu().item()), \
                        'dangle = {:.2f}'.format(self.curr_error[2].detach().cpu().item()),\
                        'reward = {:.3f}'.format(self.reward_buf.detach().cpu().item()), \
                        'position_reward = {:.3f}'.format(-(torch.sum(self.curr_error[0:2] ** 2) * 10000).detach().cpu().item()), \
                        'rotation_reward = {:.3f}'.format(-(torch.sum(self.curr_error[2] ** 2) * 20).detach().cpu().item()))
            self.done_buf = True
            self.info_buf['success'] = True
        else:
            if self.verbose:
                print_info('[Epoch {}/{} - Step {}] [Curriculum stage {}] [Failure] dx {:.5f}, dy {:.5f}'.format(epoch,num_updates,step_num,self.stage,self.curr_error[0].detach().cpu().item(), self.curr_error[1].detach().cpu().item()), \
                        'dangle = {:.2f}'.format(self.curr_error[2].detach().cpu().item()),\
                        'reward = {:.3f}'.format(self.reward_buf.detach().cpu().item()), \
                        'position_reward = {:.3f}'.format(-(torch.sum(self.curr_error[0:2] ** 2) * 10000).detach().cpu().item()), \
                        'rotation_reward = {:.3f}'.format(-(torch.sum(self.curr_error[2] ** 2) * 20).detach().cpu().item()))
                # print("当前高度;",current_obj_pose[2])
            self.done_buf = False
            self.info_buf['success'] = False
        if success or current_obj_pose[2] <= self.h_threshold:
            # if success:
            #     _ = self.ur5_control.move_ee1(det_pos=[0, 0, -3*self.insertion],det_ori=[0,0,0])
            #     _ = self.ur5_control.move_ee1(det_pos=[0, 0, 4*self.insertion],det_ori=[0,0,0])
            # else:
            _ = self.ur5_control.move_ee1(det_pos=[0, 0, self.insertion], det_ori=[0, 0, 0])
        else:
            if self.info_buf['EE_Z'] > 0.130 and abs(self.info_buf['obj_rx']) < 0.12:
                _ = self.ur5_control.move_ee1(det_pos=[0, 0, self.insertion-0.002], det_ori=[0,0,0])
                # _ = self.ur5_control.move_ee2(det_pos=[0, 0,  0.157], det_ori=[0,0,0])

            else:
                self.done_buf = True

    '''
    input: dimension (T, 2, nrows, ncols, 2)
    output: dimension (T, 2, nrows, ncols, 2)
    '''
    def success_check(self, curr_error):
        if not self.allow_rotation or self.env_name == "round_insertion.xml":
            success = (abs(curr_error[0]) <= 0.0015 and abs(curr_error[1]) <= 0.0015)
                       # abs(torch.mean(origin_torques[-8:, 0])) < 0.005 and abs(torch.mean(origin_torques[-8:, 1])) < 0.005)
        else:

            success = (abs(curr_error[0]) <= 0.0015 and abs(curr_error[1]) <= 0.0015 and abs(curr_error[2]) < np.pi / 48)
                       # abs(torch.mean(origin_torques[-8:, 0])) < 0.005 and abs(torch.mean(origin_torques[-8:, 1])) < 0.005)
        return success

    def select_data(self, tac_data, tor_data, start_index, final_index):
        step0 = (tac_data[start_index:final_index + 1].shape[0] - start_index) // 6
        selected_tactiles = []
        while not selected_tactiles:
            try:
                selected_tactiles = [tac_data[start_index + i * step0] for i in range(6)]
            except IndexError:
                print("IndexError: Reducing step size.")
                step0 = max(1, step0 - 1)  # 如果步长超出范围，将步长减去1

        selected_torques = []
        step1 = (tor_data[start_index:final_index + 1].shape[0] - start_index) // 33
        while not selected_torques:
            try:
                selected_torques = [tor_data[start_index + i * step1] for i in range(33)]
            except IndexError:
                print("IndexError: Reducing step size.")
                step1 = max(1, step1 - 1)  # 如果步长超出范围，将步长减去1
        return selected_tactiles, selected_torques

    def get_clip_xy(self):
        if self.curr_error[0] > 0:
            if self.curr_error[0] >= self.working_space_boundary[0]:
                clip_hx = 0
            else:
                clip_hx = self.working_space_boundary[0] - self.curr_error[0]
            clip_lx = -(self.curr_error[0] + self.working_space_boundary[0])
        else:
            if self.curr_error[0] <= -self.working_space_boundary[0]:
                clip_lx = 0
            else:
                clip_lx = -self.working_space_boundary[0] - self.curr_error[0]
            clip_hx = self.working_space_boundary[0] - self.curr_error[0]
        if self.curr_error[1] > 0:
            if self.curr_error[1] >= self.working_space_boundary[1]:
                clip_hy = 0
            else:
                clip_hy = self.working_space_boundary[1] - self.curr_error[1]
            clip_ly = -(self.curr_error[1] + self.working_space_boundary[1])
        else:
            if self.curr_error[1] <= -self.working_space_boundary[1]:
                clip_ly = 0
            else:
                clip_ly = -self.working_space_boundary[1] - self.curr_error[1]
            clip_hy = self.working_space_boundary[1] - self.curr_error[1]
        return clip_hx, clip_lx, clip_hy, clip_ly

    def get_clip_r(self):
        if self.curr_error[2] > 0:
            if self.curr_error[2] >= self.working_rotation_boundary:
                clip_hr = 0
            else:
                clip_hr = self.working_rotation_boundary - self.curr_error[2]
            clip_lr = -(self.curr_error[2] + self.working_rotation_boundary)
        else:
            if self.curr_error[2] <= -self.working_rotation_boundary:
                clip_lr = 0
            else:
                clip_lr = -self.working_rotation_boundary - self.curr_error[2]
            clip_hr = self.working_rotation_boundary - self.curr_error[2]
        return clip_hr,clip_lr

    def normalize_tactile(self, tactile_arrays,scale_factor=2.):
        lengths = torch.norm(tactile_arrays, dim=-1)
        
        max_length = np.max(lengths.numpy()) + 1e-5
        normalized_tactile_arrays = tactile_arrays / (max_length / scale_factor)
        
        return normalized_tactile_arrays

    def normalize_torque(self, torque_arrays, scale_factor=5.):
        lengths = torch.norm(torque_arrays, dim=-1)

        max_length = np.max(lengths.numpy()) + 1e-5
        normalized_torque_arrays = torque_arrays / (max_length / scale_factor)

        return normalized_torque_arrays
    def draw_tactile_force(self, img, loc0, loc1, tactile_force):
        end_loc0 = loc0 + int(tactile_force[0].item())
        end_loc1 = loc1 + int(tactile_force[1].item())
        
        color = (0., 1., 0.)
        
        img[:, :, :] = cv2.arrowedLine(img, (loc0, loc1), (end_loc0, end_loc1), color, 2, tipLength = 0.3)

    def draw_tactile_forces(self, img, tactile_forces, resolution):
        for i in range(tactile_forces.shape[0]):
            for j in range(tactile_forces.shape[1]):
                self.draw_tactile_force(img, i * resolution + resolution // 2, j * resolution + resolution // 2, tactile_forces[i, j])

    def plot_torques_and_tactiles_with_highlight(self,origin_tactiles, origin_torques,start_index, end_index ):
        # 获取三列扭矩数据
        torque_1 = origin_torques[:, 0]
        torque_2 = origin_torques[:, 1]
        torque_3 = origin_torques[:, 2]
        # fz = force[:,2]
        # 计算触觉数据的绝对值和（对第三维度的xy切向力求模，然后将每个(4, 4)区域的模相加）
        origin_tactiles_np = origin_tactiles.numpy()
        tactile_sum = np.sum(origin_tactiles_np, axis=(1, 2, 3))
        # tactile_sum = np.sum(np.linalg.norm(origin_tactiles[:, :, :, 0:2], axis=-1), axis=(1, 2))

        # 创建两行一列的图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # 找到急剧上升的转折点
        # gradient = np.gradient(tactile_sum)
        # upturn_points = np.where(gradient > 2)[0]
        #
        # # 找到第一个急剧上升的转折点
        # first_upturn_point = upturn_points[0]
        #
        # # 在图表中用红色点标出第一个急剧上升的转折点
        # ax1.plot(first_upturn_point, tactile_sum[first_upturn_point], 'ro')  # 'ro'表示红色圆点
        # 绘制第一行：触觉数据的和
        ax1.plot(tactile_sum, label='Tactile Sum')
        ax1.axvspan(start_index, end_index, color='gray', alpha=0.3, label='Selected Interval')
        ax1.set_title('Tactile Sum with Highlighted Sampling Interval')
        ax1.set_ylabel('Tactile Sum')

        # 绘制第二行：扭矩数据
        ax2.plot(torque_1, label='Torque 1')
        ax2.plot(torque_2, label='Torque 2')
        ax2.plot(torque_3, label='Torque 3')
        ax2.axvspan(start_index, end_index, color='gray', alpha=0.3, label='Selected Interval')

        # 创建次坐标轴用于绘制 Fz 数据
        # ax2_fz = ax2.twinx()
        # ax2_fz.plot(fz, label='Fz', linestyle='--', color='purple')  # 使用虚线区分 Fz
        # ax2_fz.set_ylabel('Fz')
        # ax2_fz.tick_params(axis='y', labelcolor='purple')  # 设置 Fz 坐标轴颜色
        #
        # ax2.set_title('3-axis Torques and Fz with Highlighted Sampling Interval')
        # ax2.set_xlabel('Time Steps')
        # ax2.set_ylabel('Torque Values')

        # 显示图例
        ax1.legend()
        # 显示图例
        ax2.legend(loc='upper left')
        # ax2_fz.legend(loc='upper right')

        # 调整布局
        plt.tight_layout()

        # 显示图表
        plt.show()

    def visualize_tactiles(self, tactile_force_buf):
        tactiles = tactile_force_buf.cpu().detach()
        # 获取向量场的尺寸
        size = tactiles.size()
        # 将 PyTorch tensor 转换为 NumPy array
        vector_field = tactiles.numpy()

        # 创建网格点坐标
        x, y = np.meshgrid(np.arange(size[2]), np.arange(size[1]))

        # 可视化每一帧的向量场
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        # fig.patch.set_facecolor('black')
        for i in range(5):
            axs[i].quiver(x, y, vector_field[i, :, :, 0], vector_field[i, :, :, 1], angles='xy', scale_units='xy',
                          scale=1, color='g', linewidth=4)
            axs[i].set_title(f'Frame {i}')
            axs[i].set_xlabel('X-axis')
            axs[i].set_ylabel('Y-axis')
            axs[i].set_facecolor('black')  # 设置子图的背景颜色为黑色
        # 调整子图之间的间距
        plt.tight_layout()
        # 显示图形
        plt.show()

    def render(self, mode='once'):
        if self.render_tactile:
            tactile_forces = self.get_tactile_forces_array()
            tactile_obs = self.get_tactile_obs_array()
            img_tactile_clean = self.visualize_tactile(tactile_forces)
            img_tactile_noise = self.visualize_tactile(tactile_obs)
            img_tactile = np.zeros(
                (img_tactile_clean.shape[0] + img_tactile_noise.shape[0], img_tactile_clean.shape[1], 3))
            img_tactile[0:img_tactile_clean.shape[0], :] = img_tactile_clean
            img_tactile[img_tactile_noise.shape[0]:, :] = img_tactile_noise

            cv2.imshow("tactile", img_tactile)
            print_info('Press [Esc] to continue...')
            cv2.waitKey(0)

        if mode == 'loop':
            print_info('Press [Esc] to continue...')

        super().render(mode)

        # return tactile obs array: shape (T, 2, nrows, ncols, 2)
    def get_tactile_obs_array(self):
        if self.observation_type == 'tactile_flatten':
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2)
        elif self.observation_type == 'tactile_map':
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, 2, self.tactile_rows, self.tactile_cols) \
                                .permute(0, 1, 3, 4, 2)
        
        return tactile_obs.detach().cpu().numpy()

    def get_tactile_forces_array(self):
        return self.tactile_force_buf.detach().cpu().numpy() / 0.000002

