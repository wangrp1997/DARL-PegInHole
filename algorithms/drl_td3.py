import sys, os

import torch

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(base_dir)

from utils.common import *

import gym

gym.logger.set_level(40)
import numpy as np
import time
from collections import deque
import copy
import yaml
from tensorboardX import SummaryWriter
from datetime import datetime  # 修改为直接导入 datetime 类
import csv

# from td3.td3 import TD3
# from td3.td3 import Replay_buffer
from td3.td3 import *

from utils import model
from tactile_insertion_env import *
# from data_sample import save_data
from torch.utils.data import DataLoader
from train_sl_insertion import load_data


class TD3Agent:
    def __init__(self, cfg):
        torch.set_num_threads(1)  # TODO: check the effect

        self.seed = cfg['params']['general']['seed']
        self.env_name = cfg['params']['env']['name']
        self.device = cfg['params']['general'].get('device', 'cpu')
        env_params = copy.deepcopy(cfg['params']['env'])
        env_params.pop('name', None)

        set_random_seed(self.seed)
        # create render env
        # self.render_env = gym.make(self.env_name, verbose=cfg['params']['general']['render'], **env_params)
        self.env_para = input("please input env name：")

        self.render_env = TactileInsertionEnv(env_name=self.env_name)
        self.render_env.seed(self.seed)
        # create policy
        tac_obs_shape = (5, 2, 4, 4)
        tor_obs_shape = (32, 3)
        self.tac_input_channels = tac_obs_shape[1]
        self.torque_input_channels = tor_obs_shape[1]
        self.tactile_hidden_size = 64
        self.torque_hidden_size = 64
        self.output_size = 2

        actor_critic_fn = getattr(model, cfg['params']['network']['actor_critic'])

        """
        self.pretrained_model = model.TacTorModel(self.tac_input_channels, self.torque_input_channels,
                                                  self.tactile_hidden_size,
                                                  self.torque_hidden_size, self.output_size)
        # 加载保存的模型权重
        self.pretrained_model.load_state_dict(torch.load(cfg['params']['network']['pretrained_model']))

        # 加载特征提取器部分全参数
        self.feature_extractor = model.FeatureExtractor(self.tac_input_channels, self.torque_input_channels,
                                                        self.tactile_hidden_size,
                                                        self.torque_hidden_size)

        self.feature_extractor.cnn.load_state_dict(self.pretrained_model.cnn.state_dict())
        self.feature_extractor.tactile_lstm.load_state_dict(self.pretrained_model.tactile_lstm.state_dict())
        self.feature_extractor.torque_lstm.load_state_dict(self.pretrained_model.torque_lstm.state_dict())
        self.feature_extractor.fc_cnn.load_state_dict(self.pretrained_model.fc_cnn.state_dict())
        self.feature_extractor.dropout_cnn.load_state_dict(self.pretrained_model.dropout_cnn.state_dict())
        """

        self.actor_critic = actor_critic_fn(128, self.render_env.action_space.shape[0], cfg['params']['network'])
        # self.actor_critic.actor_net.mean_net.load_state_dict(self.pretrained_model.fc_layers.state_dict()) # 新的动作输出为三维的

        self.actor_critic.to(self.device)
        discriminator = model.Discriminator()
        self.adversarial_trainer = model.AdversarialTrainer(self.feature_extractor, discriminator)

        self.target_domain_file = os.path.join("data_folder", self.env_name.split("_")[0] + "_domain_data.pkl")
        self.adv_update_frequency = 101
        self.adv_num_epochs = cfg['params']['network']['adv_num_epochs']

        if cfg['params']['general']['train']:
            # total num steps
            self.total_env_steps = cfg['params']['config']['num_env_steps']
            self.num_steps = cfg['params']['config']['num_steps']
            self.num_processes = cfg['params']['config']['num_processes']
            self.use_linear_lr_decay = cfg['params']['config'].get('use_linear_lr_decay', True)
            self.use_proper_time_limits = cfg['params']['config'].get('use_proper_time_limits', True)
            self.lr = cfg['params']['config'].get('lr', 3e-4)
            self.use_gae = cfg['params']['config'].get('use_gae', False)
            self.gae_lambda = cfg['params']['config'].get('gae_lambda', 0.95)
            self.gamma = cfg['params']['config'].get('gamma', 0.99)

            # create envs
            # self.envs = make_vec_envs(
            #     env_name = self.env_name,
            #     seed = self.seed,
            #     num_processes = self.num_processes,
            #     gamma = self.gamma,
            #     log_dir = None,
            #     device = self.device,
            #     allow_early_resets = False,
            #     norm_reward = cfg['params']['config'].get('norm_reward', True),
            #     norm_obs = cfg['params']['config'].get('norm_obs', True),
            #     clip_obs = cfg['params']['config'].get('clip_obs', 10.),
            #     clip_reward = cfg['params']['config'].get('clip_reward', 10.),
            #     **env_params)
            self.log_dir = cfg["params"]["general"]["logdir"]

            self.log_dir = f"{self.log_dir}-{self.env_para}"

            # create TD3 agent
            self.agent = TD3(128, self.render_env.action_space.shape[0], float(self.render_env.action_space.high[0]),
                             self.feature_extractor, self.log_dir, self.pretrained_model)

            self.rollouts = Replay_buffer()

            # interval-related arguments
            self.save_interval = cfg['params']['general'].get('save_interval', 5)
            self.log_interval = cfg['params']['general'].get('log_interval', 1)
            self.render_interval = cfg['params']['general'].get('render_interval', 20)

            # create logging folder

            os.makedirs(self.log_dir, exist_ok=True)

            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))

            # create logging file
            self.training_log_path = os.path.join(self.log_dir, 'logs.txt')
            fp_log = open(self.training_log_path, 'w')
            fp_log.close()

            # initialize writer
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))

            # 初始化Matplotlib图形
            # self.fig, self.ax = plt.subplots()
            # self.line, = self.ax.plot([], [], '-', label='Reward')
            # self.ax.set_title('Real-time Reward Visualization')
            # self.ax.set_xlabel('Step')
            # self.ax.set_ylabel('Reward')
            # self.ax.legend()

    def train_reset(self, num_updates, step):
        while True:
            obs, sl_action = self.render_env.reset(num_updates)
            if self.render_env.info_buf['success'] == True or abs(self.render_env.info_buf['obj_rx']) >= 0.3:
                print("Invalid initialization, reset again.")
                # self.render_env.interface.disconnect()
                # self.render_env.generate_initial_pose()
            else:
                break
        # save_data(self.render_env.info_buf['success'], obs, sl_action, self.target_domain_file)

        return obs, sl_action

    def visualize_reward(self, reward, visual=False):
        # 记录奖励
        with open(os.path.join(self.log_dir, 'step_rewards.txt'), 'a') as file:
            file.write(f'{reward}\n')

        if visual:
            # 读取奖励并更新图形
            with open(os.path.join(self.log_dir, 'step_rewards.txt'), 'r') as file:
                rewards = [float(line.strip()) for line in file.readlines()]

            # 更新图形
            x_data = list(range(len(rewards)))
            self.line.set_data(x_data, rewards)
            # 动态调整 x 和 y 轴范围
            self.ax.relim()
            self.ax.autoscale_view()
            plt.show(block=False)
            # 暂停一段时间后自动关闭
            plt.pause(2)

    def train(self):

        model_save_dir = os.path.join(self.log_dir, 'models')
        os.makedirs(model_save_dir, exist_ok=True)
        # os.makedirs(os.path.dirname(self.target_domain_file), exist_ok=True)
        source_data = load_data("data_folder/round_hole_data10000.pkl")
        source_dataset = [data for data in source_data if data[0] == 0]

        episode_rewards = deque(maxlen=200)
        episode_lens = deque(maxlen=200)
        # num_updates = int(
        #     self.total_env_steps) // self.num_steps // self.num_processes
        num_updates = int(input("input epoch num:"))

        scale_factor_1 = 0.01
        scale_factor_2 = 0.01
        scale_factor_3 = torch.pi / 36
        # state, sl_action = self.train_reset(num_updates,0)

        start = time.time()

        last_eval_rew = 0.
        # best_eval_rew = -1000.
        best_success_rate = -0.5
        best_len = np.inf
        success_cnt = 0
        reset = True

        for j in range(0, num_updates):

            success_episode_cnt = 0
            total_episode_cnt = 0
            episode_reward = 0
            episode_len = 0

            state, sl_action = self.render_env.reset(num_updates)
            sl_action[0] /= -scale_factor_1
            sl_action[1] /= -scale_factor_2
            sl_action[2] /= -scale_factor_3
            # save_data(self.render_env.info_buf['success'], state, sl_action, self.target_domain_file)

            state[0] = state[0].clone().detach().to(torch.float64).unsqueeze(0)
            state[1] = state[1].clone().detach().to(torch.float64).unsqueeze(0)

            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    feature_output = self.feature_extractor(state[0], state[1])
                    feature_input = feature_output.to(self.device)
                    action = self.agent.select_action(feature_input)
                    action = action + np.random.normal(0, args.exploration_noise,
                                                       size=self.render_env.action_space.shape[0])
                    action = action.clip(self.render_env.action_space.low, self.render_env.action_space.high)
                    action = torch.tensor(action, dtype=torch.float64)
                print("输入给环境的action", action)
                next_state, reward, done, infos = self.render_env.step(action, num_updates, j, step)
                # save_data(self.render_env.info_buf['success'], next_state, infos['real_action'],
                #           self.target_domain_file)

                next_state[0] = next_state[0].clone().detach().to(torch.float64).unsqueeze(0)
                next_state[1] = next_state[1].clone().detach().to(torch.float64).unsqueeze(0)
                # 实时可视化奖励
                # self.visualize_reward(reward, visual=False)

                episode_reward += reward
                episode_len += 1

                self.agent.memory.push(
                    (state[0], state[1], next_state[0], next_state[1], action, reward, np.float64(done)))

                state = next_state

                if done or step == self.num_steps - 1:
                    if infos['success'] == True:
                        success_episode_cnt += 1
                    total_episode_cnt += 1
                    episode_lens.append(episode_len)
                    episode_rewards.append(episode_reward)
                    # 回合终止，重置环境
                    # episode_len = 0
                    # episode_reward = 0
                    self.render_env.interface.disconnect()
                    self.render_env.generate_initial_pose()
                    break

            if len(self.agent.memory.storage) >= self.num_steps:
                self.agent.update(10)

            if ((j+1) % self.adv_update_frequency == 0):
                target_data = load_data(self.target_domain_file)
                target_dataset = [data for data in target_data if data[0] == 0]

                if len(target_dataset) > 640:
                    target_dataset = random.sample(target_dataset, 640)

                source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
                target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)
                combined_loader = zip(source_loader, target_loader)
                # 执行对抗网络的更新
                mean_feature_extractor_loss = []  # 假设你在训练循环中记录了特征提取器的损失值
                for epoch in range(self.adv_num_epochs):
                    # 遍历源域数据加载器
                    for (source_batch, target_batch) in combined_loader:
                        # 获取目标域数据
                        # for target_batch in target_loader:
                        #     # 使用对抗训练器更新模型
                        discriminator_loss, feature_extractor_loss = self.adversarial_trainer.update(source_batch,
                                                                                                     target_batch)
                        print(f"adv train epoch: {epoch} D loss:{discriminator_loss} G loss:{feature_extractor_loss}")
                        mean_feature_extractor_loss.append(feature_extractor_loss)

                torch.save(self.feature_extractor.state_dict(),
                           os.path.join(model_save_dir, 'feature_extractor_iter{}'.format(j) + "_loss{:.1f}".format(
                               np.mean(mean_feature_extractor_loss)) + ".pt"))

            try:
                if (success_episode_cnt / total_episode_cnt) * 100. >= 50:  # 记录成功率大于50的连续次数
                    success_cnt += 1
                else:
                    success_cnt = 0  # 如果这次的成功率低于80,重置
            except:
                success_cnt = 0
            # save for every interval-th episode or for the last epoch
            if (j % self.save_interval == 0 or j == num_updates - 1):
                self.agent.save(os.path.join(model_save_dir,
                                             'model_iter{}'.format(j) + "_reward{:.1f}".format(
                                                 np.mean(episode_rewards))))
            if success_episode_cnt / total_episode_cnt > best_success_rate + 0.002 or \
                    success_episode_cnt / total_episode_cnt > best_success_rate - 0.001 and np.mean(
                episode_lens) < best_len:
                # best_eval_rew = np.mean(episode_rewards)
                best_success_rate = success_episode_cnt / total_episode_cnt
                best_len = np.mean(episode_lens)
                self.agent.save_best(os.path.join(model_save_dir, 'best_model_iter{}'.format(j)))

            # logging to console
            if j % self.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                end = time.time()
                # if np.mean(episode_rewards) > best_eval_rew:
                if success_episode_cnt / total_episode_cnt > best_success_rate + 0.002 or \
                        success_episode_cnt / total_episode_cnt > best_success_rate - 0.001 and np.mean(
                    episode_lens) < best_len:
                    print_info(
                        "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:.1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                        "success rate {:.1f}% "
                        # "dist_entropy {:.1f}, value_loss {:.4f}, action_loss {:.4f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                (end - start) / 60.,
                                len(episode_rewards),
                                np.mean(episode_lens), np.median(episode_lens),
                                np.min(episode_lens), np.max(episode_lens),
                                np.mean(episode_rewards), np.median(episode_rewards),
                                np.min(episode_rewards), np.max(episode_rewards),
                                (success_episode_cnt / total_episode_cnt) * 100.))
                    # dist_entropy, value_loss,
                    # action_loss))
                else:
                    print(
                        "Updates {}, num timesteps {}, FPS {}, time {} minutes \n Last {} training episodes: mean/median length {:.1f}/{}, min/max length {}/{} mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} "
                        "success rate {:.1f}% "
                        # "dist_entropy {:.1f}, value_loss {:.4f}, action_loss {:.4f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                (end - start) / 60.,
                                len(episode_rewards),
                                np.mean(episode_lens), np.median(episode_lens),
                                np.min(episode_lens), np.max(episode_lens),
                                np.mean(episode_rewards), np.median(episode_rewards),
                                np.min(episode_rewards), np.max(episode_rewards),
                                (success_episode_cnt / total_episode_cnt) * 100.))
                    # dist_entropy, value_loss,
                    # action_loss))

                # logging to writer
                time_elapse = end - start
                self.writer.add_scalar('curriculum/step', self.render_env.stage, total_num_steps)
                self.writer.add_scalar('rewards/step', np.mean(episode_rewards), total_num_steps)
                self.writer.add_scalar('rewards/time', np.mean(episode_rewards), time_elapse)
                self.writer.add_scalar('rewards/iter', np.mean(episode_rewards), j)
                self.writer.add_scalar('episode_lengths/step', np.mean(episode_lens), total_num_steps)
                self.writer.add_scalar('episode_lengths/time', np.mean(episode_lens), time_elapse)
                self.writer.add_scalar('episode_lengths/iter', np.mean(episode_lens), j)
                self.writer.add_scalar('success_rate/step', (success_episode_cnt / total_episode_cnt) * 100.,
                                       total_num_steps)
                self.writer.add_scalar('success_rate/time', (success_episode_cnt / total_episode_cnt) * 100.,
                                       time_elapse)
                self.writer.add_scalar('success_rate/iter', (success_episode_cnt / total_episode_cnt) * 100., j)
                # self.writer.add_scalar('stats/value_loss/step', value_loss, total_num_steps)
                # self.writer.add_scalar('stats/dist_entropy/step', dist_entropy, total_num_steps)
                # self.writer.add_scalar('stats/action_loss/step', action_loss, total_num_steps)

            # save logs of every episode
            if (self.log_interval is not None and self.log_interval > 0 and (j + 1) % self.log_interval == 0):
                end = time.time()
                fp_log = open(self.training_log_path, 'a')
                total_num_steps = (j + 1) * self.num_processes * self.num_steps
                len_mean, len_min, len_max = np.mean(episode_lens), np.min(episode_lens), np.max(episode_lens)
                reward_mean, reward_min, reward_max = np.mean(episode_rewards), np.min(episode_rewards), np.max(
                    episode_rewards)
                fp_log.write(
                    'curriculum stage = {}, success_cnt = {}, num_steps = {}, time = {}, mean(len) = {:.1f}, min(len) = {}, max(len) = {}, mean(reward) = {:.3f}, min(reward) = {:.3f}, max(reward) = {:.3f}, eval = {:.3f}, success_rate = {:.1f} \n'.format(
                        self.render_env.stage, success_cnt, total_num_steps, end - start, len_mean, len_min, len_max,
                        reward_mean, reward_min, reward_max, last_eval_rew, (
                                                                                         success_episode_cnt / total_episode_cnt) * 100.))
                fp_log.close()
            # if success_episode_cnt / total_episode_cnt >= 0.75:
            #     print("Success rate satisfied the requirement，over training")
            #     break

        self.writer.close()
        self.render_env.close()
        # self.envs.close()

    def test(self, cfg):
        # env = TactileInsertionEnv()
        # env.seed(self.seed)

        # tac_obs_shape = (5, 2, 4, 4)
        # tor_obs_shape = (32, 3)
        # 指定保存日志文件的文件夹
        log_dir = "./baseline_results"
        os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在，则创建

        # 创建一个时间戳用于文件命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 使用 datetime.now()
        log_filename = os.path.join(log_dir, f"{timestamp}_{self.env_para}_RL.csv")

        # 创建并初始化CSV文件
        with open(log_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 写入表头
            csv_writer.writerow(["Test Round", "Attempt", "Success", "Cumulative Success Rate (%)"])

        actor_critic_fn = getattr(model, cfg['params']['network']['actor_critic'])
        ac = actor_critic_fn(128, self.render_env.action_space.shape[0], cfg['params']['network'])
        ac.to(self.device)

        # feature_extractor = model.FeatureExtractor(self.tac_input_channels, self.torque_input_channels,
        #                                            self.tactile_hidden_size,
        #                                            self.torque_hidden_size)
        # feature_extractor.to(self.device)

        feature_extractor = self.feature_extractor
        # 加载检查点
        model_save_dir = "./RL_models/12-17-2024-22-34-03-tr00/models/"
        # checkpoint1 = torch.load(os.path.join(model_save_dir, "feature_extractor_iter60_loss0.8.pt"))
        # create TD3 agent
        agent = TD3(128, self.render_env.action_space.shape[0], float(self.render_env.action_space.high[0]),
                    feature_extractor, model_save_dir, self.pretrained_model)
        # 将提取的对象加载到你的模型和相关对象中
        agent.load(os.path.join(model_save_dir, 'model_iter55_reward-37.2'))
        # feature_extractor.load_state_dict(checkpoint1)

        while True:
            state, sl_action = self.render_env.reset(1000)  # 重置环境
            if self.render_env.info_buf['success'] == True or abs(self.render_env.info_buf['obj_rx']) >= 0.12:
                print("Invalid initialization, reset again.")
                self.render_env.interface.disconnect()
                self.render_env.generate_initial_pose()
            else:
                break

        test_steps = 20
        attempt_time = 15
        # 假设您有新的测试数据 test_data，根据您的数据格式进行处理
        # 示例：tac_data, torque_data = preprocess_test_data(test_data)
        success_cnt = 0



        for i in range(0, test_steps):
            state[0] = state[0].clone().detach().to(torch.float64).unsqueeze(0)
            state[1] = state[1].clone().detach().to(torch.float64).unsqueeze(0)
            for j in range(attempt_time):
                # 进行前向传播
                with torch.no_grad():
                    feature_output = feature_extractor(state[0], state[1])
                    feature_input = feature_output.to(self.device)
                    action = agent.select_action(feature_input)
                    # action = action.clip(self.render_env.action_space.low, self.render_env.action_space.high)
                    print("原始的action:", action)
                    # s = input("sss")
                    action = torch.tensor(action, dtype=torch.float64)

                print(
                    f"第{i + 1}/{test_steps}轮测试-第{j + 1}/{attempt_time}次尝试-当前成功率{success_cnt / (i + 0.0001) * 100}%")
                print("输入给环境的动作：", action)
                next_state, reward, done, infos = self.render_env.step(action, test_steps - 1, j, -1)
                next_state[0] = next_state[0].clone().detach().to(torch.float64).unsqueeze(0)
                next_state[1] = next_state[1].clone().detach().to(torch.float64).unsqueeze(0)
                state = next_state
                if done:
                    if infos["success"] == True:
                        success_cnt += 1
                    # 写入当前测试结果到CSV
                    with open(log_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(
                            [i + 1, j + 1, 1 if infos["success"] else 0, success_cnt / (i + 1) * 100])
                    self.render_env.interface.disconnect()
                    if i != test_steps - 1:
                        self.render_env.generate_initial_pose()
                        while True:
                            state, sl_action = self.render_env.reset(1000)  # 重置环境
                            if self.render_env.info_buf['success'] == True or abs(
                                    self.render_env.info_buf['obj_rx']) >= 0.3:
                                print("Invalid initialization, reset again.")
                                self.render_env.interface.disconnect()
                                self.render_env.generate_initial_pose()
                            else:
                                break
                    break
                if j == attempt_time - 1:
                    # 写入当前测试结果到CSV
                    with open(log_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(
                            [i + 1, j + 1, 1 if infos["success"] else 0, success_cnt / (i + 1) * 100])
                    self.render_env.interface.disconnect()
                    if i != test_steps - 1:
                        self.render_env.generate_initial_pose()
                        while True:
                            state, sl_action = self.render_env.reset(1000)  # 重置环境
                            if self.render_env.info_buf['success'] == True or abs(
                                    self.render_env.info_buf['obj_rx']) >= 0.3:
                                print("Invalid initialization, reset again.")
                                self.render_env.interface.disconnect()
                                self.render_env.generate_initial_pose()
                            else:
                                break

        print(f"Test success rate = {success_cnt / test_steps * 100} %")


