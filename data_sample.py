from tactile_insertion_env import *
import pickle

scale_factor_1 = 0.01
scale_factor_2 = 0.01
scale_factor_3 = torch.pi / 36
def sample_reset():
    while True:
        obs, sl_action = env.reset(1000)  # 重置环境
        if env.info_buf['success'] == True or abs(env.info_buf['obj_rx']) > 0.1:
            print("Invalid initialization, reset again.")
            env.interface.disconnect()
            env.generate_initial_pose()
        else:
            break
    sl_action[0] /= -scale_factor_1
    sl_action[1] /= -scale_factor_2
    sl_action[2] /= -scale_factor_3
    return obs, sl_action


def save_data(state, obs, action, file_path):
    # 将布尔值转换为整数（0表示False，1表示True）
    state_as_int = int(state)
    data = (state_as_int, obs, action)
    with open(file_path, 'ab') as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    seed = 3  # 3 to 11
    env = TactileInsertionEnv(env_name="round_insertion.xml")
    env.seed(seed)
    sample_steps = 10000
    file_path = "data_folder/round_hole_data10000.pkl"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    obs0, sl_action = sample_reset()
    save_data(env.info_buf['success'], obs0, sl_action, file_path)

    for j in range(0, sample_steps):
        print(f"当前采集了{j}个数据")
        action = torch.tensor(env.action_space.sample())
        obs, reward, done, infos = env.step(action, sample_steps-1, j, -1)
        sl_action = infos['real_action']
        # 对每个分量进行放大
        sl_action[0] /= -scale_factor_1
        sl_action[1] /= -scale_factor_2
        sl_action[2] /= -scale_factor_3

        if done:
            env.interface.disconnect()
            env.generate_initial_pose()
            obs0, sl_action = sample_reset()
            save_data(env.info_buf['success'], obs0, sl_action, file_path)
        else:
            save_data(env.info_buf['success'], obs, sl_action, file_path)

    env.interface.disconnect()
