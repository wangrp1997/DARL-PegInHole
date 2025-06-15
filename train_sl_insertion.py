import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tactile_insertion_env import *
from datetime import datetime  # 修改为直接导入 datetime 类
import csv

def load_data(file_path):
    loaded_data = []
    with open(file_path, 'rb') as file:

        try:
            while True:
                data = pickle.load(file)
                loaded_data.append(data)
        except EOFError:
            pass
    return loaded_data


class TacTorModel(nn.Module):
    def __init__(self, tac_input_channels, torque_input_channels, tactile_hidden_size,torque_hidden_size, output_size,
                 dropout_rate=0.5):
        # def __init__(self, 2, 3, 64, 64, 3):

        super(TacTorModel, self).__init__()

        # CNN layers for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(tac_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_cnn = nn.Linear(64, 32)
        self.dropout_cnn = nn.Dropout(p=dropout_rate)  # 添加Dropout

        # LSTM layer for temporal modeling of tactile features
        self.tactile_lstm = nn.LSTM(input_size=32, hidden_size=tactile_hidden_size, num_layers=2, batch_first=True)

        # LSTM layer for temporal modeling of torque features
        self.torque_lstm = nn.LSTM(input_size=torque_input_channels, hidden_size=torque_hidden_size, num_layers=2, batch_first=True)

        # Additional fully connected layers for output
        self.fc_layers = nn.Sequential(
            nn.Linear(tactile_hidden_size + torque_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加Dropout
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x_tactile, x_torque):

        if x_tactile.dim() == 4:
            x_tactile = x_tactile.unsqueeze(0)
        batch_size, seq_len, _, _, _ = x_tactile.size()
        # CNN for spatial feature extraction of tactile features
        cnn_features = []
        for t in range(seq_len):
            x_t = self.cnn(x_tactile[:,t,:,:,:])
            x_t = x_t.view(batch_size, -1)
            x_t = self.fc_cnn(x_t)
            x_t = self.dropout_cnn(x_t)  # 添加Dropout
            cnn_features.append(x_t.unsqueeze(1))
        cnn_features = torch.cat(cnn_features, dim=1)

        # Reshape for LSTM input (batch_size, sequence_length, features)
        cnn_features = cnn_features.view(cnn_features.size(0), x_tactile.size(1), -1)

        # LSTM for temporal modeling of tactile features
        tactile_features, _ = self.tactile_lstm(cnn_features)

        # Take the last output of the LSTM sequence for tactile features
        tactile_feature = tactile_features[:, -1, :]

        # LSTM for temporal modeling of torque features
        torque_features, _ = self.torque_lstm(x_torque)

        # Take the last output of the LSTM sequence for torque features
        torque_feature = torque_features[:, -1, :]

        # Concatenate tactile and torque features
        x = torch.cat((tactile_feature, torque_feature), dim=1)

        # Fully connected layer for output
        x = self.fc_layers(x)

        return x



# 绘制双Y轴的函数
# 修改绘图函数
def plot_loss(train_loss_values, test_loss_values):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(train_loss_values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Test Loss', color=color)
    ax2.plot(test_loss_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def train():

    file_path0 = './data_folder/round_hole_data10000.pkl'
    # file_path1 = './data_folder/new_data5000_1.pkl'
    # file_path2 = 'data_folder/data3000.pkl'
    # file_path3 = 'data_folder/data3883.pkl'

    # 加载数据
    loaded_data0 = load_data(file_path0)
    # loaded_data1 = load_data(file_path1)
    # loaded_data2 = load_data(file_path2)
    # loaded_data3 = load_data(file_path3)

    # filtered_data = [(data[0], [normalize_tactile(data[1][0]), normalize_torque(data[1][1])], data[2]) for data in loaded_data
    #                  if data[0] == 0]
    filtered_data0 = [data for data in loaded_data0 if data[0] == 0]
    # filtered_data1 = [data for data in loaded_data1 if data[0] == 0]
    # filtered_data2 = [data for data in loaded_data2 if data[0] == 0]
    # filtered_data3 = [data for data in loaded_data3 if data[0] == 0]

    filtered_data = filtered_data0

    print(len(filtered_data))
    # 划分训练集和测试集
    train_data = filtered_data[:int(0.8 * len(filtered_data))]
    test_data = filtered_data[int(0.8 * len(filtered_data)):]
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # 创建模型
    tac_input_channels = 2  # 根据实际情况调整
    torque_input_channels = 3  # 根据实际情况调整
    tactile_hidden_size = 64
    torque_hidden_size = 64
    output_size = 2
    model = TacTorModel(tac_input_channels, torque_input_channels, tactile_hidden_size, torque_hidden_size, output_size,
                        dropout_rate=0.5)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # 在定义优化器后添加学习率衰减
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)    # 用于存储训练和测试损失值以供绘图
    train_loss_values = []  # 训练损失
    test_loss_values = []  # 测试损失
    # 训练模型
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            tac_data, torque_data, labels = batch[1][0], batch[1][1], batch[2][:,:2]
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(tac_data, torque_data)
            outputs = outputs.to(torch.float64)

            # 计算损失
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            # 反向传播和优化
            loss.backward()
            optimizer.step()
        average_train_loss = epoch_loss / len(train_loader)
        train_loss_values.append(average_train_loss)

        # 在测试集上评估模型
        model.eval()
        total_samples = 0
        correct_samples = 0
        with torch.no_grad():
            total_loss = 0
            for batch in test_loader:
                tac_data, torque_data, labels = batch[1][0], batch[1][1], batch[2][:,:2]
                outputs = model(tac_data, torque_data)
                total_samples += labels.size(0)
                # 计算符合差值条件的样本数量
                total_loss += criterion(outputs, labels).item()

        average_test_loss = total_loss / len(test_loader)
        test_loss_values.append(average_test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss}, Test Loss: {average_test_loss}')
    plot_loss(train_loss_values, test_loss_values)
    # 保存模型
    torch.save(model.state_dict(), 'trained_models/dynamic_round_model-10000.pth')

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


if __name__ == '__main__':
    # 设置随机种子
    seed = 36
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    a = input("train?")
    if a == 'y':
        train()
    """
    env_list = ['round_insertion.xml']
    results=[]
    for k in range(len(env_list)):
        env_name = env_list[k]
        env = TactileInsertionEnv(env_name=env_name)
        env.seed(seed)
        scale_factor_1 = 0.01
        scale_factor_2 = 0.01
        scale_factor_3 = torch.pi / 36

        # 创建模型实例
        tac_input_channels = 2
        torque_input_channels = 3
        tactile_hidden_size = 64
        torque_hidden_size = 64
        output_size = 2
        model = TacTorModel(tac_input_channels, torque_input_channels, tactile_hidden_size, torque_hidden_size, output_size)

        # 加载保存的模型权重
        model.load_state_dict(torch.load('trained_models/dynamic_round_model-10000.pth'))
        # 设置模型为评估模式
        model.eval()
        obs, sl_action = sample_reset()
        test_steps = 20
        attempt_time = 10
        # 假设您有新的测试数据 test_data，根据您的数据格式进行处理
        # 示例：tac_data, torque_data = preprocess_test_data(test_data)
        success_cnt = 0

        env_para = input("please input env name：")
        # 指定保存日志文件的文件夹
        log_dir = "./base_line_results"
        os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在，则创建

        # 创建一个时间戳用于文件命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 使用 datetime.now()
        log_filename = os.path.join(log_dir, f"{timestamp}_{env_para}_SL.csv")

        for i in range(1, test_steps+1):

            for j in range(attempt_time):
                print(f"Current step {i}: SL-Test success rate = {(success_cnt / (i+0.00001)) * 100} %")

            # 进行前向传播
                with torch.no_grad():
                    action = model(obs[0].unsqueeze(0), obs[1].unsqueeze(0))
                print("输入给环境的动作：",action[0])
                obs, reward, done, infos = env.step(action[0], test_steps - 1, j, -1)
                if done:
                    if infos["success"] == True:
                        success_cnt += 1
                    # 写入当前测试结果到CSV
                    with open(log_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(
                            [i + 1, j + 1, 1 if infos["success"] else 0, success_cnt / (i + 1) * 100])
                    env.interface.disconnect()
                    if i != test_steps:
                        env.generate_initial_pose()
                        obs, sl_action = sample_reset()
                    break
                if j == attempt_time-1:
                    # 写入当前测试结果到CSV
                    with open(log_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(
                            [i + 1, j + 1, 1 if infos["success"] else 0, success_cnt / (i + 1) * 100])
                    env.interface.disconnect()
                    env.generate_initial_pose()
                    obs, sl_action = sample_reset()
            print(f"Current step {i}: SL-Test success rate = {(success_cnt / (i+0.00001)) * 100} %")


        env.interface.disconnect()
        success_rate = success_cnt / test_steps * 100

        print(f"{env_name} SL-Test success rate = {success_rate} %")
    """