import torch
import torch.nn as nn
from utils import model_utils
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
# Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean

class MLP(nn.Module):
    def __init__(self, in_dim, cfg_network):
        super(MLP, self).__init__()
        
        layer_sizes = cfg_network['layer_sizes']
        modules = []
        for i in range(len(layer_sizes)):
            modules.append(nn.Linear(in_dim, layer_sizes[i]))
            modules.append(model_utils.get_activation_func(cfg_network['activation']))
            modules.append(nn.Dropout(p=0.5))
            if cfg_network.get('layernorm', False):
                modules.append(torch.nn.LayerNorm(layer_sizes[i]))
            in_dim = layer_sizes[i]

        self.body = nn.Sequential(*modules)
        self.out_features = layer_sizes[-1]

    def forward(self, inputs):
        return self.body(inputs)

class CNN(nn.Module):
    def __init__(self,input_shape, cfg_network):
        super(CNN, self).__init__()
        hidden_size = cfg_network['hidden_size']

        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 添加线性层
        self.fc = nn.Linear(32, 10)  # 假设输出维度为10
        self.out_features = hidden_size

    def forward(self, x):
        # 检查输入是否为四维，如果是，则在第0个维度上插入一个维度

        if x.dim() == 4:
            x = x.unsqueeze(0)
        batch_size, seq_len, _, _, _ = x.size()
        outputs = []

        for t in range(seq_len):
            x_t = self.conv1(x[:,t,:,:,:])
            x_t = torch.relu(x_t)
            x_t = self.pool(x_t)
            x_t = self.conv2(x_t)
            x_t = torch.relu(x_t)
            x_t = self.pool(x_t)
            # 将特征展平成一维向量
            x_t = x_t.view(x.size(0), -1)
            # 通过线性层
            x_t = self.fc(x_t)
            outputs.append(x_t.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)

        return outputs

class CNNActor(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        self.feature_net = CNN(obs_shape, cfg_network['actor_cnn'])

        self.mean_net = nn.Linear(self.feature_net.out_features, action_dim)
        self.logstd = nn.Parameter(torch.ones(action_dim) * cfg_network.get('actor_logstd_init', -1.0))

    def forward(self, inputs):
        features = self.feature_net(inputs)
        mean = self.mean_net(features)
        std = self.logstd.expand_as(mean).exp()
        dist = FixedNormal(loc=mean, scale=std)
        
        return dist
    
    def act(self, inputs, deterministic = False):
        action_dist = self.forward(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.rsample()

        return action

class CNNCritic(nn.Module):
    def __init__(self,
                 obs_shape,
                 cfg_network):
        super().__init__()
        self.feature_net = CNN(obs_shape, cfg_network['critic_cnn'])

        self.value_net = nn.Linear(self.feature_net.out_features, 1)

    def forward(self, inputs):
        features = self.feature_net(inputs)
        values = self.value_net(features)
        
        return values
    
    def act(self, inputs, deterministic = False):
        action_dist = self.forward(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.rsample()

        return action


class MLPCritic(nn.Module):
    def __init__(self,
                 obs_shape, 
                 cfg_network):
        super().__init__()
        self.feature_net = MLP(obs_shape[0], cfg_network['critic_mlp'])

        self.value_net = nn.Linear(self.feature_net.out_features, 1)

    def forward(self, inputs):
        features = self.feature_net(inputs)
        value = self.value_net(features)

        return value

class ActorCritic(nn.Module):
    def __init__(self, 
                 actor_net,
                 critic_net):

        super().__init__()
        self.actor_net = actor_net
        self.critic_net = critic_net
    
    def act(self, inputs, rnn_hxs = None, masks = None, deterministic = False):
        action_dist = self.actor_net(inputs)
        value = self.critic_net(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()
        
        action_log_probs = action_dist.log_probs(action)
        
        return value, action, action_log_probs, rnn_hxs
    
    def get_value(self, inputs, rnn_hxs = None, masks = None):
        value = self.critic_net(inputs)

        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        action_dist = self.actor_net(inputs)
        value = self.critic_net(inputs)

        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
        
    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1


class ActorCriticRNN(nn.Module):
    def __init__(self, 
                 tac_obs_shape,
                 tor_obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        self.feature_net = CNN(tac_obs_shape, cfg_network['feature_cnn'])
        self.rnn_hidden_size = cfg_network['rnn_hidden_size']
        self.rnn_hidden_layers = cfg_network['rnn_hidden_layers']

        self.touch_rnn = nn.GRU(self.feature_net.out_features, self.rnn_hidden_size, self.rnn_hidden_layers)
        self.torque_rnn = nn.GRU(tor_obs_shape, self.rnn_hidden_size, self.rnn_hidden_layers)
        self.actor_net = DiagGaussianActor((self.rnn_hidden_size*2,), action_dim, cfg_network)
        self.critic_net = MLPCritic((self.rnn_hidden_size*2,), cfg_network)

    def _forward_rnn0(self, x, hxs, masks):
        batch_size, seq_len, _= x.size()
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(1) / N)
        # unflatten
        x = x.view(batch_size, T, N, x.size(2))

        # Same deal with masks
        masks = masks.view(batch_size,1, 1)

        # hxs = hxs.unsqueeze(0)
        hxs = hxs.view(N, self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)

        final_outputs = []
        final_hxs = []

        start_idx = 0
        end_idx = T
        for j in range(batch_size):
            outputs = []
                # We can now process steps that don't have any zeros in masks together!
                # This is much faste
            rnn_scores, hxs = self.touch_rnn(
                x[j, start_idx:end_idx],
                hxs * masks[j,start_idx].view(1, -1, 1))

            outputs.append(rnn_scores)
            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x_t = torch.cat(outputs, dim=0)
            # flatten
            x_t = x_t.view(T * N, -1)

            x_t = x_t[-1,:]
            hxs_t = hxs.permute(1, 0, 2).contiguous().view(N, -1)
            final_outputs.append(x_t)
            final_hxs.append(hxs_t)
        x = torch.stack(final_outputs, dim=0)
        hxs = torch.stack(final_hxs, dim=0)

        return x, hxs

    def _forward_rnn1(self, x, hxs, masks):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.size()
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        N = hxs.size(0)
        T = int(x.size(1) / N)  # T=32
        # unflatten
        x = x.view(batch_size, T, N, x.size(2))

        # Same deal with masks
        masks = masks.view(batch_size, 1,1)

        hxs = hxs.view(N, self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)

        final_outputs = []
        final_hxs = []

        start_idx = 0
        end_idx = T
        for j in range(batch_size):
            outputs = []
                # We can now process steps that don't have any zeros in masks together!
                # This is much fast
            rnn_scores, hxs = self.torque_rnn(
                x[j, start_idx:end_idx],
                hxs * masks[j,start_idx].view(1, -1, 1))

            outputs.append(rnn_scores)
            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x_t = torch.cat(outputs, dim=0)
            # flatten
            x_t = x_t.view(T*N, -1)
            x_t = x_t[-1,:]  # 1x32
            hxs_t = hxs.permute(1, 0, 2).contiguous().view(N, -1)
            final_outputs.append(x_t)
            final_hxs.append(hxs_t)
        x = torch.stack(final_outputs, dim=0)
        hxs = torch.stack(final_hxs, dim=0)

        return x, hxs

    def act(self, inputs0, inputs1, rnn_hxs0, rnn_hxs1, masks, deterministic = False):
        features = self.feature_net(inputs0)
        tac_rnn_output, rnn_hxs_new0 = self._forward_rnn0(features, rnn_hxs0, masks)
        tor_rnn_output, rnn_hxs_new1 = self._forward_rnn1(inputs1, rnn_hxs1, masks)
        combined_output = torch.cat((tac_rnn_output, tor_rnn_output), dim=1)

        action_dist = self.actor_net(combined_output)
        value = self.critic_net(combined_output)
        
        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()
        
        action_log_probs = action_dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs_new0, rnn_hxs_new1
    
    def get_value(self, inputs0, inputs1, rnn_hxs0,rnn_hxs1, masks):
        features = self.feature_net(inputs0)
        tac_rnn_output, _ = self._forward_rnn0(features, rnn_hxs0, masks)
        tor_rnn_output, _ = self._forward_rnn1(inputs1, rnn_hxs1, masks)
        combined_output = torch.cat((tac_rnn_output, tor_rnn_output), dim=1)

        value = self.critic_net(combined_output)
        
        return value
    
    def evaluate_actions(self, inputs0, inputs1, rnn_hxs0, rnn_hxs1, masks, action):
        features = self.feature_net(inputs0)
        tac_rnn_output, rnn_hxs_new0 = self._forward_rnn0(features, rnn_hxs0, masks)
        tor_rnn_output, rnn_hxs_new1 = self._forward_rnn1(inputs1, rnn_hxs1, masks)
        combined_output = torch.cat((tac_rnn_output, tor_rnn_output), dim=1)

        action_dist = self.actor_net(combined_output)
        value = self.critic_net(combined_output)
        
        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs_new0, rnn_hxs_new1
    
    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.rnn_hidden_size * self.rnn_hidden_layers


class ActorCriticMLPRNN(nn.Module):
    def __init__(self, 
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()
        self.feature_net = MLP(obs_shape[0], cfg_network['feature_mlp'])
        self.rnn_hidden_size = cfg_network['rnn_hidden_size']
        self.rnn_hidden_layers = cfg_network['rnn_hidden_layers']

        self.rnn = nn.GRU(self.feature_net.out_features, self.rnn_hidden_size, self.rnn_hidden_layers)
        self.actor_net = DiagGaussianActor((self.rnn_hidden_size, ), action_dim, cfg_network)
        self.critic_net = MLPCritic((self.rnn_hidden_size, ), cfg_network)

    def _forward_rnn(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            N = hxs.size(0)
            hxs = (hxs * masks).view(hxs.size(0), self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)
            x, hxs = self.rnn(x.unsqueeze(0), hxs)
            x = x.squeeze(0)
            hxs = hxs.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.unsqueeze(0)
            hxs = hxs.view(N, self.rnn_hidden_layers, self.rnn_hidden_size).permute(1, 0, 2)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                
                rnn_scores, hxs = self.rnn(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.permute(1, 0, 2).contiguous().view(N, -1)

        return x, hxs

    def act(self, inputs, rnn_hxs, masks, deterministic = False):
        features = self.feature_net(inputs)
        rnn_output, rnn_hxs_new = self._forward_rnn(features, rnn_hxs, masks)

        action_dist = self.actor_net(rnn_output)
        value = self.critic_net(rnn_output)
        
        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()
        
        action_log_probs = action_dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs_new
    
    def get_value(self, inputs, rnn_hxs, masks):
        features = self.feature_net(inputs)
        rnn_output, _ = self._forward_rnn(features, rnn_hxs, masks)
        
        value = self.critic_net(rnn_output)
        
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        features = self.feature_net(inputs)
        rnn_output, rnn_hxs_new = self._forward_rnn(features, rnn_hxs, masks)

        action_dist = self.actor_net(rnn_output)
        value = self.critic_net(rnn_output)
        
        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs_new
    
    @property
    def is_recurrent(self):
        return True

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.rnn_hidden_size * self.rnn_hidden_layers

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


class AdversarialTrainer:
    def __init__(self, feature_extractor, discriminator, lr=0.0002):
        self.G = feature_extractor
        self.D = discriminator
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr)

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
    def update(self, source_data, target_data):
        # https: // github.com / eriklindernoren / PyTorch - GAN
        source_features = self.G(torch.tensor(np.array(source_data[1][0]), dtype=torch.double),
                                                 torch.tensor(np.array(source_data[1][1]), dtype=torch.double))
        target_features = self.G(torch.tensor(np.array(target_data[1][0]), dtype=torch.double),
                                                 torch.tensor(np.array(target_data[1][1]), dtype=torch.double))

        source_valid_labels = torch.ones(source_features.size(0), 1)
        target_valid_labels = torch.ones(target_features.size(0), 1)

        fake_labels = torch.zeros(target_features.size(0), 1)

        # Update discriminator
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.D(source_features), source_valid_labels)
        fake_loss = self.adversarial_loss(self.D(target_features.detach()), fake_labels)
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        self.optimizer_D.step()

        # Update feature extractor
        self.optimizer_G.zero_grad()
        G_loss = self.adversarial_loss(self.D(target_features), target_valid_labels)

        G_loss.backward()
        self.optimizer_G.step()

        #
        return D_loss.item(), G_loss.item()

class DiagGaussianActor(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_dim,
                 cfg_network):
        super().__init__()

        # self.feature_net = MLP(obs_shape[0], cfg_network['actor_mlp'])
        self.mean_net = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 添加Dropout
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # self.mean_net = nn.Linear(cfg_network['actor_mlp']['layer_sizes'][-1], action_dim)
        self.logstd = nn.Parameter(torch.ones(action_dim) * cfg_network.get('actor_logstd_init', -1.0))

    def forward(self, inputs):

        mean = self.mean_net(inputs)
        # mean = self.mean_net(features)
        std = self.logstd.expand_as(mean).exp()
        dist = FixedNormal(loc=mean, scale=std)

        return dist

    def act(self, inputs, deterministic = False):
        action_dist = self.forward(inputs)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.rsample()

        return action


class Discriminator(nn.Module):
    def __init__(self, input_size=128, hidden_size0=512, hidden_size1=256):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size0, hidden_size1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(self, tac_input_channels, torque_input_channels, tactile_hidden_size, torque_hidden_size,
                 dropout_rate=0.5):
        super(FeatureExtractor, self).__init__()
        # def __init__(self, 2, 3, 64, 64, 3):

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
        combined_output = torch.cat((tactile_feature, torque_feature), dim=1)

        return combined_output

class ActorCriticMLP(nn.Module):
    def __init__(self, in_features, action_dim, cfg_network):
        super().__init__()

        self.actor_net = DiagGaussianActor(in_features, action_dim, cfg_network)
        # self.actor_net = self.pretrained_model.fc_layers
        self.critic_net = MLPCritic((in_features, ), cfg_network)

    def act(self, combined_intput, deterministic=False):

        action_dist = self.actor_net(combined_intput)
        value = self.critic_net(combined_intput)

        if deterministic:
            action = action_dist.mode()
        else:
            action = action_dist.sample()

        action_log_probs = action_dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, combined_intput):
        value = self.critic_net(combined_intput)

        return value

    def evaluate_actions(self, combined_intput, action):
        action_dist = self.actor_net(combined_intput)
        value = self.critic_net(combined_intput)

        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    @property
    def is_recurrent(self):
        return False