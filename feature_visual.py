import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import model
import os
from train_sl_insertion import load_data
import numpy as np
import random
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

# 设置随机种子
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

target_domain_file = "./data_folder/round_domain_data41.pkl"


class LoadModel:
    def __init__(self):
        self.tac_obs_shape = (5, 2, 4, 4)
        self.tor_obs_shape = (32, 3)
        self.tac_input_channels = self.tac_obs_shape[1]
        self.torque_input_channels = self.tor_obs_shape[1]
        self.tactile_hidden_size = 64
        self.torque_hidden_size = 64
        self.output_size = 2
    def Load_trained_model(self):
        # 加载保存的参数
        model_save_dir = "./trained_models/05-16-2024-00-20-54-rr41/models/"
        checkpoint1 = torch.load(os.path.join(model_save_dir, "feature_extractor_iter30_loss0.9.pt"))

        feature_extractor = model.FeatureExtractor(self.tac_input_channels, self.torque_input_channels,
                                                   self.tactile_hidden_size,
                                                   self.torque_hidden_size)

        feature_extractor.load_state_dict(checkpoint1)

        return feature_extractor

    def Load_origin_model(self):
        # 加载保存的参数
        model_save_dir = "./trained_models/"

        pretrained_model = model.TacTorModel(self.tac_input_channels,self.torque_input_channels, self.tactile_hidden_size,
                                                  self.torque_hidden_size, self.output_size)
        checkpoint0 = torch.load(os.path.join(model_save_dir, "dynamic_round_model-10000.pth"))
        pretrained_model.load_state_dict(checkpoint0)

        # 加载特征提取器部分全参数
        feature_extractor = model.FeatureExtractor(self.tac_input_channels, self.torque_input_channels,
                                                        self.tactile_hidden_size,
                                                        self.torque_hidden_size)

        feature_extractor.cnn.load_state_dict(pretrained_model.cnn.state_dict())
        feature_extractor.tactile_lstm.load_state_dict(pretrained_model.tactile_lstm.state_dict())
        feature_extractor.torque_lstm.load_state_dict(pretrained_model.torque_lstm.state_dict())
        feature_extractor.fc_cnn.load_state_dict(pretrained_model.fc_cnn.state_dict())
        feature_extractor.dropout_cnn.load_state_dict(pretrained_model.dropout_cnn.state_dict())

        return feature_extractor


    def extract_feature(self,feature_extractor):
        # 提取源域数据和目标域数据的特征
        source_features = []
        target_features = []

        for target_data in target_dataset:
            with torch.no_grad():
                features = feature_extractor(torch.tensor(np.array(target_data[1][0]), dtype=torch.float).unsqueeze(0),
                                              torch.tensor(np.array(target_data[1][1]), dtype=torch.float).unsqueeze(0))
            target_features.append(features.numpy())
        print(len(target_features))

        for source_data in source_dataset:
            with ((torch.no_grad())):
                features = feature_extractor(torch.tensor(np.array(source_data[1][0]), dtype=torch.float).unsqueeze(0),
                                              torch.tensor(np.array(source_data[1][1]), dtype=torch.float).unsqueeze(0))
            source_features.append(features.numpy())
        print(len(source_features))
        source_features = np.array(source_features)
        target_features = np.array(target_features)
        # 压缩第二维的大小为 1 的维度
        source_features = np.squeeze(source_features, axis=1)
        target_features = np.squeeze(target_features, axis=1)

        print(source_features.shape)
        print(target_features.shape)

        return source_features, target_features


class DownSample:
    def __init__(self, n_components, perplexity, random_seed):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_seed = random_seed

    def tsne(self, source_features, target_features):
        # 将高维特征映射到二维空间
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, random_state=self.random_seed)
        source_tsne = tsne.fit_transform(source_features)
        target_tsne = tsne.fit_transform(target_features)

        return source_tsne, target_tsne

    def pca(self,source_features,target_features):
        # 使用 PCA 将高维特征映射到二维空间
        pca = PCA(n_components=self.n_components, random_state=self.random_seed)

        source_pca = pca.fit_transform(source_features)
        target_pca = pca.fit_transform(target_features)

        return source_pca, target_pca

    def mds_fun(self, source_features, target_features):
        # 将高维特征映射到二维空间（使用 MDS）
        mds = MDS(n_components=self.n_components, random_state=self.random_seed)
        source_mds = mds.fit_transform(source_features)
        target_mds = mds.fit_transform(target_features)

        return source_mds, target_mds


def compute_kl_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return entropy(p, qk=q)


def compute_js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, qk=m) + entropy(q, qk=m))


def compute_cosine_similarity(source_features, target_features):
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(source_features, target_features)
    average_similarity = similarity_matrix.mean()
    return average_similarity


def compute_mmd_rbf(source, target, gamma=1.0):
    """计算使用RBF核的MMD"""
    # 计算核矩阵
    K_ss = rbf_kernel(source, source, gamma=gamma)
    K_tt = rbf_kernel(target, target, gamma=gamma)
    K_st = rbf_kernel(source, target, gamma=gamma)

    # 计算MMD
    mmd = np.mean(K_ss) + np.mean(K_tt) - 2 * np.mean(K_st)
    return mmd


lm = LoadModel()
feature_extractor0 = lm.Load_origin_model()
feature_extractor1 = lm.Load_trained_model()

# 提取目标域数据的特征
target_data = load_data(target_domain_file)
target_dataset = [data for data in target_data if data[0] == 0]

source_data = load_data("data_folder/round_hole_data10000.pkl")
# 提取源域数据的特征
source_dataset = [data for data in source_data if data[0] == 0]
source_dataset = random.sample(source_dataset, len(target_dataset))


source_features0, target_features0 = lm.extract_feature(feature_extractor0)
source_features1, target_features1 = lm.extract_feature(feature_extractor1)


ds = DownSample(2, 30, random_seed)

source_tsne0, target_tsne0 = ds.tsne(source_features0, target_features0)
source_tsne1, target_tsne1 = ds.tsne(source_features1, target_features1)
source_pca0, target_pca0 = ds.pca(source_features0, target_features0)
source_pca1, target_pca1 = ds.pca(source_features1, target_features1)
# source_mds0, target_mds0 = ds.mds_fun(source_features0, target_features0)
# source_mds1, target_mds1 = ds.mds_fun(source_features1, target_features1)


# 对特征进行归一化处理
# scaler = StandardScaler()
# source_features0 = scaler.fit_transform(source_features0)
# target_features0 = scaler.transform(target_features0)
# source_features1 = scaler.fit_transform(source_features1)
# target_features1 = scaler.transform(target_features1)
# Compute distributions (histogram based for simplicity)
source_distribution0 = np.histogram(source_features0, bins=20, density=True)[0]
target_distribution0 = np.histogram(target_features0, bins=20, density=True)[0]
source_distribution1 = np.histogram(source_features1, bins=20, density=True)[0]
target_distribution1 = np.histogram(target_features1, bins=20, density=True)[0]


# Compute divergences and distances
kl_divergence0 = compute_kl_divergence(source_distribution0, target_distribution0)
js_divergence0 = compute_js_divergence(source_distribution0, target_distribution0)
mmd_value0 = compute_mmd_rbf(source_features0, target_features0)

kl_divergence1 = compute_kl_divergence(source_distribution1, target_distribution1)
js_divergence1 = compute_js_divergence(source_distribution1, target_distribution1)
mmd_value1 = compute_mmd_rbf(source_features1, target_features1)

# 计算源域数据和目标域数据之间的欧氏距离
domain_distance0 = euclidean(source_features0.flatten(), target_features0.flatten())
domain_distance1 = euclidean(source_features1.flatten(), target_features1.flatten())


print(f"KL Divergence-0: {kl_divergence0:.4f}")
print(f"KL Divergence-1: {kl_divergence1:.4f}", )
print(f"JS Divergence-0: {js_divergence0:.4f}")
print(f"Js Divergence-1: {js_divergence1:.4f}")
print(f"Euclidean Distance-0: {domain_distance0:.4f}")
print(f"Euclidean Distance-1: {domain_distance1:.4f}")
# Print the MMD values with four decimal places
print("MMD between Feature Spaces 0: {:.4f}".format(mmd_value0))
print("MMD between Feature Spaces 1: {:.4f}".format(mmd_value1))

"""
# 计算 t-SNE 映射后的特征空间的相关性
pearson_corr_tsne0, _ = pearsonr(source_tsne0.flatten(), target_tsne0.flatten())
print(f"Pearson Correlation between t-SNE Feature Spaces 0: {pearson_corr_tsne0:.4f}")
pearson_corr_tsne1, _ = pearsonr(source_tsne1.flatten(), target_tsne1.flatten())
print(f"Pearson Correlation between t-SNE Feature Spaces 1: {pearson_corr_tsne1:.4f}")
# 计算 PCA 映射后的特征空间的相关性
pearson_corr_pca0, _ = pearsonr(source_pca0.flatten(), target_pca0.flatten())
print(f"Pearson Correlation between PCA Feature Spaces 0: {pearson_corr_pca0:.4f}")
pearson_corr_pca1, _ = pearsonr(source_pca1.flatten(), target_pca1.flatten())
print(f"Pearson Correlation between PCA Feature Spaces 1: {pearson_corr_pca1:.4f}")

# 可视化特征空间中的数据分布
plt.figure(figsize=(12,12))

# t-SNE 可视化图 - Model 0
plt.subplot(2, 2, 1)
plt.scatter(source_tsne0[:, 0], source_tsne0[:, 1], label='Source Domain')
plt.scatter(target_tsne0[:, 0], target_tsne0[:, 1], label='Target Domain')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Feature Space Visualization - Model 0')
plt.legend()

# t-SNE 可视化图 - Model 1
plt.subplot(2, 2, 2)
plt.scatter(source_tsne1[:, 0], source_tsne1[:, 1], label='Source Domain')
plt.scatter(target_tsne1[:, 0], target_tsne1[:, 1], label='Target Domain')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Feature Space Visualization - Model 1')
plt.legend()

# PCA 可视化图 - Model 0
plt.subplot(2, 2, 3)
plt.scatter(source_pca0[:, 0], source_pca0[:, 1], label='Source Domain')
plt.scatter(target_pca0[:, 0], target_pca0[:, 1], label='Target Domain')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('PCA Feature Space Visualization - Model 0')
plt.legend()

# PCA 可视化图 - Model 1
plt.subplot(2, 2, 4)
plt.scatter(source_pca1[:, 0], source_pca1[:, 1], label='Source Domain')
plt.scatter(target_pca1[:, 0], target_pca1[:, 1], label='Target Domain')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('PCA Feature Space Visualization - Model 1')
plt.legend()

plt.tight_layout()
plt.show()
"""
