import time
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision import datasets as tv_datasets, transforms as tv_transforms
from torchvision.transforms import ToPILImage
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import copy
import torch.nn as nn  # 新增这一行！

# =========================================================
# 全局配置
# =========================================================
# CIFAR-10数据集根路径（自动下载，路径不存在时创建）
Path = '../CIFAR-10/Train/'
# 固定随机种子（保证实验可复现）
np.random.seed(0)
torch.manual_seed(0)

def seed_everything(seed=42):
    # nn.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# =========================================================
# 双视图增强类（对比学习专用）
# =========================================================
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]  # 返回 [view1, view2] 双视图


# =========================================================
# CIFAR-10专属SimCLR风格增强（3通道彩色图）
# =========================================================
def get_simclr_augment_cifar10(input_size=32):
    """CIFAR-10专属双视图增强策略"""
    s = 1
    color_jitter = tv_transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
    return tv_transforms.Compose([
        tv_transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),  # 随机缩放裁剪
        tv_transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        tv_transforms.RandomRotation(degrees=15),  # 轻微随机旋转
        tv_transforms.RandomApply([color_jitter], p=0.8),  # 随机颜色抖动
        tv_transforms.RandomGrayscale(p=0.2),  # 随机灰度化
        tv_transforms.ToTensor(),  # 转为Tensor
        # CIFAR-10官方归一化参数（3通道）
        tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


# =========================================================
# 核心数据加载函数（适配CIFAR-10，对比学习专用）
# =========================================================
def DataLoader_1280(batch_size=32, num_workers=4, input_size=32, Train_num=100, Test_num=20, category=10):
    """
    返回CIFAR-10的DataLoader（训练集双视图增强，测试集单视图）
    参数说明：
        batch_size: 批次大小
        num_workers: 数据加载线程数
        input_size: 输入图像尺寸（CIFAR-10默认32）
        Train_num: 每个类别选取的训练样本数 → 总训练样本数=Train_num*category
        Test_num: 每个类别选取的测试样本数 → 总测试样本数=Test_num*category
        category: CIFAR-10固定10类
    返回：
        train_loader: 训练集DataLoader（每张图返回[view1, view2]双视图）
        test_loader: 测试集DataLoader（单视图+标签，用于评估）
        classes: CIFAR-10类别名称列表
    """
    g = torch.Generator()
    g.manual_seed(0)  # 固定随机种子，保证可复现

    # ---- 训练集：双视图增强（对比学习专用） ----
    aug_train = get_simclr_augment_cifar10(input_size)
    train_full = tv_datasets.CIFAR10(
        root=Path,
        train=True,
        download=True,  # 路径不存在时自动下载
        transform=TwoCropsTransform(aug_train)  # 生成双视图
    )

    # 随机采样指定数量的训练样本
    total_train_idx = np.arange(len(train_full))
    selected_train_idx = np.random.choice(total_train_idx, size=Train_num * category, replace=False)
    train_set = Subset(train_full, selected_train_idx)
    print(f"CIFAR-10训练集样本数量: {len(train_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    # ---- 测试集：单视图（无增强）+ 标签（用于准确率评估） ----
    test_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # 轻微放大后中心裁剪，保持尺寸一致
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # CIFAR-10归一化参数
        tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_full = tv_datasets.CIFAR10(
        root=Path,
        train=False,
        download=True,
        transform=test_transform
    )

    # 随机采样指定数量的测试样本
    total_test_idx = np.arange(len(test_full))
    selected_test_idx = np.random.choice(total_test_idx, size=Test_num * category, replace=False)
    test_set = Subset(test_full, selected_test_idx)
    print(f"CIFAR-10测试集样本数量: {len(test_set)}")

    test_loader = DataLoader(
        test_set,
        batch_size=Test_num * category,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    # CIFAR-10类别名称
    classes = (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    return train_loader, test_loader, classes


# =========================================================
# 准确率计算函数（适配CIFAR-10）
# =========================================================
def CaculateAccuracy(TestLoader, device, Net):
    """
    计算模型在CIFAR-10测试集上的准确率
    参数：
        TestLoader: 测试集DataLoader
        device: 计算设备（cuda/cpu）
        Net: 待评估的模型
    返回：
        accuracy: 测试集准确率（百分比）
    """
    correct = 0  # 预测正确的样本数
    total = 0  # 总样本数
    # 测试阶段不计算梯度，提升效率、节约显存
    with torch.no_grad():
        for data in TestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = Net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'CIFAR-10测试集准确率: {accuracy:.5f}%')
    return accuracy


# =========================================================
# 相似度任务专用工具函数
# =========================================================
def build_similarity_pairs(dataset, num_pairs_per_class=100, num_neg_pairs=100):
    """
    从CIFAR-10数据集构建相似/不相似样本对
    :param dataset: Subset类型的CIFAR-10数据集（带标签）
    :param num_pairs_per_class: 每个类别内的相似样本对数量
    :param num_neg_pairs: 跨类别的不相似样本对数量
    :return: 样本对列表 [(img1, img2, label), ...]，label=1（相似）/0（不相似）
    """
    # 按类别分组样本
    class2idx = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset.dataset[dataset.indices[idx]]  # Subset取原数据
        class2idx[label].append(idx)

    pairs = []
    classes = list(class2idx.keys())

    # 1. 构建相似样本对（同类别）
    for c in classes:
        idxs = class2idx[c]
        if len(idxs) < 2:
            continue
        # 随机采样同类别样本对
        for _ in range(num_pairs_per_class):
            i1, i2 = np.random.choice(idxs, 2, replace=False)
            img1, _ = dataset[i1]
            img2, _ = dataset[i2]
            pairs.append((img1, img2, 1.0))  # label=1表示相似

    # 2. 构建不相似样本对（跨类别）
    for _ in range(num_neg_pairs):
        c1, c2 = np.random.choice(classes, 2, replace=False)
        i1 = np.random.choice(class2idx[c1])
        i2 = np.random.choice(class2idx[c2])
        img1, _ = dataset[i1]
        img2, _ = dataset[i2]
        pairs.append((img1, img2, 0.0))  # label=0表示不相似

    # 打乱样本对
    np.random.shuffle(pairs)
    return pairs


class CosineSimilarityLoss(nn.Module):
    """带边际的余弦相似度损失（对齐对比损失）"""

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, z1, z2, y):
        """
        :param z1: 样本1的特征向量 [B, D]
        :param z2: 样本2的特征向量 [B, D]
        :param y: 相似性标签 [B]（1=相似，0=不相似）
        :return: 批次损失值
        """
        sim = self.cos(z1, z2)  # [B]
        # 相似样本：损失=1-sim（拉近）；不相似样本：损失=max(0, sim-margin)（推远）
        loss = y * (1 - sim) + (1 - y) * torch.clamp(sim - self.margin, min=0.0)
        return torch.mean(loss)


def spearman_correlation(pred_sim, true_sim):
    """
    计算Spearman相关系数（等级相关）
    :param pred_sim: 模型预测的相似度 [N]（Tensor/ndarray）
    :param true_sim: 真实相似度标签 [N]（Tensor/ndarray）
    :return: Spearman系数（标量）
    """
    # 转为numpy数组
    if isinstance(pred_sim, torch.Tensor):
        pred_sim = pred_sim.detach().cpu().numpy()
    if isinstance(true_sim, torch.Tensor):
        true_sim = true_sim.detach().cpu().numpy()

    # 生成排名（降序，相似度越高排名越前）
    pred_rank = np.argsort(np.argsort(-pred_sim))  # 负号实现降序排名
    true_rank = np.argsort(np.argsort(-true_sim))

    # 计算皮尔逊相关系数（Spearman核心）
    cov = np.cov(pred_rank, true_rank)[0, 1]
    std_pred = np.std(pred_rank)
    std_true = np.std(true_rank)

    if std_pred == 0 or std_true == 0:
        return 0.0  # 避免除零
    return cov / (std_pred * std_true)


# 适配DataLoader的样本对加载
class SimilarityPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]
        return img1, img2, torch.tensor(label, dtype=torch.float32)

