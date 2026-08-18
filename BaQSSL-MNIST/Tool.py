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

# ========== 核心修改1：MNIST数据集路径（自动创建子文件夹） ==========
Path = '../MNIST/Train/'
# 固定随机种子（保证可复现）
np.random.seed(0)
torch.manual_seed(0)

# ========== 双视图增强类（无需修改，与数据集无关） ==========
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]  # 返回 [view1, view2] 双视图

# ========== 核心修改2：适配MNIST的SimCLR风格增强 ==========
def get_simclr_augment_mnist(input_size=28):
    """MNIST专属双视图增强（单通道灰度图，手写数字适配）"""
    return tv_transforms.Compose([
        tv_transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),  # 随机裁剪
        # 移除水平翻转：手写数字翻转会导致语义丢失（如6→9）
        tv_transforms.RandomRotation(degrees=5),   # 旋转角度降低至5°，避免数字识别失效
        tv_transforms.ToTensor(),
        # MNIST专属归一化参数（单通道，官方统计值）
        tv_transforms.Normalize((0.1307,), (0.3081,)),
    ])

# ========== 核心修改3：重写DataLoader_1280，适配MNIST ==========
def DataLoader_1280(batch_size=32, num_workers=4, input_size=28, Train_num=100, Test_num=20, category=10):
    """
    返回MNIST的无标签 train / test DataLoader（双视图训练）。
    每张图返回 [view1, view2]，后续训练直接用 images[0], images[1]。
    参数说明：
        input_size: MNIST输入尺寸（默认28，标准尺寸）
        Train_num * category: 训练集总样本数（随机选取）
        Test_num * category: 测试集总样本数（随机选取）
        category: MNIST固定10类数字
    """
    g = torch.Generator()
    g.manual_seed(0)  # 固定随机种子

    # ---- 训练集：双视图增强 ----
    aug_train = get_simclr_augment_mnist(input_size)  # 调用MNIST增强
    train_full = tv_datasets.MNIST(
        root=Path,
        train=True,
        download=True,  # 自动下载（路径不存在时创建）
        transform=TwoCropsTransform(aug_train)  # 双视图生成
    )

    # 随机采样训练集样本
    total_idx = np.arange(len(train_full))
    selected_idx = np.random.choice(total_idx, size=Train_num * category, replace=False)
    train_set = Subset(train_full, selected_idx)
    print(f"MNIST训练集样本数量: {len(train_set)}")  # 验证样本数

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    # ---- 测试集：单视图（无增强）+ 标签（用于评估） ----
    test_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # 轻微放大后中心裁剪（保持尺寸一致）
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        tv_transforms.Normalize((0.1307,), (0.3081,)),  # 同步MNIST归一化参数
    ])

    test_full = tv_datasets.MNIST(
        root=Path,
        train=False,
        download=True,
        transform=test_transform
    )

    # 随机采样测试集样本
    test_idx = np.arange(len(test_full))
    selected_test_idx = np.random.choice(test_idx, size=Test_num * category, replace=False)
    test_set = Subset(test_full, selected_test_idx)
    print(f"MNIST测试集样本数量: {len(test_set)}")  # 验证样本数

    test_loader = DataLoader(
        test_set,
        batch_size=Test_num * category,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    # ========== 核心修改4：MNIST类别名称（10类数字） ==========
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return train_loader, test_loader, classes


