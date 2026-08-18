import time
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision import datasets as tv_datasets, transforms as tv_transforms
from torchvision.transforms import ToPILImage
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset

# ========== 核心修改1：Fashion-MNIST数据集路径（自动创建子文件夹） ==========
Path = '../FashionMNIST/'

# ========== 双视图增强类（无需修改，与数据集无关） ==========
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]  # 返回 [view1, view2] 双视图

# ========== 核心修改2：适配Fashion-MNIST的SimCLR风格增强 ==========
def get_simclr_augment_fashionmnist(input_size=28):
    """Fashion-MNIST专属双视图增强（单通道灰度图，衣物适配）"""
    return tv_transforms.Compose([
        tv_transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),  # 随机裁剪
        tv_transforms.RandomHorizontalFlip(p=0.5),  # 衣物水平翻转合理（不改变语义）
        tv_transforms.RandomRotation(degrees=15),   # 轻微旋转（适配衣物姿态变化）
        tv_transforms.ToTensor(),
        # Fashion-MNIST专属归一化参数（单通道，官方统计值）
        tv_transforms.Normalize((0.2860,), (0.3530,)),
    ])

# ========== 核心修改3：重写DataLoader_1280，适配Fashion-MNIST ==========
def DataLoader_1280(batch_size=32, num_workers=4, input_size=28, Train_num=100, Test_num=20, category=10):

    g = torch.Generator()
    g.manual_seed(0)  # 固定随机种子

    # ---- 训练集：双视图增强 ----
    aug_train = get_simclr_augment_fashionmnist(input_size)  # 调用Fashion-MNIST增强
    train_full = tv_datasets.FashionMNIST(
        root=Path,
        train=True,
        download=True,  # 自动下载（路径不存在时创建）
        transform=TwoCropsTransform(aug_train)  # 双视图生成
    )

    # 随机采样训练集样本
    total_idx = np.arange(len(train_full))
    selected_idx = np.random.choice(total_idx, size=Train_num * category, replace=False)
    train_set = Subset(train_full, selected_idx)
    print(f"Fashion-MNIST训练集样本数量: {len(train_set)}")  # 验证样本数

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
        tv_transforms.Normalize((0.2860,), (0.3530,)),  # 同步训练集归一化
    ])

    test_full = tv_datasets.FashionMNIST(
        root=Path,
        train=False,
        download=True,
        transform=test_transform
    )

    # 随机采样测试集样本
    test_idx = np.arange(len(test_full))
    selected_test_idx = np.random.choice(test_idx, size=Test_num * category, replace=False)
    test_set = Subset(test_full, selected_test_idx)
    print(f"Fashion-MNIST测试集样本数量: {len(test_set)}")  # 验证样本数

    test_loader = DataLoader(
        test_set,
        batch_size=Test_num * category,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    # ========== 核心修改4：Fashion-MNIST类别名称（10类衣物） ==========
    classes = (
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    )

    return train_loader, test_loader, classes

# ========== 准确率计算函数（适配Fashion-MNIST，无需修改逻辑） ==========
def CaculateAccuracy(TestLoader, device, Net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in TestLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = Net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Fashion-MNIST测试集中的准确率为: {correct / total * 100:.5f}%')

# ========== 主函数测试 ==========
if __name__ == "__main__":
    # 加载Fashion-MNIST数据（batch_size=100，默认input_size=28）
    train_loader, test_loader, classes = DataLoader_1280(batch_size=100)
    # 打印Fashion-MNIST类别
    print(f"Fashion-MNIST类别: {classes}")

    # 可选：验证数据格式（查看第一个batch的双视图形状）
    for (views, _) in train_loader:
        view1, view2 = views[0], views[1]
        print(f"View1形状: {view1.shape} (batch_size, channel, height, width)")
        print(f"View2形状: {view2.shape}")
        break
