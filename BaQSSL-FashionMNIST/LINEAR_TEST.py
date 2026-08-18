import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tv_transforms
from torch.utils.data import Subset, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import pennylane as qml
from pennylane.qnn import TorchLayer


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cb_quantum_circuit(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    k = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    beta_q_raw = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    return k + [beta_q_raw]

def style_perturb_circuit(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumWrapper:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.__name__ = "cb_quantum_circuit_wrapper"

    def __call__(self, inputs, weights):
        return cb_quantum_circuit(inputs, weights, self.n_qubits)

class PerturbCircuitWrapper:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.__name__ = "style_perturb_circuit"

    def __call__(self, inputs, weights):
        return style_perturb_circuit(inputs, weights, self.n_qubits)

class PQCStylePerturb(nn.Module):
    def __init__(self, in_dim, n_qubits=6, n_perturb=3, device='cuda:0'):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_perturb = n_perturb
        self.device = device





def get_qlayer_output_dim(qlayer, input_dim, device):
    test_input = torch.randn(1, input_dim).to(device, dtype=torch.float32)
    with torch.no_grad():
        output = qlayer(test_input)
    return output.size(-1)

class QuantumFeatHead(nn.Module):
    def __init__(self, classic_in_dim, q_input_dim, qlayer, device, use_EA=True, eps=0.1):
        super().__init__()
        self.use_EA = use_EA
        self.eps = eps
        self.device = device
        self.pre_linear = nn.Linear(classic_in_dim, q_input_dim).to(device)
        self.ea = nn.Linear(q_input_dim, q_input_dim, bias=False).to(device)
        nn.init.zeros_(self.ea.weight)
        self.q_layer = qlayer.to(device)
        self.q_output_dim = get_qlayer_output_dim(qlayer, q_input_dim, device)
        self.proj = nn.Sequential(
            nn.Linear(self.q_output_dim, 128).to(device),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128).to(device)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.pre_linear(x)
        if self.use_EA:
            x = x + self.ea(x) * self.eps
        z = self.q_layer(x)
        z = self.proj(z)
        return nn.functional.normalize(z, dim=1)

INPUT_SIZE = 10

def get_simclr_augment(input_size=INPUT_SIZE):
    return tv_transforms.Compose([
        tv_transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        tv_transforms.RandomHorizontalFlip(p=0.5),
        tv_transforms.RandomRotation(degrees=15),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.2860,), (0.3530,)),
    ])

def get_val_transform(input_size=INPUT_SIZE):
    return tv_transforms.Compose([
        tv_transforms.Resize(int(input_size * 1.14)),
        tv_transforms.CenterCrop(input_size),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.2860,), (0.3530,)),
    ])



class LinearProbe(nn.Module):
    def __init__(self, encoder, feature_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feature_dim, num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)
        logits = self.fc(feat)
        return logits

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.encoder.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    n_perturb = 3
    n_qubits = 6

    model_path = "./Models_ddl_qhalf_10_pqcturb/model_86.pt"
    assert os.path.exists(model_path), f"未找到预训练模型：{model_path}"

    pretrain_model = torch.load(model_path, map_location=device, weights_only=False)
    pretrain_model.to(device)
    pretrain_model.eval()
    print(f"✅ 预训练模型加载完成：{model_path}")

    with torch.no_grad():
        test_input = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE).to(device)
        test_feat = pretrain_model(test_input)
        print(f"✅ 编码器输出维度验证：{test_feat.shape}（应为[1, 128]）")
        assert test_feat.shape == (1, 128), "编码器输出维度不匹配，检查模型结构"

    linear_probe_model = LinearProbe(
        encoder=pretrain_model,
        feature_dim=128,
        num_classes=2
    ).to(device)

    train_set, test_set = load_fmnist01_datasets(
        train_samples_per_class=500,
        test_samples_per_class=100
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        linear_probe_model.fc.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    num_epochs = 80
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    save_dir = "./linear_probe_quantum_cb_pqcturb_10"
    os.makedirs(save_dir, exist_ok=True)
    log_path = f"{save_dir}/training_log_pqcturb_fmnist01.txt"

    with open(log_path, "w") as f:
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tTest_Loss\tTest_Acc\n")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                linear_probe_model, train_loader, optimizer, criterion, device
            )
            test_loss, test_acc = evaluate(
                linear_probe_model, test_loader, criterion, device
            )

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            log_line = f"{epoch}\t{train_loss:.6f}\t{train_acc:.6f}\t{test_loss:.6f}\t{test_acc:.4f}"
            f.write(log_line + "\n")
            print(f"[Epoch {epoch:03d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                  f"Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.4f}")

        final_model_path = f"{save_dir}/linear_probe_pqcturb_fmnist01.pt"
        torch.save(linear_probe_model, final_model_path)
        print(f"✅ 线性探测模型已保存：{final_model_path}")
